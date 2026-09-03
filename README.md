# Prompt channels bypass access control in retrieval-augmented generation

Code and data for a study of access control in retrieval-augmented generation.
We construct a multi-level security policy over 13,012 passages of Abu Dhabi
Global Market financial regulation, hold that policy fixed, and vary the
generation paradigm across nine deployed pipelines to ask what each of them
actually puts in front of a reader who is cleared for part of the corpus.

The finding is not about the retriever. Every enforcement mechanism in the
literature is a predicate over retrieval results, and every metric is computed
over the passages a retriever returns. That is sound only if the retriever is
the sole route by which corpus text reaches the model, and in two of these nine
pipelines it is not.

| Channel | Carrier | Instruments | Tokens | Reader authorised for all of it |
|---|---|---|---|---|
| exemplar | few-shot pool | 7 | 996 | **0.42%** |
| compiled | dspy llama3.1 seed0 | 5 | 1,512 | 2.36% |
| compiled | dspy qwen3 seed0 | 4 | 2,658 | 5.30% |
| compiled | dspy qwen3 seed1 | 7 | 3,513 | 0.42% |
| compiled | dspy qwen3 seed2 | 7 | 3,491 | 0.42% |

The few-shot arm builds its demonstrations by concatenating the gold supporting
passages of training questions. That is verbatim regulatory text, assembled once
when the pipeline is deployed and sent identically to every reader. The compiled
arm is worse in a specific way: a prompt optimiser selects demonstrations during
compilation and freezes them, so which instruments a deployment exposes is
decided by a random seed before any reader exists. Both score perfectly on
retrieval-side authorisation correctness while emitting regulatory text their
reader is not cleared to see on every query.

Attribution is by exact eight-word match against the corpus, of which 95.8
percent of sequences occur in only one instrument, so a match identifies a
source rather than a topic.

## What the retriever results are, and are not

Three properties of the standard mechanisms follow from their definitions, and
they are stated as propositions with proofs rather than reported as
findings:

- On questions whose supporting evidence a reader may see, post-filtering cannot
  remove that evidence, so its recall equals that of no enforcement at all. Over
  five embedders, two labellings and five clearance breadths, fifty conditions,
  the measured difference is exactly zero in every one.
- Partitioned per-compartment indexes return under exact search what
  pre-filtering returns.
- Post-filtering with enough over-fetch is pre-filtering.

What is measured instead is what the propositions leave open. Post-filtering's
price is a thinner prompt, 3.28 of five passages at a tenth clearance, which
makes it the only mechanism that costs less per query than enforcing nothing, 39
percent fewer context tokens. Pre-filtering costs 9 percent more and buys 0.054
recall at a tenth clearance, falling to 0.005 at nine tenths. And the real
utility cost belongs to the policy rather than to any mechanism: at a tenth
clearance only 9.0 percent of questions have all their supporting evidence
authorised at all.

## From context to answer

A passage in the context window is a risk only if the model repeats it.
Measuring that under every clearance would mean regenerating thousands of times.
It is avoidable: whether an answer discloses a passage is a property of the
answer and the passage, and the clearance decides only whether that passage was
one the reader may see. So disclosure is judged once per pair with no clearance
named in the prompt, and leakage under any policy is recovered afterwards by
intersecting the matrix with the authorised set.

The answer differs sharply by route. A retrieved passage is repeated 57.5
percent of the time overall, 88.7 percent when it supports the answer and 52.2
percent when it was merely retrieved. A demonstration is repeated 2.0 percent of
the time, judged with the same rubric on the same answers.

That twenty-eight-fold gap matters for how these systems are assessed. Every
measure in the access-control literature counts what enters the prompt, and on
these results such a measure ranks the two channels in the wrong order: it
reports the demonstration channel as a large continuous exposure, which it is,
while missing that almost none of it reaches the reader. The two channels also
differ in relevance, since retrieved passages are selected for the question and
demonstrations are fixed at deployment, and this study cannot separate that from
position in the prompt.

## The policy

The corpus is public regulation and carries no classification, so one is
imposed, and its structure comes from the publisher rather than from us.
Compartments are the forty ADGM instruments, so a clearance is a coherent
need-to-know set. Levels come from the instrument taxonomy the dataset's own
document map records:

| Level | Instrument type | Documents | Passages |
|---|---|---|---|
| L0 | Regulations, binding on everyone | 5 | 2,320 |
| L1 | Rulebooks, binding on firm categories | 16 | 9,149 |
| L2 | Guidance, narrow activities | 19 | 1,543 |

A passage is readable when its level does not exceed the clearance and its
compartment is granted, which is no-read-up intersected with need-to-know. We do
not claim ADGM classifies these documents and nothing here depends on it doing
so. Clearance breadth is swept rather than fixed, and every headline number is
repeated under two further labellings we did not construct: one derived from
whom the regulatory text says it binds, and the topic classification the dataset
authors released.

## Reproducing it

Everything on the retrieval and channel side runs on CPU from cached embeddings.
No GPU, no model weights, no corpus download.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python src/access_control.py --embedders bge-m3 --cost   # enforcement sweep
.venv/bin/python src/overfetch.py                                  # over-fetch depth
.venv/bin/python src/prompt_channels.py                            # the channel decomposition
.venv/bin/python src/leakage.py                                    # leakage at every clearance
.venv/bin/python src/enforced_quality.py                           # answer quality under enforcement
./scripts/run_analysis.sh                                          # the whole chain
```

`run_analysis.sh` reports each stage's exit status and prints `ANALYSIS_DONE`
only when every one succeeded, so a partial failure cannot pass quietly. It
rewrites every table under `results/tables/` and every figure under `figures/`
from the committed generations, verdicts and ledger. Comparing what it writes
against what is committed is the check that a clone is intact, and
`src/results_facts.py` prints every measured number beside the table it came
from when you want to check one figure rather than rerun the chain.

Regenerating the disclosure matrix needs a GPU and the judge weights:

```bash
PROJECT_GPU=1 ./scripts/start_ollama.sh &                          # pin to a free device
.venv/bin/python src/disclosure.py --arm a4 --validate 200         # three-judge agreement gate
.venv/bin/python src/disclosure.py --arm a4 --workers 4            # the 5,000-pair matrix
```

The gate exists because the disclosure rubric must be shown to be more
reproducible than the correctness panel before a single judge is trusted with
it. Three judges over 200 pairs give pairwise Cohen's kappa of 0.625 to 0.713,
against 0.404 to 0.462 for the correctness panel on the same corpus.

## Layout

```
src/
  Policy and enforcement
    access_control.py         the lattice, four enforcement mechanisms, the sweep
    prompt_channels.py        which channels carry corpus text, and to which instrument
    disclosure.py             the clearance-independent disclosure matrix
    leakage.py                leakage at any clearance, recomputed on CPU
    run_enforced.py           generation under each enforcement mechanism

  The pipelines under test
    engine.py                 ollama client and the per-call token ledger
    retrieval.py              dense, BM25 and reciprocal-rank-fusion retrieval
    arms.py                   prompt construction for the fixed-schedule arms
    agentic.py                the two agentic arms and their step budget
    run_generation.py         generation driver, resumable, optionally concurrent
    run_dspy.py               the compiled arm, MIPROv2
    run_judging.py            the three-judge correctness panel
    judge.py                  grading rubric and verdict parsing

  Analysis and output
    analyze.py stats.py economics.py dominance.py retrieval_stats.py
    judge_validity.py check_annotations.py compute_budget.py
    figures.py figures_access_control.py results_facts.py summarize_results.py

scripts/      pipeline entry points, GPU guard, ollama launcher
results/      generations, judgments, disclosure matrix, token ledger, tables
```

## GPU etiquette

`scripts/gpu_guard.sh` refuses to start work on a device held by a job this
project did not start, and `scripts/start_ollama.sh` takes the device as a
parameter, because which GPU is free changes. Both learned this the hard way.
The guard once matched a PyTorch training job as its own server, because
`pgrep -f` matches the whole command line and the word llama appeared in the
job's arguments, and it reported a device free while an 18.6 GB tenant was on
it. It now matches the executable name. `nvidia-smi --query-compute-apps` also
hangs on some drivers, emitting the first device's rows and then blocking, so
every call is wrapped in a timeout and a guard that cannot read a device refuses
rather than assuming it is free.

`ALLOW_SHARED=1` shares a device deliberately and prints the utilisation when it
does, because sharing is a judgement about someone else's work and should be on
the record.

## Data

The corpus is not redistributed. ObliQA ships no licence file, so it is cloned
at build time; see [DATA_LICENSES.md](DATA_LICENSES.md). The generated answers
record only passage identifiers and never retrieved passage text.

