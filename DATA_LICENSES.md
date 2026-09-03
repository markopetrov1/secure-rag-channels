# Data provenance and licensing

The MIT licence in `LICENSE` covers the code in this repository. It does not
cover the benchmark, the corpus or the model weights, which belong to their
respective owners and carry their own terms. Nothing third-party is
redistributed here. `scripts/build_corpora.sh` fetches the source from its
original location so that anyone can rebuild an identical corpus and index.

## Benchmark

| Source | Licence | Redistributed here |
|---|---|---|
| ObliQA / RIRAG (`RegNLP/ObliQADataset`) | No licence file present | No, cloned at build time |

ObliQA carries no licence file in its repository. It is cloned rather than
vendored, and anyone intending to redistribute it or build on it commercially
should contact the RegNLP maintainers first. The questions are drawn from the
published rulebooks of the Abu Dhabi Global Market, and each ships with the gold
regulatory passages that answer it, which is what allows retrieval here to be
scored against an external target rather than against a judge.

The sampling and cleaning applied on top of the released files is in
`src/prepare_obliqa.py`, and the resulting per-split counts are recorded in
`results/obliqa_prep_stats.json`.

## A benchmark this study does not use

An earlier draft of this work carried a second track built on the ISO27K-QnA
dataset (`dimitarjovanovski/ISO27K-QnA-Benchmark-dataset`), and that track has
been withdrawn. Its dataset card claims CC BY 4.0 in prose, with no licence file
and no machine-readable tag, while also stating that the items were manually
extracted and constructed from nine named commercial exam-preparation books.
Questions derived from copyrighted books are not the extractor's to relicense,
so the claim cannot be relied on. No item, answer, generation or verdict from
that benchmark is present in this repository, and no reported result depends
on it. It is recorded here because a reader comparing this repository
against the published predecessors of this study will find that track in them
and should know why it is absent from this one.

The normative text of ISO/IEC 27001 and 27002 is copyrighted and paywalled. It
is quoted nowhere in this repository.

## Model weights

Each model is pulled from its own distributor under its own licence. None are
redistributed here.

| Role | Model | Family |
|---|---|---|
| Generator | `qwen3:8b` | Alibaba |
| Generator | `llama3.1:8b` | Meta |
| Embedder | `bge-m3` | BAAI |
| Judge | `gemma3:12b` | Google |
| Judge | `gpt-oss:20b` | OpenAI |
| Judge | `phi4:14b` | Microsoft |

No judge shares a model family with either generator. Judges prefer text whose
distribution resembles their own, so a judge grading its own family's output
would flatter it.

## What this repository does contain

Code, configuration, prompts, the token ledger, model-generated answers, judge
verdicts, the two authors' blinded human annotations, and the derived tables and
figures. The generated answers are the models' own text and record only passage
identifiers, never retrieved passage text, so the source corpus cannot be
reconstructed from them.
