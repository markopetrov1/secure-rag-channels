"""Arm A7: DSPy-optimized RAG (MIPROv2-compiled ChainOfThought over the same
dense retriever as A4). Compile-time LM/embedding tokens are ledger-accounted
as setup cost; evaluation calls as query cost.

Usage:
  python src/run_dspy.py --track obliqa --generator qwen3:8b --stage compile --seed 0
  python src/run_dspy.py --track obliqa --generator qwen3:8b --stage eval --seed 0
"""
import argparse
import json
import random
import re
import threading
import time
from pathlib import Path

import numpy as np
import dspy
import litellm
from dspy.clients.base_lm import GLOBAL_HISTORY
from dspy.utils.callback import BaseCallback

from engine import OllamaClient
from retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
CLIENT = OllamaClient()
TOP_K = 5
N_TRAIN, N_VAL = 120, 60

# Ledger labels for whichever stage is running. These are read by the retriever
# call inside RAGProgram.forward, by the metric, and by the LM-history flush.
# They are module-level rather than program attributes because MIPROv2 deep-copies
# the student program, which would detach any label the program carried, and
# because the flush runs from a dspy callback with no access to our call stack.
LABEL = {"run_id": "dspy", "lm_purpose": "setup:dspy_compile",
         "retrieve_purpose": "setup:dspy_compile"}

# Running totals of what has been flushed, so the eval loop can attribute the
# tokens of one query without re-reading history the callback may already have
# consumed.
TOTALS = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

_logged = set()
_flush_lock = threading.Lock()


def sanitize(m):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", m)


def make_lm(generator):
    # api_key is a placeholder, not a credential. DSPy speaks the OpenAI wire
    # format and its client refuses to start without the field, while the local
    # ollama server on 11435 never reads it. Nothing in this project
    # authenticates against anything.
    kw = dict(api_base="http://127.0.0.1:11435", api_key="x",
              temperature=0.0, max_tokens=1024, cache=False)
    if "qwen3" in generator:
        kw["think"] = False
    return dspy.LM(f"ollama_chat/{generator}", **kw)


def count_tokens(model, text):
    """Tokeniser count used only when the provider returned no usage block."""
    if not text:
        return 0
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        # Better a coarse count than a zero that silently deflates setup cost.
        return max(1, len(text) // 4)


def estimate_usage(h):
    """Token counts for a history entry whose usage block came back empty."""
    model = h.get("model") or ""
    msgs = h.get("messages") or []
    try:
        if msgs:
            pt = litellm.token_counter(model=model, messages=msgs)
        else:
            pt = count_tokens(model, h.get("prompt") or "")
    except Exception:
        # Multimodal or otherwise unexpected message shapes still have to cost
        # something, so fall back to the joined text.
        pt = count_tokens(model, " ".join(
            str(m.get("content", "")) for m in msgs))
    outs = h.get("outputs") or []
    ct = count_tokens(model, " ".join(
        o if isinstance(o, str) else str(o) for o in outs))
    return pt, ct


def ledger_flush():
    """Log every dspy LM call not yet logged, under the current LABEL.

    dspy 3.2 appends each call to a process-wide list (dspy.clients.base_lm
    GLOBAL_HISTORY) as well as to the calling LM's own .history. MIPROv2 proposes
    instructions through prompt_model.copy(), which returns an LM with a fresh
    empty .history, so an offset into one LM's history loses every instruction
    proposal. Deduplicating the global list on the per-entry uuid catches calls on
    any LM instance, in any thread, and stays correct if a flush overlaps another.
    """
    with _flush_lock:
        fresh = [h for h in GLOBAL_HISTORY if h.get("uuid") not in _logged]
        for h in fresh:
            _logged.add(h.get("uuid"))
        purpose, run_id = LABEL["lm_purpose"], LABEL["run_id"]
        for h in fresh:
            u = h.get("usage") or {}
            pt = u.get("prompt_tokens") or 0
            ct = u.get("completion_tokens") or 0
            source = "measured"
            if not pt and not ct:
                pt, ct = estimate_usage(h)
                source = "estimated"
            TOTALS["prompt_tokens"] += pt
            TOTALS["completion_tokens"] += ct
            TOTALS["calls"] += 1
            CLIENT._log({"kind": "chat", "model": h.get("model"),
                         "purpose": purpose, "run_id": run_id,
                         "prompt_tokens": pt, "completion_tokens": ct,
                         "token_source": source, "wall_s": 0.0})
    return len(fresh)


class LedgerCallback(BaseCallback):
    """Flush after every LM call rather than only at the end of a stage.

    A compile that raises part-way (as the 2026-07-31 run did) has still spent
    the tokens, and dspy caps GLOBAL_HISTORY at 10,000 entries by dropping the
    oldest, so a long compile flushed only at the end could lose both.
    """

    def on_lm_end(self, call_id, outputs, exception):
        ledger_flush()


class RAGProgram(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        ps = self.retriever.dense_topk(
            question, TOP_K, purpose=LABEL["retrieve_purpose"],
            run_id=LABEL["run_id"])
        ctx = "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(ps))
        pred = self.answer(context=ctx, question=question)
        pred.retrieved_ids = [p.get("uid") for p in ps]
        return pred


def sem_metric(example, pred, trace=None):
    """bge-m3 cosine similarity between predicted answer and gold text."""
    a = (pred.answer or "").strip()
    if not a:
        return 0.0
    e = CLIENT.embed("bge-m3", [a, example.gold_text],
                     purpose="setup:dspy_compile:metric",
                     run_id=LABEL["run_id"])
    v = np.asarray(e, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return float(v[0] @ v[1])


def load_train():
    train = json.load((ROOT / "data/raw/ObliQADataset/ObliQA_train.json").open())
    rng = random.Random(123)
    pick = rng.sample(train, N_TRAIN + N_VAL)
    exs = [dspy.Example(
        question=q["Question"],
        gold_text=" ".join(p["Passage"] for p in q["Passages"])[:2000],
    ).with_inputs("question") for q in pick]
    return exs[:N_TRAIN], exs[N_TRAIN:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="obliqa", choices=["obliqa"])
    ap.add_argument("--generator", required=True)
    ap.add_argument("--stage", required=True, choices=["compile", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    gen_s = sanitize(a.generator)
    prog_path = ROOT / f"results/dspy/{a.track}_{gen_s}_seed{a.seed}.json"
    prog_path.parent.mkdir(parents=True, exist_ok=True)
    lm = make_lm(a.generator)
    dspy.configure(lm=lm, callbacks=[LedgerCallback()])
    retriever = Retriever(a.track, client=CLIENT)
    program = RAGProgram(retriever)

    if a.stage == "compile":
        # economics.py parses this run_id as dspy_compile_<track>_<gen>_seed<k>.
        LABEL["run_id"] = f"dspy_compile_{a.track}_{gen_s}_seed{a.seed}"
        LABEL["lm_purpose"] = "setup:dspy_compile"
        LABEL["retrieve_purpose"] = "setup:dspy_compile"
        trainset, valset = load_train()
        t0 = time.time()
        opt = dspy.MIPROv2(metric=sem_metric, auto="light", num_threads=1,
                           seed=a.seed)
        try:
            compiled = opt.compile(program, trainset=trainset, valset=valset,
                                   requires_permission_to_run=False)
        finally:
            ledger_flush()
        compiled.save(str(prog_path))
        print(f"COMPILE_DONE seed={a.seed} in {time.time()-t0:.0f}s "
              f"-> {prog_path}", flush=True)
        return

    # eval
    arm_name = "a7" if a.seed == 0 else f"a7s{a.seed}"
    # As elsewhere in the repo the run_id is the output file stem, which keeps the
    # query cost of each optimiser seed separable instead of pooling all three.
    LABEL["run_id"] = f"{a.track}_open_{arm_name}_{gen_s}"
    LABEL["lm_purpose"] = "query:generate"
    LABEL["retrieve_purpose"] = "query:retrieve"
    program.load(str(prog_path))
    test = [json.loads(l) for l in
            (ROOT / "data/processed/obliqa_test_sample.jsonl").open()]
    if a.limit:
        test = test[:a.limit]
    out = ROOT / f"results/generations/{a.track}_open_{arm_name}_{gen_s}.jsonl"
    done = {json.loads(l)["qid"] for l in out.open()} if out.exists() else set()
    todo = [q for q in test if q["qid"] not in done]
    print(f"== dspy eval: {len(todo)} to do ({len(done)} done)", flush=True)
    with out.open("a") as f:
        try:
            for i, q in enumerate(todo):
                t0 = time.time()
                before = dict(TOTALS)
                pred = program(question=q["question"])
                ledger_flush()
                # Per-query usage from the ledger totals, since ChainOfThought can
                # issue more than one LM call for one question.
                rec = {"qid": q["qid"], "arm": arm_name, "mode": "open",
                       "track": a.track, "generator": a.generator,
                       "answer": (pred.answer or "").strip(),
                       "usage": {"prompt_tokens": TOTALS["prompt_tokens"]
                                 - before["prompt_tokens"],
                                 "completion_tokens": TOTALS["completion_tokens"]
                                 - before["completion_tokens"],
                                 "wall_s": time.time() - t0},
                       "meta": {"retrieved_ids": pred.retrieved_ids,
                                "dspy_seed": a.seed}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 20 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        finally:
            ledger_flush()
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
