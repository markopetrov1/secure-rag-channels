"""Run generation experiments: arm x generator, resumable.

Usage:
  python src/run_generation.py --track obliqa --mode open --arms a1,a2,a4,a5 \
      --generators qwen3:8b,llama3.1:8b

The study runs one benchmark in one mode, but track and mode stay in the run id
because every downstream reader recovers them by splitting the filename, and
because a second benchmark would otherwise overwrite the first arm for arm.

Outputs one JSONL per (track, mode, arm, generator) under results/generations/.
"""
import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine import OllamaClient
import agentic
import arms as A

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "results/generations"
SEED = 42
K_FEWSHOT = 5
TOP_K = 5

_retriever_cache = {}
_doc_texts_cache = {}


def sanitize(model):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model)


def load_obliqa():
    test = [json.loads(l) for l in
            (ROOT / "data/processed/obliqa_test_sample.jsonl").open()]
    fewshot = [json.loads(l) for l in
               (ROOT / "data/processed/obliqa_fewshot_dev.jsonl").open()]
    # exemplar reference answer = concatenated gold passages, lightly framed
    for ex in fewshot:
        ex["reference_answer"] = " ".join(
            p["Passage"] for p in ex["gold_passages"])[:1200]
    return test, fewshot[:K_FEWSHOT]


def get_retriever(track):
    if track not in _retriever_cache:
        from retrieval import Retriever
        _retriever_cache[track] = Retriever(track)
    return _retriever_cache[track]


def get_doc_texts(track):
    """Full-document texts for the long-context arm, plus a BM25 doc ranker."""
    if track in _doc_texts_cache:
        return _doc_texts_cache[track]
    from rank_bm25 import BM25Okapi
    meta = [json.loads(l) for l in
            (ROOT / f"data/index/{track}_meta.jsonl").open()]
    docs = {}
    for r in meta:
        docs.setdefault(str(r["doc_id"]), []).append(r["text"])
    names = sorted(docs, key=lambda d: int(d) if d.isdigit() else 0)
    texts = {n: "\n".join(docs[n]) for n in names}
    tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
    bm25 = BM25Okapi([tok(texts[n]) for n in names])
    _doc_texts_cache[track] = (names, texts, bm25, tok)
    return _doc_texts_cache[track]


def build_prompt(arm, q, track, exemplars, run_id, client):
    if arm == "a1":
        return A.a1_zero_shot(q)
    if arm == "a2":
        return A.a2_few_shot(q, exemplars)
    if arm == "a2o":
        # One-shot. Same template and the same held-out pool as a2, truncated to a
        # single demonstration, so the only thing that varies across a1, a2o and
        # a2 is the number of demonstrations. That makes the in-context ladder a
        # dose-response curve rather than two unrelated points.
        return A.a2_few_shot(q, exemplars[:1])
    if arm == "a3":
        names, texts, bm25, tok = get_doc_texts(track)
        scores = bm25.get_scores(tok(q["question"]))
        order = sorted(range(len(names)), key=lambda i: -scores[i])
        ranked = [(f"Doc {names[i]}", texts[names[i]]) for i in order[:3]]
        return A.a3_long_context(q, ranked)
    if arm in ("a4", "a5"):
        r = get_retriever(track)
        if arm == "a4":
            passages = r.dense_topk(q["question"], TOP_K,
                                    purpose="query:retrieve", run_id=run_id)
        else:
            passages = r.hybrid_topk(q["question"], TOP_K,
                                     purpose="query:retrieve", run_id=run_id)
        return A.a45_rag(q, passages)
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["obliqa"])
    ap.add_argument("--mode", required=True, choices=["open"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--generators", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests. Token counts are unaffected by "
                         "concurrency, but per-record wall_s is, so records "
                         "produced with workers>1 are marked concurrent=True "
                         "and excluded from latency reporting; latency comes "
                         "from bench_latency.py instead.")
    args = ap.parse_args()

    test, exemplars = load_obliqa()
    if args.limit:
        test = test[:args.limit]

    client = OllamaClient()
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    for gen_model in args.generators.split(","):
        for arm in args.arms.split(","):
            run_id = f"{args.track}_{args.mode}_{arm}_{sanitize(gen_model)}"
            out = GEN_DIR / f"{run_id}.jsonl"
            done = set()
            if out.exists():
                done = {json.loads(l)["qid"] for l in out.open()}
            todo = [q for q in test if q["qid"] not in done]
            print(f"== {run_id}: {len(todo)} to do ({len(done)} done)", flush=True)
            num_ctx = 32768 if arm == "a3" else 8192
            num_predict = 512
            lock = threading.Lock()
            counter = {"n": 0}

            def do_one(q):
                try:
                    if arm in agentic.RUNNERS:
                        # The agent runs its own loop and returns usage summed
                        # over every call the question cost, so a multi-call arm
                        # is priced on what it actually spent.
                        text, usage, meta = agentic.run(
                            arm, q, args.track, gen_model, run_id,
                            client, get_retriever(args.track))
                    else:
                        msgs, meta = build_prompt(arm, q, args.track,
                                                  exemplars, run_id, client)
                        think = False if "qwen3" in gen_model else None
                        text, usage = client.chat(
                            gen_model, msgs, purpose="query:generate", run_id=run_id,
                            num_ctx=num_ctx, num_predict=num_predict, think=think)
                except Exception as e:
                    print(f"  error qid={q['qid']}: {e}", flush=True)
                    return None
                rec = {"qid": q["qid"], "arm": arm, "mode": args.mode,
                       "track": args.track, "generator": gen_model,
                       "answer": text.strip(), "usage": usage, "meta": meta,
                       "concurrent": args.workers > 1}
                return rec

            with out.open("a") as f:
                def sink(rec):
                    if rec is None:
                        return
                    with lock:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        counter["n"] += 1
                        if counter["n"] % 20 == 0:
                            print(f"  {counter['n']}/{len(todo)} "
                                  f"({rec['usage'].get('wall_s', 0):.1f}s/q)",
                                  flush=True)

                if args.workers > 1:
                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        for rec in ex.map(do_one, todo):
                            sink(rec)
                else:
                    for q in todo:
                        sink(do_one(q))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
