"""Controlled serial latency benchmark.

Throughput runs are executed with concurrent requests, which makes their
per-request wall time meaningless. Token counts are unaffected by concurrency,
so the economics of token cost come from the full runs, while every reported
latency comes from this benchmark instead: a fixed number of questions per
configuration, issued strictly one at a time against an otherwise idle GPU with
the server running a single slot.

Writes results/latency_bench.jsonl and results/tables/latency_bench.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine import OllamaClient
import run_generation as rg

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/latency_bench.jsonl"
N_DEFAULT = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--arms", default="a1,a2,a4,a5,a3")
    ap.add_argument("--generators", default="qwen3:8b,llama3.1:8b")
    args = ap.parse_args()

    # Kept in the record and in the run id because economics.py joins the
    # benchmark on (track, mode, arm, generator), the same key it recovers from
    # a generation filename.
    track, mode = "obliqa", "open"

    client = OllamaClient()
    done = set()
    if OUT.exists():
        for l in OUT.open():
            r = json.loads(l)
            done.add((r["track"], r["mode"], r["arm"], r["generator"], r["qid"]))

    test, exemplars = rg.load_obliqa()
    sample = test[:args.n]
    with OUT.open("a") as f:
        for gen_model in args.generators.split(","):
            for arm in args.arms.split(","):
                run_id = f"bench_{track}_{mode}_{arm}_{rg.sanitize(gen_model)}"
                num_ctx = 32768 if arm == "a3" else 8192
                pending = [q for q in sample
                           if (track, mode, arm, gen_model, q["qid"]) not in done]
                if not pending:
                    continue
                print(f"== {run_id}: {len(pending)}", flush=True)
                for q in pending:
                    try:
                        msgs, meta = rg.build_prompt(
                            arm, q, track, exemplars, run_id, client)
                        think = False if "qwen3" in gen_model else None
                        t0 = time.time()
                        text, usage = client.chat(
                            gen_model, msgs, purpose="bench:latency",
                            run_id=run_id, num_ctx=num_ctx,
                            num_predict=512, think=think)
                        usage["wall_s"] = time.time() - t0
                    except Exception as e:
                        print(f"  error {q['qid']}: {e}", flush=True)
                        continue
                    f.write(json.dumps({
                        "track": track, "mode": mode, "arm": arm,
                        "generator": gen_model, "qid": q["qid"],
                        "usage": usage}, ensure_ascii=False) + "\n")
                    f.flush()

    rows = [json.loads(l) for l in OUT.open()]
    df = pd.DataFrame([{
        "track": r["track"], "mode": r["mode"], "arm": r["arm"],
        "generator": r["generator"],
        "wall_s": r["usage"].get("wall_s", 0),
        "prompt_eval_s": r["usage"].get("prompt_eval_s", 0),
        "eval_s": r["usage"].get("eval_s", 0),
        "prompt_tokens": r["usage"].get("prompt_tokens", 0),
        "completion_tokens": r["usage"].get("completion_tokens", 0),
    } for r in rows])
    # The samples file is append-only and still holds a withdrawn benchmark's
    # runs, so the table is scoped the same way the benchmark itself is.
    df = df[df["track"] == track]
    agg = df.groupby(["track", "mode", "arm", "generator"]).agg(
        n=("wall_s", "size"),
        latency_mean_s=("wall_s", "mean"),
        latency_median_s=("wall_s", "median"),
        latency_p95_s=("wall_s", lambda x: float(np.percentile(x, 95))),
        prompt_eval_s=("prompt_eval_s", "mean"),
        decode_s=("eval_s", "mean"),
        prompt_tokens=("prompt_tokens", "mean"),
        completion_tokens=("completion_tokens", "mean"),
    ).reset_index()
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    agg.to_csv(ROOT / "results/tables/latency_bench.csv", index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
