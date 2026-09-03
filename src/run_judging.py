"""Batch multi-judge evaluation over generation files.

For every record in the given generation JSONLs, obtains a correctness verdict
from each judge (resumable per (qid, judge)).

Usage:
  python src/run_judging.py --files "results/generations/obliqa_open_*.jsonl"
  python src/run_judging.py --files "..." --judges gemma3:12b
"""
import argparse
import glob
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engine import OllamaClient
from judge import judge_correctness

ROOT = Path(__file__).resolve().parent.parent
JUDGES = ["gemma3:12b", "gpt-oss:20b", "phi4:14b"]


def sanitize(m):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", m)


def load_references():
    """qid -> (question, reference_text).

    The dataset ships no answer strings, so the reference a judge grades
    against is the set of gold supporting passages the question came with.
    """
    refs = {}
    for l in (ROOT / "data/processed/obliqa_test_sample.jsonl").open():
        r = json.loads(l)
        gold = "\n".join(f"- {p['Passage']}" for p in r["gold_passages"])
        refs[r["qid"]] = {"question": r["question"], "reference": gold}
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent judge requests (safe: judging runs after "
                         "all generation timings are collected)")
    args = ap.parse_args()

    client = OllamaClient()
    out_dir = ROOT / "results/judgments"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(ROOT / args.files)))
    print(f"{len(files)} generation files", flush=True)
    refs = load_references()

    def unload_others(current):
        """Free VRAM: unload every model except the current judge."""
        import requests as rq
        try:
            ps = rq.get("http://127.0.0.1:11435/api/ps", timeout=10).json()
            for m in ps.get("models", []):
                if m["name"] != current:
                    rq.post("http://127.0.0.1:11435/api/generate",
                            json={"model": m["name"], "keep_alive": 0}, timeout=30)
        except Exception as e:
            print(f"  (unload_others: {e})", flush=True)

    # Judge-major order: one judge stays loaded across all files, then the next.
    for judge_model in args.judges.split(","):
        unload_others(judge_model)
        for fp in files:
            stem = Path(fp).stem
            if stem.split("_")[1] != "open":
                continue
            gens = [json.loads(l) for l in open(fp)]
            out = out_dir / f"{stem}__{sanitize(judge_model)}.jsonl"
            done = {json.loads(l)["qid"] for l in out.open()} if out.exists() else set()
            todo = [g for g in gens if g["qid"] in refs and g["qid"] not in done]
            if todo:
                print(f"== judge {judge_model} on {stem}: {len(todo)} to do "
                      f"({len(done)} done)", flush=True)
            lock = threading.Lock()
            counter = {"n": 0}

            def do_one(g, f=None):
                r = refs[g["qid"]]
                try:
                    v, raw, usage = judge_correctness(
                        client, judge_model, r["question"], g["answer"],
                        r["reference"], run_id=f"judge_{stem}")
                except Exception as e:
                    print(f"  judge error qid={g['qid']}: {e}", flush=True)
                    return None
                return {"qid": g["qid"], "judge": judge_model, "verdict": v,
                        "gen_file": stem, "raw_tail": raw[-200:]}

            with out.open("a") as f:
                def sink(rec):
                    if rec is None:
                        return
                    with lock:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        counter["n"] += 1
                        if counter["n"] % 25 == 0:
                            print(f"  {counter['n']}/{len(todo)}", flush=True)

                if args.workers > 1:
                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        for rec in ex.map(do_one, todo):
                            sink(rec)
                else:
                    for g in todo:
                        sink(do_one(g))

    print("JUDGING_DONE", flush=True)


if __name__ == "__main__":
    main()
