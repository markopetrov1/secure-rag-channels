"""Generate answers under each access-control enforcement mechanism.

The propositions settle what enforcement does to retrieval. They cannot settle
what it does to an answer, because post-filtering returns fewer passages than it
was asked for and a thinner context is not the same prompt. Whether that costs
answer quality is the one question on this axis that has to be run rather than
derived.

One clearance is fixed for the whole run and recorded, rather than swept, since
sweeping would multiply generation by the number of draws. It is drawn at half
breadth on the compartment labelling with the seed below, and the retrieval
sweep reports where that draw sits in the distribution so a reader can see it is
not a favourable pick.

Arms:
  none  the existing A4 generations, already on disk, regenerated here only if
        asked, so the comparison is against the same pipeline and not a rerun
  post  rank over the whole corpus, then drop what the clearance forbids
  pre   rank only within the authorised set

Writes results/generations/obliqa_open_<arm>_<generator>.jsonl in the same
schema every other arm uses, so the existing judging and analysis chain reads
them without modification.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import arms
import engine

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data/index"
GEN = ROOT / "results/generations"
PROC = ROOT / "data/processed"

CLEARANCE_SEED = 20260829
CLEARANCE_FRACTION = 0.50
EMBEDDER = "bge-m3"


def clearance(labels, frac=CLEARANCE_FRACTION, seed=CLEARANCE_SEED):
    """The one pre-registered clearance this run is conditioned on."""
    universe = sorted(set(labels))
    rng = np.random.default_rng(seed)
    n = max(1, int(round(frac * len(universe))))
    granted = set(rng.choice(universe, n, replace=False).tolist())
    return granted, np.isin(labels, list(granted))


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import access_control as ac

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="post,pre")
    ap.add_argument("--generators", default="qwen3:8b,llama3.1:8b")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    meta = ac.load_corpus()
    qs = [json.loads(l) for l in (PROC / "obliqa_test_sample.jsonl").open()]
    if a.limit:
        qs = qs[:a.limit]
    D, Q = ac.load_embeddings(EMBEDDER)
    labels = ac.labels_compartment(meta)
    granted, allowed_mask = clearance(labels)
    allowed_rows = np.flatnonzero(allowed_mask)
    S = Q @ D.T

    print(f"clearance: {len(granted)} of {len(set(labels))} compartments, "
          f"seed {CLEARANCE_SEED}, {allowed_mask.mean():.1%} of passages authorised")
    print(f"granted: {','.join(sorted(granted, key=int))}\n")

    GEN.mkdir(parents=True, exist_ok=True)
    client = engine.OllamaClient()
    lock = threading.Lock()

    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        for gen_model in [g.strip() for g in a.generators.split(",") if g.strip()]:
            tag = gen_model.replace(":", "_").replace(".", ".")
            path = GEN / f"obliqa_open_{arm}_{tag}.jsonl"
            done = set()
            if path.exists():
                done = {json.loads(l)["qid"] for l in path.open()}
            todo = [(i, q) for i, q in enumerate(qs) if q["qid"] not in done]
            print(f"{arm}/{gen_model}: {len(done)} done, {len(todo)} to go", flush=True)
            if not todo:
                continue
            run_id = f"obliqa_open_{arm}_{tag}"
            counter = {"n": 0}

            def work(item, fh):
                qi, q = item
                got = ac.retrieve(arm, S[qi], a.k, allowed_mask, allowed_rows)
                passages = [{"uid": meta[r]["uid"], "text": meta[r]["text"],
                             "doc_id": meta[r]["doc_id"]} for r in got]
                msgs, pmeta = arms.a45_rag(q, passages)
                txt, usage = client.chat(gen_model, msgs,
                                         purpose="query:generate", run_id=run_id)
                rec = {"qid": q["qid"], "arm": arm, "mode": "open",
                       "track": "obliqa", "generator": gen_model,
                       "answer": txt, "usage": usage,
                       "meta": {**pmeta,
                                "clearance_seed": CLEARANCE_SEED,
                                "clearance_fraction": CLEARANCE_FRACTION,
                                "n_authorised_passages": int(allowed_mask.sum())},
                       "concurrent": a.workers > 1}
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    counter["n"] += 1
                    if counter["n"] % 50 == 0:
                        print(f"  {counter['n']}/{len(todo)}", flush=True)

            with path.open("a") as fh:
                if a.workers > 1:
                    with ThreadPoolExecutor(max_workers=a.workers) as ex:
                        list(ex.map(lambda it: work(it, fh), todo))
                else:
                    for it in todo:
                        work(it, fh)
            print(f"  wrote {path.name}", flush=True)

    print("ENFORCED_GENERATION_DONE")


if __name__ == "__main__":
    main()
