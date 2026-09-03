"""Retrieval-quality evaluation on ObliQA: recall@k and MRR vs gold passage IDs,
for dense (bge-m3), BM25, and hybrid RRF. Writes results/tables/retrieval_quality.csv"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import Retriever

ROOT = Path(__file__).resolve().parent.parent
KS = [1, 3, 5, 10]


def gold_uids(q, meta_by_key):
    out = set()
    for p in q["gold_passages"]:
        key = (str(p["DocumentID"]), str(p["PassageID"]))
        if key in meta_by_key:
            out.add(meta_by_key[key])
    return out


def main():
    r = Retriever("obliqa")
    meta_by_key = {(str(m["doc_id"]), str(m["passage_id"])): m["uid"]
                   for m in r.meta}
    test = [json.loads(l) for l in
            (ROOT / "data/processed/obliqa_test_sample.jsonl").open()]
    rows = []
    miss = 0
    for method in ["dense", "bm25", "hybrid"]:
        hits = {k: [] for k in KS}
        rr = []
        for q in test:
            gold = gold_uids(q, meta_by_key)
            if not gold:
                miss += 1
                continue
            if method == "dense":
                res = r.dense_topk(q["question"], max(KS),
                                   purpose="eval:retrieval", run_id="eval_retr")
            elif method == "bm25":
                res = r.bm25_topk(q["question"], max(KS))
            else:
                res = r.hybrid_topk(q["question"], max(KS),
                                    purpose="eval:retrieval", run_id="eval_retr")
            ids = [x["uid"] for x in res]
            for k in KS:
                hits[k].append(len(gold & set(ids[:k])) / len(gold))
            pos = next((i for i, u in enumerate(ids) if u in gold), None)
            rr.append(1.0 / (pos + 1) if pos is not None else 0.0)
        row = {"method": method, "n": len(rr), "mrr@10": np.mean(rr)}
        for k in KS:
            row[f"recall@{k}"] = np.mean(hits[k])
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/tables/retrieval_quality.csv", index=False)
    print(f"questions with unmatched gold passages: {miss // 3}")


if __name__ == "__main__":
    main()
