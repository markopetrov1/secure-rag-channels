"""Confidence intervals and paired tests for the retrieval comparison.

`eval_retrieval.py` reports point estimates only, which is not enough to support
the claim that lexical retrieval beats a strong dense embedder on
regulatory text. This adds paired inference over the same 500 questions.

Dense and fused rankings are reconstructed from the `meta.retrieved_ids` written
by the A4 and A5 generation runs, so no model call is needed and the ranking
tested is exactly the one the generation arms saw. BM25 is recomputed directly,
since it needs no model at all. Rankings are therefore compared at the depth the
arms actually retrieved, which is five.

Writes results/tables/retrieval_significance.csv.
"""
import collections
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retrieval import _tok
from rank_bm25 import BM25Okapi
from stats import holm

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"
KS = [1, 3, 5]
DEPTH = 5
B = 10000
SEED = 42


def load_index():
    meta = [json.loads(l) for l in
            (ROOT / "data/index/obliqa_meta.jsonl").open()]
    by_key = {(str(m["doc_id"]), str(m["passage_id"])): m["uid"] for m in meta}
    return meta, by_key


def load_rankings(arm):
    """qid -> ordered uid list, from whichever generator file carries it.

    Retrieval is a function of the question and the index, so the two generator
    runs of one arm must agree; any disagreement is checked rather than assumed.
    """
    out, seen = {}, collections.Counter()
    for gen in ("qwen3_8b", "llama3.1_8b"):
        fp = ROOT / f"results/generations/obliqa_open_{arm}_{gen}.jsonl"
        if not fp.exists():
            continue
        for l in fp.open():
            r = json.loads(l)
            ids = (r.get("meta") or {}).get("retrieved_ids") or []
            if not ids:
                continue
            if r["qid"] in out and out[r["qid"]] != ids:
                seen["disagree"] += 1
            out.setdefault(r["qid"], ids)
    if seen["disagree"]:
        print(f"  warning: {arm} rankings differ between generators on "
              f"{seen['disagree']} questions; using the first seen")
    return out


def metrics(ranked, gold, ks=KS):
    """Recall at each k, and reciprocal rank within the retrieved depth."""
    out = {}
    for k in ks:
        out[f"recall@{k}"] = len(gold & set(ranked[:k])) / len(gold)
    pos = next((i for i, u in enumerate(ranked) if u in gold), None)
    out[f"rr@{len(ranked) if ranked else 0}"] = \
        1.0 / (pos + 1) if pos is not None else 0.0
    return out


def paired_bootstrap(a, b, rng):
    """Percentile interval for the mean paired difference a minus b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), tuple(np.percentile(boot, [2.5, 97.5]))


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank p over the non-zero differences.

    Recall and reciprocal rank are bounded and heavily tied, so a signed-rank
    test on the paired differences is the appropriate choice over a t-test.
    """
    from scipy import stats as sps
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    if len(nz) < 10:
        return np.nan, len(nz)
    return float(sps.wilcoxon(nz).pvalue), len(nz)


def main():
    meta, by_key = load_index()
    test = [json.loads(l) for l in
            (ROOT / "data/processed/obliqa_test_sample.jsonl").open()]
    dense = load_rankings("a4")
    hybrid = load_rankings("a5")
    bm25 = BM25Okapi([_tok(m["text"]) for m in meta])
    uids = [m["uid"] for m in meta]

    per = {m: collections.defaultdict(list) for m in ("dense", "bm25", "hybrid")}
    used = 0
    for q in test:
        gold = {by_key[(str(p["DocumentID"]), str(p["PassageID"]))]
                for p in q["gold_passages"]
                if (str(p["DocumentID"]), str(p["PassageID"])) in by_key}
        if not gold or q["qid"] not in dense or q["qid"] not in hybrid:
            continue
        scores = bm25.get_scores(_tok(q["question"]))
        b_rank = [uids[i] for i in np.argsort(-scores)[:DEPTH]]
        used += 1
        for name, ranked in (("dense", dense[q["qid"]][:DEPTH]),
                             ("bm25", b_rank),
                             ("hybrid", hybrid[q["qid"]][:DEPTH])):
            for k, v in metrics(list(ranked), gold).items():
                per[name][k].append(v)

    print(f"paired over {used} questions, rankings truncated at depth {DEPTH}")
    measures = [f"recall@{k}" for k in KS] + [f"rr@{DEPTH}"]

    print("\n== point estimates with bootstrap intervals ==")
    rows = []
    rng = np.random.default_rng(SEED)
    for name in ("dense", "bm25", "hybrid"):
        for m in measures:
            v = np.asarray(per[name][m], float)
            idx = rng.integers(0, len(v), size=(B, len(v)))
            lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
            rows.append({"method": name, "measure": m, "n": len(v),
                         "value": v.mean(), "ci_lo": lo, "ci_hi": hi})
    est = pd.DataFrame(rows)
    print(est.pivot(index="measure", columns="method", values="value")
          .to_string(float_format=lambda x: f"{x:.4f}"))
    est.to_csv(TAB / "retrieval_estimates.csv", index=False)

    print("\n== paired contrasts, Holm corrected within each measure ==")
    rows = []
    for m in measures:
        rng = np.random.default_rng(SEED)
        sub = []
        for a, b in itertools.combinations(("dense", "bm25", "hybrid"), 2):
            diff, (lo, hi) = paired_bootstrap(per[a][m], per[b][m], rng)
            p, nz = wilcoxon(per[a][m], per[b][m])
            sub.append({"measure": m, "method_a": a, "method_b": b,
                        "n": len(per[a][m]), "diff": diff,
                        "ci_lo": lo, "ci_hi": hi, "n_discordant": nz,
                        "p_raw": p})
        d = pd.DataFrame(sub)
        d["p_holm"] = holm(d["p_raw"].values)
        d["m_family"] = len(d)
        d["sig_05"] = d["p_holm"] < 0.05
        rows.append(d)
    sig = pd.concat(rows, ignore_index=True)
    sig.to_csv(TAB / "retrieval_significance.csv", index=False)
    print(sig[["measure", "method_a", "method_b", "n", "diff", "ci_lo", "ci_hi",
               "p_raw", "p_holm", "sig_05"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    n_sig = int(sig["sig_05"].sum())
    print(f"\nsignificant after Holm: {n_sig} of {len(sig)} contrasts")
    print("wrote results/tables/retrieval_estimates.csv and "
          "retrieval_significance.csv")


if __name__ == "__main__":
    main()
