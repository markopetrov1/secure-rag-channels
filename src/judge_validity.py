"""How far the judges agree with each other, and how they are biased.

The benchmark ships gold supporting passages but no gold answer strings, so
nothing here is validity against a key. What can still be measured without one
is whether the three judges are interchangeable and whether any of them pays
for length, and both bear on how much weight a panel verdict can carry.

Outputs (results/tables/):
  judge_agreement.csv       pairwise Cohen kappa + Krippendorff alpha
  judge_bias.csv            verdict distribution + answer-length effect
"""
import glob
import json
import collections
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import panel

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results/generations"
JUD = ROOT / "results/judgments"
TAB = ROOT / "results/tables"


def cohen_kappa(a, b):
    a = np.asarray(a); b = np.asarray(b)
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return np.nan
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    obs = np.mean(a == b)
    pa = np.bincount([idx[x] for x in a], minlength=len(cats)) / n
    pb = np.bincount([idx[x] for x in b], minlength=len(cats)) / n
    exp = float(np.sum(pa * pb))
    return (obs - exp) / (1 - exp) if exp < 1 else np.nan


def load_judgments():
    """(gen_file, qid) -> judge -> verdict, for the benchmark this study reports.

    Verdicts from the withdrawn track can still be sitting in results/judgments
    on a machine that ran it, and pooling them would let the mix of benchmarks
    move a coefficient that is supposed to be a property of the judges.
    """
    return {k: d for k, d in panel.load_verdicts().items()
            if str(k[0]).startswith("obliqa")}


def load_answer_lengths():
    lens = {}
    for fp in glob.glob(str(GEN / "obliqa_open_*.jsonl")):
        stem = Path(fp).stem
        for l in open(fp):
            r = json.loads(l)
            lens[(stem, r["qid"])] = len(r["answer"].split())
    return lens


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    per = load_judgments()
    lens = load_answer_lengths()
    if not per:
        print("no judgments yet")
        return
    judges = sorted({j for d in per.values() for j in d})
    print(f"judges: {judges}; judged items: {len(per)}")

    # ---- verdict distribution + verbosity bias
    rows = []
    for j in judges:
        vs = [d[j] for d in per.values() if j in d]
        keys = [k for k, d in per.items() if j in d]
        if not vs:
            continue
        wl = [lens.get(k, np.nan) for k in keys]
        binary = [1 if per[k][j] == "correct" else 0 for k in keys]
        ok = ~np.isnan(np.array(wl, dtype=float))
        r = sps.pointbiserialr(np.array(binary)[ok], np.array(wl, dtype=float)[ok]) \
            if ok.sum() > 10 else (np.nan, np.nan)
        c = collections.Counter(vs)
        rows.append({"judge": j, "n": len(vs),
                     "pct_correct": c["correct"] / len(vs),
                     "pct_missing": c["missing"] / len(vs),
                     "pct_incorrect": c["incorrect"] / len(vs),
                     "len_corr_r": r[0], "len_corr_p": r[1],
                     "unparsed": sum(1 for v in vs if v is None)})
    pd.DataFrame(rows).to_csv(TAB / "judge_bias.csv", index=False)
    print("\n== judge verdict distribution / verbosity bias ==")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- pairwise agreement
    rows = []
    for a, b in itertools.combinations(judges, 2):
        keys = [k for k, d in per.items()
                if a in d and b in d and d[a] and d[b]]
        if len(keys) < 20:
            continue
        va = [per[k][a] for k in keys]
        vb = [per[k][b] for k in keys]
        rows.append({"judge_a": a, "judge_b": b, "n": len(keys),
                     "raw_agreement": np.mean([x == y for x, y
                                               in zip(va, vb)]),
                     "cohen_kappa": cohen_kappa(va, vb)})
    if rows:
        agr = pd.DataFrame(rows)
        try:
            import krippendorff
            vmap = {"correct": 0, "missing": 1, "incorrect": 2}
            ks = list(per)
            mat = np.array([[vmap.get(per[k].get(j), np.nan) for k in ks]
                            for j in judges], dtype=float)
            # alpha needs at least two coders with overlapping judgements
            coded = (~np.isnan(mat)).sum(axis=0)
            alpha = (krippendorff.alpha(
                reliability_data=mat, level_of_measurement="nominal")
                if (coded >= 2).sum() >= 20 else np.nan)
        except Exception as e:
            alpha = np.nan
            print(f"(krippendorff failed: {e})")
        agr["krippendorff_alpha_panel"] = alpha
        agr.to_csv(TAB / "judge_agreement.csv", index=False)
        print("\n== pairwise agreement ==")
        print(agr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
