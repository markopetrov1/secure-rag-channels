"""Does post-filtering's thinner context cost answer quality?

The three propositions settle what enforcement does to retrieval.
They cannot settle this, because post-filtering returns fewer passages than it
was asked for and a shorter prompt is not the same prompt.

The comparison is paired: both arms answered the same 500 questions under the
same clearance with the same generator, so a question contributes to both sides
or to neither. It is also stratified, because post-filtering sometimes returns
nothing at all and an answer written with no context is a closed-book answer
rather than a thinner-context one. Pooling those two cases hides both effects,
which cancel.

Writes results/tables/enforced_quality.csv.
"""
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

import panel

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results/generations"
JUD = ROOT / "results/judgments"
TAB = ROOT / "results/tables"
J = list(panel.PANEL_JUDGES)
B = 10000


def panel_correct():
    """(gen_file, qid) -> 1 if the panel majority is correct, complete cases only."""
    per = collections.defaultdict(dict)
    for f in sorted(JUD.glob("obliqa_open_p*__*.jsonl")):
        if f.stem.endswith("__extract"):
            continue
        for line in f.open():
            r = json.loads(line)
            per[(r["gen_file"], r["qid"])][r["judge"]] = r["verdict"]
    out = {}
    for k, v in per.items():
        if not all(v.get(j) for j in J):
            continue
        lab, n = collections.Counter(v[j] for j in J).most_common(1)[0]
        out[k] = 1.0 if (n >= 2 and lab == "correct") else 0.0
    return out


def context_sizes():
    out = {}
    for f in sorted(GEN.glob("obliqa_open_p*_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            out[(f.stem, r["qid"])] = r["meta"]["context_passages"]
    return out


def paired_boot(a, b, seed=17):
    """Percentile interval on a paired difference in proportions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = a - b
    stats = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ok = panel_correct()
    ctx = context_sizes()
    rows = []
    for gen in ("qwen3_8b", "llama3.1_8b"):
        pf, rf = f"obliqa_open_post_{gen}", f"obliqa_open_pre_{gen}"
        qids = sorted({q for (g, q) in ok if g == pf} & {q for (g, q) in ok if g == rf})
        strata = {"all": qids,
                  "context delivered": [q for q in qids if ctx.get((pf, q), 5) > 0],
                  "context empty": [q for q in qids if ctx.get((pf, q), 5) == 0]}
        for name, qs in strata.items():
            if len(qs) < 5:
                continue
            a = [ok[(pf, q)] for q in qs]
            b = [ok[(rf, q)] for q in qs]
            d, lo, hi = paired_boot(a, b)
            rows.append({"generator": gen.replace("_", ":", 1), "stratum": name,
                         "n": len(qs), "post": float(np.mean(a)),
                         "pre": float(np.mean(b)), "diff": d, "lo": lo, "hi": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "enforced_quality.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nempty-context share of questions:")
    for gen in ("qwen3_8b", "llama3.1_8b"):
        pf = f"obliqa_open_post_{gen}"
        qs = [q for (g, q) in ok if g == pf]
        e = sum(1 for q in qs if ctx.get((pf, q), 5) == 0)
        print(f"  {gen}: {e}/{len(qs)} = {e/len(qs):.1%}")
    print(f"\nwrote results/tables/enforced_quality.csv")


if __name__ == "__main__":
    main()
