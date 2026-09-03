"""How deep must post-filtering search to match pre-filtering?

Proposition 3 states that post-filtering with sufficient
over-fetch returns what pre-filtering returns. The proposition fixes that such a
depth exists; it says nothing about how large it is, and a deployment has to
choose one value in advance. The naive rule sets it to k divided by the
authorised fraction, on the assumption that authorised passages are spread
uniformly through the ranking. They are not, because relevance and compartment
membership are correlated.

Reports the distribution of the required depth rather than its mean, since the
mean is what a deployment would wrongly budget for and the tail is what
determines whether the budget is ever exceeded.

Writes results/tables/overfetch.csv.
"""
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"
sys.path.insert(0, str(ROOT / "src"))
import access_control as ac


def required_depth(order, allowed_mask, k):
    """Rank at which the kth authorised passage appears in the global ranking."""
    seen = 0
    for depth, idx in enumerate(order, start=1):
        if allowed_mask[idx]:
            seen += 1
            if seen == k:
                return depth
    return len(order)


def main(k=5, draws=25, seed=13):
    meta = ac.load_corpus()
    qs = ac.load_questions(meta)
    D, Q = ac.load_embeddings("bge-m3")
    labels = ac.labels_compartment(meta)
    universe = sorted(set(labels))
    S = Q @ D.T
    rng = np.random.default_rng(seed)
    rows = []
    for frac in (0.10, 0.25, 0.50, 0.75, 0.90):
        n_lab = max(1, int(round(frac * len(universe))))
        depths = []
        for _ in range(draws):
            granted = set(rng.choice(universe, n_lab, replace=False).tolist())
            allowed = np.isin(labels, list(granted))
            for qi in range(len(qs)):
                order = np.argsort(-S[qi])
                depths.append(required_depth(order, allowed, k))
        a = np.array(depths, float)
        rows.append({
            "granted_fraction": frac, "k": k, "n": len(a),
            "naive_rule": k / frac,
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "share_above_naive": float((a > k / frac).mean()),
        })
        print(f"  f={frac:.0%}  median {np.median(a):.0f}  p90 {np.percentile(a,90):.0f}  "
              f"p99 {np.percentile(a,99):.0f}  max {a.max():.0f}  "
              f"(naive rule says {k/frac:.0f}; exceeded on "
              f"{(a > k/frac).mean():.1%} of queries)", flush=True)
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "overfetch.csv", index=False)
    print(f"\nwrote results/tables/overfetch.csv")


if __name__ == "__main__":
    print("over-fetch depth required for post-filtering to match pre-filtering")
    main()
