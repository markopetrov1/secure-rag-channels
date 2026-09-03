"""Which retrieval paradigms can ever repay what they cost to build.

Cost comparisons in this literature report tokens per query. That is the wrong
object for a deployment decision, because paradigms differ less in what they cost
in total than in where the cost falls. An index is paid for once and reused; a
stuffed context is paid for on every question. Writing the total cost of
ownership over N queries as S + Nm, with S the one-time setup and m the mean
marginal cost, two arms cross at

    N* = (S_a - S_b) / (m_b - m_a)

whenever the two differences have opposite signs, and never cross when they do
not. The second case is the interesting one and is invisible to per-query
reporting. An arm with both a higher setup cost and a higher marginal cost is
dominated: no query volume makes it rational, and no amount of amortisation
rescues it.

Outputs (results/tables/):
  dominance_pairs.csv    every pair, with the crossing volume or why none exists
  dominance_summary.csv  per arm, how often it dominates and is dominated
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import arm_labels

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"
VOLUMES = [100, 1_000, 10_000, 100_000, 1_000_000]


def load():
    f = TAB / "economics_per_arm.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    need = {"track", "mode", "generator", "arm", "setup_tokens",
            "tokens_per_query"}
    if not need <= set(d.columns):
        return pd.DataFrame()
    return d[d.tokens_per_query > 0].copy()


def dominated(sa, ma, qa, sb, mb, qb):
    """True when b beats a outright: no dearer to build, no dearer to run, and
    no worse in quality, with at least one of the three strictly better.

    Cost alone cannot answer this. Ranking arms by tokens says a model that
    retrieves nothing wins, which is true and useless. What a deployment needs
    to know is which paradigms are beaten on cost by something that is also at
    least as accurate, because those are the ones no query volume rescues.
    """
    if np.isnan(qa) or np.isnan(qb):
        return False
    no_worse = sb <= sa and mb <= ma and qb >= qa
    strictly = sb < sa or mb < ma or qb > qa
    return bool(no_worse and strictly)


def classify(sa, ma, sb, mb):
    """Where two arms cross on cost, ignoring quality.

    A crossing exists only when one arm is cheaper to build and dearer to run
    than the other; otherwise one of them is cheaper at every volume.
    """
    ds, dm = sa - sb, ma - mb
    if np.isclose(ds, 0) and np.isclose(dm, 0):
        return "identical", np.nan
    if ds >= 0 and dm >= 0:
        return "costs more on both axes", np.nan
    if ds <= 0 and dm <= 0:
        return "costs less on both axes", np.nan
    return "crosses", ds / (mb - ma)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    d = load()
    if d.empty:
        print("no economics table yet; run src/economics.py first")
        return

    rows = []
    for (track, mode, gen), g in d.groupby(["track", "mode", "generator"]):
        cols = ["setup_tokens", "tokens_per_query"]
        if "quality" in g.columns:
            cols.append("quality")
        arms = g.set_index("arm")[cols].to_dict("index")
        for a, b in itertools.permutations(sorted(arms), 2):
            sa, ma = arms[a]["setup_tokens"], arms[a]["tokens_per_query"]
            sb, mb = arms[b]["setup_tokens"], arms[b]["tokens_per_query"]
            qa = arms[a].get("quality", np.nan)
            qb = arms[b].get("quality", np.nan)
            rel, n = classify(sa, ma, sb, mb)
            row = {"track": track, "mode": mode, "generator": gen,
                   "arm_a": a, "arm_b": b, "relation": rel,
                   "quality_a": qa, "quality_b": qb,
                   "a_dominated_by_b": dominated(sa, ma, qa, sb, mb, qb),
                   "setup_a": sa, "setup_b": sb,
                   "marginal_a": ma, "marginal_b": mb,
                   "breakeven_queries": n}
            for v in VOLUMES:
                row[f"a_cheaper_at_{v}"] = bool((sa + v * ma) < (sb + v * mb))
            rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(TAB / "dominance_pairs.csv", index=False)

    summ = []
    for (track, mode, gen, a), g in pairs.groupby(
            ["track", "mode", "generator", "arm_a"]):
        dominated_by = sorted(g.loc[g.a_dominated_by_b, "arm_b"])
        dominates = sorted(pairs.loc[(pairs.track == track) &
                                     (pairs["mode"] == mode) &
                                     (pairs.generator == gen) &
                                     (pairs.arm_b == a) &
                                     pairs.a_dominated_by_b, "arm_a"])
        summ.append({"track": track, "mode": mode, "generator": gen, "arm": a,
                     "arm_name": arm_labels.name(a),
                     "setup_tokens": g.setup_a.iloc[0],
                     "tokens_per_query": g.marginal_a.iloc[0],
                     "quality": g.quality_a.iloc[0],
                     "n_dominates": len(dominates),
                     "n_dominated_by": len(dominated_by),
                     "dominated_by": ",".join(dominated_by),
                     "strictly_dominated": len(dominated_by) > 0})
    s = pd.DataFrame(summ).sort_values(
        ["track", "mode", "generator", "n_dominated_by"])
    s.to_csv(TAB / "dominance_summary.csv", index=False)

    print("== Pareto-dominated arms: beaten on cost by something no less accurate ==")
    dom = s[s.strictly_dominated]
    if len(dom):
        for _, r in dom.iterrows():
            print(f"  {r.track}/{r.generator:<12} {r.arm:<5} setup "
                  f"{r.setup_tokens:>11,.0f} marginal {r.tokens_per_query:>8,.0f} "
                  f"quality {r.quality:.3f}   dominated by {r.dominated_by}")
    else:
        print("  none")

    print("\n== pairs that genuinely cross, and where ==")
    cx = pairs[(pairs.relation == "crosses") & (pairs.breakeven_queries > 0)]
    seen = set()
    for _, r in cx.sort_values("breakeven_queries").iterrows():
        key = (r.track, r.generator, tuple(sorted((r.arm_a, r.arm_b))))
        if key in seen:
            continue
        seen.add(key)
        print(f"  {r.track}/{r.generator:<12} {r.arm_a:<5} vs {r.arm_b:<5} "
              f"crosses at {r.breakeven_queries:>12,.0f} queries")


if __name__ == "__main__":
    main()
