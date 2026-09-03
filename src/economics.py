"""RQ2: setup vs marginal cost economics, break-even analysis, Pareto frontier.

Reads results/token_ledger.jsonl (every LLM/embedding call tagged setup:* or
query:*) and joins with quality from results/tables/.

Key quantity: total cost of ownership for N queries
    C(N) = S + N * m
where S is one-time setup token cost (index embedding, DSPy compile) and m is
mean per-query token cost. Two arms cross at
    N* = (S_a - S_b) / (m_b - m_a)
which is the break-even query volume - the number a deployment must exceed
before a setup-heavy paradigm pays for itself.

Break-even and the Pareto frontier are token-cost quantities; latency is
reported alongside them but is not a frontier axis.

Reported latency is net of ollama model-load time. The serial benchmark ran
with OLLAMA_MAX_LOADED_MODELS=1, so the arms that embed each query with bge-m3
(a4, a5) evict the generator and pay a reload on every request; that reload is
an artifact of the benchmark harness, not of the paradigm, and it is 80 percent
of their wall time. load_duration_s isolates it. Wall time is kept as a second
column so the artifact stays visible.

Outputs (results/tables/): economics_per_arm.csv, breakeven_matrix.csv,
pareto_frontier.csv
"""
import glob
import json
import collections
from pathlib import Path

import arm_labels
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results/generations"
TAB = ROOT / "results/tables"
LEDGER = ROOT / "results/token_ledger.jsonl"
BENCH = ROOT / "results/latency_bench.jsonl"

def ledger_marginal(led, gen_file, n_items):
    """Query-phase tokens per question for one cell, read from the ledger.

    The fallback for a runner that wrote no token counts into its generation
    records. Returns (tokens_per_query, source) or (0.0, None) when the ledger
    has nothing either, which is the genuine phantom-arm case the caller drops.
    """
    if led.empty or n_items <= 0:
        return 0.0, None
    q = led[led["phase"] == "query"]
    hit = q[q["base_run"] == gen_file]
    if len(hit):
        return float(hit["total_tokens"].sum()) / n_items, "ledger"
    return 0.0, None


def load_ledger():
    if not LEDGER.exists():
        return pd.DataFrame()
    df = pd.DataFrame(json.loads(l) for l in LEDGER.open())
    df["base_run"] = df["run_id"].astype(str).str.split(":").str[0]
    df["phase"] = df["purpose"].astype(str).str.split(":").str[0]
    df["total_tokens"] = df["prompt_tokens"].fillna(0) + df["completion_tokens"].fillna(0)
    return df


def setup_costs(df):
    """arm-level one-time token cost, keyed (track, generator, arm)."""
    out = collections.defaultdict(float)
    setup = df[df["phase"] == "setup"]
    for _, r in setup.iterrows():
        pur, run = str(r["purpose"]), str(r["base_run"])
        tok = r["total_tokens"]
        if pur.startswith("setup:index:"):
            track = pur.split(":")[2]
            # the dense index is shared by the retrieval arms
            for arm in ("a4", "a5", "a7", "a8", "a8c", "a8p"):
                for g in ("qwen3:8b", "llama3.1:8b"):
                    out[(track, g, arm)] += tok
        elif pur.startswith("setup:dspy"):
            parts = run.split("_")  # dspy_compile_<track>_<gen>_seed<k>
            if len(parts) >= 4:
                track = parts[2]
                gen = "_".join(parts[3:-1]).replace("_8b", ":8b")
                out[(track, gen, "a7")] += tok
    return out


def _p95(x):
    return float(np.percentile(x, 95)) if len(x) > 1 else float(np.mean(x))


def bench_latency():
    """Per-cell serial-benchmark latency, both net of model load and raw wall.

    Reads the per-request jsonl rather than tables/latency_bench.csv because the
    aggregated csv drops load_duration_s, which is the only way to separate the
    model reload from the work the arm actually does.
    """
    key = ["track", "mode", "arm", "generator"]
    if BENCH.exists():
        recs = [json.loads(l) for l in BENCH.open()]
        b = pd.DataFrame([{
            "track": r["track"], "mode": r["mode"], "arm": r["arm"],
            "generator": r["generator"],
            "wall_s": r["usage"].get("wall_s", 0) or 0.0,
            "load_s": r["usage"].get("load_duration_s", 0) or 0.0,
        } for r in recs])
    else:
        csv = TAB / "latency_bench.csv"
        if not csv.exists():
            return pd.DataFrame()
        b = pd.read_csv(csv)[key + ["latency_mean_s"]].rename(
            columns={"latency_mean_s": "wall_s"})
        b["load_s"] = np.nan  # no load column in the aggregate, nothing to net out
    if b.empty:
        return pd.DataFrame()
    b["net_s"] = b["wall_s"] - b["load_s"]
    out = b.groupby(key).agg(
        bench_n=("wall_s", "size"),
        bench_latency_net_mean_s=("net_s", "mean"),
        bench_latency_net_p95_s=("net_s", _p95),
        bench_latency_wall_mean_s=("wall_s", "mean"),
        bench_latency_wall_p95_s=("wall_s", _p95),
        bench_model_load_mean_s=("load_s", "mean"),
    ).reset_index()
    out["bench_load_share_of_wall"] = (out["bench_model_load_mean_s"] /
                                       out["bench_latency_wall_mean_s"])
    return out


def per_query_costs():
    """From generation records: mean prompt/completion tokens and latency."""
    rows = []
    # Scoped to the benchmark this study reports. A withdrawn track's generations
    # may still be on disk from an earlier run, and pricing them would put arms
    # in the shipped cost table that are reported nowhere.
    for fp in sorted(glob.glob(str(GEN / "obliqa_open_*.jsonl"))):
        recs = [json.loads(l) for l in open(fp)]
        if not recs:
            continue
        stem = Path(fp).stem
        parts = stem.split("_")
        track, mode, arm = parts[0], parts[1], parts[2]
        gen = "_".join(parts[3:]).replace("_8b", ":8b")
        pt = np.array([r["usage"].get("prompt_tokens", 0) or 0 for r in recs], float)
        ct = np.array([r["usage"].get("completion_tokens", 0) or 0 for r in recs], float)
        # Latency is only meaningful for records produced serially; concurrent
        # runs are marked and their wall time is ignored here. Reported latency
        # comes from the controlled benchmark (bench_latency.py) instead.
        serial = [r for r in recs if not r.get("concurrent")]
        w = np.array([r["usage"].get("wall_s", 0) or 0 for r in serial], float)
        # same reload artifact as in the benchmark, so net it out here too
        ld = np.array([r["usage"].get("load_duration_s", 0) or 0 for r in serial],
                      float)
        net = w - ld
        rows.append({"track": track, "mode": mode, "arm": arm, "generator": gen,
                     "gen_file": stem, "n": len(recs),
                     "n_serial": len(serial),
                     "prompt_tokens_mean": pt.mean(),
                     "completion_tokens_mean": ct.mean(),
                     "tokens_per_query": pt.mean() + ct.mean(),
                     "latency_net_mean_s": net.mean() if len(net) else np.nan,
                     "latency_net_p95_s": _p95(net) if len(net) else np.nan,
                     "latency_wall_mean_s": w.mean() if len(w) else np.nan,
                     "latency_wall_p95_s": _p95(w) if len(w) else np.nan,
                     "model_load_mean_s": ld.mean() if len(ld) else np.nan,
                     "load_share_of_wall": (ld.mean() / w.mean()
                                            if len(w) and w.mean() > 0
                                            else np.nan)})
    df = pd.DataFrame(rows)
    lat = ["latency_net_mean_s", "latency_net_p95_s", "latency_wall_mean_s",
           "latency_wall_p95_s", "model_load_mean_s", "load_share_of_wall"]
    # Prefer the controlled serial benchmark wherever it exists.
    b = bench_latency()
    if not b.empty and not df.empty:
        df = df.merge(b, on=["track", "mode", "arm", "generator"], how="left")
        # every latency field of a row comes from one measurement, so the net and
        # the raw column can never describe different sources
        has = df["bench_n"].notna().to_numpy()
        for c in lat:
            src = "bench_" + c
            if src in df.columns:
                df[c] = np.where(has, df[src], df[c])
        df["latency_source"] = np.where(has, "serial benchmark", "run (serial)")
        df = df.drop(columns=[c for c in df.columns if c.startswith("bench_")
                              and c != "bench_n"])
    elif not df.empty:
        df["latency_source"] = "run (serial)"
    if not df.empty:
        df.loc[df["latency_net_mean_s"].isna() &
               df["latency_wall_mean_s"].isna(), "latency_source"] = "unmeasured"
        # Downstream tables and figure inputs read latency_mean_s / latency_p95_s;
        # point those names at the net figure so nothing reports the reload
        # artifact. Where load_duration_s is unavailable the net figure stays
        # empty rather than silently falling back to contaminated wall time.
        df["latency_mean_s"] = df["latency_net_mean_s"]
        df["latency_p95_s"] = df["latency_net_p95_s"]
    return df


def quality_lookup():
    """gen_file -> judged majority-correct rate."""
    q = {}
    f = TAB / "judged_correctness.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            q[r["gen_file"]] = r["maj_correct"]
    return q


def pareto_front(df, cost_col, qual_col):
    """Non-dominated set: lower cost, higher quality."""
    keep = []
    for i, r in df.iterrows():
        dominated = ((df[cost_col] <= r[cost_col]) & (df[qual_col] >= r[qual_col]) &
                     ((df[cost_col] < r[cost_col]) | (df[qual_col] > r[qual_col]))).any()
        keep.append(not dominated)
    return np.array(keep)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    led = load_ledger()
    if led.empty:
        print("empty ledger")
        return
    setup = setup_costs(led)
    pq = per_query_costs()
    if pq.empty:
        print("no generations")
        return
    qual = quality_lookup()

    # An extra optimiser seed and the critique-revise variant carry their own
    # results but draw on the base arm's one-time spend, so the lookup is by
    # base arm.
    pq["setup_tokens"] = [
        setup.get((r.track, r.generator, arm_labels.base_arm(r.arm)), 0.0)
        for r in pq.itertuples()]
    pq["quality"] = [qual.get(r.gen_file, np.nan) for r in pq.itertuples()]
    pq["arm_name"] = [arm_labels.name(a) for a in pq["arm"]]
    # An arm with no marginal cost never actually answered anything: it is a
    # partial or purged run whose records survived. Pricing it produces
    # break-even rows against a phantom, which is how a withdrawn arm once came
    # back as a 203,531-token setup cost with no generations behind it.
    # A runner that issues several model calls per question and writes only its
    # own wall time leaves tokens_per_query at zero, which is indistinguishable
    # from a phantom arm until the ledger is consulted. Recover from the ledger
    # first and drop only what the ledger cannot account for either, so that the
    # most expensive arm in the study is priced rather than silently absent.
    pq["cost_source"] = np.where(pq["tokens_per_query"] > 0, "record", "")
    for i, r in pq[pq["tokens_per_query"] <= 0].iterrows():
        tpq, src = ledger_marginal(led, r.gen_file, r.n)
        if src:
            pq.at[i, "tokens_per_query"] = tpq
            pq.at[i, "cost_source"] = src
            print(f"  {r['track']}/{r['mode']}/{r['arm']}/{r['generator']}: records carry "
                  f"no token counts; recovered {tpq:,.0f} tokens per query "
                  f"from the ledger ({src})")
    empty = pq[(pq["tokens_per_query"] <= 0) | (pq["n"] <= 0)]
    if len(empty):
        for r in empty.itertuples():
            print(f"  dropping {r.track}/{r.mode}/{r.arm}/{r.generator}: "
                  f"{r.n} records, {r.tokens_per_query:.0f} tokens per query, "
                  f"and nothing in the ledger, so it has no marginal cost")
        pq = pq[(pq["tokens_per_query"] > 0) & (pq["n"] > 0)].copy()
    for n in (100, 1000, 10000, 100000):
        pq[f"total_tokens_at_{n}"] = pq["setup_tokens"] + n * pq["tokens_per_query"]
    pq = pq.sort_values(["track", "mode", "generator", "arm"])
    pq.to_csv(TAB / "economics_per_arm.csv", index=False)
    print("== economics per arm ==")
    print("latency_net_mean_s is wall time minus ollama model-load time and is "
          "the reported figure;\nlatency_wall_mean_s is raw wall time, "
          "load_share_of_wall shows how much of it was reload.")
    cols = ["track", "mode", "arm", "generator", "n", "setup_tokens",
            "tokens_per_query", "latency_net_mean_s", "latency_net_p95_s",
            "latency_wall_mean_s", "load_share_of_wall", "latency_source",
            "quality"]
    print(pq[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    # a1 is the no-retrieval reference; the net and raw ratios differ a lot
    ref = pq[pq["arm"] == "a1"]["latency_net_mean_s"].mean()
    ref_w = pq[pq["arm"] == "a1"]["latency_wall_mean_s"].mean()
    print("\n== latency relative to a1 (pooled over cells) ==")
    for arm, sub_a in pq.groupby("arm"):
        print(f"  {arm}: net {sub_a['latency_net_mean_s'].mean() / ref:5.2f}x  "
              f"raw {sub_a['latency_wall_mean_s'].mean() / ref_w:5.2f}x  "
              f"load share {sub_a['load_share_of_wall'].mean():.1%}")

    # break-even matrix within each (track, mode, generator)
    rows = []
    for (t, m, g), sub in pq.groupby(["track", "mode", "generator"]):
        for i, a in sub.iterrows():
            for j, b in sub.iterrows():
                if a["arm"] >= b["arm"]:
                    continue
                ds = a["setup_tokens"] - b["setup_tokens"]
                dm = b["tokens_per_query"] - a["tokens_per_query"]
                n_star = ds / dm if dm != 0 else np.nan
                rows.append({"track": t, "mode": m, "generator": g,
                             "arm_a": a["arm"], "arm_b": b["arm"],
                             "setup_diff": ds, "per_query_diff": dm,
                             "breakeven_queries": n_star,
                             "meaningful": bool(np.isfinite(n_star) and n_star > 0)})
    if rows:
        be = pd.DataFrame(rows)
        be.to_csv(TAB / "breakeven_matrix.csv", index=False)
        print("\n== break-even (queries before the setup-heavy arm pays off) ==")
        show = be[be["meaningful"]].sort_values("breakeven_queries")
        print(show.head(25).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # Pareto frontier at several volumes. An arm with no quality score cannot be
    # placed on or off the frontier, but dropping it silently would change which
    # arms appear, so unscored arms are carried through with an explicit status.
    fr = []
    for (t, m, g), grp in pq.groupby(["track", "mode", "generator"]):
        scored = grp[grp["quality"].notna()]
        unscored = grp[grp["quality"].isna()]
        for n in (100, 1000, 10000, 100000):
            col = f"total_tokens_at_{n}"
            if len(scored) >= 2:
                mask = pareto_front(scored, col, "quality")
                status = ["on_frontier" if k else "dominated" for k in mask]
            else:
                # a single scored arm has nothing to be compared against
                status = ["not_evaluated_too_few_scored_arms"] * len(scored)
            for st, (_, r) in zip(status, scored.iterrows()):
                fr.append({"track": t, "mode": m, "generator": g, "arm": r["arm"],
                           "n_queries": n, "total_tokens": r[col],
                           "quality": r["quality"],
                           "on_frontier": st == "on_frontier",
                           "frontier_status": st,
                           "n_scored_arms": len(scored),
                           "n_unscored_arms": len(unscored)})
            for _, r in unscored.iterrows():
                fr.append({"track": t, "mode": m, "generator": g, "arm": r["arm"],
                           "n_queries": n, "total_tokens": r[col],
                           "quality": np.nan, "on_frontier": False,
                           "frontier_status": "quality_missing",
                           "n_scored_arms": len(scored),
                           "n_unscored_arms": len(unscored)})
    if fr:
        pf = pd.DataFrame(fr)
        pf.to_csv(TAB / "pareto_frontier.csv", index=False)
        print("\n== Pareto-optimal arms by query volume ==")
        for (t, m, g, n), s in pf.groupby(["track", "mode", "generator", "n_queries"]):
            arms = ", ".join(sorted(s[s["frontier_status"] == "on_frontier"]["arm"]))
            miss = sorted(s[s["frontier_status"] == "quality_missing"]["arm"])
            n_scored = int(s["n_scored_arms"].iloc[0])
            note = ""
            if n_scored < 2:
                # an empty arm list here means unevaluable, not an empty frontier
                arms = f"(not evaluated, only {n_scored} scored arms)"
            if miss:
                note = f"   [unscored, excluded from the frontier: {', '.join(miss)}]"
            print(f"  {t}/{m}/{g} @ N={n:>6}: {arms}{note}")
        n_miss = int((pf["frontier_status"] == "quality_missing").sum() / 4)
        if n_miss:
            print(f"  {n_miss} arm-cells carry no quality score and are reported "
                  f"as quality_missing rather than dropped")


if __name__ == "__main__":
    main()
