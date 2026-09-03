"""Aggregate results: judge verdicts + agreement, token economics.

Run anytime; consumes whatever generation/judgment files exist.
Outputs CSVs under results/tables/ and prints a summary.
"""
import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd

import panel

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"


def bootstrap_ci(flags, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return (np.nan, np.nan)
    means = rng.choice(flags, size=(n, len(flags)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def judge_tables():
    per_item = panel.load_verdicts()
    rows = []
    for gf, items in sorted(panel.by_gen_file(per_item).items()):
        coverage, complete, reason = panel.panel_coverage(items)
        counts = collections.Counter()
        maj_correct = []
        n_undecided = 0
        for jd in items.values():
            label, _, _ = panel.panel_verdict(jd)
            flag = panel.is_correct(label)
            if flag is None:
                n_undecided += 1
                continue
            counts[label] += 1
            maj_correct.append(bool(flag))
        n_dec = len(maj_correct)
        # A file the full panel never saw cannot have a panel figure. The three
        # ObliQA files were graded by gemma3 alone, and a one-judge "majority"
        # printed under a three-judge caption misstates how the number was
        # produced, so the panel columns stay empty and the per-judge columns
        # carry the evidence instead.
        if complete:
            lo, hi = bootstrap_ci(maj_correct)
            row = {"gen_file": gf, "n": len(items),
                   "maj_correct": np.mean(maj_correct) if maj_correct else np.nan,
                   "ci_lo": lo, "ci_hi": hi,
                   "maj_missing": counts["missing"] / max(n_dec, 1),
                   "maj_incorrect": counts["incorrect"] / max(n_dec, 1)}
        else:
            row = {"gen_file": gf, "n": len(items), "maj_correct": np.nan,
                   "ci_lo": np.nan, "ci_hi": np.nan,
                   "maj_missing": np.nan, "maj_incorrect": np.nan}
        row["panel_complete"] = complete
        row["panel_note"] = reason
        # Undecided is a panel concept, so it is left empty rather than zero
        # for a file one judge graded alone: a single voter can never tie.
        row["n_decided"] = n_dec if complete else np.nan
        row["n_undecided"] = n_undecided if complete else np.nan
        row["undecided_rate"] = (n_undecided / max(len(items), 1)) if complete \
            else np.nan
        for j in panel.PANEL_JUDGES:
            short = j.split(":")[0]
            flags = [jd[j] == "correct" for jd in items.values() if jd.get(j)]
            row[f"coverage_{short}"] = coverage[j]
            row[f"n_{short}"] = len(flags)
            row[f"correct_{short}"] = np.mean(flags) if flags else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(TAB / "judged_correctness.csv", index=False)

    # agreement: krippendorff alpha (nominal) + pairwise kappa, pooled over files
    try:
        import krippendorff
        judges = sorted({j for jd in per_item.values() for j in jd})
        mat = []
        vmap = {"correct": 0, "missing": 1, "incorrect": 2}
        for jd in per_item.values():
            mat.append([vmap.get(jd.get(j), np.nan) for j in judges])
        arr = np.array(mat, dtype=float).T
        alpha = krippendorff.alpha(reliability_data=arr,
                                   level_of_measurement="nominal") if arr.size else np.nan
    except Exception as e:
        alpha = f"error: {e}"
    return df, alpha


def ledger_table():
    led = ROOT / "results/token_ledger.jsonl"
    if not led.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in led.open()]
    df = pd.DataFrame(rows)
    df["base_run"] = df["run_id"].str.split(":").str[0]
    df["phase"] = df["purpose"].str.split(":").str[0]
    agg = df.groupby(["base_run", "phase"]).agg(
        calls=("purpose", "size"),
        prompt_tokens=("prompt_tokens", "sum"),
        completion_tokens=("completion_tokens", "sum"),
        wall_s=("wall_s", "sum"),
    ).reset_index()
    agg.to_csv(TAB / "token_ledger_summary.csv", index=False)
    return agg


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    jt, alpha = judge_tables()
    print("\n== Judged correctness (panel majority, no tie-break) ==")
    print(jt.to_string(index=False) if len(jt) else "(none yet)")
    if len(jt):
        print("\nundecided items per file (no majority among the judges who voted):")
        for _, r in jt[jt["panel_complete"]].iterrows():
            print(f"  {r['gen_file']}: {int(r['n_undecided'])} of {int(r['n'])} "
                  f"undecided ({r['undecided_rate']:.1%}), "
                  f"{int(r['n_decided'])} decided")
        for _, r in jt[~jt["panel_complete"]].iterrows():
            print(f"  panel figure suppressed for {r['gen_file']}: {r['panel_note']}")
    print(f"\nKrippendorff alpha (pooled, nominal): {alpha}")
    lt = ledger_table()
    print("\n== Token ledger by run/phase ==")
    print(lt.to_string(index=False) if len(lt) else "(none yet)")
