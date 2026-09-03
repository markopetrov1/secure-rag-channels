"""Print a compact, factual digest of every result table.

Used as the single source of truth when writing the results and discussion, so
no reported number is written from memory.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"


def show(name, path, cols=None, sort=None, n=40):
    p = TAB / path
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    if not p.exists():
        print("(not produced yet)")
        return None
    df = pd.read_csv(p)
    if df.empty:
        print("(empty)")
        return df
    if sort:
        df = df.sort_values(sort)
    view = df[cols] if cols and all(c in df for c in cols) else df
    print(view.head(n).to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
    return df


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    show("JUDGED OPEN-ENDED CORRECTNESS (majority vote)", "judged_correctness.csv")
    show("RETRIEVAL QUALITY (ObliQA, vs gold passages)", "retrieval_quality.csv")
    show("JUDGE AGREEMENT", "judge_agreement.csv")
    show("JUDGE BIAS (verdict mix, verbosity correlation)", "judge_bias.csv")
    show("DOMINANCE SUMMARY", "dominance_summary.csv")
    show("AGENTIC BEHAVIOUR", "agentic_behaviour.csv")
    show("ECONOMICS PER ARM", "economics_per_arm.csv",
         ["track", "mode", "arm", "generator", "n", "setup_tokens",
          "tokens_per_query", "latency_mean_s", "latency_p95_s",
          "latency_source", "quality"],
         sort=["track", "mode", "generator", "arm"])
    show("LATENCY BENCHMARK (serial, idle GPU)", "latency_bench.csv")

    be = show("BREAK-EVEN VOLUMES", "breakeven_matrix.csv")
    if be is not None and not be.empty and "meaningful" in be:
        m = be[be["meaningful"]].sort_values("breakeven_queries")
        print("\nmeaningful crossings only, smallest first:")
        print(m.head(20).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    pf = TAB / "pareto_frontier.csv"
    if pf.exists():
        d = pd.read_csv(pf)
        print(f"\n{'='*70}\nPARETO-OPTIMAL ARMS BY QUERY VOLUME\n{'='*70}")
        for k, s in d.groupby(["track", "mode", "generator", "n_queries"]):
            arms = ", ".join(sorted(s[s["on_frontier"]]["arm"]))
            print(f"  {k[0]}/{k[1]}/{k[2]} @ N={k[3]:>7,}: {arms}")

    d = show("SIGNIFICANCE, JUDGED", "significance_judged.csv",
             ["track", "generator", "arm_a", "arm_b", "n", "acc_a", "acc_b",
              "diff", "ci_lo", "ci_hi", "p_raw", "p_holm", "sig_05"])
    if d is not None and not d.empty and "sig_05" in d:
        print(f"  significant after Holm correction: "
              f"{int(d['sig_05'].sum())} of {len(d)} contrasts")


if __name__ == "__main__":
    main()
