"""Emit the specific sentences-worth of fact needed for the results section.

Every number the results prose will state is printed here with its provenance,
so the section can be written by transcription rather than recall. Anything not
printed here has not been measured and must not be reported.
"""
from pathlib import Path

import arm_labels
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"


def get(name):
    p = TAB / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size else None


def fact(label, value):
    print(f"  {label}: {value}")


def main():
    print("FACTS FOR THE RESULTS SECTION")
    print("Only these numbers may appear in the prose.\n")

    d = get("judged_correctness.csv")
    if d is not None and len(d):
        print("\n[judged open-ended correctness, majority verdict]")
        for _, r in d.iterrows():
            # The rate and its interval are computed over the items the panel
            # actually decided, so quote that denominator rather than the number
            # of answers graded; the two differ by the undecided count and
            # quoting the larger one would overstate the interval's basis.
            if pd.isna(r.get("maj_correct")):
                note = str(r.get("panel_note") or "no panel figure")
                fact(r["gen_file"], f"no panel verdict, {note}")
                continue
            nd = r.get("n_decided")
            nd = int(nd) if pd.notna(nd) else int(r["n"])
            und = r.get("n_undecided")
            tail = (f", {int(und)} of {int(r['n'])} undecided"
                    if pd.notna(und) and int(und) else "")
            fact(r["gen_file"], f"{r['maj_correct']:.1%} "
                                f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] "
                                f"n={nd} decided{tail}")

    d = get("retrieval_quality.csv")
    if d is not None:
        print("\n[retrieval quality, ObliQA vs gold passages]")
        for _, r in d.iterrows():
            fact(r["method"], f"R@1={r['recall@1']:.3f} R@5={r['recall@5']:.3f} "
                              f"R@10={r['recall@10']:.3f} MRR={r['mrr@10']:.3f}")

    d = get("judge_agreement.csv")
    if d is not None and len(d):
        print("\n[inter-judge agreement]")
        for _, r in d.iterrows():
            fact(f"{r['judge_a']} vs {r['judge_b']}",
                 f"kappa={r['cohen_kappa']:.3f} "
                 f"raw={r['raw_agreement']:.3f} n={int(r['n'])}")
        a = d["krippendorff_alpha_panel"].iloc[0]
        fact("Krippendorff alpha (panel)",
             "not estimable" if pd.isna(a) else f"{a:.3f}")

    d = get("economics_per_arm.csv")
    if d is not None:
        print("\n[cost per arm]")
        for _, r in d.sort_values(["track", "mode", "generator", "arm"]).iterrows():
            fact(f"{r['track']}/{r['mode']}/{r['generator']}/{arm_labels.name(r['arm'], short=True)}",
                 f"setup={r['setup_tokens']:,.0f} tok, "
                 f"marginal={r['tokens_per_query']:,.0f} tok/query, "
                 f"latency={r['latency_mean_s']:.2f}s "
                 f"({r.get('latency_source', 'run')})")
        sub = d[d["setup_tokens"] > 0]
        if len(sub):
            fact("largest setup cost",
                 f"{sub.loc[sub['setup_tokens'].idxmax(), 'arm']} at "
                 f"{sub['setup_tokens'].max():,.0f} tokens")
        fact("largest marginal cost",
             f"{d.loc[d['tokens_per_query'].idxmax(), 'arm']} at "
             f"{d['tokens_per_query'].max():,.0f} tokens/query")
        fact("smallest marginal cost",
             f"{d.loc[d['tokens_per_query'].idxmin(), 'arm']} at "
             f"{d['tokens_per_query'].min():,.0f} tokens/query")

    d = get("breakeven_matrix.csv")
    if d is not None and len(d) and "meaningful" in d:
        m = d[d["meaningful"]].sort_values("breakeven_queries")
        print("\n[break-even volumes, smallest first]")
        for _, r in m.head(8).iterrows():
            fact(f"{arm_labels.name(r['arm_a'], short=True)} vs "
                 f"{arm_labels.name(r['arm_b'], short=True)}"
                 f" ({r['track']}/{r['mode']}/{r['generator']})",
                 f"{r['breakeven_queries']:,.0f} queries")

    d = get("pareto_frontier.csv")
    if d is not None and len(d):
        print("\n[Pareto-optimal arms by query volume]")
        for k, s in d.groupby(["track", "mode", "generator", "n_queries"]):
            arms = ", ".join(sorted(arm_labels.name(a, short=True)
                                    for a in s[s["on_frontier"]]["arm"]))
            fact(f"{k[0]}/{k[1]}/{k[2]} at N={k[3]:,}", arms)

    print("\n(absent sections above have not been measured yet)")


if __name__ == "__main__":
    main()
