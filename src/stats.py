"""Statistical comparison of arms: paired tests with multiple-comparison control.

Consumes results/judgments/*.jsonl (judged correctness), builds per-item paired
vectors, and runs:
  - McNemar exact test for paired binary outcomes (judged correct/not)
  - paired bootstrap CIs on the accuracy difference
  - Holm-Bonferroni correction across the comparison family

protocol.tex pre-registers the correction family as the contrasts within one
track and one generator, so ten contrasts per cell. Earlier code applied Holm
once to the concatenated frame, which is a larger family and therefore a more
conservative correction than the protocol asks for. Both are now reported side
by side with the family size that produced them, so the pre-registered numbers
are recoverable and the conservative ones remain visible.

Writes results/tables/significance_*.csv and
results/tables/panel_tiebreak_sensitivity.csv
"""
import itertools
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import panel

ROOT = Path(__file__).resolve().parent.parent
JUD = ROOT / "results/judgments"
TAB = ROOT / "results/tables"
B = 10000
SEED = 42


def mcnemar_exact(b, c):
    """b = A right/B wrong, c = A wrong/B right. Exact binomial two-sided."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5)))


def paired_bootstrap_diff(x, y, n_boot=B, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    d = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    return float(np.mean(x) - np.mean(y)), tuple(np.percentile(d, [2.5, 97.5]))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def judged_vectors(tiebreak=panel.CANONICAL_RULE):
    """system_key -> {qid: 0/1} from the canonical panel verdict.

    The panel label is shared with analyze.py through src/panel.py so the two
    scripts cannot drift. Items the panel left undecided carry no evidence
    about the answer and are dropped from the paired vector rather than being
    counted as failures, which is why n varies a little between contrasts.
    """
    out = {}
    meta = {}
    for gf, items in panel.by_gen_file(panel.load_verdicts()).items():
        _, complete, reason = panel.panel_coverage(items)
        vec = {}
        n_undecided = 0
        for qid, jd in items.items():
            flag = panel.is_correct(panel.panel_verdict(jd, tiebreak)[0])
            if flag is None:
                n_undecided += 1
                continue
            vec[qid] = flag
        if vec:
            out[gf] = vec
            meta[gf] = {"panel_complete": complete, "panel_note": reason,
                        "n_undecided": n_undecided}
    return out, meta


def split_by_panel_coverage(vectors, meta):
    """Separate files the full panel graded from those it did not.

    A file graded by one judge still yields a "majority" of one, which would
    enter the contrast table indistinguishable from a real panel verdict and
    carry a p-value earned by a single grader. While ObliQA judging was partway
    through, that produced three contrasts at p below 1e-19 from gemma3 alone.
    The two sets are therefore tested as separate families and written to
    separate files, so a partial panel can never be read as a panel result.
    """
    full = {k: v for k, v in vectors.items() if meta[k]["panel_complete"]}
    partial = {k: v for k, v in vectors.items() if not meta[k]["panel_complete"]}
    return full, partial


def compare_family(vectors, label, meta=None, write=True):
    """All pairwise comparisons within each (track, generator) group."""
    groups = collections.defaultdict(list)
    for key in vectors:
        parts = key.split("_")
        track = parts[0]
        gen = "_".join(parts[3:]) if len(parts) > 3 else "?"
        groups[(track, gen)].append(key)

    rows = []
    for (track, gen), keys in sorted(groups.items()):
        for a, b_key in itertools.combinations(sorted(keys), 2):
            va, vb = vectors[a], vectors[b_key]
            common = sorted(set(va) & set(vb))
            if len(common) < 20:
                continue
            xa = [va[q] for q in common]
            xb = [vb[q] for q in common]
            b_cnt = sum(1 for i in range(len(common)) if xa[i] == 1 and xb[i] == 0)
            c_cnt = sum(1 for i in range(len(common)) if xa[i] == 0 and xb[i] == 1)
            p = mcnemar_exact(b_cnt, c_cnt)
            diff, (lo, hi) = paired_bootstrap_diff(xa, xb)
            row = {
                "track": track, "generator": gen,
                "arm_a": a.split("_")[2], "arm_b": b_key.split("_")[2],
                "n": len(common), "acc_a": np.mean(xa), "acc_b": np.mean(xb),
                "diff": diff, "ci_lo": lo, "ci_hi": hi,
                "b": b_cnt, "c": c_cnt, "p_raw": p,
            }
            if meta is not None:
                ma = meta.get(a, {}); mb = meta.get(b_key, {})
                row["panel_complete"] = bool(ma.get("panel_complete")) and \
                    bool(mb.get("panel_complete"))
                row["n_undecided_a"] = ma.get("n_undecided")
                row["n_undecided_b"] = mb.get("n_undecided")
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Pre-registered family: contrasts within one track and one generator.
    df["m_prereg"] = df.groupby(["track", "generator"])["p_raw"].transform("size")
    df["p_holm_prereg"] = df.groupby(["track", "generator"])["p_raw"].transform(
        lambda s: holm(s.values))
    # Pooled family: every contrast in the frame at once. Larger m, so this is
    # the more conservative reading, kept because earlier runs reported it.
    df["m_pooled"] = len(df)
    df["p_holm_pooled"] = holm(df["p_raw"].values)
    # p_holm and sig_05 stay as column names because figures.py, make_tables.py
    # and results_facts.py read them; they now hold the pre-registered family,
    # and holm_family records that so a reader knows which m was applied.
    df["p_holm"] = df["p_holm_prereg"]
    df["holm_family"] = "prereg_within_track_and_generator"
    df["holm_m_applied"] = df["m_prereg"]
    df["sig_05"] = df["p_holm_prereg"] < 0.05
    df["sig_05_pooled"] = df["p_holm_pooled"] < 0.05
    df = df.sort_values(["track", "generator", "p_holm"])
    if write:
        df.to_csv(TAB / f"significance_{label}.csv", index=False)
    return df


def tiebreak_sensitivity(label="judged"):
    """How many judged contrasts survive Holm under each tie-break rule.

    The tie-break is not a detail: a substantial share of items have no
    majority verdict, and assigning them by any fixed rule instead of reporting
    them as undecided moves the number of surviving contrasts. Reporting the
    whole set is the honest form of that finding.

    Only files the full panel graded are included, since a tie cannot arise
    among one judge and a single-judge contrast would otherwise pad every row
    with the same value.
    """
    rows = []
    for rule, desc in panel.TIEBREAK_RULES.items():
        vec, meta = judged_vectors(rule)
        vec, _ = split_by_panel_coverage(vec, meta)
        meta = {k: m for k, m in meta.items() if k in vec}
        df = compare_family(vec, label, meta=meta, write=False)
        if not len(df):
            continue
        sig = df[df["sig_05"]]
        rows.append({
            "tiebreak": rule, "rule_description": desc,
            "undecided_items": sum(m["n_undecided"] for m in meta.values()),
            "contrasts": len(df),
            "sig_prereg_family": int(df["sig_05"].sum()),
            "sig_pooled_family": int(df["sig_05_pooled"].sum()),
            "min_p_holm_prereg": df["p_holm_prereg"].min(),
            "min_p_holm_pooled": df["p_holm_pooled"].min(),
            "sig_contrasts_prereg": "; ".join(
                f"{r.track}/{r.generator} {r.arm_a}-{r.arm_b}"
                for r in sig.itertuples()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(TAB / "panel_tiebreak_sensitivity.csv", index=False)
    return out


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    cols = ["track", "generator", "arm_a", "arm_b", "n", "acc_a", "acc_b",
            "diff", "ci_lo", "ci_hi", "b", "c", "p_raw", "p_holm_prereg",
            "m_prereg", "p_holm_pooled", "m_pooled", "sig_05", "sig_05_pooled"]
    jv, jmeta = judged_vectors()
    full, partial = split_by_panel_coverage(jv, jmeta)
    print(f"\n== judged systems: {len(jv)} "
          f"({len(full)} graded by the full panel, {len(partial)} not)")
    d2 = compare_family(full, "judged", meta=jmeta)
    if len(d2):
        print(d2[cols + ["panel_complete", "n_undecided_a", "n_undecided_b"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"significant after Holm: {int(d2['sig_05'].sum())} of {len(d2)} "
              f"in the pre-registered family, "
              f"{int(d2['sig_05_pooled'].sum())} pooled")
    if partial:
        # Tested and written separately. These are one judge's opinion, so they
        # are not panel contrasts and must not be reported beside those above.
        d2p = compare_family(partial, "judged_single_judge", meta=jmeta)
        print(f"\n== single-judge contrasts, NOT panel results "
              f"({len(d2p)} contrasts over {len(partial)} files) ==")
        if len(d2p):
            print(d2p[cols].to_string(index=False,
                                      float_format=lambda x: f"{x:.4f}"))
            for gf in sorted(partial):
                print(f"  {gf}: {jmeta[gf]['panel_note']}")
    st = tiebreak_sensitivity()
    print("\n== judged significance by panel tie-break rule ==")
    print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}")
          if len(st) else "(none yet)")
