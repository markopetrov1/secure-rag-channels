"""Validate returned annotation sheets and turn them into the reported numbers.

Reports what the study cannot get any other way: how often two humans agree with
each other, which is the ceiling any automated grader is being measured against;
how often each judge and the panel agree with a human; and how the graders'
verdict levels compare once stratum weights are restored.

Stratum weights matter here. The sheet oversamples items the panel split on,
because those carry most of the information about where automated grading breaks
down, so an unweighted figure would misstate the population by construction.
Population estimates restore the weights; the raw figures are reported beside
them.

The annotation pack offers a fourth label beyond the panel's three: "unsure",
for items where the reference does not settle the question, which both
annotators used sparingly. Those are reported and then excluded pairwise rather
than folded into one of the three real categories, because forcing them either
way would manufacture agreement or disagreement that the annotator declined to
assert. How much that choice is worth is reported beside it.

Usage:
  python src/check_annotations.py
  python src/check_annotations.py --files a.xlsx b.csv
"""
import argparse
import glob
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

import panel as P

ROOT = Path(__file__).resolve().parent.parent
HV = ROOT / "results/human_validation"
TAB = ROOT / "results/tables"
VERDICTS = {"correct", "missing", "incorrect"}
ABSTAIN = "unsure"
LEGAL = VERDICTS | {ABSTAIN}
JUDGE_COLS = {"judge_gemma3": "gemma3:12b", "judge_gpt-oss": "gpt-oss:20b",
              "judge_phi4": "phi4:14b"}
SEED = 7
BOOT = 5000

RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------- loading

def read_sheet(path):
    """Return (item_id, human_verdict) and the annotator's name.

    The blank sheet ships a column called human_verdict, but a returned sheet
    is per annotator and names the column after them (ema_verdict,
    marko_verdict), which is also how the annotator is identified. Both spellings
    are accepted so that the blank instrument and the returned sheets can be fed
    to the same script.
    """
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(p, sheet_name="Grade")
    else:
        df = pd.read_csv(p)
    cols = [c for c in df.columns if c.endswith("_verdict")]
    if "item_id" not in df.columns or not cols:
        raise ValueError(f"{p.name} lacks item_id or a *_verdict column")
    col = cols[0]
    who = col[: -len("_verdict")]
    if who == "human":
        who = p.stem.replace("annotation_", "").replace("grading_", "")
    out = df[["item_id", col]].rename(columns={col: "human_verdict"}).copy()
    out["human_verdict"] = (out["human_verdict"].astype(str).str.strip()
                            .str.lower().replace({"nan": ""}))
    return out, who


def validate(df, who):
    n = len(df)
    blank = int((df["human_verdict"] == "").sum())
    bad = sorted(set(df.loc[~df["human_verdict"].isin(LEGAL | {""}),
                            "human_verdict"]))
    dup = int(df["item_id"].duplicated().sum())
    unsure = int((df["human_verdict"] == ABSTAIN).sum())
    print(f"  {who}: {n} rows, {n - blank} graded, {blank} blank, "
          f"{unsure} unsure")
    if bad:
        print(f"    unrecognised verdicts: {bad}")
    if dup:
        print(f"    duplicate item ids: {dup}")
    return blank == 0 and not bad and not dup


# ---------------------------------------------------------------- statistics

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return np.nan
    obs = float(np.mean(a == b))
    pa = np.array([np.mean(a == c) for c in cats])
    pb = np.array([np.mean(b == c) for c in cats])
    exp = float(pa @ pb)
    return (obs - exp) / (1 - exp) if exp < 1 else np.nan


def boot_kappa(a, b, n=BOOT):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 10:
        return np.nan, np.nan
    out = []
    for _ in range(n):
        i = RNG.integers(0, len(a), len(a))
        k = kappa(a[i], b[i])
        if not np.isnan(k):
            out.append(k)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
        if out else (np.nan, np.nan)


def weighted_rate(flags, strata, weights):
    """Population estimate of a rate, restoring stratum weights."""
    flags = np.asarray(flags, dtype=float)
    strata = np.asarray(strata)
    num = den = 0.0
    for s in set(strata):
        m = strata == s
        if not m.any() or s not in weights:
            continue
        num += weights[s] * float(flags[m].mean()) * m.sum()
        den += weights[s] * m.sum()
    return num / den if den else np.nan


def boot_weighted(flags, strata, weights, n=BOOT):
    """Stratified bootstrap: resample within stratum, so the design is kept."""
    flags = np.asarray(flags, dtype=float)
    strata = np.asarray(strata)
    if len(flags) < 10:
        return np.nan, np.nan
    groups = [np.where(strata == s)[0] for s in sorted(set(strata))]
    out = []
    for _ in range(n):
        idx = np.concatenate([RNG.choice(g, len(g), replace=True)
                              for g in groups])
        out.append(weighted_rate(flags[idx], strata[idx], weights))
    out = [x for x in out if not np.isnan(x)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
        if out else (np.nan, np.nan)


def panel_label(row):
    votes = {j: str(row[c]).lower() for c, j in JUDGE_COLS.items()
             if c in row.index and str(row.get(c, "")) not in ("", "nan")}
    return P.panel_verdict(votes)[0] if votes else None


def decided(series):
    """Mask of grader labels that assert one of the three real verdicts."""
    s = series.astype(str).str.lower()
    return s.isin(VERDICTS)


# ---------------------------------------------------------------- reporting

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    a = ap.parse_args()

    files = a.files or (sorted(glob.glob(str(HV / "grading_*.csv")))
                        or sorted(glob.glob(str(HV / "pack" / "annotation_*.xlsx"))
                                  + glob.glob(str(HV / "pack" / "annotation_*.csv"))))
    picked, seen = [], set()
    for f in files:
        stem = Path(f).stem
        if stem in seen:
            continue
        seen.add(stem)
        picked.append(f)
    if not picked:
        print("no annotation sheets found under results/human_validation")
        return 1

    print("== sheets ==")
    sheets, ok = {}, True
    for f in picked:
        df, who = read_sheet(f)
        ok &= validate(df, who)
        sheets[who] = df
    graded = {w: d[d.human_verdict.isin(LEGAL)] for w, d in sheets.items()}
    if all(len(d) == 0 for d in graded.values()):
        print("\nnothing graded yet, so there is nothing to report")
        return 0

    key = pd.read_csv(HV / "judge_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv")
    # The study now rests on one benchmark, so the annotated items belonging to
    # the removed track are excluded rather than pooled into figures that would
    # then describe a set the study no longer reports.
    if "track" in key.columns:
        before = len(key)
        key = key[key["track"] == "obliqa"].copy()
        print(f"  restricted to the ObliQA track: {len(key)} of {before} "
              f"annotated items")
    wdf = pd.read_csv(HV / "stratum_weights.csv")
    weights = dict(zip(wdf.stratum, wdf.weight))
    TAB.mkdir(parents=True, exist_ok=True)

    # One frame carrying the key, the panel label and one column per annotator.
    d = key.copy()
    for who, g in graded.items():
        d = d.merge(g.rename(columns={"human_verdict": who}), on="item_id",
                    how="left")
    d[who] = d[who]  # no-op, keeps the merge chain readable
    d["panel"] = [panel_label(r) for _, r in d.iterrows()]
    d["stratum"] = d["stratum"].fillna("random_core")
    annotators = sorted(graded)

    rows = []

    # ---- human against human: the ceiling
    print("\n== annotator agreement, the ceiling ==")
    for x, y in itertools.combinations(annotators, 2):
        m = d[decided(d[x]) & decided(d[y])]
        dropped = int(len(d) - len(m))
        if len(m) < 10:
            print(f"  {x} vs {y}: only {len(m)} shared decided items")
            continue
        agree = (m[x] == m[y]).values
        raw = float(np.mean(agree))
        wtd = weighted_rate(agree, m.stratum, weights)
        k = kappa(m[x], m[y])
        lo, hi = boot_kappa(m[x].values, m[y].values)
        print(f"  {x} vs {y}: n={len(m)} raw {raw:.3f}, weighted {wtd:.3f}, "
              f"kappa {k:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"({dropped} dropped as unsure by one or both)")
        rows.append({"comparison": f"{x} vs {y}", "kind": "human-human",
                     "n": len(m), "raw_agreement": raw,
                     "weighted_agreement": wtd, "cohen_kappa": k,
                     "kappa_lo": lo, "kappa_hi": hi, "dropped_unsure": dropped})
        for st, s in m.groupby("stratum"):
            if len(s) >= 10:
                print(f"      {st:<13} n={len(s):>3} raw "
                      f"{float(np.mean(s[x] == s[y])):.3f}, "
                      f"kappa {kappa(s[x], s[y]):.3f}")
                rows.append({"comparison": f"{x} vs {y} [{st}]",
                             "kind": "human-human-stratum", "n": len(s),
                             "raw_agreement": float(np.mean(s[x] == s[y])),
                             "cohen_kappa": kappa(s[x], s[y])})
        for tr, s in m.groupby("track"):
            if len(s) >= 10:
                rows.append({"comparison": f"{x} vs {y} [{tr}]",
                             "kind": "human-human-track", "n": len(s),
                             "raw_agreement": float(np.mean(s[x] == s[y])),
                             "cohen_kappa": kappa(s[x], s[y])})

    # ---- how much the ceiling depends on the unsure items
    # Excluding them pairwise is a choice, so the alternative is reported
    # rather than assumed away: forcing them into each of the three real
    # categories in turn bounds how much the choice is worth.
    if len(annotators) == 2:
        x, y = annotators
        both = d[d[x].isin(LEGAL) & d[y].isin(LEGAL)]
        if int((both[x] == ABSTAIN).sum() + (both[y] == ABSTAIN).sum()):
            print("  forcing the unsure items instead of dropping them:")
            for forced in sorted(VERDICTS):
                a = both[x].replace(ABSTAIN, forced)
                b = both[y].replace(ABSTAIN, forced)
                print(f"      as {forced:<9} n={len(both)} raw "
                      f"{float(np.mean(a == b)):.3f}, kappa {kappa(a, b):.3f}")
                rows.append({"comparison": f"{x} vs {y} [unsure as {forced}]",
                             "kind": "human-human-sensitivity", "n": len(both),
                             "raw_agreement": float(np.mean(a == b)),
                             "cohen_kappa": kappa(a, b)})

    # ---- each judge, and the panel, against each human
    print("\n== automated graders against each human ==")
    for who in annotators:
        for col, judge in list(JUDGE_COLS.items()) + [("panel", "PANEL")]:
            if col not in d:
                continue
            sub = d[decided(d[col]) & decided(d[who])]
            if len(sub) < 10:
                continue
            g = sub[col].astype(str).str.lower()
            agree = (g == sub[who]).values
            raw = float(np.mean(agree))
            wtd = weighted_rate(agree, sub.stratum, weights)
            k = kappa(g, sub[who])
            lo, hi = boot_kappa(g.values, sub[who].values)
            print(f"  {judge:<14} vs {who:<8} n={len(sub):>4} raw {raw:.3f}, "
                  f"weighted {wtd:.3f}, kappa {k:.3f} [{lo:.3f}, {hi:.3f}]")
            rows.append({"comparison": f"{judge} vs {who}",
                         "kind": "panel-human" if col == "panel"
                         else "judge-human",
                         "n": len(sub), "raw_agreement": raw,
                         "weighted_agreement": wtd, "cohen_kappa": k,
                         "kappa_lo": lo, "kappa_hi": hi})

    # ---- graders against the items both humans agreed on
    if len(annotators) == 2:
        x, y = annotators
        cons = d[decided(d[x]) & decided(d[y]) & (d[x] == d[y])].copy()
        cons["consensus"] = cons[x]
        print(f"\n== graders against human consensus ({len(cons)} items both "
              f"annotators labelled the same way) ==")
        for col, judge in list(JUDGE_COLS.items()) + [("panel", "PANEL")]:
            sub = cons[decided(cons[col])]
            if len(sub) < 10:
                continue
            g = sub[col].astype(str).str.lower()
            raw = float(np.mean(g == sub.consensus))
            k = kappa(g, sub.consensus)
            lo, hi = boot_kappa(g.values, sub.consensus.values)
            print(f"  {judge:<14} vs consensus n={len(sub):>4} raw {raw:.3f}, "
                  f"kappa {k:.3f} [{lo:.3f}, {hi:.3f}]")
            rows.append({"comparison": f"{judge} vs consensus",
                         "kind": "grader-consensus", "n": len(sub),
                         "raw_agreement": raw, "cohen_kappa": k,
                         "kappa_lo": lo, "kappa_hi": hi})

    pd.DataFrame(rows).to_csv(TAB / "human_validation.csv", index=False)

    # ---- verdict levels: is the panel harsher or softer than a human reader
    print("\n== verdict levels, population-weighted ==")
    lev = []
    for name, col in ([(w, w) for w in annotators]
                      + [("PANEL", "panel")]
                      + [(j, c) for c, j in JUDGE_COLS.items()]):
        sub = d[decided(d[col])]
        g = sub[col].astype(str).str.lower()
        r = {"grader": name, "n": len(sub)}
        for v in ("correct", "missing", "incorrect"):
            flags = (g == v).values
            r[f"raw_{v}"] = float(np.mean(flags))
            r[f"weighted_{v}"] = weighted_rate(flags, sub.stratum, weights)
        lo, hi = boot_weighted((g == "correct").values, sub.stratum, weights)
        r["weighted_correct_lo"], r["weighted_correct_hi"] = lo, hi
        lev.append(r)
        print(f"  {name:<12} n={len(sub):>3} correct {r['weighted_correct']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  missing {r['weighted_missing']:.3f}  "
              f"incorrect {r['weighted_incorrect']:.3f}")
    pd.DataFrame(lev).to_csv(TAB / "human_leniency.csv", index=False)

    # ---- the gap, paired item by item
    if len(annotators) == 2:
        x, y = annotators
        p = d[decided(d.panel) & decided(d[x]) & decided(d[y])].copy()
        p["panel_ok"] = (p.panel == "correct").astype(int)
        p["both_ok"] = ((p[x] == "correct") & (p[y] == "correct")).astype(int)
        p["either_ok"] = ((p[x] == "correct") | (p[y] == "correct")).astype(int)
        gap = (p.panel_ok - p.both_ok).values
        glo, ghi = boot_weighted(gap, p.stratum, weights)
        b = int(((p.panel_ok == 1) & (p.both_ok == 0)).sum())
        c = int(((p.panel_ok == 0) & (p.both_ok == 1)).sum())
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else np.nan
        print(f"\n== the gap, paired on {len(p)} items ==")
        print(f"  panel correct           "
              f"{weighted_rate(p.panel_ok, p.stratum, weights):.3f}")
        print(f"  both annotators correct "
              f"{weighted_rate(p.both_ok, p.stratum, weights):.3f}")
        print(f"  either annotator correct "
              f"{weighted_rate(p.either_ok, p.stratum, weights):.3f}")
        print(f"  gap {weighted_rate(gap, p.stratum, weights):+.3f} "
              f"[{glo:+.3f}, {ghi:+.3f}]; discordant pairs: panel alone {b}, "
              f"annotators alone {c}, McNemar p={pv:.3g}")
        sub = []
        for grp in ("track", "generator", "stratum"):
            for lv, s in p.groupby(grp):
                sub.append({"split": grp, "level": lv, "n": len(s),
                            "panel_correct": float(s.panel_ok.mean()),
                            "humans_correct": float(s.both_ok.mean()),
                            "gap": float(s.panel_ok.mean() - s.both_ok.mean())})
        print("\n== where the gap is largest ==")
        for r in sub:
            print(f"  {r['split']}={str(r['level']):<14} n={r['n']:>3} "
                  f"panel {r['panel_correct']:.3f} humans "
                  f"{r['humans_correct']:.3f} gap {r['gap']:+.3f}")
        # ---- what the panel credits and both annotators refuse
        # The direction of the discordance is the finding, not just its size,
        # so the items are counted by the label the annotators agreed on and
        # by track. On this subset the disagreement is one-sided and it
        # concentrates on the track whose reference is a set of passages.
        both_ref = p[(p[x] == p[y]) & (p[x] != "correct") & (p.panel == "correct")]
        print(f"\n== items the panel credits and both annotators refuse: "
              f"{len(both_ref)} ==")
        for grp in ("track", x):
            print("  by " + ("annotator label" if grp == x else grp) + ": "
                  + ", ".join(f"{k} {v}" for k, v in
                              both_ref[grp].value_counts().items()))
        # Against each annotator on their own the panel is occasionally the
        # stricter party, so the one-directional claim is only true of the two
        # of them jointly and is counted that way rather than asserted.
        for who in (x, y):
            n_strict = int(((p[who] == "correct") & (p.panel != "correct")).sum())
            n_lenient = int(((p[who] != "correct") & (p.panel == "correct")).sum())
            print(f"  against {who} alone: panel refuses {n_strict} the "
                  f"annotator credits, credits {n_lenient} the annotator refuses")
            sub.append({"split": "discordance",
                        "level": f"panel_stricter_than_{who}", "n": n_strict,
                        "panel_correct": np.nan, "humans_correct": np.nan,
                        "gap": np.nan})

        rev = p[(p[x] == p[y]) & (p[x] == "correct") & (p.panel != "correct")]
        print(f"  the reverse, both annotators credit and the panel does not: "
              f"{len(rev)}")
        sub.append({"split": "discordance", "level": "panel_credits_both_refuse",
                    "n": len(both_ref), "panel_correct": np.nan,
                    "humans_correct": np.nan, "gap": np.nan})
        for tr, cnt in both_ref["track"].value_counts().items():
            sub.append({"split": "discordance",
                        "level": f"panel_credits_both_refuse[{tr}]", "n": int(cnt),
                        "panel_correct": np.nan, "humans_correct": np.nan,
                        "gap": np.nan})
        sub.append({"split": "discordance", "level": "both_credit_panel_refuses",
                    "n": len(rev), "panel_correct": np.nan,
                    "humans_correct": np.nan, "gap": np.nan})

        pd.DataFrame([{"metric": "weighted_panel_correct",
                       "value": weighted_rate(p.panel_ok, p.stratum, weights)},
                      {"metric": "weighted_both_humans_correct",
                       "value": weighted_rate(p.both_ok, p.stratum, weights)},
                      {"metric": "weighted_either_human_correct",
                       "value": weighted_rate(p.either_ok, p.stratum, weights)},
                      {"metric": "weighted_gap",
                       "value": weighted_rate(gap, p.stratum, weights)},
                      {"metric": "gap_lo", "value": glo},
                      {"metric": "gap_hi", "value": ghi},
                      {"metric": "discordant_panel_alone", "value": b},
                      {"metric": "discordant_humans_alone", "value": c},
                      {"metric": "mcnemar_p", "value": pv},
                      {"metric": "n_paired", "value": len(p)}]
                     + sub).to_csv(TAB / "human_gap.csv", index=False)


        # ---- does the level shift move the ranking, or only the level
        arm = p.groupby("arm").agg(n=("panel_ok", "size"),
                                   panel=("panel_ok", "mean"),
                                   humans=("both_ok", "mean")).reset_index()
        keep = arm[arm.n >= 10]
        if len(keep) >= 4:
            rho, pval = spearmanr(keep.panel, keep.humans)
            print(f"\n== rank preservation across arms ==")
            print(f"  Spearman rho {rho:.3f} (p={pval:.3g}) over "
                  f"{len(keep)} arms with at least 10 annotated items")
            arm["spearman_rho"] = rho
            arm["spearman_p"] = pval
        arm.to_csv(TAB / "human_rank_preservation.csv", index=False)

    # ---- the alternative annotator test
    if len(annotators) >= 2:
        print("\n== alternative annotator test ==")
        alt = []
        for col, judge in list(JUDGE_COLS.items()) + [("panel", "PANEL")]:
            scores = []
            for target in annotators:
                others = [o for o in annotators if o != target]
                sub = d[decided(d[col]) & decided(d[target])
                        & np.logical_and.reduce([decided(d[o]) for o in others])]
                if len(sub) < 10:
                    continue
                a_llm = float(np.mean(sub[col].astype(str).str.lower()
                                      == sub[target]))
                a_hum = float(np.mean([np.mean(sub[o] == sub[target])
                                       for o in others]))
                scores.append((target, a_llm, a_hum))
            if not scores:
                continue
            for eps in (0.0, 0.2):
                wr = float(np.mean([1.0 if l >= h - eps else 0.0
                                    for _, l, h in scores]))
                alt.append({"grader": judge, "epsilon": eps,
                            "winning_rate": wr, "n_annotators": len(scores),
                            "passes": wr >= 0.5})
            detail = "  ".join(f"[{t}: grader {l:.3f} vs human {h:.3f}]"
                               for t, l, h in scores)
            w0 = [r for r in alt if r["grader"] == judge and r["epsilon"] == 0][0]
            w2 = [r for r in alt if r["grader"] == judge and r["epsilon"] == 0.2][0]
            print(f"  {judge:<14} winning rate {w0['winning_rate']:.2f} at "
                  f"eps=0, {w2['winning_rate']:.2f} at eps=0.2   {detail}")
        if alt:
            pd.DataFrame(alt).to_csv(TAB / "human_alt_test.csv", index=False)

    print(f"\nwrote {TAB / 'human_validation.csv'} and companions")
    hv = pd.DataFrame(rows)
    hh = hv[hv.kind == "human-human"]["cohen_kappa"]
    jh = hv[hv.kind.isin(["judge-human", "panel-human"])]["cohen_kappa"]
    if len(hh) and len(jh):
        print(f"\nheadline: the two annotators agree with each other at kappa "
              f"{hh.max():.3f}; the best automated grader agrees with an "
              f"annotator at {jh.max():.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
