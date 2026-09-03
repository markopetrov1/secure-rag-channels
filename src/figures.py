"""Publication figures. Vector PDF + 300dpi PNG into figures/.

Design rules followed (dataviz skill):
  - categorical hues assigned in fixed validated order, never cycled;
  - scatter (all-pairs form) is capped at 3 categorical hues, so the Pareto
    plot uses ONE hue plus direct labels instead of a colour per arm;
  - sequential magnitude (agreement heatmap) is a single blue ramp, and every
    cell is annotated so colour is never the only channel;
  - no dual axes anywhere; grid and axes recessive; legend for >=2 series.
"""
from pathlib import Path

import arm_labels
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"
FIG = ROOT / "figures"

# validated categorical order (light mode)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
GRID = "#e3e3e0"

# Hue is keyed to the base arm, so an arm keeps its colour across figures even
# where a cell is missing one, and a variant reads as its base rather than as an
# unrelated series. CAT must stay at least as long as the base roster or two
# arms would share a hue.
BASE_ARMS = ["a1", "a2", "a3", "a4", "a5", "a7", "a8"]
assert len(CAT) >= len(BASE_ARMS), (
    f"{len(CAT)} colours for {len(BASE_ARMS)} base arms")
ARM_COLOR = {a: CAT[i] for i, a in enumerate(BASE_ARMS)}
# Every arm label the study can produce must resolve to a colour, or a variant
# is silently drawn in the fallback grey.
for _a in arm_labels.SHORT:
    assert arm_labels.HUE_OF.get(arm_labels.base_arm(_a),
                                 arm_labels.base_arm(_a)) in ARM_COLOR, _a


def arm_key(a):
    """The arm whose hue this variant shares (a7s1 -> a7, a8c -> a8, a2o -> a2).

    A variant that falls through to no colour is drawn in grey, which reads as
    "not one of the arms" rather than as the arm it is, so the mapping is
    explicit and asserted below.
    """
    b = arm_labels.base_arm(a)
    return arm_labels.HUE_OF.get(b, b)


def arm_label(a):
    """Full short label, so a variant reads as its own series in a legend."""
    return arm_labels.name(a, short=True)


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf|.png")


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", visible=False)


def fig_pareto():
    """Quality vs total token cost. One hue + direct labels (all-pairs form)."""
    src = TAB / "economics_per_arm.csv"
    if not src.exists():
        return
    df = pd.read_csv(src).dropna(subset=["quality"])
    if df.empty:
        return
    for (track, mode, gen), sub in df.groupby(["track", "mode", "generator"]):
        if len(sub) < 2:
            continue
        for n in (1000, 100000):
            col = f"total_tokens_at_{n}"
            if col not in sub:
                continue
            s = sub.sort_values(col).copy()
            front = []
            best = -np.inf
            for _, r in s.iterrows():
                if r["quality"] > best:
                    front.append(True); best = r["quality"]
                else:
                    front.append(False)
            s["front"] = front
            fig, ax = plt.subplots(figsize=(5.4, 3.6))
            f = s[s["front"]]
            ax.step(f[col], f["quality"], where="post", color=CAT[0],
                    lw=2, alpha=0.55, zorder=2)
            ax.scatter(s.loc[~s["front"], col], s.loc[~s["front"], "quality"],
                       s=52, facecolors="white", edgecolors=MUTED,
                       linewidths=1.6, zorder=3, label="dominated")
            ax.scatter(f[col], f["quality"], s=58, color=CAT[0],
                       edgecolors="white", linewidths=1.6, zorder=4,
                       label="Pareto-optimal")
            # Optimiser seeds land almost on top of one another, so they are
            # plotted to show the spread and labelled once.
            placed = []
            for _, r in s.iterrows():
                base = arm_labels.base_arm(r["arm"])
                if base in placed and base != r["arm"]:
                    continue
                placed.append(base)
                ax.annotate(arm_label(r["arm"]),
                            (r[col], r["quality"]),
                            textcoords="offset points", xytext=(7, 4),
                            fontsize=7.5, color=INK2)
            ax.set_xscale("log")
            ax.set_xlabel(f"total tokens for {n:,} queries (setup + marginal, log scale)")
            ax.set_ylabel("quality")
            ax.set_title(f"Cost and quality frontier at N={n:,} ({track}, {gen})")
            ax.legend(loc="lower right", fontsize=8, labelcolor=INK2)
            _despine(ax)
            _save(fig, f"fig_pareto_{track}_{mode}_{gen.replace(':','_')}_N{n}")


def fig_breakeven():
    """Total cost of ownership vs query volume; crossings = break-even points."""
    src = TAB / "economics_per_arm.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if df.empty:
        return
    Ns = np.logspace(1, 6, 120)
    for (track, mode, gen), sub in df.groupby(["track", "mode", "generator"]):
        if len(sub) < 2:
            continue
        sub = sub.sort_values("arm")
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        for _, r in sub.iterrows():
            y = r["setup_tokens"] + Ns * r["tokens_per_query"]
            ax.plot(Ns, y, color=ARM_COLOR.get(arm_key(r["arm"]), MUTED),
                    label=arm_label(r["arm"]))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("queries served (log scale)")
        ax.set_ylabel("cumulative tokens (log scale)")
        ax.set_title(f"Cost of ownership against query volume ({track}, {gen})")
        ax.legend(fontsize=7.5, ncols=2, loc="upper left", labelcolor=INK2)
        ax.set_xlim(10, 1e6)
        _despine(ax)
        _save(fig, f"fig_breakeven_{track}_{mode}_{gen.replace(':','_')}")


def fig_judge_heatmap():
    """Pairwise judge agreement (Cohen's kappa) — single sequential hue + labels."""
    src = TAB / "judge_agreement.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    if df.empty:
        return
    judges = sorted(set(df["judge_a"]) | set(df["judge_b"]))
    m = np.full((len(judges), len(judges)), np.nan)
    for i, a in enumerate(judges):
        m[i, i] = 1.0
    for _, r in df.iterrows():
        i, j = judges.index(r["judge_a"]), judges.index(r["judge_b"])
        m[i, j] = m[j, i] = r["cohen_kappa"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(judges)), judges, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(judges)), judges, fontsize=8)
    for i in range(len(judges)):
        for j in range(len(judges)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if m[i, j] > 0.55 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Cohen's κ", color=INK2)
    cb.outline.set_visible(False)
    alpha = df["krippendorff_alpha_panel"].iloc[0] if "krippendorff_alpha_panel" in df else np.nan
    ax.set_title(f"Inter-judge agreement (panel α = {alpha:.2f})"
                 if np.isfinite(alpha) else "Inter-judge agreement")
    _save(fig, "fig_judge_agreement")


def fig_dominance():
    """Quality against per-query cost, with setup drawn as the marker.

    Three quantities decide whether a paradigm is worth deploying and a scatter
    of two of them hides the third, so setup cost is encoded as marker area and
    stated in the label. An arm that sits below and to the right of another
    while carrying a larger marker is beaten on all three counts, and the cross
    marks exactly those.
    """
    f = TAB / "dominance_summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[(d.track == "obliqa") & d.quality.notna()]
    if d.empty:
        return
    gens = sorted(d.generator.unique())
    fig, axes = plt.subplots(1, len(gens), figsize=(4.1 * len(gens), 4.0),
                             sharey=True, squeeze=False)
    for ax, gen in zip(axes[0], gens):
        g = d[d.generator == gen]
        smax = max(g.setup_tokens.max(), 1.0)
        for r in g.itertuples():
            # Area carries setup, so a paradigm that is expensive to build is
            # visibly large even when its per-query cost looks ordinary.
            area = 40 + 620 * (r.setup_tokens / smax)
            c = ARM_COLOR.get(arm_key(r.arm), MUTED)
            ax.scatter(r.tokens_per_query, r.quality, s=area, color=c,
                       alpha=0.30 if r.strictly_dominated else 0.85,
                       edgecolors=c, linewidths=1.4, zorder=2)
            if r.strictly_dominated:
                ax.scatter(r.tokens_per_query, r.quality, marker="x", s=46,
                           color=INK, linewidths=1.6, zorder=4)
            # The extra optimiser seeds are the same paradigm at a different
            # random start, so they are drawn to show the spread but labelled
            # once, otherwise three identical labels stack on one another.
            if r.arm == arm_labels.base_arm(r.arm):
                ax.annotate(arm_label(r.arm), (r.tokens_per_query, r.quality),
                            textcoords="offset points", xytext=(0, 15),
                            ha="center", fontsize=7, color=c)
        ax.set_xscale("log")
        ax.set_xlabel("Tokens per query (log)")
        ax.set_title(gen)
        # The panels share a y axis, so the limit has to span every arm in the
        # figure; taking it per panel silently crops the other one's arms.
        lo, hi = d.quality.min(), d.quality.max()
        pad = 0.10 * (hi - lo or 0.1)
        ax.set_ylim(lo - pad, hi + 1.6 * pad)
        ax.grid(axis="x", visible=True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("Judged correct rate")
    from matplotlib.lines import Line2D
    axes[0][-1].legend(handles=[
        Line2D([], [], marker="o", ls="", color=MUTED, ms=9, alpha=0.85,
               label="on the frontier"),
        Line2D([], [], marker="x", ls="", color=INK, ms=8,
               label="dominated on all three"),
        Line2D([], [], marker="o", ls="", color=MUTED, ms=13, alpha=0.35,
               label="marker area is setup cost")],
        loc="lower right", fontsize=7, frameon=False, labelcolor=INK2)
    fig.suptitle("What a paradigm costs to build, to run, and what it buys",
                 x=0.02, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig_dominance")


if __name__ == "__main__":
    print("building figures...")
    for f in (fig_pareto, fig_breakeven, fig_judge_heatmap, fig_dominance):
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__}: {type(e).__name__}: {e}")
    print("done")
