"""Result figures, drawn from the committed tables.

Every panel reads a CSV under results/tables, so a figure cannot show something
the tables do not.
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "results/tables"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
# One colour per mechanism, fixed so a mechanism keeps its colour across figures.
MECH = {"none": "#c1121f", "post": "#005f73", "pre": "#0a9396", "part": "#94d2bd"}
MECH_LABEL = {"none": "no enforcement", "post": "post-filter",
              "pre": "pre-filter", "part": "partitioned"}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf|.png")


def fig_enforcement():
    """What each mechanism delivers as the clearance widens."""
    sw = pd.read_csv(TAB / "access_control_sweep.csv")
    b = sw[(sw.bucket == "answerable") & (sw.labelling == "compartment")
           & (sw.embedder == "bge-m3")]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))

    rec = b[b.metric == "recall"].pivot_table(index="granted_fraction",
                                              columns="mechanism", values="mean")
    for m in ("none", "post", "pre"):
        ls = "--" if m == "post" else "-"
        ax[0].plot(rec.index * 100, rec[m], ls, marker="o", ms=3.5,
                   color=MECH[m], label=MECH_LABEL[m], lw=1.6)
    ax[0].set_title("Recall against gold passages", fontsize=9)
    ax[0].set_ylabel("recall@5")
    # Annotate in axes fraction, not data coordinates: the y range here is a
    # narrow band around 0.65 to 0.78 and a data-coordinate label lands outside
    # the axes, which is how the first draft of this figure grew a blank half.
    ax[0].annotate("post-filter and no enforcement\ncoincide exactly",
                   xy=(50, rec.loc[0.5, "post"]), xycoords="data",
                   xytext=(0.30, 0.24), textcoords="axes fraction",
                   fontsize=7.5, color="#444",
                   arrowprops=dict(arrowstyle="->", color="#888", lw=0.7))

    cs = b[b.metric == "context_size"].pivot_table(index="granted_fraction",
                                                   columns="mechanism", values="mean")
    for m in ("none", "post", "pre"):
        ax[1].plot(cs.index * 100, cs[m], marker="o", ms=3.5,
                   color=MECH[m], label=MECH_LABEL[m], lw=1.6)
    ax[1].set_title("Context passages delivered of five", fontsize=9)
    ax[1].set_ylabel("passages")
    ax[1].set_ylim(0, 5.4)

    lk = b[(b.metric == "leaked") & (b.mechanism == "none")].set_index("granted_fraction")
    ax[2].plot(lk.index * 100, lk["mean"], marker="o", ms=3.5,
               color=MECH["none"], lw=1.6, label="no enforcement")
    ax[2].fill_between(lk.index * 100, lk["lo"], lk["hi"], color=MECH["none"], alpha=0.15)
    ax[2].axhline(0, color=MECH["pre"], lw=1.6, label="any enforcement")
    ax[2].set_title("Unauthorised passages in the context", fontsize=9)
    ax[2].set_ylabel("passages per query")

    for a in ax:
        a.set_xlabel("clearance breadth (% of compartments)")
        a.legend(fontsize=7.5, loc="best")
    save(fig, "fig_enforcement")


def fig_channels():
    """Corpus text that reaches the prompt without passing the retriever."""
    ch = pd.read_csv(TAB / "prompt_channels.csv")
    st = ch[~ch.per_query & (ch.documents > 0)].copy()
    st["short"] = [c.replace("obliqa_", "").replace(".json", "").replace("_", " ")
                   for c in st.carrier]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.2))

    colours = ["#bb3e03" if c == "exemplar" else "#ee9b00" for c in st.channel]
    y = np.arange(len(st))
    ax[0].barh(y, st.tokens, color=colours, height=0.62)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(st.short, fontsize=7.5)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("verbatim corpus tokens carried into every prompt")
    for i, r in enumerate(st.itertuples()):
        ax[0].text(r.tokens + 60, i, f"{int(r.documents)} instruments",
                   va="center", fontsize=7.2, color="#333")
    ax[0].set_xlim(0, st.tokens.max() * 1.42)
    ax[0].set_title("What each static channel carries", fontsize=9)

    from math import comb
    N = 40
    def hyp(n, d):
        return comb(N - d, n - d) / comb(N, n) if n >= d else 0.0
    ns = np.arange(1, N + 1)
    f = ns / N
    for r in st.itertuples():
        ax[1].plot(f * 100, [hyp(int(n), int(r.documents)) * 100 for n in ns], lw=1.5,
                   color="#bb3e03" if r.channel == "exemplar" else "#ee9b00",
                   alpha=0.95 if r.channel == "exemplar" else 0.55)
    ax[1].axvline(50, color="#888", ls=":", lw=1)
    ax[1].annotate("0.42% of readers at\nhalf clearance", xy=(50, 0.42),
                   xycoords="data", xytext=(0.42, 0.55), textcoords="axes fraction",
                   fontsize=7.5, color="#444",
                   arrowprops=dict(arrowstyle="->", color="#888", lw=0.7))
    ax[1].set_xlabel("clearance breadth (% of compartments)")
    ax[1].set_ylabel("% of readers authorised for the whole channel")
    ax[1].set_title("How rarely a reader may see all of it", fontsize=9)
    save(fig, "fig_channels")


def fig_disclosure():
    """What the model repeats from what it was shown."""
    import leakage
    disclosed, diag = leakage.load_matrix()
    import access_control as ac
    meta = ac.load_corpus()
    qs = ac.load_questions(meta)
    gold_of = {q["qid"]: q["gold"] for q in qs}
    uid_row = {m["uid"]: i for i, m in enumerate(meta)}

    by_rank = collections.defaultdict(list)
    by_gold = collections.defaultdict(list)
    for line in (ROOT / "results/disclosure/matrix_a4_gemma3_12b.jsonl").open():
        r = json.loads(line)
        k = (r["qid"], r["generator"], r["uid"])
        if r["verdict"] is None or k not in disclosed:
            continue
        by_rank[r["rank"]].append(disclosed[k])
        by_gold[uid_row.get(r["uid"]) in gold_of.get(r["qid"], set())].append(disclosed[k])

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.1))
    ranks = sorted(by_rank)
    vals = [np.mean(by_rank[r]) for r in ranks]
    ax[0].bar([r + 1 for r in ranks], vals, color="#005f73", width=0.62)
    for r, v in zip(ranks, vals):
        ax[0].text(r + 1, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)
    ax[0].set_xlabel("retrieval rank")
    ax[0].set_ylabel("fraction repeated in the answer")
    ax[0].set_title("Disclosure falls with rank", fontsize=9)
    ax[0].set_ylim(0, 0.92)

    g, ng = np.mean(by_gold[True]), np.mean(by_gold[False])
    ax[1].bar([0, 1], [g, ng], color=["#0a9396", "#bb3e03"], width=0.55)
    for i, v in enumerate([g, ng]):
        ax[1].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["supports the answer", "merely retrieved"], fontsize=8)
    ax[1].set_ylabel("fraction repeated in the answer")
    ax[1].set_title("Irrelevant material is repeated too", fontsize=9)
    ax[1].set_ylim(0, 1.0)
    save(fig, "fig_disclosure")


def fig_leakage():
    """Exposure and disclosure as the clearance widens."""
    lk = pd.read_csv(TAB / "leakage_by_clearance.csv")
    p = lk.pivot_table(index="granted_fraction", columns="metric", values="mean")
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    x = p.index * 100
    ax.plot(x, p.context_leak_per_query, marker="o", ms=4, lw=1.8,
            color="#c1121f", label="reached the context")
    ax.plot(x, p.answer_leak_per_query, marker="s", ms=4, lw=1.8,
            color="#bb3e03", label="repeated in the answer")
    ax.fill_between(x, p.answer_leak_per_query, p.context_leak_per_query,
                    color="#c1121f", alpha=0.10)
    ax.set_xlabel("clearance breadth (% of compartments)")
    ax.set_ylabel("unauthorised passages per query")
    ax.set_title("Unauthorised material under no enforcement", fontsize=9)
    ax.legend(fontsize=8)
    ax.annotate(f"conversion holds at {p.conversion.mean():.3f}\nacross the sweep",
                xy=(50, p.loc[0.5, "answer_leak_per_query"]), xycoords="data",
                xytext=(0.36, 0.62), textcoords="axes fraction",
                fontsize=7.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.7))
    save(fig, "fig_leakage")


def fig_lattice():
    """The policy: levels by compartment, sized by passage count."""
    import access_control as ac
    meta = ac.load_corpus()
    lv = ac.document_levels()
    counts = collections.Counter((lv.get(str(m["doc_id"]), 2), str(m["doc_id"]))
                                 for m in meta)
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    colours = {0: "#005f73", 1: "#0a9396", 2: "#94d2bd"}
    xpos = {0: 0, 1: 0, 2: 0}
    for (level, doc), n in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1])):
        ax.barh(level, n, left=xpos[level], height=0.62,
                color=colours[level], edgecolor="white", lw=0.7)
        if n > 600:
            ax.text(xpos[level] + n / 2, level, doc, ha="center", va="center",
                    fontsize=6.5, color="white")
        xpos[level] += n
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"L0 regulations\n{xpos[0]:,} passages",
                        f"L1 rulebooks\n{xpos[1]:,} passages",
                        f"L2 guidance\n{xpos[2]:,} passages"], fontsize=8)
    ax.set_xlabel("passages, one bar segment per instrument")
    ax.set_title("The policy lattice, taken from the publisher's instrument taxonomy",
                 fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_lattice")


if __name__ == "__main__":
    print("drawing result figures")
    fig_enforcement()
    fig_channels()
    fig_disclosure()
