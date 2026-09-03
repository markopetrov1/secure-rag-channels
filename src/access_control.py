"""Access control over a retrieval corpus, and what enforcing it costs.

The professor's brief asks what secure retrieval costs. Answering that needs a
security policy over a corpus that has none, so the policy is constructed here,
and how it is constructed is the part a reviewer will attack first. Two
independent labellings are therefore provided and every result should be
reported under both.

COMPARTMENT LABELLING. Each passage inherits the compartment of its source
document. ObliQA is drawn from 40 Abu Dhabi Global Market rulebooks, so a
compartment is a rulebook and a clearance is a coherent need-to-know set that a
role would plausibly hold. This is complete, every passage has exactly one
label, and it is not a random slice of the corpus, which is what makes it
defensible.

ADDRESSEE LABELLING. Regulatory text names who it binds. Passages that speak to
an Authorised Person, a Recognised Body, a Fund Manager, an Insurer, an Islamic
financial business or a Representative Office are labelled by that addressee.
This is the corpus segmenting itself rather than us segmenting it, which is a
stronger claim, but it is sparse: most passages name no addressee. It is used as
a robustness check on the compartment results, not as the primary policy.

ENFORCEMENT. Four mechanisms, matching the brief's taxonomy.

  none  retrieve over the whole corpus and ignore the policy. Not deployable.
        Included because it is what a pipeline without authorization does, and
        the harm it does is the quantity this study is trying to establish.
  post  retrieve over the whole corpus, then drop what the user may not see.
        Authorization-correct by construction, but it returns fewer than k
        passages, so what it costs is context, not necessarily recall.
  pre   retrieve only within the authorized set. The brief's
        Authorization-First Retrieval.
  part  one index per compartment, search the authorized ones, merge. Under
        exact search this returns the same passages as pre, so what separates
        them is build cost and staleness, not quality. Reported rather than
        hidden, because a difference that does not exist is worth stating.

CONDITIONING. Recall under enforcement falls for two unrelated reasons: the
mechanism failed to surface an authorized answer, or the answer was never
authorized. Pooling them is what produces the claim that post-filtering wrecks
recall. Every recall figure here is therefore reported separately for
answerable questions, where every gold passage is authorized, and for
unanswerable ones, where refusal is the correct behaviour.

Runs entirely on cached embeddings. No GPU, no model calls.
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data/index"
TAB = ROOT / "results/tables"
QUESTIONS = ROOT / "data/processed/obliqa_test_sample.jsonl"

MECHANISMS = ("none", "post", "pre", "part")

# Addressee patterns, in priority order: a passage takes the first that matches.
# Ordered most specific first so that a passage naming both a Fund and an
# Authorised Person is filed under the narrower of the two.
ADDRESSEE = [
    ("islamic", r"\bIslamic (?:Financial Business|Window)\b|\bShari'?ah?\b"),
    ("rep_office", r"\bRepresentative Office\b"),
    ("recognised_body", r"\bRecognised (?:Body|Investment Exchange|Clearing House)\b"),
    ("fund", r"\bFund Manager\b|\bPublic Fund\b|\bExempt Fund\b|\bQualified Investor Fund\b"),
    ("insurer", r"\bInsurer\b|\bInsurance (?:Intermediary|Manager)\b"),
    ("authorised_person", r"\bAuthorised Person\b"),
]
UNLABELLED = "general"


# ------------------------------------------------------------------ loading

def load_corpus():
    """Passage metadata in index row order, so row i is embedding row i."""
    return [json.loads(l) for l in (IDX / "obliqa_meta.jsonl").open()]


def load_questions(meta):
    """Questions with their gold passages resolved to index rows.

    A question whose gold passages cannot all be located is kept with the rows
    that did resolve; dropping it silently would bias the answerable rate.
    """
    key = {(str(m["doc_id"]), str(m["passage_id"])): i for i, m in enumerate(meta)}
    out = []
    for line in QUESTIONS.open():
        q = json.loads(line)
        rows = {key.get((str(p["DocumentID"]), str(p["PassageID"])))
                for p in q["gold_passages"]}
        rows.discard(None)
        out.append({"qid": q["qid"], "gold": rows, "n_gold_declared": len(q["gold_passages"])})
    return out


def load_embeddings(embedder):
    """Row-normalised corpus and query matrices for one embedder."""
    d = np.load(IDX / f"obliqa_{embedder}_dense.npy").astype(np.float32)
    q = np.load(IDX / f"obliqa_{embedder}_queries.npy").astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return d, q


# ------------------------------------------------------------------ labelling

def labels_compartment(meta):
    """One label per passage, taken from its source document."""
    return np.array([str(m["doc_id"]) for m in meta])


def labels_addressee(meta):
    """One label per passage, taken from whom the text says it binds."""
    pats = [(name, re.compile(p, re.I)) for name, p in ADDRESSEE]
    out = []
    for m in meta:
        t = m["text"]
        hit = next((name for name, p in pats if p.search(t)), UNLABELLED)
        out.append(hit)
    return np.array(out)


# ------------------------------------------------------------ the MLS lattice

# ADGM publishes three kinds of instrument and the corpus names them, so the
# level dimension is read off the document map rather than assigned by us:
# Regulations are primary legislation binding on everyone, Rulebooks bind
# categories of authorised firm, and Guidance notes address narrow activities
# such as virtual assets, mining or petroleum reporting. Generality of
# applicability decreasing is the ordering, so the broadest instrument sits at
# the bottom and the most specialised at the top, which is the direction that
# makes a clearance a "read everything at or below my level" rule.
#
# This is a stipulated policy. ADGM does not classify these documents and we do
# not claim it does. What is not stipulated is the STRUCTURE: the partition into
# levels and compartments comes from the publisher's own instrument taxonomy,
# which is what stops the lattice being a researcher's invention end to end.
LEVEL_NAMES = {0: "regulation", 1: "rulebook", 2: "guidance"}


def document_levels(map_path=None):
    """doc_id -> level, from the dataset's own document map."""
    import re
    p = map_path or (ROOT / "data/raw/ObliQADataset/scripts/DocumentMap.rtf")
    if not Path(p).exists():
        return {}
    raw = Path(p).read_text(errors="ignore")
    txt = re.sub(r"\\'[0-9a-f]{2}", "-", raw)
    txt = re.sub(r"\\[a-zA-Z]+-?\d*\s?", " ", txt)
    txt = txt.replace("{", " ").replace("}", " ").replace("\\", " ")
    out = {}
    for i, body in re.findall(r"(\d+)\s*:\s*\[(.*?)\]", txt, re.S):
        names = re.findall(r'"([^"]+)"', body)
        if not names:
            continue
        n = names[0]
        if re.match(r"^[A-Z]+_VER\d", n):
            lvl = 1                                   # rulebook
        elif re.search(r"regulations|FSMR", n, re.I):
            lvl = 0                                   # primary legislation
        else:
            lvl = 2                                   # guidance and supplements
        out[str(int(i))] = lvl
    return out


def labels_lattice(meta):
    """One (level, compartment) label per passage, as 'L<level>/<doc>'."""
    lv = document_levels()
    return np.array([f"L{lv.get(str(m['doc_id']), 2)}/{m['doc_id']}" for m in meta])


def lattice_readable(labels, cleared_level, cleared_compartments):
    """Bell-LaPadula no-read-up, intersected with need-to-know compartments.

    A passage is readable when its level does not exceed the clearance AND its
    compartment is granted. Both conditions, which is what separates a lattice
    from the flat compartment set this module started with.
    """
    lv = np.array([int(x.split("/")[0][1:]) for x in labels])
    comp = np.array([x.split("/")[1] for x in labels])
    return (lv <= cleared_level) & np.isin(comp, list(cleared_compartments))


LABELLERS = {"compartment": labels_compartment, "addressee": labels_addressee,
             "lattice": labels_lattice}


# ------------------------------------------------------------------ retrieval

def topk(scores, k, restrict=None):
    """Indices of the k highest scores, descending, optionally within a subset."""
    if restrict is None:
        if k >= scores.size:
            return np.argsort(-scores)
        part = np.argpartition(-scores, k)[:k]
        return part[np.argsort(-scores[part])]
    if restrict.size == 0:
        return np.empty(0, dtype=int)
    sub = scores[restrict]
    if k >= sub.size:
        return restrict[np.argsort(-sub)]
    part = np.argpartition(-sub, k)[:k]
    return restrict[part[np.argsort(-sub[part])]]


def retrieve(mech, scores, k, allowed_mask, allowed_rows):
    """The passages one enforcement mechanism puts in the context window."""
    if mech == "none":
        return topk(scores, k)
    if mech == "post":
        wide = topk(scores, k)
        return wide[allowed_mask[wide]]
    # pre and part return the same rows under exact search; they are separated
    # in the cost model, not here, and saying so is the point.
    return topk(scores, k, restrict=allowed_rows)


# ------------------------------------------------------------------ measuring

def measure(mech, scores, k, gold, allowed_mask, allowed_rows):
    """Per-question outcome for one mechanism under one clearance."""
    got = retrieve(mech, scores, k, allowed_mask, allowed_rows)
    got_set = set(got.tolist())
    unauth = int((~allowed_mask[got]).sum()) if got.size else 0
    hit = len(got_set & gold)
    return {
        "recall": hit / len(gold) if gold else np.nan,
        "any_hit": float(hit > 0) if gold else np.nan,
        "leaked": unauth,
        "authorized": float(unauth == 0),
        "context_size": int(got.size),
    }


def sweep(embedder="bge-m3", labelling="compartment", k=5,
          fractions=(0.1, 0.25, 0.5, 0.75, 0.9), draws=40, seed=7):
    """Every mechanism over a range of clearance breadths.

    A clearance is a random subset of labels rather than of passages, which is
    what makes it a need-to-know set. Results are averaged over `draws`
    independent clearances per breadth so no single lucky assignment carries a
    conclusion, and the spread across draws is reported alongside the mean.
    """
    meta = load_corpus()
    qs = load_questions(meta)
    D, Q = load_embeddings(embedder)
    labels = LABELLERS[labelling](meta)
    universe = sorted(set(labels))
    S = Q @ D.T
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        n_lab = max(1, int(round(frac * len(universe))))
        per_draw = collections.defaultdict(list)
        for _ in range(draws):
            granted = set(rng.choice(universe, n_lab, replace=False).tolist())
            allowed_mask = np.isin(labels, list(granted))
            allowed_rows = np.flatnonzero(allowed_mask)
            acc = collections.defaultdict(lambda: collections.defaultdict(list))
            for qi, q in enumerate(qs):
                gold = q["gold"]
                if not gold:
                    continue
                answerable = all(allowed_mask[g] for g in gold)
                bucket = "answerable" if answerable else "unanswerable"
                s = S[qi]
                for mech in MECHANISMS:
                    m = measure(mech, s, k, gold, allowed_mask, allowed_rows)
                    for key, val in m.items():
                        acc[(mech, bucket)][key].append(val)
                    acc[(mech, "all")]["recall"].append(m["recall"])
                    acc[(mech, "all")]["leaked"].append(m["leaked"])
                    acc[(mech, "all")]["authorized"].append(m["authorized"])
                    acc[(mech, "all")]["context_size"].append(m["context_size"])
                acc[("_q", bucket)]["n"].append(1.0)
            for (mech, bucket), d in acc.items():
                for key, vals in d.items():
                    per_draw[(mech, bucket, key)].append(float(np.mean(vals)) if vals else np.nan)
            n_ans = len(acc[("_q", "answerable")]["n"])
            n_tot = n_ans + len(acc[("_q", "unanswerable")]["n"])
            per_draw[("_policy", "all", "answerable_rate")].append(n_ans / n_tot)
            per_draw[("_policy", "all", "authorized_passages")].append(
                float(allowed_mask.mean()))
        for (mech, bucket, key), vals in sorted(per_draw.items()):
            a = np.array(vals, dtype=float)
            rows.append({
                "embedder": embedder, "labelling": labelling, "k": k,
                "granted_fraction": frac, "n_labels_granted": n_lab,
                "mechanism": mech, "bucket": bucket, "metric": key,
                "mean": float(np.nanmean(a)),
                "lo": float(np.nanpercentile(a, 2.5)),
                "hi": float(np.nanpercentile(a, 97.5)),
                "draws": int(np.isfinite(a).sum()),
            })
    return rows


def passage_tokens(meta, encoding="cl100k_base"):
    """Token count of every passage, so a context can be priced exactly.

    Counted once and reused. The absolute figure depends on the tokeniser, but
    every mechanism is priced with the same one, and what is reported is
    the ratio between mechanisms rather than an absolute bill.
    """
    import tiktoken
    enc = tiktoken.get_encoding(encoding)
    return np.array([len(enc.encode(m["text"])) for m in meta], dtype=np.int64)


def cost_sweep(embedder="bge-m3", labelling="compartment", k=5,
               fractions=(0.1, 0.25, 0.5, 0.75, 0.9), draws=40, seed=7):
    """What each mechanism puts in the prompt, in tokens.

    Post-filtering returns fewer passages than it was asked for, so it is the
    only mechanism that makes a query cheaper than having no access control at
    all. Whether that discount costs answer quality is the question generation
    has to settle; here it is only priced.
    """
    meta = load_corpus()
    qs = load_questions(meta)
    D, Q = load_embeddings(embedder)
    labels = LABELLERS[labelling](meta)
    tok = passage_tokens(meta)
    universe = sorted(set(labels))
    S = Q @ D.T
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions:
        n_lab = max(1, int(round(frac * len(universe))))
        per_draw = collections.defaultdict(lambda: collections.defaultdict(list))
        for _ in range(draws):
            granted = set(rng.choice(universe, n_lab, replace=False).tolist())
            allowed_mask = np.isin(labels, list(granted))
            allowed_rows = np.flatnonzero(allowed_mask)
            acc = collections.defaultdict(list)
            for qi, q in enumerate(qs):
                if not q["gold"]:
                    continue
                if not all(allowed_mask[g] for g in q["gold"]):
                    continue                      # answerable questions only
                s = S[qi]
                for mech in MECHANISMS:
                    got = retrieve(mech, s, k, allowed_mask, allowed_rows)
                    acc[mech].append(int(tok[got].sum()) if got.size else 0)
            for mech, vals in acc.items():
                per_draw[mech]["context_tokens"].append(float(np.mean(vals)))
        for mech, d in per_draw.items():
            for key, vals in d.items():
                a = np.array(vals, dtype=float)
                rows.append({
                    "embedder": embedder, "labelling": labelling, "k": k,
                    "granted_fraction": frac, "mechanism": mech, "metric": key,
                    "mean": float(np.nanmean(a)),
                    "lo": float(np.nanpercentile(a, 2.5)),
                    "hi": float(np.nanpercentile(a, 97.5)),
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedders", default="bge-m3")
    ap.add_argument("--labellings", default="compartment,addressee")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--out", default="access_control_sweep.csv")
    ap.add_argument("--cost", action="store_true",
                    help="also price the context each mechanism delivers")
    a = ap.parse_args()

    meta = load_corpus()
    print(f"corpus {len(meta):,} passages")
    for name, fn in LABELLERS.items():
        lab = fn(meta)
        c = collections.Counter(lab)
        cov = 1.0 - (c.get(UNLABELLED, 0) / len(lab))
        print(f"  {name:12s} {len(c):3d} labels, "
              f"{cov:5.1%} of passages carry a specific one, "
              f"largest {max(c.values()):,}")

    rows = []
    for emb in [e.strip() for e in a.embedders.split(",") if e.strip()]:
        for lab in [l.strip() for l in a.labellings.split(",") if l.strip()]:
            print(f"\nsweeping {emb} / {lab} ...")
            rows += sweep(emb, lab, k=a.k, draws=a.draws)
    import pandas as pd
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / a.out, index=False)
    print(f"\nwrote results/tables/{a.out}: {len(df):,} rows")

    piv = df[(df.metric == "recall") & (df.bucket == "answerable")]
    print("\n== recall@k on answerable questions, the enforcement mechanism's own cost ==")
    for (emb, lab), g in piv.groupby(["embedder", "labelling"]):
        print(f"\n{emb} / {lab}")
        t = g.pivot_table(index="granted_fraction", columns="mechanism", values="mean")
        print(t.to_string(float_format=lambda x: f"{x:.4f}"))
    if a.cost:
        crows = []
        for emb in [e.strip() for e in a.embedders.split(",") if e.strip()]:
            for lab in [l.strip() for l in a.labellings.split(",") if l.strip()]:
                crows += cost_sweep(emb, lab, k=a.k, draws=a.draws)
        cdf = pd.DataFrame(crows)
        cdf.to_csv(TAB / "access_control_cost.csv", index=False)
        print(f"\nwrote results/tables/access_control_cost.csv: {len(cdf):,} rows")
        print("\n== context tokens per query, answerable questions ==")
        for (emb, lab), g in cdf.groupby(["embedder", "labelling"]):
            t = g.pivot_table(index="granted_fraction", columns="mechanism", values="mean")
            t["post/none"] = t["post"] / t["none"]
            print(f"\n{emb} / {lab}")
            print(t.to_string(float_format=lambda x: f"{x:,.1f}"))

    leak = df[(df.metric == "leaked") & (df.bucket == "answerable")
              & (df.mechanism == "none")]
    print("\n== unauthorized passages in the context under no access control ==")
    for (emb, lab), g in leak.groupby(["embedder", "labelling"]):
        s = g.set_index("granted_fraction")["mean"]
        print(f"{emb} / {lab}: " + ", ".join(f"{i:.0%}->{v:.2f}" for i, v in s.items()))


if __name__ == "__main__":
    main()
