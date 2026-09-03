"""Which channels carry corpus text into a prompt, and which of them authorisation can see.

Access control in retrieval-augmented generation is enforced at one place, where
the retriever meets the index, and every metric the field defines is computed
over the set of passages that retriever returns. That is sound only if the
retriever is the sole route by which corpus text reaches the model. It is not.

This module decomposes each deployed pipeline into the channels through which
regulatory text actually enters the prompt, attributes every channel to its
source documents, and asks of each whether a retrieval-side authorisation hook
would inspect it.

  retrieved   passages the retriever returned for this question. Per query,
              clearance-dependent, and the only channel any published
              enforcement mechanism inspects.
  exemplar    in-context demonstrations. Their reference answers are built from
              the gold passages of training questions, so they are verbatim
              corpus text. Fixed when the pipeline is deployed, identical for
              every user, and never re-derived per query.
  compiled    demonstrations an optimiser selected at compile time. Which
              documents they come from is decided by the optimiser seed, before
              any user exists and before any clearance is known.
  routed      whole-document prefixes a long-context arm stuffs into the window.

Attribution is by exact 8-gram match against the corpus. At that length a
coincidental match between unrelated regulatory prose is vanishingly unlikely,
and the distinctiveness of the matched grams is reported so the reader can check
that rather than take it on trust.

Runs on committed artefacts. No GPU, no model calls.
"""
import collections
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data/index"
GEN = ROOT / "results/generations"
DSPY = ROOT / "results/dspy"
TAB = ROOT / "results/tables"
PROC = ROOT / "data/processed"
NGRAM = 8

INSPECTED = {"retrieved": True, "exemplar": False, "compiled": False, "routed": False}


def words(t):
    return re.findall(r"[a-z0-9]+", (t or "").lower())


def grams(t, n=NGRAM):
    w = words(t)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_index(meta):
    """8-gram -> set of documents containing it.

    A gram in more than one document cannot attribute text to a single source,
    so those are counted separately rather than assigned arbitrarily.
    """
    ix = collections.defaultdict(set)
    for m in meta:
        d = m["doc_id"]
        for g in grams(m["text"]):
            ix[g].add(d)
    return ix


def attribute(text, ix):
    """Documents whose verbatim text appears in this string."""
    gs = grams(text)
    if not gs:
        return set(), 0, 0
    hit = [ix[g] for g in gs if g in ix]
    unique = {next(iter(s)) for s in hit if len(s) == 1}
    ambiguous = sum(1 for s in hit if len(s) > 1)
    return unique, len(hit), ambiguous


def tokens(text, enc=None):
    if enc is None:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


# ------------------------------------------------------------- the channels

def exemplar_channel():
    """Reference answers of the few-shot pool, exactly as run_generation builds them."""
    p = PROC / "obliqa_fewshot_dev.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.open():
        ex = json.loads(line)
        ref = " ".join(g["Passage"] for g in ex["gold_passages"])[:1200]
        out.append({"qid": ex["qid"], "text": ref,
                    "declared_docs": sorted({str(g["DocumentID"]) for g in ex["gold_passages"]})})
    return out


def compiled_channel():
    """Demonstration text inside each committed optimiser artefact."""
    out = []
    for f in sorted(DSPY.glob("*.json")):
        blob = json.dumps(json.load(f.open()))
        out.append({"artefact": f.name, "text": blob})
    return out


def retrieved_channel(meta):
    """Passages each generation record says it retrieved."""
    uid_doc = {m["uid"]: m["doc_id"] for m in meta}
    uid_tok = {m["uid"]: m["text"] for m in meta}
    out = collections.defaultdict(list)
    for f in sorted(GEN.glob("obliqa_open_*.jsonl")):
        stem = f.stem
        parts = stem.split("_")
        arm, gen = parts[2], "_".join(parts[3:])
        for line in f.open():
            r = json.loads(line)
            ids = (r.get("meta") or {}).get("retrieved_ids") or []
            out[(arm, gen)].append([(i, uid_doc.get(i), uid_tok.get(i, "")) for i in ids])
    return out


def main():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    meta = [json.loads(l) for l in (IDX / "obliqa_meta.jsonl").open()]
    print(f"corpus {len(meta):,} passages, building {NGRAM}-gram index ...")
    ix = build_index(meta)
    print(f"  {len(ix):,} distinct grams, "
          f"{sum(1 for v in ix.values() if len(v) == 1) / len(ix):.1%} unique to one document\n")

    rows = []

    # ---- exemplar channel -------------------------------------------------
    ex = exemplar_channel()
    if ex:
        docs, hits, amb = set(), 0, 0
        tok = 0
        for e in ex:
            d, h, a = attribute(e["text"], ix)
            docs |= d; hits += h; amb += a
            tok += tokens(e["text"], enc)
        rows.append({"channel": "exemplar", "carrier": "few-shot pool",
                     "n_items": len(ex), "documents": len(docs),
                     "doc_list": ",".join(str(x) for x in sorted(docs, key=int)),
                     "tokens": tok, "grams_matched": hits, "grams_ambiguous": amb,
                     "per_query": False, "clearance_aware": False,
                     "inspected_by_retrieval_hook": INSPECTED["exemplar"]})

    # ---- compiled channel -------------------------------------------------
    for c in compiled_channel():
        d, h, a = attribute(c["text"], ix)
        rows.append({"channel": "compiled", "carrier": c["artefact"],
                     "n_items": 1, "documents": len(d),
                     "doc_list": ",".join(str(x) for x in sorted(d, key=int)),
                     "tokens": tokens(c["text"], enc), "grams_matched": h,
                     "grams_ambiguous": a, "per_query": False,
                     "clearance_aware": False,
                     "inspected_by_retrieval_hook": INSPECTED["compiled"]})

    # ---- retrieved channel ------------------------------------------------
    ret = retrieved_channel(meta)
    for (arm, gen), per_q in sorted(ret.items()):
        if not any(per_q):
            continue
        docs = {d for q in per_q for _, d, _ in q if d is not None}
        tok = int(np.mean([sum(tokens(t, enc) for _, _, t in q) for q in per_q])) if per_q else 0
        rows.append({"channel": "retrieved", "carrier": f"{arm}/{gen}",
                     "n_items": len(per_q), "documents": len(docs),
                     "doc_list": "", "tokens": tok, "grams_matched": -1,
                     "grams_ambiguous": -1, "per_query": True,
                     "clearance_aware": True,
                     "inspected_by_retrieval_hook": INSPECTED["retrieved"]})

    import pandas as pd
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "prompt_channels.csv", index=False)

    print("== channels that carry verbatim corpus text into the prompt ==")
    show = df[~df.per_query][["channel", "carrier", "documents", "doc_list",
                              "tokens", "grams_matched", "inspected_by_retrieval_hook"]]
    print(show.to_string(index=False))

    print("\n== what a retrieval-side authorisation hook inspects ==")
    for ch in ("retrieved", "exemplar", "compiled"):
        sub = df[df.channel == ch]
        if sub.empty:
            continue
        print(f"  {ch:10s} {'inspected' if INSPECTED[ch] else 'NOT INSPECTED':14s} "
              f"{len(sub)} carrier(s), "
              f"{'re-derived per query' if INSPECTED[ch] else 'fixed before any clearance exists'}")

    # ---- the number that matters -----------------------------------------
    print("\n== probability a clearance authorises a static channel outright ==")
    print("Independent uniform draw over the 40 compartments, which is the")
    print("assumption the role-correlated draws in the policy layer will test.")
    static = df[~df.per_query & (df.documents > 0)]
    print(f"\n{'carrier':40s} {'docs':>5s} " + " ".join(f"{f:>7.0%}" for f in (0.25, 0.5, 0.75, 0.9)))
    for r in static.itertuples():
        ps = " ".join(f"{f ** r.documents:7.3%}" for f in (0.25, 0.5, 0.75, 0.9))
        print(f"{r.carrier:40s} {r.documents:5d} {ps}")
    print(f"\nwrote results/tables/prompt_channels.csv")


if __name__ == "__main__":
    main()
