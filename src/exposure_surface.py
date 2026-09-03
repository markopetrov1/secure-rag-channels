"""What every generation design puts in front of a reader, channel by channel.

The central claim concerns nine deployed designs, but only the two carrying
static channels are interesting enough to tabulate individually. This module
produces the full surface, so the claim that the remaining seven are confined to
the retrieved channel is shown rather than asserted, and so the one design whose
channel is neither static nor a normal retrieval result is not overlooked.

Channels, and whether a retrieval-side authorisation hook inspects them:

  retrieved  passages the retriever returned for this question. Inspected.
  routed     whole documents a coarse router stuffs into a long context window.
             Inspected in principle, since a router selected them, but at
             document rather than passage granularity, so a clearance expressed
             over passages cannot be applied without splitting the document.
  exemplar   in-context demonstrations built from the collection. Not inspected.
  compiled   demonstrations an optimiser froze at compile time. Not inspected.

The long-context design records only how many documents it stuffed and how many
tokens they occupied, not which documents, so its routing is recomputed here
from the same BM25 ranking over concatenated documents that produced it.
"""
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data/index"
GEN = ROOT / "results/generations"
PROC = ROOT / "data/processed"
TAB = ROOT / "results/tables"

INSPECTED = {"retrieved": True, "routed": True, "exemplar": False, "compiled": False}
DESIGNS = [
    ("a1", "Zero-shot", []),
    ("a2o", "One-shot ICL", ["exemplar"]),
    ("a2", "Few-shot ICL", ["exemplar"]),
    ("a3", "Long context", ["routed"]),
    ("a4", "Dense RAG", ["retrieved"]),
    ("a5", "Hybrid RAG", ["retrieved"]),
    ("a7", "Compiled RAG", ["compiled", "retrieved"]),
    ("a8", "Agentic ReAct", ["retrieved"]),
    ("a8c", "Agentic critique", ["retrieved"]),
]


def tok(text, enc=None):
    import tiktoken
    enc = enc or tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def route_documents(meta, budget=24000, k_docs=8):
    """Recompute the long-context router: BM25 over whole documents, stuffed to budget."""
    from rank_bm25 import BM25Okapi
    docs = collections.defaultdict(list)
    for m in meta:
        docs[str(m["doc_id"])].append(m["text"])
    names = sorted(docs, key=int)
    joined = {n: "\n".join(docs[n]) for n in names}
    corpus = [re.findall(r"[a-z0-9]+", joined[n].lower()) for n in names]
    bm = BM25Okapi(corpus)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    dtok = {n: len(enc.encode(joined[n])) for n in names}
    out = {}
    for line in (PROC / "obliqa_test_sample.jsonl").open():
        q = json.loads(line)
        scores = bm.get_scores(re.findall(r"[a-z0-9]+", q["question"].lower()))
        order = [names[i] for i in np.argsort(-scores)[:k_docs]]
        used, picked = 0, []
        for n in order:
            if used + dtok[n] > budget:
                if budget - used > 500:
                    picked.append(n)
                break
            picked.append(n)
            used += dtok[n]
        out[q["qid"]] = (picked, min(used, budget))
    return out


def main():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    meta = [json.loads(l) for l in (IDX / "obliqa_meta.jsonl").open()]
    uid_doc = {m["uid"]: str(m["doc_id"]) for m in meta}
    uid_tok = {m["uid"]: len(enc.encode(m["text"])) for m in meta}

    ch = pd.read_csv(TAB / "prompt_channels.csv")
    static = ch[~ch.per_query & (ch.documents > 0)]
    fewshot = static[static.channel == "exemplar"].iloc[0]
    compiled = static[static.channel == "compiled"]

    print("recomputing the long-context router ...", flush=True)
    routed = route_documents(meta)
    r_docs = float(np.mean([len(v[0]) for v in routed.values()]))
    r_tok = float(np.mean([v[1] for v in routed.values()]))
    r_uniq = len({d for v in routed.values() for d in v[0]})

    rows = []
    for arm, label, channels in DESIGNS:
        f = GEN / f"obliqa_open_{arm}_qwen3_8b.jsonl"
        recs = [json.loads(l) for l in f.open()] if f.exists() else []
        for chan in channels or ["none"]:
            if chan == "retrieved":
                ids = [i for r in recs for i in (r.get("meta") or {}).get("retrieved_ids", [])]
                per_q = np.mean([len((r.get("meta") or {}).get("retrieved_ids", [])) for r in recs])
                d = len({uid_doc[i] for i in ids if i in uid_doc})
                t = float(np.mean([sum(uid_tok.get(i, 0) for i in (r.get("meta") or {}).get("retrieved_ids", []))
                                   for r in recs]))
                fixed = False
            elif chan == "routed":
                d, t, per_q, fixed = r_uniq, r_tok, r_docs, False
            elif chan == "exemplar":
                n = 1 if arm == "a2o" else 5
                d = int(fewshot.documents) if n == 5 else None
                t = int(fewshot.tokens) if n == 5 else int(fewshot.tokens / 5)
                d = d if d is not None else 2
                per_q, fixed = n, True
            elif chan == "compiled":
                d = int(compiled.documents.mean().round())
                t = float(compiled.tokens.mean())
                per_q, fixed = 1, True
            else:
                d, t, per_q, fixed = 0, 0, 0, False
            rows.append({"arm": arm, "design": label, "channel": chan,
                         "instruments": d, "tokens_per_query": round(t, 1),
                         "units_per_query": round(float(per_q), 2),
                         "fixed_before_any_query": fixed,
                         "inspected": INSPECTED.get(chan, False)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "exposure_surface.csv", index=False)
    print()
    print(df.to_string(index=False))
    print()
    un = df[(~df.inspected) & (df.channel != "none")]
    print(f"designs with an uninspected channel: "
          f"{sorted(set(un.design))}")
    print(f"\nlong-context router: {r_docs:.1f} documents per query, "
          f"{r_tok:,.0f} tokens, {r_uniq} distinct instruments across the sample")
    print(f"wrote results/tables/exposure_surface.csv")


if __name__ == "__main__":
    main()
