"""Embedding-model ablation.

The retrieval results raised a question the rest of the study cannot answer: on
this regulatory corpus, dense retrieval with bge-m3 is beaten by plain BM25.
Either dense retrieval is the wrong tool for regulatory text, or the embedder is
simply not good enough. This module settles it by re-indexing both corpora with
stronger embedders and re-scoring retrieval against the same gold passages.

Models compared:
  bge-m3                     the study's default, and the predecessor's stack
  qwen3-embedding:8b         via ollama
  nvidia/Nemotron-3-Embed-1B via sentence-transformers, "query: "/"passage: " prefixes

Usage:
  python src/embed_ablation.py --model bge-m3 --track obliqa
  python src/embed_ablation.py --model qwen3-embedding:8b --track obliqa
  python src/embed_ablation.py --model nemotron-1b --track obliqa
  python src/embed_ablation.py --report
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data/index"
TAB = ROOT / "results/tables"
KS = [1, 3, 5, 10]

HF_MODELS = {
    "nemotron-1b": "nvidia/Nemotron-3-Embed-1B-BF16",
    "nemotron-8b": "nvidia/Nemotron-3-Embed-8B-BF16",
    "qwen3-emb-8b": "Qwen/Qwen3-Embedding-8B",
    "qwen3-emb-4b": "Qwen/Qwen3-Embedding-4B",
    "bge-m3-hf": "BAAI/bge-m3",
}

# Every model is truncated to the same window. These models advertise 32,768
# tokens, and one ObliQA chunk is 24,312 words long, so sentence-transformers
# pads a batch to that length and asks for 32 GiB of attention. The 99th
# percentile chunk is 339 words, so 1,024 tokens truncates 68 of 13,012 chunks
# and costs nothing that matters, while an uncapped window costs the whole run.
MAX_SEQ = 1024
# Batch sizes chosen so the attention allocation stays small on a 48 GB device
# shared with other work. The 8B models are the constraint.
BATCH = {"nemotron-8b": 8, "qwen3-emb-8b": 8, "qwen3-emb-4b": 16,
         "nemotron-1b": 32, "bge-m3-hf": 32}

# Asymmetric query and passage prefixes, taken from each model card. A wrong
# prefix silently handicaps a model, which would invalidate the comparison the
# ablation exists to make, so they are declared here and echoed at run time.
QWEN_TASK = ("Instruct: Given a regulatory compliance question, retrieve the "
             "passage of the rulebook that answers it\nQuery:")
PREFIXES = {
    "nemotron-1b": ("query: ", "passage: "),
    "nemotron-8b": ("query: ", "passage: "),
    "qwen3-emb-8b": (QWEN_TASK, ""),
    "qwen3-emb-4b": (QWEN_TASK, ""),
    "bge-m3-hf": ("", ""),
}

# RTEB leaderboard scores, for the transfer question the ablation asks. Recorded
# here so the published ranking can be compared against the ranking these
# models achieve on a regulatory corpus with real gold passages.
RTEB = {"nemotron-8b": 78.5, "qwen3-emb-8b": 73.1, "nemotron-1b": 72.4}


def slug(model):
    return model.replace(":", "_").replace("/", "_")


def load_corpus(track):
    return [json.loads(l) for l in (IDX / f"{track}_meta.jsonl").open()]


def load_queries(track):
    rows = [json.loads(l) for l in
            (ROOT / f"data/processed/{track}_test_sample.jsonl").open()]
    return [(r["qid"], r["question"], r["gold_passages"]) for r in rows]


def embed_ollama(model, texts, purpose, run_id, batch=32):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from engine import OllamaClient
    c = OllamaClient()
    out = []
    for i in range(0, len(texts), batch):
        out.extend(c.embed(model, texts[i:i + batch], purpose=purpose, run_id=run_id))
        if (i // batch) % 25 == 0:
            print(f"  {i}/{len(texts)}", flush=True)
    return np.asarray(out, dtype=np.float32)


def embed_hf(model_key, texts, is_query, batch=None):
    """Embed with a HuggingFace model. Precision falls back where needed.

    This device is Turing, which has no native bfloat16, so a model published in
    bf16 is loaded in fp16 instead. The precision actually used is printed,
    because it is a property of the measurement and not an implementation detail.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    name = HF_MODELS[model_key]
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        m = SentenceTransformer(name, device="cuda", trust_remote_code=True,
                                model_kwargs={"torch_dtype": dtype})
    except Exception as e:
        print(f"  fp16 load failed ({type(e).__name__}), retrying in fp32")
        dtype = torch.float32
        m = SentenceTransformer(name, device="cuda", trust_remote_code=True,
                                model_kwargs={"torch_dtype": dtype})
    # Truncate identically for every model, or a single very long chunk decides
    # how much memory the whole pass needs.
    m.max_seq_length = min(MAX_SEQ, m.max_seq_length)
    if batch is None:
        batch = BATCH.get(model_key, 8)
    qp, pp = PREFIXES.get(model_key, ("", ""))
    prefix = qp if is_query else pp
    print(f"  {name} in {dtype}, max_seq_length {m.max_seq_length}, "
          f"batch {batch}, {'query' if is_query else 'passage'} prefix "
          f"{prefix[:60]!r}, {len(texts)} texts", flush=True)
    embs = m.encode([prefix + t for t in texts], batch_size=batch,
                    show_progress_bar=True, normalize_embeddings=True,
                    convert_to_numpy=True)
    del m
    torch.cuda.empty_cache()
    return np.asarray(embs, dtype=np.float32)


def build(model, track):
    corpus = load_corpus(track)
    texts = [r["text"] for r in corpus]
    out = IDX / f"{track}_{slug(model)}_dense.npy"
    t0 = time.time()
    if out.exists():
        print(f"{out.name} exists, skipping")
    else:
        if model in HF_MODELS:
            arr = embed_hf(model, texts, is_query=False)
        else:
            arr = embed_ollama(model, texts,
                               purpose=f"setup:index_ablation:{track}:{model}",
                               run_id=f"ablation_{slug(model)}_{track}")
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        np.save(out, arr)
        print(f"corpus embedded {arr.shape} in {time.time()-t0:.0f}s")

    qs = load_queries(track)
    qout = IDX / f"{track}_{slug(model)}_queries.npy"
    if qout.exists():
        print(f"{qout.name} exists, skipping")
        return
    qtexts = [q[1] for q in qs]
    if model in HF_MODELS:
        qarr = embed_hf(model, qtexts, is_query=True)
    else:
        qarr = embed_ollama(model, qtexts,
                            purpose=f"query:retrieve_ablation:{track}:{model}",
                            run_id=f"ablation_q_{slug(model)}_{track}")
    qarr /= np.linalg.norm(qarr, axis=1, keepdims=True) + 1e-12
    np.save(qout, qarr)
    print(f"queries embedded {qarr.shape}")


def gold_sets(track, corpus):
    """qid -> set of gold passage uids, using the same mapping as the main eval."""
    key_to_uid = {(str(r["doc_id"]), str(r["passage_id"])): r["uid"]
                  for r in corpus}
    out = {}
    for qid, _, gold in load_queries(track):
        if not gold:
            continue
        g = {key_to_uid[(str(p["DocumentID"]), str(p["PassageID"]))]
             for p in gold
             if (str(p["DocumentID"]), str(p["PassageID"])) in key_to_uid}
        if g:
            out[qid] = g
    return out


def per_question(ranked_by_qid, golds, ks=KS):
    """Recall at each k and reciprocal rank, per question, for paired tests."""
    out = {f"recall@{k}": [] for k in ks}
    out["rr"] = []
    order = sorted(ranked_by_qid)
    for qid in order:
        g = golds[qid]
        ranked = ranked_by_qid[qid]
        for k in ks:
            out[f"recall@{k}"].append(len(g & set(ranked[:k])) / len(g))
        pos = next((i for i, u in enumerate(ranked) if u in g), None)
        out["rr"].append(1.0 / (pos + 1) if pos is not None else 0.0)
    return out, order


def dense_rankings(model, track, corpus, golds, depth=max(KS)):
    C = np.load(IDX / f"{track}_{slug(model)}_dense.npy")
    Q = np.load(IDX / f"{track}_{slug(model)}_queries.npy")
    uids = [r["uid"] for r in corpus]
    qs = load_queries(track)
    out = {}
    for i, (qid, _, _) in enumerate(qs):
        if qid not in golds:
            continue
        sims = C @ Q[i]
        out[qid] = [uids[j] for j in np.argsort(-sims)[:depth]]
    return out, int(C.shape[1])


def bm25_rankings(track, corpus, golds, depth=max(KS)):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from retrieval import _tok
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([_tok(r["text"]) for r in corpus])
    uids = [r["uid"] for r in corpus]
    out = {}
    for qid, question, _ in load_queries(track):
        if qid not in golds:
            continue
        sc = bm.get_scores(_tok(question))
        out[qid] = [uids[j] for j in np.argsort(-sc)[:depth]]
    return out


def fuse(a, b, depth=max(KS), rrf_k=60, pool=50):
    """Reciprocal rank fusion of two ranking dicts, as the main hybrid arm does."""
    out = {}
    for qid in a:
        if qid not in b:
            continue
        ra = {u: i for i, u in enumerate(a[qid][:pool])}
        rb = {u: i for i, u in enumerate(b[qid][:pool])}
        sc = {}
        for u in set(ra) | set(rb):
            sc[u] = (1.0 / (rrf_k + ra.get(u, pool)) +
                     1.0 / (rrf_k + rb.get(u, pool)))
        out[qid] = sorted(sc, key=sc.get, reverse=True)[:depth]
    return out


def report(track="obliqa"):
    """Compare every embedder that has been built against BM25, and fuse each.

    The question is not which embedder is best in the abstract. It is whether the
    published leaderboard order survives on a regulatory corpus with real gold
    passages, and whether any dense model clears the lexical baseline the main
    study found ahead of bge-m3.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from stats import holm
    from scipy import stats as sps

    corpus = load_corpus(track)
    golds = gold_sets(track, corpus)
    built = []
    for f in sorted(IDX.glob(f"{track}_*_queries.npy")):
        m = f.name[len(track) + 1:-len("_queries.npy")]
        # slugs replace ':' with '_', so restore it only where a model uses one
        cand = [m, m.replace("_", ":", 1)]
        built.append(next((c for c in cand
                           if (IDX / f"{track}_{slug(c)}_dense.npy").exists()),
                          m))
    if not built:
        print("no ablation indices built yet")
        return
    print(f"embedders built: {built}")
    print(f"questions with resolvable gold passages: {len(golds)}")

    rank = {}
    dims = {}
    for m in built:
        rank[m], dims[m] = dense_rankings(m, track, corpus, golds)
    rank["bm25"] = bm25_rankings(track, corpus, golds)
    for m in built:
        rank[f"{m}+bm25"] = fuse(rank[m], rank["bm25"])

    rows = []
    B, rng = 10000, np.random.default_rng(42)
    vecs = {}
    for name, r in rank.items():
        pq, order = per_question(r, golds)
        vecs[name] = (pq, order)
        row = {"method": name, "n": len(order),
               "dim": dims.get(name, np.nan),
               "rteb": RTEB.get(name.replace("+bm25", ""), np.nan),
               "mrr@10": float(np.mean(pq["rr"]))}
        for k in KS:
            row[f"recall@{k}"] = float(np.mean(pq[f"recall@{k}"]))
        v = np.asarray(pq["recall@5"], float)
        idx = rng.integers(0, len(v), size=(B, len(v)))
        lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
        row["recall@5_ci_lo"], row["recall@5_ci_hi"] = lo, hi
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mrr@10", ascending=False)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "embedding_ablation.csv", index=False)
    print("\n== retrieval quality by embedder, ObliQA gold passages ==")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # The comparison that matters: every dense model against the lexical
    # baseline, on the same questions, with multiplicity handled.
    contrasts = []
    base = "bm25"
    for name in rank:
        if name == base:
            continue
        pa, oa = vecs[name]
        pb, ob = vecs[base]
        common = [i for i, q in enumerate(oa) if q in set(ob)]
        a5 = np.asarray(pa["recall@5"], float)
        b5 = np.asarray(pb["recall@5"], float)
        n = min(len(a5), len(b5))
        d = a5[:n] - b5[:n]
        idx = rng.integers(0, n, size=(B, n))
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        nz = d[d != 0]
        p = float(sps.wilcoxon(nz).pvalue) if len(nz) >= 10 else np.nan
        contrasts.append({"method": name, "baseline": base, "measure": "recall@5",
                          "n": n, "diff": float(d.mean()),
                          "ci_lo": lo, "ci_hi": hi,
                          "n_discordant": len(nz), "p_raw": p})
    cf = pd.DataFrame(contrasts)
    if len(cf):
        cf["p_holm"] = holm(cf["p_raw"].values)
        cf["sig_05"] = cf["p_holm"] < 0.05
        cf = cf.sort_values("diff", ascending=False)
        cf.to_csv(TAB / "embedding_ablation_contrasts.csv", index=False)
        print("\n== each configuration against BM25 at recall@5 ==")
        print(cf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        beat = cf[(cf["diff"] > 0) & cf["sig_05"]]
        print(f"\nconfigurations significantly ahead of BM25 alone: "
              f"{len(beat)} of {len(cf)}"
              + (f" ({', '.join(beat['method'])})" if len(beat) else ""))

    # Does the published leaderboard order survive here? Only the standalone
    # dense models can answer that: a fused configuration inherits its
    # partner's RTEB score without being the model the leaderboard scored, so
    # including the fused rows double-counts each embedder and makes the
    # correlation meaningless.
    have = df[df["rteb"].notna() & ~df["method"].str.contains(r"\+")]
    if len(have) >= 2:
        r, pv = sps.spearmanr(have["rteb"], have["mrr@10"])
        print(f"\nRTEB score against reciprocal rank on this corpus, standalone "
              f"dense models only: Spearman rho {r:.3f}, p {pv:.3f}, over "
              f"{len(have)} models")
        print("  " + "; ".join(
            f"{r2.method} RTEB {r2.rteb:.1f} -> MRR {r2['mrr@10']:.4f}"
            for _, r2 in have.sort_values("rteb", ascending=False).iterrows()))
        print("  With three scored models this is an observation about the "
              "ordering, not a powered test of transfer.")
        have.to_csv(TAB / "embedding_rteb_transfer.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--track", default="obliqa")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.track)
    else:
        build(a.model, a.track)
