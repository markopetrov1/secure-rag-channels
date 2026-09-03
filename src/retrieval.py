"""Dense (bge-m3), BM25, and hybrid (RRF) retrieval over a built index."""
import json
import re
import numpy as np
from pathlib import Path

from rank_bm25 import BM25Okapi

from engine import OllamaClient

ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "bge-m3"


def _tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, name, client=None):
        idx = ROOT / "data/index"
        self.name = name
        self.meta = [json.loads(l) for l in (idx / f"{name}_meta.jsonl").open()]
        self.dense = np.load(idx / f"{name}_dense.npy")
        self.bm25 = BM25Okapi([_tok(r["text"]) for r in self.meta])
        self.client = client or OllamaClient()

    def _embed_query(self, query, purpose, run_id):
        e = self.client.embed(EMBED_MODEL, [query], purpose=purpose, run_id=run_id)[0]
        v = np.asarray(e, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def dense_topk(self, query, k, purpose="query:retrieve", run_id="adhoc"):
        v = self._embed_query(query, purpose, run_id)
        scores = self.dense @ v
        order = np.argsort(-scores)[:k]
        return [dict(self.meta[i], score=float(scores[i])) for i in order]

    def bm25_topk(self, query, k):
        scores = self.bm25.get_scores(_tok(query))
        order = np.argsort(-scores)[:k]
        return [dict(self.meta[i], score=float(scores[i])) for i in order]

    def hybrid_topk(self, query, k, pool=50, rrf_k=60,
                    purpose="query:retrieve", run_id="adhoc"):
        """Reciprocal-rank fusion of dense and BM25 rankings."""
        v = self._embed_query(query, purpose, run_id)
        d_scores = self.dense @ v
        b_scores = self.bm25.get_scores(_tok(query))
        d_rank = {int(i): r for r, i in enumerate(np.argsort(-d_scores)[:pool])}
        b_rank = {int(i): r for r, i in enumerate(np.argsort(-b_scores)[:pool])}
        fused = {}
        for i in set(d_rank) | set(b_rank):
            fused[i] = (1.0 / (rrf_k + d_rank.get(i, pool)) +
                        1.0 / (rrf_k + b_rank.get(i, pool)))
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [dict(self.meta[i], score=fused[i]) for i in order]
