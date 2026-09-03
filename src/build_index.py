"""Embed a corpus with bge-m3 (setup cost, logged to the token ledger) and save
the dense index (float32 .npy + metadata jsonl).

Usage: python src/build_index.py obliqa
"""
import json
import sys
import time
import numpy as np
from pathlib import Path

from engine import OllamaClient

ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "bge-m3"
BATCH = 64

CORPORA = {
    "obliqa": ROOT / "data/processed/obliqa_corpus.jsonl",
}


def main(name):
    src = CORPORA[name]
    rows = [json.loads(l) for l in src.open()]
    client = OllamaClient()
    out_dir = ROOT / "data/index"
    out_dir.mkdir(exist_ok=True)

    shard_dir = out_dir / f"{name}_shards"
    shard_dir.mkdir(exist_ok=True)
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        shard = shard_dir / f"{i:07d}.npy"
        if shard.exists():
            continue
        batch = [r["text"] for r in rows[i:i + BATCH]]
        e = client.embed(EMBED_MODEL, batch,
                        purpose=f"setup:index:{name}", run_id=f"index_{name}")
        np.save(shard, np.asarray(e, dtype=np.float32))
        if (i // BATCH) % 20 == 0:
            print(f"{i}/{len(rows)} embedded, {time.time()-t0:.0f}s", flush=True)

    embs = [np.load(shard_dir / f"{i:07d}.npy") for i in range(0, len(rows), BATCH)]
    arr = np.concatenate(embs).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    np.save(out_dir / f"{name}_dense.npy", arr)
    with (out_dir / f"{name}_meta.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"done: {arr.shape} in {time.time()-t0:.0f}s -> {out_dir}/{name}_dense.npy")


if __name__ == "__main__":
    main(sys.argv[1])
