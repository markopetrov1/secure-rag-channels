"""Ollama client with per-call token/latency accounting.

Every LLM/embedding call is logged to a JSONL ledger with a `purpose` tag
("setup:*" or "query:*") so setup-vs-marginal cost can be reconstructed exactly.
"""
import json
import time
import threading
from pathlib import Path

import requests

OLLAMA = "http://127.0.0.1:11435"
LEDGER = Path(__file__).resolve().parent.parent / "results/token_ledger.jsonl"
_lock = threading.Lock()


class OllamaClient:
    def __init__(self, base_url=OLLAMA, ledger_path=LEDGER):
        self.base = base_url
        self.ledger = Path(ledger_path)
        self.ledger.parent.mkdir(exist_ok=True)

    def _log(self, rec):
        rec["ts"] = time.time()
        with _lock, self.ledger.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def chat(self, model, messages, purpose, run_id, temperature=0.0, seed=42,
             num_ctx=8192, num_predict=1024, retries=3, think=None):
        """Returns (text, usage). Greedy by default. `think` False disables
        qwen3/gpt-oss thinking mode where supported."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        if think is not None:
            payload["think"] = think
        last_err = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = requests.post(f"{self.base}/api/chat", json=payload, timeout=1800)
                r.raise_for_status()
                d = r.json()
                usage = {
                    "prompt_tokens": d.get("prompt_eval_count", 0),
                    "completion_tokens": d.get("eval_count", 0),
                    "load_duration_s": d.get("load_duration", 0) / 1e9,
                    "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
                    "eval_s": d.get("eval_duration", 0) / 1e9,
                    "wall_s": time.time() - t0,
                }
                msg = d["message"]
                text = msg.get("content") or ""
                if not text.strip():
                    # Reasoning models can spend the whole completion budget in
                    # their reasoning channel and return empty content. The
                    # verdict is often still present there, so fall back to it
                    # rather than discarding the call.
                    text = msg.get("reasoning") or msg.get("thinking") or ""
                    usage["used_reasoning_fallback"] = bool(text.strip())
                self._log({"kind": "chat", "model": model, "purpose": purpose,
                           "run_id": run_id, **usage})
                return text, usage
            except (requests.RequestException, KeyError) as e:
                last_err = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"chat failed after {retries} attempts: {last_err}")

    def embed(self, model, texts, purpose, run_id):
        t0 = time.time()
        r = requests.post(f"{self.base}/api/embed",
                          json={"model": model, "input": texts}, timeout=600)
        r.raise_for_status()
        d = r.json()
        self._log({"kind": "embed", "model": model, "purpose": purpose,
                   "run_id": run_id, "n_texts": len(texts),
                   "prompt_tokens": d.get("prompt_eval_count", 0),
                   "wall_s": time.time() - t0})
        return d["embeddings"]
