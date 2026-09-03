"""Fixed-schedule arms A1-A5. The agentic arms live in agentic.py and the
compiled arm in run_dspy.py.

Each arm returns (messages, meta) — the prompt to send and bookkeeping about
retrieved context. Prompt templates are deliberately minimal and identical in
structure across arms; only the context block varies.
"""
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

SYSTEM_OPEN = (
    "You are an expert assistant for regulatory and information-security "
    "compliance. Answer the question accurately and concisely (2-6 sentences). "
    "If the provided context does not contain the answer, say so explicitly "
    "rather than guessing."
)
SYSTEM_OPEN_CLOSED_BOOK = (
    "You are an expert assistant for regulatory and information-security "
    "compliance. Answer the question accurately and concisely (2-6 sentences). "
    "If you do not know the answer, say so explicitly rather than guessing."
)


def ntokens(text):
    return len(ENC.encode(text))


def fmt_context(passages):
    return "\n\n".join(
        f"[{i+1}] ({p.get('doc_id','')} {p.get('passage_id','')}) {p['text']}"
        for i, p in enumerate(passages))


# ---------------- arms ----------------

def a1_zero_shot(q):
    return [{"role": "system", "content": SYSTEM_OPEN_CLOSED_BOOK},
            {"role": "user", "content": q["question"]}], {"context_passages": 0}


def a2_few_shot(q, exemplars):
    msgs = [{"role": "system", "content": SYSTEM_OPEN_CLOSED_BOOK}]
    for ex in exemplars:
        msgs.append({"role": "user", "content": ex["question"]})
        msgs.append({"role": "assistant", "content": ex["reference_answer"]})
    msgs.append({"role": "user", "content": q["question"]})
    return msgs, {"context_passages": 0, "n_exemplars": len(exemplars)}


def a3_long_context(q, doc_texts, budget_tokens=24000):
    """Coarse document-level routing (BM25 over whole docs, done by caller) then
    stuff whole documents up to the token budget."""
    blocks, used = [], 0
    for name, text in doc_texts:
        t = ntokens(text)
        if used + t > budget_tokens:
            remaining = budget_tokens - used
            if remaining > 500:
                text = ENC.decode(ENC.encode(text)[:remaining])
                blocks.append((name, text))
                used = budget_tokens
            break
        blocks.append((name, text))
        used += t
    ctx = "\n\n".join(f"### Document: {n}\n{t}" for n, t in blocks)
    user = f"Context documents:\n{ctx}\n\nQuestion: {q['question']}"
    return [{"role": "system", "content": SYSTEM_OPEN},
            {"role": "user", "content": user}], {
        "context_passages": len(blocks), "context_tokens_approx": used}


def a45_rag(q, passages):
    """Shared template for naive dense (A4) and hybrid (A5) — caller retrieves."""
    ctx = fmt_context(passages)
    user = f"Context passages:\n{ctx}\n\nQuestion: {q['question']}"
    return [{"role": "system", "content": SYSTEM_OPEN},
            {"role": "user", "content": user}], {
        "context_passages": len(passages),
        "retrieved_ids": [p.get("uid") or p.get("chunk_id") for p in passages]}
