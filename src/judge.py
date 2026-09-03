"""Multi-judge evaluation: pointwise, reference-guided, CRAG-style 3-way verdict.

Two rubrics:
  correctness  - answer vs reference (the question's gold supporting passages)
  groundedness - answer vs retrieved context (only for retrieval arms)

Judges output brief reasoning then a JSON block; temperature 0.
"""
import json
import re

CORRECTNESS_PROMPT = """You are grading an answer to a regulatory-compliance question.

Question:
{question}

Reference material (treat as ground truth):
{reference}

Candidate answer:
{answer}

Grade STRICTLY against the reference material. Answer length is irrelevant; do
not reward verbosity. First reason briefly (2-4 sentences), then output a JSON
object on the last line:
{{"verdict": "..."}}

Choose exactly one verdict:
- "correct": the answer's substantive claims agree with the reference and it
  answers the question asked.
- "missing": the answer declines, says it cannot answer, or omits the key
  information required by the question.
- "incorrect": the answer makes at least one substantive claim that contradicts
  the reference or invents unsupported facts."""

GROUNDEDNESS_PROMPT = """You are checking whether an answer is grounded in the
context passages it was given.

Context passages:
{context}

Question:
{question}

Candidate answer:
{answer}

Ignore whether the answer is actually correct; judge ONLY whether its
substantive claims are supported by the context passages above. Answer length is
irrelevant. First reason briefly (2-4 sentences), then output a JSON object on
the last line:
{{"grounded": "..."}}

Choose exactly one:
- "full": every substantive claim is supported by the context.
- "partial": some claims are supported, some are not found in the context.
- "none": the substantive claims are not supported by the context."""

VALID_VERDICTS = {"correct", "missing", "incorrect"}
VALID_GROUNDED = {"full", "partial", "none"}


# Models that emit a separate reasoning channel and will otherwise spend the
# completion budget there instead of on the verdict.
REASONING_MODELS = ("qwen3", "gpt-oss", "deepseek-r", "magistral")


def no_think(model):
    return False if any(m in model for m in REASONING_MODELS) else None


def _parse_json_tail(text, key, valid):
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get(key, "")).lower().strip()
            if v in valid:
                return v
        except json.JSONDecodeError:
            continue
    tail = text[-200:].lower()
    hits = [v for v in valid if v in tail]
    return hits[0] if len(hits) == 1 else None


def judge_correctness(client, judge_model, question, answer, reference, run_id):
    prompt = CORRECTNESS_PROMPT.format(
        question=question, reference=reference, answer=answer)
    text, usage = client.chat(
        judge_model, [{"role": "user", "content": prompt}],
        purpose="query:judge:correctness", run_id=run_id,
        num_ctx=8192, num_predict=900, think=no_think(judge_model))
    return _parse_json_tail(text, "verdict", VALID_VERDICTS), text, usage


def judge_groundedness(client, judge_model, question, answer, context, run_id):
    prompt = GROUNDEDNESS_PROMPT.format(
        question=question, context=context, answer=answer)
    text, usage = client.chat(
        judge_model, [{"role": "user", "content": prompt}],
        purpose="query:judge:groundedness", run_id=run_id,
        num_ctx=8192, num_predict=900, think=no_think(judge_model))
    return _parse_json_tail(text, "grounded", VALID_GROUNDED), text, usage
