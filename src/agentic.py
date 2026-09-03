"""Agentic arms: pipelines that decide for themselves when to retrieve.

The arms in arms.py retrieve once, on a schedule fixed before the question is
read. The arms here do not. A8 chooses whether to search at all, what to search
for, and when it has enough; A8C retrieves once and then criticises and revises
its own draft. Both therefore issue several model calls per question, which is
what makes them expensive and what makes them worth pricing.

Two design choices are deliberate.

The agent speaks a one-line text protocol rather than using the server's native
tool-calling API. Both mechanisms were tested and both work on qwen3:8b and
llama3.1:8b before either arm was built, but the text protocol goes through
engine.OllamaClient.chat unchanged, so every step lands in the token ledger with
its own tokens and latency. Native tool calling would have needed a second code
path around the ledger, and an unledgered call is exactly the failure that left
the graph arm unpriced.

Usage is accumulated across every call a question costs. A runner that writes
only the last call's usage, or only wall time, understates a multi-call arm by
roughly the number of calls it made, and economics.py then either misprices the
arm or drops it. That is not hypothetical: it is what happened to a6.

What the agent searched for is recorded verbatim. The queries it composes are
not the question it was asked, and on the track that ships gold passages they
can be scored against the same gold set as the question, which is how we find
out whether agentic reformulation earns its cost or merely spends it.
"""
import re
import time

import arms

TOP_K = 5
MAX_SEARCHES = 3          # hard cap; the step budget is a reported quantity
MAX_TURNS = 8             # protocol-violation backstop, never reached normally

SYSTEM_AGENT = (
    "You are an expert assistant for regulatory and information-security "
    "compliance, working against a searchable corpus.\n\n"
    "Reply with EXACTLY ONE line, in one of these two forms:\n"
    "SEARCH: <what to look for>\n"
    "ANSWER: <your answer>\n\n"
    "Use SEARCH when you need source text you have not been given yet. Write "
    "the search as keywords and terminology, not as a question. Use ANSWER "
    "once the material you have is enough, and answer in 2 to 6 sentences. If "
    "the material does not contain the answer, say so explicitly in the ANSWER "
    "rather than guessing. Never write both forms, and never add commentary."
)

SYSTEM_CRITIC = (
    "You are reviewing a draft answer to a compliance question against the "
    "source passages it was written from. List, as terse bullet points, every "
    "claim in the draft that the passages do not support, and every part of "
    "the question the draft leaves unanswered. If the draft is fully supported "
    "and complete, reply with exactly: OK"
)

ACTION = re.compile(r"^\s*(SEARCH|ANSWER)\s*:\s*(.*)$", re.I | re.S)


def _blank_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "wall_s": 0.0,
            "load_duration_s": 0.0, "eval_s": 0.0, "prompt_eval_s": 0.0}


def _add(total, u):
    """Accumulate one call's usage into the per-question total."""
    for k in total:
        total[k] += u.get(k, 0) or 0
    return total


def _parse(text):
    """(kind, payload) for the first line that states an action, else (None, raw).

    Models occasionally prefix the action with a stray blank line or wrap it in
    a code fence, which is a formatting slip rather than a refusal to follow the
    protocol, so those are tolerated. Anything else counts as a violation and is
    reported.
    """
    for line in (text or "").splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        m = ACTION.match(line)
        if m:
            return m.group(1).upper(), m.group(2).strip()
        break
    return None, (text or "").strip()


def _observation(passages):
    if not passages:
        return "OBSERVATION: no passages matched that search."
    return "OBSERVATION:\n" + arms.fmt_context(passages)


def _final_instruction(q):
    """How the agent is told to shape its final answer, matching the fixed arms."""
    return ("Now answer the question using everything above, in 2 to 6 "
            "sentences.\n\nQUESTION: " + q["question"])


def _react(client, model, retriever, q, run_id, think):
    """A8: search when it decides to, up to a cap, then answer."""
    question = q["question"]
    msgs = [{"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": f"QUESTION: {question}"}]
    usage, queries, passages, violations = _blank_usage(), [], [], 0
    calls = 0
    stop = "answered"

    for _turn in range(MAX_TURNS):
        text, u = client.chat(model, msgs, purpose="query:generate",
                              run_id=run_id, num_ctx=16384, num_predict=512,
                              think=think)
        _add(usage, u)
        calls += 1
        kind, payload = _parse(text)

        if kind == "ANSWER":
            return payload, usage, dict(
                n_llm_calls=calls, n_searches=len(queries), queries=queries,
                retrieved_ids=[p.get("uid") for p in passages],
                protocol_violations=violations, stop_reason=stop,
                agent="react")

        if kind == "SEARCH" and len(queries) < MAX_SEARCHES:
            queries.append(payload)
            hits = retriever.hybrid_topk(payload, TOP_K,
                                         purpose="query:retrieve",
                                         run_id=run_id)
            passages.extend(hits)
            msgs.append({"role": "assistant", "content": f"SEARCH: {payload}"})
            msgs.append({"role": "user", "content": _observation(hits)})
            continue

        if kind == "SEARCH":
            # Budget spent. Stop searching and require an answer from what it
            # has, so the cap shows up as a forced answer rather than a crash.
            stop = "search_budget"
            msgs.append({"role": "assistant", "content": f"SEARCH: {payload}"})
            msgs.append({"role": "user", "content":
                         "OBSERVATION: no further searches are available. "
                         + _final_instruction(q)})
            continue

        # Neither form. Restate the contract once per violation and continue;
        # the count is reported rather than hidden.
        violations += 1
        stop = "protocol_violation"
        msgs.append({"role": "assistant", "content": (text or "")[:800]})
        msgs.append({"role": "user", "content":
                     "That was not one of the two allowed forms. Reply with a "
                     "single line beginning SEARCH: or ANSWER:."})

    # Ran out of turns: take one last unconstrained answer so the cell is never
    # empty, and mark why.
    msgs.append({"role": "user", "content": _final_instruction(q)})
    text, u = client.chat(model, msgs, purpose="query:generate", run_id=run_id,
                          num_ctx=16384, num_predict=512, think=think)
    _add(usage, u)
    calls += 1
    return text.strip(), usage, dict(
        n_llm_calls=calls, n_searches=len(queries), queries=queries,
        retrieved_ids=[p.get("uid") for p in passages],
        protocol_violations=violations, stop_reason="turn_cap", agent="react")


def _critique(client, model, retriever, q, run_id, think):
    """A8C: retrieve once, draft, criticise the draft, revise.

    The schedule is fixed rather than chosen, which is the point: it isolates
    the cost of self-criticism from the cost of deciding to retrieve. Against
    A8 it says whether autonomy or iteration is what an agentic arm is paying
    for.
    """
    question = q["question"]
    usage, calls = _blank_usage(), 0

    hits = retriever.hybrid_topk(q["question"], TOP_K,
                                 purpose="query:retrieve", run_id=run_id)
    ctx = arms.fmt_context(hits)
    sysmsg = arms.SYSTEM_OPEN

    draft, u = client.chat(model, [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"},
    ], purpose="query:generate", run_id=run_id, num_ctx=16384,
        num_predict=512, think=think)
    _add(usage, u); calls += 1

    crit, u = client.chat(model, [
        {"role": "system", "content": SYSTEM_CRITIC},
        {"role": "user", "content":
         f"PASSAGES:\n{ctx}\n\nQUESTION: {question}\n\nDRAFT: {draft.strip()}"},
    ], purpose="query:generate", run_id=run_id, num_ctx=16384,
        num_predict=384, think=think)
    _add(usage, u); calls += 1

    clean = crit.strip().upper().startswith("OK")
    if clean:
        return draft.strip(), usage, dict(
            n_llm_calls=calls, n_searches=1, queries=[q["question"]],
            retrieved_ids=[p.get("uid") for p in hits],
            protocol_violations=0, stop_reason="critic_accepted",
            revised=False, agent="critique")

    final, u = client.chat(model, [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content":
         f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nDRAFT: {draft.strip()}\n\n"
         f"REVIEW OF THE DRAFT:\n{crit.strip()}\n\n"
         + _final_instruction(q)},
    ], purpose="query:generate", run_id=run_id, num_ctx=16384,
        num_predict=512, think=think)
    _add(usage, u); calls += 1

    return final.strip(), usage, dict(
        n_llm_calls=calls, n_searches=1, queries=[q["question"]],
        retrieved_ids=[p.get("uid") for p in hits],
        protocol_violations=0, stop_reason="revised", revised=True,
        agent="critique")


RUNNERS = {"a8": _react, "a8c": _critique}


def run(arm, q, track, generator, run_id, client, retriever):
    """Answer one question with an agentic arm.

    Returns (text, usage, meta) in the shape run_generation.py already writes,
    with usage summed over every model call the question cost and meta carrying
    the step counts and issued queries that make the arm's behaviour reportable.
    """
    fn = RUNNERS.get(arm)
    if fn is None:
        raise ValueError(f"unknown agentic arm {arm!r}; known: {sorted(RUNNERS)}")
    think = False if str(generator).startswith(("qwen3", "gpt-oss")) else None
    t0 = time.time()
    text, usage, meta = fn(client, generator, retriever, q,
                           f"{run_id}:{q['qid']}", think)
    # Wall time is measured around the whole question, not summed per call, so
    # it includes the retrieval and parsing between calls.
    usage["wall_s"] = time.time() - t0
    meta["context_passages"] = len(meta.get("retrieved_ids") or [])
    return text, usage, meta
