"""What the agentic arms did, and whether their own search queries were better.

Two questions the fixed-schedule arms cannot raise.

How much autonomy did the agent actually exercise? An arm that searches three
times on every question is a different object from one that searches on a tenth
of them, and both average the same way, so the step count is reported as a
distribution and the stop reasons and protocol violations are reported beside it
rather than folded away.

Did composing its own query help? The agent does not search for the question it
was asked; it writes keywords. Track B ships gold supporting passages, so the
passages the agent's first search returned can be scored against exactly the same
gold set, at exactly the same cutoff, as the passages the raw question returns.
Most agentic work cannot ask this, having no gold passages to score against. The
comparison is k-matched on purpose: the agent's later searches add passages, and
crediting reformulation with the recall those extra passages bring would measure
budget rather than phrasing.

Outputs (results/tables/):
  agentic_behaviour.csv        steps, searches, stop reasons, violations per cell
  agentic_reformulation.csv    first-search vs raw question against gold, k-matched
  agentic_cost.csv             tokens and calls per question per cell
"""
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import arm_labels

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results/generations"
TAB = ROOT / "results/tables"
KS = [1, 3, 5]
TOP_K = 5          # must match agentic.TOP_K, or the k-matching below is a lie


def agentic_files():
    out = []
    for fp in sorted(glob.glob(str(GEN / "obliqa_*_a8*.jsonl"))):
        parts = Path(fp).stem.split("_")
        out.append({"path": fp, "track": parts[0], "mode": parts[1],
                    "arm": parts[2], "generator": "_".join(parts[3:])})
    return out


def load(fp):
    recs = []
    for line in open(fp):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a run still in flight can leave a partial line
    return recs


def behaviour():
    rows = []
    for f in agentic_files():
        recs = [r for r in load(f["path"]) if r.get("meta")]
        if not recs:
            continue
        calls = np.array([r["meta"].get("n_llm_calls", 0) for r in recs], float)
        srch = np.array([r["meta"].get("n_searches", 0) for r in recs], float)
        stops = Counter(r["meta"].get("stop_reason", "?") for r in recs)
        viol = sum(r["meta"].get("protocol_violations", 0) for r in recs)
        row = {**{k: f[k] for k in ("track", "mode", "arm", "generator")},
               "n": len(recs),
               "calls_mean": calls.mean(), "calls_median": np.median(calls),
               "searches_mean": srch.mean(),
               "pct_zero_search": float(np.mean(srch == 0)),
               "pct_one_search": float(np.mean(srch == 1)),
               "pct_multi_search": float(np.mean(srch > 1)),
               "protocol_violations": viol,
               "pct_items_with_violation": float(np.mean(
                   [r["meta"].get("protocol_violations", 0) > 0 for r in recs]))}
        for k, v in stops.items():
            row[f"stop_{k}"] = v / len(recs)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["track", "mode", "arm", "generator"])
    df.to_csv(TAB / "agentic_behaviour.csv", index=False)
    return df


def _gold_and_questions():
    """qid -> (gold passage uids, question text) for the ObliQA sample."""
    meta = [json.loads(l) for l in
            (ROOT / "data/index/obliqa_meta.jsonl").open()]
    by_key = {(str(m["doc_id"]), str(m["passage_id"])): m["uid"] for m in meta}
    out = {}
    for line in (ROOT / "data/processed/obliqa_test_sample.jsonl").open():
        q = json.loads(line)
        ids = {by_key[(str(p["DocumentID"]), str(p["PassageID"]))]
               for p in q["gold_passages"]
               if (str(p["DocumentID"]), str(p["PassageID"])) in by_key}
        if ids:
            out[q["qid"]] = (ids, q["question"])
    return out


def _score(ranked, gold):
    """recall@k for each k in KS, plus reciprocal rank, for one ranked list."""
    out = {}
    for k in KS:
        out[f"recall@{k}"] = len(set(ranked[:k]) & gold) / len(gold)
    rr = 0.0
    for i, uid in enumerate(ranked, 1):
        if uid in gold:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    return out


def reformulation():
    """The agent's first search against the raw question, at the same cutoff.

    The agent's own retrieval is scored from what it actually saw, recorded per
    question, so nothing is re-embedded on its side and the figure is what the
    generator was given rather than a reconstruction. The raw-question ranking
    is computed once per question with the same retriever the arm used, and the
    two are compared on the same items, paired, at the same cutoff.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    gq = _gold_and_questions()
    if not gq:
        print("  no ObliQA gold passages available; skipping reformulation")
        return pd.DataFrame()
    from retrieval import Retriever
    retr, raw_cache, rows = None, {}, []
    for f in agentic_files():
        if f["track"] != "obliqa" or f["mode"] != "open":
            continue
        recs = [r for r in load(f["path"])
                if r.get("meta", {}).get("queries") and r["qid"] in gq]
        if len(recs) < 20:
            continue
        if retr is None:
            retr = Retriever("obliqa")
        agent_s, raw_s, changed = [], [], 0
        for r in recs:
            gold, question = gq[r["qid"]]
            first = (r["meta"].get("retrieved_ids") or [])[:TOP_K]
            if not first:
                continue
            if r["qid"] not in raw_cache:
                hits = retr.hybrid_topk(question, TOP_K,
                                        purpose="eval:retrieval",
                                        run_id="agentic_reformulation")
                raw_cache[r["qid"]] = [p["uid"] for p in hits]
            agent_s.append(_score(first, gold))
            raw_s.append(_score(raw_cache[r["qid"]], gold))
            q0 = (r["meta"]["queries"] or [""])[0]
            if q0.strip().lower() != question.strip().lower():
                changed += 1
        if not agent_s:
            continue
        row = {**{k: f[k] for k in ("track", "arm", "generator")},
               "n": len(agent_s), "pct_query_rewritten": changed / len(agent_s)}
        for k in KS:
            a = float(np.mean([x[f"recall@{k}"] for x in agent_s]))
            b = float(np.mean([x[f"recall@{k}"] for x in raw_s]))
            row[f"agent_recall@{k}"] = a
            row[f"question_recall@{k}"] = b
            row[f"delta_recall@{k}"] = a - b
        am = float(np.mean([x["mrr"] for x in agent_s]))
        qm = float(np.mean([x["mrr"] for x in raw_s]))
        row["agent_mrr"], row["question_mrr"], row["delta_mrr"] = am, qm, am - qm
        # Paired bootstrap on the item-level recall@5 difference, because the
        # two rankings are computed on the same questions.
        d = np.array([x["recall@5"] for x in agent_s]) - \
            np.array([x["recall@5"] for x in raw_s])
        rng = np.random.default_rng(7)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        row["delta_recall@5_lo"] = float(np.percentile(boot, 2.5))
        row["delta_recall@5_hi"] = float(np.percentile(boot, 97.5))
        row["delta_recall@5_p"] = float(2 * min(
            np.mean(np.array(boot) <= 0), np.mean(np.array(boot) >= 0)))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "agentic_reformulation.csv", index=False)
    return df


def cost():
    rows = []
    for f in agentic_files():
        recs = [r for r in load(f["path"]) if r.get("usage")]
        if not recs:
            continue
        pt = np.array([r["usage"].get("prompt_tokens", 0) for r in recs], float)
        ct = np.array([r["usage"].get("completion_tokens", 0) for r in recs], float)
        rows.append({**{k: f[k] for k in ("track", "mode", "arm", "generator")},
                     "n": len(recs),
                     "prompt_tokens_per_query": pt.mean(),
                     "completion_tokens_per_query": ct.mean(),
                     "tokens_per_query": pt.mean() + ct.mean(),
                     "arm_name": arm_labels.name(f["arm"])})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["track", "mode", "arm", "generator"])
    df.to_csv(TAB / "agentic_cost.csv", index=False)
    return df


def channel_decomposition():
    """Where the reformulation penalty falls: dense, lexical, or the fusion.

    The agent rewrites a natural-language question into keywords, which is what
    a person types into a search engine. A BM25 index rewards that. A dense
    bi-encoder trained on question and passage pairs does not, because keywords
    sit off the manifold it was trained on. Scoring both query forms through
    each channel separately says which of those two effects dominates, and it is
    the difference between reporting that reformulation hurt and explaining why.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    gq = _gold_and_questions()
    from retrieval import Retriever
    retr, rows = None, []
    for f in agentic_files():
        if f["track"] != "obliqa" or f["mode"] != "open" or f["arm"] != "a8":
            continue
        recs = [r for r in load(f["path"])
                if r.get("meta", {}).get("queries") and r["qid"] in gq]
        if len(recs) < 50:
            continue
        if retr is None:
            retr = Retriever("obliqa")
        acc = {c: {"agent": [], "question": []}
               for c in ("dense", "bm25", "hybrid")}
        for r in recs:
            gold, question = gq[r["qid"]]
            aq = (r["meta"]["queries"] or [""])[0]
            if not aq.strip():
                continue
            for name, text in (("agent", aq), ("question", question)):
                d = retr.dense_topk(text, TOP_K, purpose="eval:retrieval",
                                    run_id="agentic_channels")
                b = retr.bm25_topk(text, TOP_K)
                h = retr.hybrid_topk(text, TOP_K, purpose="eval:retrieval",
                                     run_id="agentic_channels")
                acc["dense"][name].append(_score([x["uid"] for x in d], gold))
                acc["bm25"][name].append(_score([x["uid"] for x in b], gold))
                acc["hybrid"][name].append(_score([x["uid"] for x in h], gold))
        for chan in ("dense", "bm25", "hybrid"):
            a = acc[chan]["agent"]
            q = acc[chan]["question"]
            if not a:
                continue
            am = float(np.mean([x["recall@5"] for x in a]))
            qm = float(np.mean([x["recall@5"] for x in q]))
            d = np.array([x["recall@5"] for x in a]) - \
                np.array([x["recall@5"] for x in q])
            rng = np.random.default_rng(7)
            boot = [d[rng.integers(0, len(d), len(d))].mean()
                    for _ in range(10000)]
            rows.append({"generator": f["generator"], "channel": chan,
                         "n": len(a), "agent_recall@5": am,
                         "question_recall@5": qm, "delta": am - qm,
                         "lo": float(np.percentile(boot, 2.5)),
                         "hi": float(np.percentile(boot, 97.5)),
                         "agent_mrr": float(np.mean([x["mrr"] for x in a])),
                         "question_mrr": float(np.mean([x["mrr"] for x in q]))})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "agentic_channels.csv", index=False)
    return df


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    fmt = lambda x: f"{x:.3f}"

    b = behaviour()
    print("== agentic behaviour ==")
    print(b.to_string(index=False, float_format=fmt) if len(b)
          else "  no agentic generations yet")

    c = cost()
    print("\n== agentic cost per question ==")
    print(c.to_string(index=False, float_format=fmt) if len(c) else "  none")

    r = reformulation()
    print("\n== agentic query reformulation against gold passages ==")
    print(r.to_string(index=False, float_format=fmt) if len(r)
          else "  not enough ObliQA agentic records yet")

    ch = channel_decomposition()
    print("\n== where the reformulation penalty falls, by retrieval channel ==")
    print(ch.to_string(index=False, float_format=fmt) if len(ch)
          else "  not enough records yet")


if __name__ == "__main__":
    main()
