"""What this study cost to run, taken from the token ledger rather than estimated.

Every model and embedding call in the pipeline is ledgered with its token counts
and its wall time, so the compute budget is a measurement and not a guess. This
turns that ledger into the table the README quotes.

Two things it is careful about.

The ledger is the record of everything the machine ever did for this project,
which is not the same as what this study reports. It still holds the calls of a
second benchmark and of a graph-retrieval arm, both withdrawn. Charging the
reported study for work no reported result depends on would overstate what a
repetition costs, so those calls are excluded here and the excluded total is
printed beside the kept one rather than quietly dropped.

Post-hoc analysis is separated from the run. agentic_analysis.py re-ranks the
agent's own search queries through the live retriever, which is measurement
about the study rather than part of it, and it appends its calls to the same
ledger. Folding those into generation would make the agentic arms look several
times more expensive to operate than they are.

Writes results/tables/compute_budget.csv.
"""
import collections
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/token_ledger.jsonl"
TAB = ROOT / "results/tables"

TRACK = "obliqa"
AGENTIC_ARMS = ("a8", "a8c")

# Order is the order the README prints.
STAGES = ["Corpus indexing, one-time",
          "DSPy compilation, one-time, three seeds",
          "Generation, the seven fixed arms",
          "Generation, the two agentic arms",
          "Judging, three judges over every arm",
          "Retrieval evaluation and embedder ablation",
          "Serial latency benchmark",
          "Agentic behaviour analysis, post hoc"]


def arm_of(run_id):
    """The arm a generation or judging run belongs to, or None."""
    parts = run_id.split("_")
    for i, p in enumerate(parts):
        if p == TRACK and i + 2 < len(parts):
            return parts[i + 2]
    return None


def stage_of(run_id, purpose):
    """Which budget line a ledger record belongs to, or None to exclude it.

    Excluded on purpose: anything naming the withdrawn benchmark, and the
    graph arm, whose index build and queries support no reported result.
    """
    if "iso27k" in run_id or "lightrag" in run_id:
        return None
    if run_id.startswith("agentic_"):
        return "Agentic behaviour analysis, post hoc"
    if run_id.startswith("bench_"):
        return "Serial latency benchmark"
    # The compiler's own retrieval calls are tagged query:retrieve but belong to
    # the one-time compile, not to serving, so they are matched on the run first.
    if purpose.startswith("setup:dspy_compile") or run_id == "dspy":
        return "DSPy compilation, one-time, three seeds"
    if purpose.startswith("setup:index_ablation") or "ablation" in run_id:
        return "Retrieval evaluation and embedder ablation"
    if purpose.startswith("setup:index"):
        return "Corpus indexing, one-time"
    if run_id.startswith("eval_") or run_id.startswith("probe"):
        return "Retrieval evaluation and embedder ablation"
    if run_id.startswith("judge_"):
        return "Judging, three judges over every arm"
    if run_id.startswith(TRACK + "_"):
        arm = arm_of(run_id)
        return ("Generation, the two agentic arms" if arm in AGENTIC_ARMS
                else "Generation, the seven fixed arms")
    return None


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    keep = collections.defaultdict(lambda: {"calls": 0, "tok": 0, "wall": 0.0})
    dropped = {"calls": 0, "tok": 0, "wall": 0.0}
    unclassified = collections.Counter()
    for line in LEDGER.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        run_id = str(r.get("run_id", ""))
        purpose = str(r.get("purpose", ""))
        tok = (r.get("prompt_tokens", 0) or 0) + (r.get("completion_tokens", 0) or 0)
        wall = r.get("wall_s", 0.0) or 0.0
        stage = stage_of(run_id, purpose)
        if stage is None:
            dropped["calls"] += 1
            dropped["tok"] += tok
            dropped["wall"] += wall
            if "iso27k" not in run_id and "lightrag" not in run_id:
                unclassified[run_id or purpose] += 1
            continue
        a = keep[stage]
        a["calls"] += 1
        a["tok"] += tok
        a["wall"] += wall

    rows = [{"stage": s, "worker_hours": keep[s]["wall"] / 3600,
             "tokens": keep[s]["tok"], "calls": keep[s]["calls"]}
            for s in STAGES if s in keep]
    df = pd.DataFrame(rows)
    total = {"stage": "Total", "worker_hours": df.worker_hours.sum(),
             "tokens": int(df.tokens.sum()), "calls": int(df.calls.sum())}
    df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    df.to_csv(TAB / "compute_budget.csv", index=False)

    print("| Stage | Worker-hours | Tokens | Calls |")
    print("|---|---|---|---|")
    for r in df.itertuples():
        bold = "**" if r.stage == "Total" else ""
        print(f"| {bold}{r.stage}{bold} | {bold}{r.worker_hours:.1f}{bold} "
              f"| {bold}{r.tokens / 1e6:.1f} M{bold} | {bold}{r.calls:,}{bold} |")
    print(f"\nexcluded as withdrawn or unreported: {dropped['calls']:,} calls, "
          f"{dropped['tok'] / 1e6:.1f} M tokens, {dropped['wall'] / 3600:.1f} h")
    if unclassified:
        print("unclassified and therefore excluded, which should be empty:")
        for k, v in unclassified.most_common(10):
            print(f"  {v:,}  {k}")


if __name__ == "__main__":
    main()
