"""Canonical panel verdict: one definition, shared by every reporting script.

The panel is three judges (see JUDGES in src/run_judging.py). Each judge returns
one of three labels for an answer, and the panel label is the majority of the
judges who actually voted. The three-way label is kept rather than collapsed to
correct/not because protocol.tex argues that collapsing "missing" into
"incorrect" would score an arm that hallucinates the same as an arm that
declines to answer, and those are not the same failure.

The point of this module is that there is no hidden tie-break. When no label
holds a strict majority of the votes cast, the canonical rule returns UNDECIDED
and the item is reported as undecided rather than being assigned to whichever
judge happened to sort first. The earlier code path used
collections.Counter.most_common on a judge list sorted alphabetically, which
silently handed every tie to gemma3 (the first name alphabetically, and the most
lenient judge in eight of ten files). That decided 9.0 percent of ISO27K items
and moved the number of judged contrasts surviving Holm, so it cannot be an
undocumented default.

The alternative rules are kept so the tie-break can be reported as a
sensitivity analysis instead of an assumption. LEGACY_RULE reproduces the old
behaviour exactly, which is what makes the old numbers auditable.
"""
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUD = ROOT / "results/judgments"

# Mirrors JUDGES in src/run_judging.py. Declared here rather than derived from
# whatever files happen to be on disk, so a partially judged run is detected as
# incomplete instead of quietly redefining the panel to its own subset.
PANEL_JUDGES = ("gemma3:12b", "gpt-oss:20b", "phi4:14b")

CORRECT = "correct"
MISSING = "missing"
INCORRECT = "incorrect"
UNDECIDED = "undecided"

VERDICTS = (CORRECT, MISSING, INCORRECT)

# Generosity order, most generous first. Used only by the lenient/strict
# tie-break rules in the sensitivity analysis.
GENEROSITY = (CORRECT, MISSING, INCORRECT)

CANONICAL_RULE = "undecided"
LEGACY_RULE = "alphabetical"

TIEBREAK_RULES = {
    "undecided": "no majority is reported as undecided (canonical)",
    "alphabetical": "tie goes to the first judge alphabetically (legacy)",
    "lenient": "tie goes to the most generous tied label",
    "strict": "tie goes to the least generous tied label",
    "declined": "tie is reported as missing, i.e. the panel declines",
}


def panel_verdict(votes, tiebreak=CANONICAL_RULE):
    """Panel label for one item.

    votes is a mapping judge -> verdict; judges that never returned a verdict
    are either absent or map to None, and both mean the same thing, that the
    judge said nothing. Returns (label, n_votes, tied) where tied says whether
    the label needed the tie-break rule to be produced at all.
    """
    if tiebreak not in TIEBREAK_RULES:
        raise ValueError(f"unknown tie-break rule: {tiebreak}")
    cast = [votes[j] for j in sorted(votes) if votes.get(j)]
    if not cast:
        return UNDECIDED, 0, False
    counts = collections.Counter(cast)
    top = max(counts.values())
    winners = [v for v in counts if counts[v] == top]
    if len(winners) == 1:
        return winners[0], len(cast), False
    if tiebreak == "undecided":
        return UNDECIDED, len(cast), True
    if tiebreak == "alphabetical":
        # Counter preserves insertion order and most_common is a stable sort,
        # so the old code returned the tied label cast by the alphabetically
        # first judge. cast is built in sorted-judge order, so this reproduces
        # it exactly.
        return counts.most_common(1)[0][0], len(cast), True
    if tiebreak == "lenient":
        return next(v for v in GENEROSITY if v in winners), len(cast), True
    if tiebreak == "strict":
        return next(v for v in reversed(GENEROSITY) if v in winners), len(cast), True
    return MISSING, len(cast), True


def is_correct(label):
    """1 if the panel called the answer correct, 0 if not, None if undecided.

    None is not 0. An undecided item carries no information about the answer,
    so it must be dropped from a correctness rate rather than counted against
    the arm.
    """
    if label == UNDECIDED:
        return None
    return int(label == CORRECT)


def load_verdicts():
    """(gen_file, qid) -> judge -> verdict, over the reported benchmark.

    Scoped by name rather than by whatever is on disk. A withdrawn track's
    verdicts may still be sitting in results/judgments from an earlier run, and
    a glob that swept them in would make the numbers depend on which files a
    given machine happens to hold, which is the opposite of reproducible.
    """
    per = collections.defaultdict(dict)
    for fp in sorted(glob.glob(str(JUD / "obliqa_*__*.jsonl"))):
        if fp.endswith("__extract.jsonl"):
            continue
        for l in open(fp):
            r = json.loads(l)
            per[(r["gen_file"], r["qid"])][r["judge"]] = r["verdict"]
    return per


def by_gen_file(per):
    """Regroup load_verdicts output as gen_file -> qid -> judge -> verdict."""
    out = collections.defaultdict(dict)
    for (gf, qid), jd in per.items():
        out[gf][qid] = jd
    return out


def panel_coverage(items):
    """Which panel judges graded a file, and how much of it.

    items is qid -> judge -> verdict for one generation file. Returns
    (coverage, complete, reason): coverage maps each panel judge to the
    fraction of items it graded, complete says whether every panel judge
    graded at least one item, and reason names the missing judges. A file
    graded by one judge cannot yield a majority of three, so a panel figure
    must not be printed for it however many items it has.
    """
    n = max(len(items), 1)
    coverage = {}
    for j in PANEL_JUDGES:
        coverage[j] = sum(1 for jd in items.values() if jd.get(j)) / n
    absent = [j for j in PANEL_JUDGES if coverage[j] == 0.0]
    if absent:
        graded = [j for j in PANEL_JUDGES if coverage[j] > 0.0]
        reason = (f"not graded by the full panel: {', '.join(absent)} absent, "
                  f"only {', '.join(graded) or 'no panel judge'} graded this file")
        return coverage, False, reason
    return coverage, True, ""
