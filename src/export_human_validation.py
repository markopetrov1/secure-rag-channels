"""Export a stratified subset of judged answers for human annotation.

Produces results/human_validation/annotation_sheet.csv - one row per
(generation file, question), with the answer, the reference, and BLANK columns
for the human verdict. Judge verdicts are withheld from the sheet so the
annotator is not anchored; they live in a separate key file.

Stratification: proportional across (arm, generator), and within that
oversampling items where the judge panel disagreed (those are the informative
ones for the alt-test), while keeping a random core for unbiased estimation.
"""
import argparse
import glob
import json
import random
import collections
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "results/generations"
JUD = ROOT / "results/judgments"
OUT = ROOT / "results/human_validation"

N_RANDOM_CORE = 120     # unbiased random stratified core
N_DISAGREEMENT = 60     # oversampled panel-disagreement items
SEED = 7


def returned_sheets():
    """Annotation sheets that have come back with verdicts in them."""
    return sorted(glob.glob(str(OUT / "grading_*.csv")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-export even though annotated sheets exist, which "
                         "reassigns item ids and orphans those labels")
    a = ap.parse_args()

    # Once the sheet has been annotated, its item ids are the only link between
    # a human verdict and the answer it was written about. The sample is drawn
    # from whatever judgments are on disk, so re-exporting after any new
    # judgment lands reassigns every id and silently detaches the labels from
    # the answers. This is a no-op in that case rather than a destructive one.
    done = returned_sheets()
    if done and not a.force:
        print("annotated sheets present, so the instrument is frozen:")
        for f in done:
            print(f"  {Path(f).name}")
        print("re-exporting would reassign item ids and orphan those verdicts; "
              "pass --force if that is really what you want")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    verdicts = collections.defaultdict(dict)
    for fp in glob.glob(str(JUD / "*__*.jsonl")):
        if fp.endswith("__extract.jsonl"):
            continue
        for l in open(fp):
            r = json.loads(l)
            verdicts[(r["gen_file"], r["qid"])][r["judge"]] = r["verdict"]
    if not verdicts:
        print("no judgments yet - run judging first")
        return

    answers, refs = {}, {}
    for fp in glob.glob(str(GEN / "*_open_*.jsonl")):
        stem = Path(fp).stem
        for l in open(fp):
            r = json.loads(l)
            answers[(stem, r["qid"])] = r["answer"]
    questions = {}
    # The frozen annotation sheet was drawn before the second benchmark was
    # withdrawn, so its item ids span both. Those items are still in the sheet
    # the annotators returned and are reported in check_annotations.py, but
    # their source file is no longer redistributed, so the sheet can only be
    # rebuilt for the benchmark this paper reports. Regenerating it here yields
    # the ObliQA half of the instrument, not the whole of it.
    withdrawn = ROOT / "data/processed/iso27k_mcq.jsonl"
    if withdrawn.exists():
        for l in withdrawn.open():
            r = json.loads(l)
            refs[("iso27k", r["qid"])] = r["answer_text"]
            questions[("iso27k", r["qid"])] = r["question"]
    for l in (ROOT / "data/processed/obliqa_test_sample.jsonl").open():
        r = json.loads(l)
        refs[("obliqa", r["qid"])] = "\n".join(
            f"- {p['Passage']}" for p in r["gold_passages"])
        questions[("obliqa", r["qid"])] = r["question"]

    keys = [k for k in verdicts if k in answers]
    disagreed = [k for k in keys if len(set(verdicts[k].values())) > 1]
    agreed = [k for k in keys if k not in set(disagreed)]

    by_strat = collections.defaultdict(list)
    for k in agreed:
        parts = k[0].split("_")
        by_strat[(parts[0], parts[2])].append(k)
    core = []
    per = max(1, N_RANDOM_CORE // max(len(by_strat), 1))
    for s, items in by_strat.items():
        core.extend(rng.sample(items, min(per, len(items))))
    extra = rng.sample(disagreed, min(N_DISAGREEMENT, len(disagreed)))

    # The annotator must not see which arm produced an answer, so the sheet
    # carries an opaque item id only; arm identity, stratum and judge verdicts
    # live in the key file.
    selected = [(k, "random_core") for k in core] + \
               [(k, "disagreement") for k in extra]
    rng.shuffle(selected)

    sheet_rows, key_rows = [], []
    for i, (k, stratum) in enumerate(selected):
        gen_file, qid = k
        track = gen_file.split("_")[0]
        item_id = f"HV{i:04d}"
        sheet_rows.append({
            "item_id": item_id,
            "question": questions.get((track, qid), ""),
            "reference": refs.get((track, qid), "")[:1500],
            "candidate_answer": answers[k],
            "human_verdict": "",          # correct | missing | incorrect
            "human_notes": "",
        })
        key_rows.append({
            "item_id": item_id, "gen_file": gen_file, "qid": qid,
            "stratum": stratum, "track": track,
            "arm": gen_file.split("_")[2],
            "generator": "_".join(gen_file.split("_")[3:]),
            **{f"judge_{j.split(':')[0]}": v for j, v in verdicts[k].items()}})

    pd.DataFrame(sheet_rows).to_csv(OUT / "annotation_sheet.csv", index=False)
    pd.DataFrame(key_rows).to_csv(
        OUT / "judge_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv", index=False)

    # Sampling fractions, so stratum weights can be restored when the human
    # labels are used to estimate population quantities. The disagreement
    # stratum is deliberately oversampled and must be down-weighted.
    n_core_pop, n_dis_pop = len(agreed), len(disagreed)
    n_core_s = sum(1 for _, st in selected if st == "random_core")
    n_dis_s = sum(1 for _, st in selected if st == "disagreement")
    pd.DataFrame([
        {"stratum": "random_core", "population": n_core_pop,
         "sampled": n_core_s,
         "weight": (n_core_pop / n_core_s) if n_core_s else float("nan")},
        {"stratum": "disagreement", "population": n_dis_pop,
         "sampled": n_dis_s,
         "weight": (n_dis_pop / n_dis_s) if n_dis_s else float("nan")},
    ]).to_csv(OUT / "stratum_weights.csv", index=False)
    df = pd.DataFrame(sheet_rows)

    (OUT / "INSTRUCTIONS.md").write_text("""# Human validation annotation

Fill the `human_verdict` column in `annotation_sheet.csv` with exactly one of:

- `correct`   - the answer's substantive claims agree with the reference and it
                answers the question asked.
- `missing`   - the answer declines, says it cannot answer, or omits the key
                information the question asks for.
- `incorrect` - the answer makes at least one substantive claim that contradicts
                the reference, or invents unsupported facts.

Grade against the reference only. Ignore style and length. Do not open the judge
key file until you have finished - it exists so we can measure human-judge
agreement, and looking first would invalidate that.

The sheet is blind by construction: it carries an opaque item id and shows only
the question, the reference and the candidate answer, so you cannot tell which
pipeline produced an answer. `judge_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv` maps
item ids back to arms and judge verdicts, and `stratum_weights.csv` records the
sampling fractions needed to restore stratum weights, because the disagreement
stratum is deliberately oversampled and must be down-weighted when the labels
are used to estimate population quantities.

A second annotator on the same sheet lets us report human-human agreement as the
ceiling for judge performance (strongly recommended; reviewers ask for it).
""")
    print(f"wrote {len(df)} rows -> {OUT}/annotation_sheet.csv")
    print(pd.DataFrame(key_rows)["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
