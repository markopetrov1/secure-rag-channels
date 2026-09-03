"""Prepare the ObliQA/RIRAG track.

Outputs:
  data/processed/obliqa_test_sample.jsonl  - seeded stratified sample of test questions
  data/processed/obliqa_fewshot_dev.jsonl  - k few-shot exemplars drawn from train
  data/processed/obliqa_corpus.jsonl       - flattened passage corpus (one row per passage)
  results/obliqa_prep_stats.json
"""
import json
import random
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/ObliQADataset"
OUT = ROOT / "data/processed"
RESULTS = ROOT / "results"

SEED = 42
SAMPLE_SIZE = 500
N_FEWSHOT = 8  # pool; arms use k=5


def load(split):
    return json.load((RAW / f"ObliQA_{split}.json").open())


def flatten_corpus():
    rows = []
    for f in sorted((RAW / "StructuredRegulatoryDocuments").glob("*.json"),
                    key=lambda p: int(p.stem)):
        doc = json.load(f.open())
        for el in doc:
            passage = (el.get("Passage") or "").strip()
            if not passage:
                continue
            rows.append({
                "uid": el["ID"],
                "doc_id": el["DocumentID"],
                "passage_id": el["PassageID"],
                "text": passage,
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    test = load("test")
    # Stratify by Group (question-generation group in ObliQA; keeps mix of 1-passage
    # and multi-passage questions representative).
    by_group = collections.defaultdict(list)
    for q in test:
        by_group[q["Group"]].append(q)
    total = len(test)
    sample = []
    for g, items in sorted(by_group.items()):
        k = round(SAMPLE_SIZE * len(items) / total)
        sample.extend(rng.sample(items, min(k, len(items))))
    # top up / trim to exact size deterministically
    if len(sample) < SAMPLE_SIZE:
        chosen = {q["QuestionID"] for q in sample}
        rest = [q for q in test if q["QuestionID"] not in chosen]
        sample.extend(rng.sample(rest, SAMPLE_SIZE - len(sample)))
    sample = sample[:SAMPLE_SIZE]
    rng.shuffle(sample)

    with (OUT / "obliqa_test_sample.jsonl").open("w") as f:
        for q in sample:
            f.write(json.dumps({
                "qid": q["QuestionID"],
                "question": q["Question"],
                "gold_passages": q["Passages"],
                "group": q["Group"],
            }, ensure_ascii=False) + "\n")

    train = load("train")
    fewshot = rng.sample(train, N_FEWSHOT)
    with (OUT / "obliqa_fewshot_dev.jsonl").open("w") as f:
        for q in fewshot:
            f.write(json.dumps({
                "qid": q["QuestionID"],
                "question": q["Question"],
                "gold_passages": q["Passages"],
            }, ensure_ascii=False) + "\n")

    corpus = flatten_corpus()
    with (OUT / "obliqa_corpus.jsonl").open("w") as f:
        for r in corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    npass = [len(q["Passages"]) for q in sample]
    stats = {
        "seed": SEED,
        "test_total": total,
        "sample_size": len(sample),
        "group_dist_sample": dict(collections.Counter(q["Group"] for q in sample)),
        "gold_passages_per_q": dict(collections.Counter(npass)),
        "corpus_passages": len(corpus),
        "corpus_docs": len({r["doc_id"] for r in corpus}),
        "corpus_words": sum(len(r["text"].split()) for r in corpus),
    }
    (RESULTS / "obliqa_prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
