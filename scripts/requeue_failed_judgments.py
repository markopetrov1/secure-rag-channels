"""Drop judgment records with no parsed verdict so the next judging pass retries
them. Records are keyed by (gen_file, qid, judge) and the runner skips any qid
already present, so an unparsed record would otherwise be permanent."""
import json, glob
from pathlib import Path

removed = 0
for f in sorted(glob.glob("results/judgments/*__*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    # Only panel verdict files are rewritten here. A file of some other shape
    # carries no "verdict" key at all, so the filter below would read as "every
    # record failed" and empty it.
    if not all("verdict" in r for r in rows):
        continue
    keep = [r for r in rows if r.get("verdict") is not None]
    if len(keep) != len(rows):
        with open(f, "w") as fh:
            for r in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{Path(f).name}: dropped {len(rows)-len(keep)}")
        removed += len(rows) - len(keep)
print(f"total requeued: {removed}")
