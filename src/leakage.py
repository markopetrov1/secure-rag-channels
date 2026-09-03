"""Leakage at any clearance, recomputed from one pass of judging.

Whether an answer repeats a passage is a property of the answer and the passage.
The clearance decides only whether that passage was one the reader may see. So
the disclosure matrix is judged once, with the clearance never mentioned, and
leakage under any policy is an intersection computed here on CPU. One GPU pass
buys every clearance, every labelling and every enforcement mechanism.

Three quantities, following the brief's metric list:

  context leakage   unauthorised passages that reached the context window
  answer leakage    unauthorised passages the answer actually repeated
  conversion        answer leakage divided by context leakage, the fraction of
                    exposure that becomes disclosure

The third is the one that matters for policy. If conversion were near zero the
context window would be a weak boundary and retrieval-side enforcement would be
worth little; if it is high, every unauthorised passage that reaches the prompt
should be treated as disclosed.
"""
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IDX = ROOT / "data/index"
DISC = ROOT / "results/disclosure"
TAB = ROOT / "results/tables"


def load_matrix(arm="a4", judge="gemma3_12b"):
    """(qid, generator, uid) -> disclosed, with repeat judgements reconciled.

    The file may hold a pair more than once, because the judging pass was run
    twice. That is reported rather than hidden: it is a test-retest measurement
    of the judge at temperature zero, and pairs the two passes disagree on are
    excluded from the point estimate and counted, since a pair the judge cannot
    reproduce should not carry a leakage claim.
    """
    seen = collections.defaultdict(list)
    for line in (DISC / f"matrix_{arm}_{judge}.jsonl").open():
        r = json.loads(line)
        if r["verdict"] is None:
            continue
        seen[(r["qid"], r["generator"], r["uid"])].append(r["verdict"])
    stable, unstable = {}, 0
    for k, vs in seen.items():
        if len(set(vs)) == 1:
            stable[k] = vs[0]
        else:
            unstable += 1
    repeats = sum(1 for vs in seen.values() if len(vs) > 1)
    return stable, {"pairs": len(seen), "repeated": repeats,
                    "unstable": unstable,
                    "retest_agreement": 1 - unstable / repeats if repeats else float("nan")}


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import access_control as ac

    meta = ac.load_corpus()
    uid_doc = {m["uid"]: str(m["doc_id"]) for m in meta}
    disclosed, diag = load_matrix()
    print(f"disclosure matrix: {diag['pairs']:,} pairs, "
          f"{diag['repeated']:,} judged twice, "
          f"test-retest agreement {diag['retest_agreement']:.4f}, "
          f"{diag['unstable']:,} unstable pairs excluded")
    print(f"overall disclosure rate: "
          f"{np.mean(list(disclosed.values())):.3f}\n")

    # context passages per (qid, generator), in rank order
    ctx = collections.defaultdict(list)
    for (qid, gen, uid) in disclosed:
        ctx[(qid, gen)].append(uid)

    labels = ac.labels_compartment(meta)
    universe = sorted(set(labels))
    doc_of_uid = uid_doc
    rng = np.random.default_rng(7)

    rows = []
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        n_lab = max(1, int(round(frac * len(universe))))
        per_draw = collections.defaultdict(list)
        for _ in range(60):
            granted = set(rng.choice(universe, n_lab, replace=False).tolist())
            ctx_leak = ans_leak = n = 0
            for key, uids in ctx.items():
                for u in uids:
                    if doc_of_uid.get(u) in granted:
                        continue
                    ctx_leak += 1
                    ans_leak += disclosed[(key[0], key[1], u)]
                n += 1
            per_draw["context_leak_per_query"].append(ctx_leak / n)
            per_draw["answer_leak_per_query"].append(ans_leak / n)
            per_draw["conversion"].append(ans_leak / ctx_leak if ctx_leak else np.nan)
        for k, v in per_draw.items():
            a = np.array(v, dtype=float)
            rows.append({"granted_fraction": frac, "metric": k,
                         "mean": float(np.nanmean(a)),
                         "lo": float(np.nanpercentile(a, 2.5)),
                         "hi": float(np.nanpercentile(a, 97.5))})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "leakage_by_clearance.csv", index=False)

    piv = df.pivot_table(index="granted_fraction", columns="metric", values="mean")
    piv = piv[["context_leak_per_query", "answer_leak_per_query", "conversion"]]
    print("== leakage under no access control, by clearance breadth ==")
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\ncontext leak: unauthorised passages reaching the prompt, per query")
    print("answer leak : unauthorised passages the answer repeated, per query")
    print("conversion  : the fraction of exposure that became disclosure")
    print(f"\nwrote results/tables/leakage_by_clearance.csv")


if __name__ == "__main__":
    main()
