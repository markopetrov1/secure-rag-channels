#!/bin/bash
# Full analysis chain over whatever results currently exist.
#
# Each stage's exit status is recorded and reported at the end. Letting a stage
# fail quietly is what hid the DSPy crash of 2026-07-31 for three weeks, and on
# 2026-08-22 it briefly hid figures.py aborting on a colour-roster assertion
# while every other stage reported success.
# Each staged command pipes through tee, and a pipeline's status is its last
# command's, so without pipefail a failing stage reports tee's zero and the
# whole point of this wrapper is lost. A fresh `bash -c` does not inherit the
# option from here, so it is passed to each one as `bash -o pipefail -c`.
set -uo pipefail
cd "$(dirname "$0")/.."
P=.venv/bin/python
FAILED=()

run () {          # $1 = label, rest = command
  local label="$1"; shift
  if ! "$@"; then
    echo "### STAGE FAILED: $label"
    FAILED+=("$label")
  fi
}

run "analyze"        bash -o pipefail -c "$P src/analyze.py         | tee results/tables/_analyze.txt"
run "stats"          bash -o pipefail -c "$P src/stats.py           | tee results/tables/_stats.txt"
run "judge agreement" bash -o pipefail -c "$P src/judge_validity.py | tee results/tables/_judge_validity.txt"
run "retrieval inference" $P src/retrieval_stats.py
run "economics"      bash -o pipefail -c "$P src/economics.py       | tee results/tables/_economics.txt"
run "compute budget" bash -o pipefail -c "$P src/compute_budget.py  | tee results/tables/_compute_budget.txt"
# The dominance table carries the headline claim and costs nothing to
# recompute, so it belongs here rather than only in scripts/finalise.sh, which
# is where it used to live and where a rerun of this script would miss it.
run "dominance"      bash -o pipefail -c "$P src/dominance.py       | tee results/tables/_dominance.txt"
# src/agentic_analysis.py is deliberately NOT in this chain. It re-ranks the
# agent's queries against the corpus index, so it needs the model server and
# GPU 0, and it writes its own retrieval calls into the token ledger this chain
# reads. Everything above runs from the committed results on a laptop, which is
# what requirements-analysis.txt promises. Run it from scripts/finish_agentic.sh
# or scripts/finalise.sh, beside the generation it describes.
run "figures"        bash -o pipefail -c "$P src/figures.py         | tee results/tables/_figures.txt"
run "human validation export" $P src/export_human_validation.py
# Reads the returned annotation sheets if they are there and is a no-op if they
# are not, so the chain runs the same way before and after annotation.
run "human validation" bash -o pipefail -c "$P src/check_annotations.py | tee results/tables/_human_validation.txt"
run "summary digest" bash -o pipefail -c "$P src/summarize_results.py > results/tables/_summary.txt 2>&1"
# Advisory, not a gate. While data is still arriving the reported numbers
# are expected to drift, and the verifier exists to say exactly what moved. A
# non-zero exit here does not mean the analysis failed, so it is reported
# without entering the failure list; run it directly before believing any number.
run "facts digest"   bash -o pipefail -c "$P src/results_facts.py > results/tables/_facts.txt 2>&1"

if [ ${#FAILED[@]} -eq 0 ]; then
  echo ANALYSIS_DONE
else
  echo "ANALYSIS_FINISHED_WITH_FAILURES"
  for f in "${FAILED[@]}"; do echo "  failed: $f"; done
  exit 1
fi
