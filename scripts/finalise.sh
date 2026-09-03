#!/bin/bash
# Wait for the judged agentic results, then regenerate every derived artefact.
#
# finish_agentic.sh stops after the analysis CSVs. This carries on through the
# remaining analysis and the figures, so that every reported number on
# disk is consistent with the data the moment judging ends.
set -uo pipefail
cd "$(dirname "$0")/.."
P=.venv/bin/python

echo "### waiting for judging $(date '+%F %T')"
while pgrep -f run_judging > /dev/null; do sleep 60; done
echo "### judging finished $(date '+%F %T')"
cat results/judgments/*a8* 2>/dev/null | wc -l | xargs echo "agentic verdicts:"

for stage in analyze economics dominance agentic_analysis figures; do
  echo "### $stage"
  $P "src/$stage.py" > "results/tables/_$stage.txt" 2>&1 || echo "### FAILED: $stage"
done

echo "### verifying claims"
tail -3 results/tables/_verify.txt

echo "### compiling"
echo "### FINALISE_DONE $(date '+%F %T')"
