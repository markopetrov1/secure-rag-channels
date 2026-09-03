#!/bin/bash
# Generate the agentic arms for both generators.
#
# The gold passages are what these arms exist to be measured against: they let
# an agent's own search queries be scored against the same target as the
# question it was handed, which is the only way to separate what autonomy costs
# from what it retrieves.
#
# Resumable in the same way as every other stage: re-running skips any question
# already on disk, so an interruption costs only the questions in flight.
set -uo pipefail
cd "$(dirname "$0")/.."
P=.venv/bin/python
W=${W:-4}
FAILED=()

run () {
  local label="$1"; shift
  echo "### $(date '+%F %T')  $label"
  if ! "$@"; then
    echo "### STAGE FAILED: $label"
    FAILED+=("$label")
  fi
}

./scripts/gpu_guard.sh || { echo "GPU 0 is busy; not starting."; exit 1; }

for ARM in a8 a8c; do
  run "obliqa open $ARM" $P src/run_generation.py --track obliqa --mode open \
      --arms "$ARM" --generators qwen3:8b,llama3.1:8b --workers "$W"
done

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "AGENTIC_GENERATION_DONE"
else
  echo "AGENTIC_GENERATION_FINISHED_WITH_FAILURES"
  for f in "${FAILED[@]}"; do echo "  failed: $f"; done
  exit 1
fi
