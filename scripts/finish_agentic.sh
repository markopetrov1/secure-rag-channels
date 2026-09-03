#!/bin/bash
# Wait for agentic generation, then judge it, then recompute the economics.
#
# The question the judging answers is whether the agentic arms join the
# dominated set. They share the retriever's index, so they carry no setup cost
# of their own; if their answers are no better than the fixed pipeline they wrap,
# then a higher marginal cost with no quality gain makes them dominated at every
# volume, and the claim closes.
set -uo pipefail
cd "$(dirname "$0")/.."
P=.venv/bin/python

echo "### waiting for agentic generation to finish"
while pgrep -f run_generation.py > /dev/null; do sleep 60; done
echo "### generation done $(date '+%F %T')"
wc -l results/generations/*a8*.jsonl

echo "### judging the agentic arms"
$P src/run_judging.py --files "results/generations/obliqa_open_a8*.jsonl" --workers 4 \
  || echo "### JUDGING FAILED"

echo "### recomputing analysis"
$P src/analyze.py            > results/tables/_analyze.txt 2>&1
$P src/economics.py          > results/tables/_economics.txt 2>&1
$P src/dominance.py          > results/tables/_dominance.txt 2>&1
$P src/agentic_analysis.py   > results/tables/_agentic.txt 2>&1
echo "### AGENTIC_PIPELINE_DONE $(date '+%F %T')"
