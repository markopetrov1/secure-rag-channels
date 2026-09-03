#!/bin/bash
# Rebuild the corpus and index from public sources.
#
# The corpus itself is deliberately not redistributed. ObliQA ships without a
# licence file, so this repository ships the build code rather than the text.
# Cloning a pinned depth-1 checkout and re-running the same preparation script
# reproduces a byte-identical corpus from the same public source; the per-split
# counts it should land on are in results/obliqa_prep_stats.json.
set -e
cd "$(dirname "$0")/.."
P=.venv/bin/python

echo "==> ObliQA (clone if absent)"
[ -d data/raw/ObliQADataset ] || git clone --depth 1 \
  https://github.com/RegNLP/ObliQADataset data/raw/ObliQADataset
$P src/prepare_obliqa.py

echo "==> dense index"
$P src/build_index.py obliqa
echo "Corpus and index rebuilt."
