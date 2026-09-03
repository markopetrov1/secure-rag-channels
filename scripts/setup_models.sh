#!/bin/bash
# Pull every model the study uses into the project-local ollama store.
# Roughly 50 GB of weights. No judge shares a model family with either generator,
# because a judge grading its own family's output would flatter it.
set -e
cd "$(dirname "$0")/.."
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$PWD/ollama-models"

MODELS=(
  "bge-m3"                 # embedder, retrieval arms
  "qwen3:8b"               # generator
  "llama3.1:8b"            # generator
  "gemma3:12b"             # judge
  "gpt-oss:20b"            # judge
  "phi4:14b"               # judge
  "qwen3-embedding:8b"     # embedding ablation
)
for m in "${MODELS[@]}"; do
  echo "==> $m"
  ollama pull "$m"
done
echo "All models pulled into $OLLAMA_MODELS"
