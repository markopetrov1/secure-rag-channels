#!/bin/bash
# Project-local ollama instance: isolated model store, own port, one pinned GPU.
#
# The device is a parameter because which GPU is free changes. On 2026-08-28 the
# project's own PyTorch training occupied GPU 0 at 14.3 GB and full utilisation
# while GPU 1 sat idle, so pinning to 0 would have made this project fight its
# own other work for the same device.
#
#   PROJECT_GPU=1 ./scripts/start_ollama.sh
#
# Defaults to GPU 0 so existing callers behave exactly as before.
PROJECT_GPU="${PROJECT_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$PROJECT_GPU"
# Ollama also enumerates GPUs through its Vulkan backend, which ignores
# CUDA_VISIBLE_DEVICES and once split layers onto the second, busy GPU
# (0.4 tok/s against 92 tok/s). Keep it off so only the pinned device is used.
export OLLAMA_VULKAN=0
export GGML_VK_VISIBLE_DEVICES=""
export OLLAMA_HOST="127.0.0.1:${PROJECT_PORT:-11435}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export OLLAMA_MODELS="$ROOT/ollama-models"
# Two resident models by default. A judging pass alternates judge and extractor
# and with one resident model the server reloads on every alternation. An 8B
# generator plus bge-m3 is about 7 GB, well inside one of these devices.
export OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-2}
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-1}
export OLLAMA_KEEP_ALIVE=10m
echo "ollama serve: GPU $PROJECT_GPU, ${OLLAMA_HOST}, models $OLLAMA_MODELS" >&2
exec /usr/local/bin/ollama serve
