#!/bin/bash
# Project-local ollama instance, pinned to GPU 0, isolated model store and port.
export CUDA_VISIBLE_DEVICES=0
# Ollama also enumerates GPUs through its Vulkan backend, which ignores
# CUDA_VISIBLE_DEVICES and made it split layers onto the second, busy GPU
# (0.4 tok/s vs 92 tok/s). Keep it off so only CUDA0 is ever used.
export OLLAMA_VULKAN=0
export GGML_VK_VISIBLE_DEVICES=""
export OLLAMA_HOST=127.0.0.1:11435
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export OLLAMA_MODELS="$ROOT/ollama-models"
# One model resident at a time. The pipeline is generator-major and the
# judging pass is judge-major, so nothing needs two models at once, and
# with four concurrent 32k-context slots a second resident model would
# not fit in 48 GB.
# Two by default. A graph build alternates between the generator and the
# embedder, and with one resident model the server reloads on every alternation,
# which blew LightRAG's embedding timeout and failed 48 of 51 documents on
# 2026-08-23. An 8B generator plus bge-m3 is about 7 GB, well inside this device.
export OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-2}
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-1}
export OLLAMA_KEEP_ALIVE=10m
exec /usr/local/bin/ollama serve
