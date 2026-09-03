#!/bin/bash
# Additions that came out of the reframing: the embedder ablation and the
# one-shot arm.
#
# The ablation asks whether the RTEB leaderboard order survives on a regulatory
# corpus with real gold passages, and whether any dense model clears the lexical
# baseline that beat bge-m3 in the main study. That question is the reason the
# main retrieval finding is exposed to a "you used a weak embedder" objection,
# and it is answerable in about two GPU hours.
#
# The one-shot arm completes the in-context ladder from zero to one to five
# demonstrations, so the collapse in answer length that confounds the few-shot
# arm becomes a dose-response curve rather than two unrelated points.
#
# Waits for any generation or judging run to finish first, because both want
# GPU 0 and because the ablation loads models outside ollama, which a watchdog
# would otherwise read as a foreign job and kill.
set -uo pipefail
cd "$(dirname "$0")/.."
P=.venv/bin/python
W=${W:-4}
G="qwen3:8b llama3.1:8b"
FAILED=()
STAGE=""
# The watchdog used to signal the parent with TERM, which bash ignored while it
# was waiting on a child. On 2026-08-23 that let the run continue through every
# remaining stage after the server had been torn down, failing each one against
# a closed port. The watchdog now also drops this sentinel, and the stage runner
# refuses to start anything once it exists.
YIELD_FLAG="logs/.yielded.$$"
rm -f "$YIELD_FLAG"

# Children of this script share its process group. The guard is told to ignore
# that group so a model this script loads directly is not mistaken for someone
# else's job.
export MINE_PGID
MINE_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
# Long-tailed chunk lengths fragment the allocator badly on this device.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log ()  { echo; echo "### $* @ $(date +%F' '%H:%M:%S)"; }
note () { echo "    $*"; }

run () {
  if [ -e "$YIELD_FLAG" ]; then
    echo "### SKIPPING '$1': the run yielded GPU 0 and is shutting down."
    return 1
  fi
  STAGE="$1"; local label="$1"; shift
  log "$label"
  if ! "$@"; then
    echo "### STAGE FAILED: $label"
    FAILED+=("$label")
    return 1
  fi
}

require_free_gpu () {
  if ! ./scripts/gpu_guard.sh; then
    echo "### HALTED at '$STAGE': GPU 0 is taken. Re-run when it is free."
    exit 3
  fi
}

start_server () {
  require_free_gpu
  for pid in $(pgrep -u "$(whoami)" -f "/usr/local/lib/ollama/llama-server"); do
    kill -9 "$pid" 2>/dev/null || true
  done
  for pid in $(pgrep -u "$(whoami)" -f "ollama serve"); do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 4
  OLLAMA_NUM_PARALLEL="$1" setsid nohup ./scripts/start_ollama_gpu0.sh \
      > logs/ollama_serve.log 2>&1 &
  sleep 12
  local n
  n=$(grep -ac 'inference compute' logs/ollama_serve.log)
  note "ollama devices listed: $n (must be 1)"
  [ "$n" = "1" ] || { echo "### ABORT: $n devices enumerated, not 1"; exit 4; }
}

start_watchdog () {
  local parent=$$
  (
    while kill -0 "$parent" 2>/dev/null; do
      sleep "${GUARD_POLL:-60}"
      if ! ./scripts/gpu_guard.sh >/dev/null 2>&1; then
        echo; echo "### YIELDING at $(date +%F' '%H:%M:%S): a job that is not ours"
        echo "### appeared on GPU 0. Re-run when it is free; every stage resumes."
        pkill -g "$MINE_PGID" -f "src/run_generation.py" 2>/dev/null
        pkill -g "$MINE_PGID" -f "src/run_judging.py" 2>/dev/null
        pkill -g "$MINE_PGID" -f "src/embed_ablation.py" 2>/dev/null
        sleep 3
        for pid in $(pgrep -u "$(whoami)" -f "/usr/local/lib/ollama/llama-server"); do
          kill -9 "$pid" 2>/dev/null || true
        done
        for pid in $(pgrep -u "$(whoami)" -f "ollama serve"); do
          kill -9 "$pid" 2>/dev/null || true
        done
        : > "$YIELD_FLAG"
        kill -TERM "$parent" 2>/dev/null
        sleep 5
        kill -KILL "$parent" 2>/dev/null
        exit 0
      fi
    done
  ) &
  WATCHDOG=$!
}

cleanup () {
  rm -f "$YIELD_FLAG"
  [ -n "${WATCHDOG:-}" ] && kill "$WATCHDOG" 2>/dev/null
  for pid in $(pgrep -u "$(whoami)" -f "/usr/local/lib/ollama/llama-server"); do
    kill -9 "$pid" 2>/dev/null || true
  done
  for pid in $(pgrep -u "$(whoami)" -f "ollama serve"); do
    kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

# ---------------------------------------------------------------- wait
log "waiting for any generation or judging run to finish"
while pgrep -u "$(whoami)" -f "src/run_generation.py\|src/run_judging.py" \
      > /dev/null 2>&1; do
  sleep 120
done
note "GPU 0 is free; starting"

# ---------------------------------------------------------------- ablation
# bge-m3 goes through the same code path as the others rather than reusing the
# main index, so every model is embedded, normalised and scored identically and
# the comparison cannot be an artefact of two different pipelines.
start_server 1
start_watchdog

run "embedder ablation, bge-m3 baseline" \
    $P src/embed_ablation.py --model bge-m3 --track obliqa

# Smallest first: if a HuggingFace load fails on this device, it fails in ten
# minutes rather than after an hour of downloading.
for m in bge-m3-hf nemotron-1b qwen3-emb-4b qwen3-emb-8b nemotron-8b; do
  run "embedder ablation, $m" \
      $P src/embed_ablation.py --model "$m" --track obliqa
done

run "embedder ablation report" \
    bash -o pipefail -c "$P src/embed_ablation.py --report --track obliqa \
        | tee results/tables/_embedding_ablation.txt"

# ---------------------------------------------------------------- one-shot arm
run "one-shot arm, ObliQA open ended" \
    $P src/run_generation.py --track obliqa --mode open --arms a2o \
       --generators "qwen3:8b,llama3.1:8b" --workers "$W"

# bench_latency defaults to the original arm list, so the new arm is named.
run "latency benchmark for the one-shot arm" \
    $P src/bench_latency.py --n 30 --arms a2o

run "judging, one-shot arm" \
    $P src/run_judging.py --files "results/generations/obliqa_open_a2o_*.jsonl" \
       --workers "$W"
run "requeue unparsed verdicts" $P scripts/requeue_failed_judgments.py

run "final analysis" ./scripts/run_analysis.sh

echo
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ENRICHMENT_COMPLETE @ $(date +%F' '%H:%M:%S)"
else
  echo "ENRICHMENT_FINISHED_WITH_FAILURES @ $(date +%F' '%H:%M:%S)"
  for f in "${FAILED[@]}"; do echo "  failed: $f"; done
  exit 1
fi
