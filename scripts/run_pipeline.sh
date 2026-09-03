#!/bin/bash
# Master pipeline.
#
# Generation runs with concurrent requests for throughput. Token counts are
# unaffected by concurrency, so the cost model is exact either way, and every
# reported latency comes instead from a dedicated serial benchmark on an idle
# GPU (step 3). Records produced concurrently are marked so the analysis never
# reads latency from them.
#
# Every step is resumable: re-running skips any question already on disk, so an
# interruption costs only the questions in flight.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
G="qwen3:8b,llama3.1:8b"
W=4
step () { echo "### STEP $* @ $(date +%F' '%H:%M:%S)"; }

# A step that raises must not be stepped over in silence. Step 4 failed on
# 2026-07-31 because MIPROv2 needs optuna, printed a traceback, and the run
# carried on to report a clean finish with the DSPy arm entirely absent.
FAILED=()
guard () {        # $1 = label, rest = command
  local label="$1"; shift
  if ! "$@"; then
    echo "### STEP FAILED: $label"
    FAILED+=("$label")
  fi
}

# Never contend with another job on GPU 0. Checked before every server start,
# so a long run also stops rather than fighting a job that begins mid-pipeline.
require_free_gpu () {
  if ! ./scripts/gpu_guard.sh; then
    echo "### HALTED: GPU 0 is taken. Re-run ./scripts/run_pipeline.sh when it is free;"
    echo "### every stage is resumable and nothing computed so far is lost."
    exit 3
  fi
}

restart_ollama () {  # $1 = OLLAMA_NUM_PARALLEL
  require_free_gpu
  for pid in $(pgrep -u "$(whoami)" -f "llama-server"); do kill -9 "$pid" 2>/dev/null || true; done
  for pid in $(pgrep -u "$(whoami)" -f "ollama serve"); do kill -9 "$pid" 2>/dev/null || true; done
  sleep 4
  OLLAMA_NUM_PARALLEL="$1" setsid nohup ./scripts/start_ollama_gpu0.sh > logs/ollama_serve.log 2>&1 &
  sleep 12
  echo "  ollama devices listed: $(grep -ac 'inference compute' logs/ollama_serve.log) (must be 1)"
}

restart_ollama $W

# The short-context arms scale well with concurrency. The long-context arm does
# not: its prompts are about 24k tokens, so four concurrent requests saturate
# prefill and thrash the KV cache, measured at 1.8 questions per minute against
# 3.3 serial. It therefore runs on a single slot, in step 1b.
step "1 Fixed-schedule arms, short context"
guard "fixed arms" $P src/run_generation.py --track obliqa --mode open \
  --arms a1,a2o,a2,a4,a5 --generators $G --workers $W

step "1b Long-context arm, serial"
restart_ollama 1
guard "long context" $P src/run_generation.py --track obliqa --mode open \
  --arms a3 --generators $G --workers 1

step "2 Agentic arms, which decide their own retrieval"
restart_ollama $W
for ARM in a8 a8c; do
  guard "agentic $ARM" $P src/run_generation.py --track obliqa --mode open \
    --arms "$ARM" --generators $G --workers $W
done

step "3 Controlled serial latency benchmark"
restart_ollama 1
guard "latency bench" $P src/bench_latency.py --n 30

step "4 DSPy compile and eval"
for g in qwen3:8b llama3.1:8b; do
  guard "dspy compile $g" $P src/run_dspy.py --track obliqa --generator $g --stage compile --seed 0
  guard "dspy eval $g"    $P src/run_dspy.py --track obliqa --generator $g --stage eval    --seed 0
done

step "5 DSPy optimiser-variance seeds"
# The compiled arm's run-to-run spread has to be visible rather than assumed,
# so the same compile is repeated under two further optimiser seeds.
for s in 1 2; do
  guard "dspy seed $s compile" $P src/run_dspy.py --track obliqa --generator qwen3:8b --stage compile --seed $s
  guard "dspy seed $s eval"    $P src/run_dspy.py --track obliqa --generator qwen3:8b --stage eval    --seed $s
done

step "6 Retrieval quality against the gold supporting passages"
guard "retrieval eval" $P src/eval_retrieval.py

step "7 Judging"
restart_ollama $W
guard "judging" $P src/run_judging.py --files "results/generations/obliqa_open_*.jsonl" --workers $W

step "8 Analysis"
./scripts/run_analysis.sh

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "PIPELINE_DONE @ $(date +%F' '%H:%M:%S)"
else
  echo "PIPELINE_FINISHED_WITH_FAILURES @ $(date +%F' '%H:%M:%S)"
  for f in "${FAILED[@]}"; do echo "  failed: $f"; done
  exit 1
fi
