#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -gt 1 ]]; then
  echo 'Use one dataset config per run.' >&2
  exit 2
fi
DATASET=${1:-gsm8k_shared_smoke}
export MODEL_PATH=${MODEL_PATH:-/root/lkd/Models/LLaDA2.0-mini}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export BATCH_SIZE=${BATCH_SIZE:-8}
export MAX_OUT_LEN=${MAX_OUT_LEN:-32}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
export BLOCK_LENGTH=${BLOCK_LENGTH:-32}
export DENOISING_STEPS=${DENOISING_STEPS:-32}
export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.8}
export ROUTE_EPSILON=${ROUTE_EPSILON:-0.1}
export JOINT_METHODS=${JOINT_METHODS:-native,independent,request,batch}
export MAX_MEMORY_PER_GPU=${MAX_MEMORY_PER_GPU:-34GiB}
export SAMPLING=${SAMPLING:-greedy}
export DETAILED_TIMING=${DETAILED_TIMING:-0}
export SEED=${SEED:-0}
RUN_DIR=${OUTPUT_DIR:-${ROOT}/results/block_joint_eval/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
export ROUTE_METRICS_DIR="$RUN_DIR/metrics"
export PYTHONPATH="$ROOT/opencompass-0.5.1.post1${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$ROOT/opencompass-0.5.1.post1"
"$PYTHON_BIN" run.py --models llada2_joint_four --datasets "$DATASET" \
  --max-num-workers 1 --work-dir "$RUN_DIR/opencompass" 2>&1 | tee "$RUN_DIR/run.log"
IFS=',' read -r -a METHOD_ARRAY <<< "$JOINT_METHODS"
SUMMARY_ARGS=()
if [[ "$DATASET" == gsm8k_shared_smoke ]]; then
  SUMMARY_ARGS+=(--expected-requests 32)
elif [[ -n "${EXPECTED_REQUESTS:-}" ]]; then
  SUMMARY_ARGS+=(--expected-requests "$EXPECTED_REQUESTS")
fi
"$PYTHON_BIN" "$ROOT/benchmark/summarize_shared_eval.py" "$ROUTE_METRICS_DIR" \
  --expected-models "${#METHOD_ARRAY[@]}" "${SUMMARY_ARGS[@]}" --output "$RUN_DIR/timing_summary.json"
