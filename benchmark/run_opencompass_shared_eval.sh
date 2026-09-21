#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -gt 1 ]]; then
  echo 'Use one dataset config per run so timing is dataset-specific.' >&2
  exit 2
fi
DATASET=${1:-gsm8k_shared_smoke}
export MODEL_PATH=${MODEL_PATH:-/root/lkd/Models/LLaDA2.0-mini}
export BATCH_SIZE=${BATCH_SIZE:-8}
export MAX_OUT_LEN=${MAX_OUT_LEN:-1024}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
export BLOCK_LENGTH=${BLOCK_LENGTH:-32}
export DENOISING_STEPS=${DENOISING_STEPS:-32}
export CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.8}
export ROUTE_EPSILON=${ROUTE_EPSILON:-0.1}
export ROUTE_METHOD=${ROUTE_METHOD:-joint}
export SHARED_MODE=${SHARED_MODE:-both}
export SAMPLING=${SAMPLING:-native}
export SEED=${SEED:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
RUN_DIR=${OUTPUT_DIR:-${ROOT}/results/shared_route_eval/$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"  # fail rather than mix old predictions/timing with a new run
RUN_DIR=$(cd "$RUN_DIR" && pwd)
export ROUTE_METRICS_DIR="$RUN_DIR/metrics"
export PYTHONPATH="$ROOT/opencompass-0.5.1.post1${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$ROOT/opencompass-0.5.1.post1"
"$PYTHON_BIN" run.py --models llada2_shared_pair --datasets "$DATASET" \
  --max-num-workers 1 --work-dir "$RUN_DIR/opencompass" 2>&1 | tee "$RUN_DIR/run.log"
EXPECTED_MODELS=1
if [[ "$SHARED_MODE" == both ]]; then EXPECTED_MODELS=2; fi
"$PYTHON_BIN" "$ROOT/benchmark/summarize_shared_eval.py" "$ROUTE_METRICS_DIR" \
  --expected-models "$EXPECTED_MODELS" --output "$RUN_DIR/timing_summary.json"
