#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dataset_id_or_path> <model_id_or_path> [output_dir]"
    exit 1
fi

DATASET=$1
MODEL=$2
OUTPUT_DIR=${3:-./results/llada2_moe_saturation}

GPU_IDS=${CUDA_VISIBLE_DEVICES:-0,1}
TP_SIZE=${TP_SIZE:-2}
BATCH_SIZES_TEXT=${BATCH_SIZES:-"1 2 4 8 16 32"}
NUM_PROMPTS=${NUM_PROMPTS:-64}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-128}
MAX_SCAN_EXAMPLES=${MAX_SCAN_EXAMPLES:-20000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
CACHE_MAX_ENTRY_COUNT=${CACHE_MAX_ENTRY_COUNT:-0.35}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.8}

read -r -a BATCH_SIZE_ARRAY <<< "${BATCH_SIZES_TEXT}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)

DATASET_ARGS=()
if [[ "${DATASET}" == *"hendrycks-MATH"* ]]; then
    DATASET_ARGS+=(--dataset-format math)
elif [[ "${DATASET}" == "openai/gsm8k" ]]; then
    DATASET_ARGS+=(--dataset-format gsm8k --hf-split test --hf-config main)
elif [[ "${DATASET}" == "google-research-datasets/mbpp" ]]; then
    DATASET_ARGS+=(--dataset-format mbpp --hf-split test --hf-config sanitized)
fi

echo "LLaDA2 MoE saturation experiment"
echo "  GPUs:              ${GPU_IDS}"
echo "  TP:                ${TP_SIZE}"
echo "  request batches:   ${BATCH_SIZES_TEXT}"
echo "  max input tokens:  ${MAX_INPUT_LEN}"
echo "  max output tokens: ${MAX_NEW_TOKENS}"
echo "  output:            ${OUTPUT_DIR}"

SUCCESSFUL_TRACES=()
FAILED_BATCHES=()
for BATCH_SIZE in "${BATCH_SIZE_ARRAY[@]}"; do
    TRACE_FILE="${OUTPUT_DIR}/routes_bs${BATCH_SIZE}.jsonl"
    LOG_FILE="${OUTPUT_DIR}/trace_run_bs${BATCH_SIZE}.log"
    ERROR_FILE="${OUTPUT_DIR}/trace_run_bs${BATCH_SIZE}.err"
    CSV_FILE="${OUTPUT_DIR}/trace_run_bs${BATCH_SIZE}.csv"
    MAX_PREFILL_TOKEN_NUM=$((BATCH_SIZE * MAX_INPUT_LEN))

    echo "Running request batch ${BATCH_SIZE}..."
    if CUDA_VISIBLE_DEVICES="${GPU_IDS}" python benchmark/profile_throughput.py \
            "${DATASET}" "${MODEL}" \
            "${DATASET_ARGS[@]}" \
            --backend pytorch \
            --tp "${TP_SIZE}" \
            --distributed-executor-backend mp \
            --dtype bfloat16 \
            --eager-mode \
            --cache-max-entry-count "${CACHE_MAX_ENTRY_COUNT}" \
            --dllm-block-length 32 \
            --dllm-denoising-steps 32 \
            --dllm-confidence-threshold "${CONFIDENCE_THRESHOLD}" \
            --dllm-enable-delayed-cache \
            --dllm-track \
            --max-new-tokens "${MAX_NEW_TOKENS}" \
            --max-input-len "${MAX_INPUT_LEN}" \
            --max-prefill-token-num "${MAX_PREFILL_TOKEN_NUM}" \
            --max-scan-examples "${MAX_SCAN_EXAMPLES}" \
            --num-prompts "${NUM_PROMPTS}" \
            --concurrency "${BATCH_SIZE}" \
            --temperature 0 \
            --no-stream-output \
            --skip-tokenize \
            --skip-detokenize \
            --moe-trace-output "${TRACE_FILE}" \
            --csv "${CSV_FILE}" \
            >"${LOG_FILE}" 2>"${ERROR_FILE}"; then
        SUCCESSFUL_TRACES+=("${TRACE_FILE}")
    else
        FAILED_BATCHES+=("${BATCH_SIZE}")
        echo "Batch ${BATCH_SIZE} failed; preserving earlier traces. See ${ERROR_FILE}." >&2
    fi
done

if [[ ${#SUCCESSFUL_TRACES[@]} -eq 0 ]]; then
    echo "No batch completed successfully." >&2
    exit 1
fi

python benchmark/analyze_moe_saturation.py \
    "${SUCCESSFUL_TRACES[@]}" \
    --output-csv "${OUTPUT_DIR}/moe_saturation_summary.csv" \
    --output-svg "${OUTPUT_DIR}/moe_saturation.svg"

if [[ ${#FAILED_BATCHES[@]} -gt 0 ]]; then
    echo "Failed batches: ${FAILED_BATCHES[*]} (inspect trace_run_bs*.err)" >&2
fi
echo "Experiment complete: ${OUTPUT_DIR}"
