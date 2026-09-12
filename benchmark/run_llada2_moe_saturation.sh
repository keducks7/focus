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
BATCH_SIZES_TEXT=${BATCH_SIZES:-"8"}
NUM_PROMPTS=${NUM_PROMPTS:-32}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-128}
MAX_SCAN_EXAMPLES=${MAX_SCAN_EXAMPLES:-20000}
MASK_BLOCK_LENGTH=${MASK_BLOCK_LENGTH:-${MAX_NEW_TOKENS:-32}}
DENOISING_STEPS=${DENOISING_STEPS:-32}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.95}
TEMPERATURE=${TEMPERATURE:-0}
MAX_MEMORY_PER_GPU=${MAX_MEMORY_PER_GPU:-38GiB}

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
elif [[ "${DATASET}" == "openai/openai_humaneval" ]]; then
    DATASET_ARGS+=(--dataset-format auto --hf-split test)
fi

echo "LLaDA2 MoE denoising-route experiment"
echo "  GPUs:              ${GPU_IDS}"
echo "  loader:            HF Accelerate balanced device map"
echo "  request batches:   ${BATCH_SIZES_TEXT}"
echo "  max input tokens:  ${MAX_INPUT_LEN}"
echo "  observed mask block: ${MASK_BLOCK_LENGTH}"
echo "  denoising steps:   ${DENOISING_STEPS}"
echo "  confidence:        ${CONFIDENCE_THRESHOLD}"
echo "  temperature:       ${TEMPERATURE}"
echo "  output:            ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python benchmark/profile_llada2_hf_moe_saturation.py \
    "${DATASET}" "${MODEL}" \
    "${DATASET_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-sizes "${BATCH_SIZE_ARRAY[@]}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-input-len "${MAX_INPUT_LEN}" \
    --max-scan-examples "${MAX_SCAN_EXAMPLES}" \
    --block-length "${MASK_BLOCK_LENGTH}" \
    --denoising-steps "${DENOISING_STEPS}" \
    --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
    --temperature "${TEMPERATURE}" \
    --max-memory-per-gpu "${MAX_MEMORY_PER_GPU}" \
    2>&1 | tee "${OUTPUT_DIR}/hf_trace_run.log"

MANIFEST="${OUTPUT_DIR}/successful_traces.txt"
if [[ ! -s "${MANIFEST}" ]]; then
    echo "No batch completed successfully. See ${OUTPUT_DIR}/hf_trace_run.log." >&2
    exit 1
fi
SUCCESSFUL_TRACES=()
while IFS= read -r trace_path; do
    [[ -n "${trace_path}" ]] && SUCCESSFUL_TRACES+=("${trace_path}")
done < "${MANIFEST}"

python benchmark/analyze_moe_denoising.py \
    "${SUCCESSFUL_TRACES[@]}" \
    --output-layer-csv "${OUTPUT_DIR}/moe_denoising_layers.csv" \
    --output-summary-csv "${OUTPUT_DIR}/moe_denoising_summary.csv" \
    --output-svg "${OUTPUT_DIR}/moe_denoising.svg"

echo "Experiment complete: ${OUTPUT_DIR}"
