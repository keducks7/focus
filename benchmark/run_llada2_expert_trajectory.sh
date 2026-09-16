#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dataset_id_or_path> <model_path> [output_dir]"
    exit 1
fi

DATASET=$1
MODEL=$2
OUTPUT_DIR=${3:-./results/llada2_expert_trajectory}
GPU_IDS=${CUDA_VISIBLE_DEVICES:-0,1}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_PROMPTS=${NUM_PROMPTS:-32}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-128}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
GEN_LENGTH=${GEN_LENGTH:-32}
DENOISING_STEPS=${DENOISING_STEPS:-32}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.95}
TEMPERATURE=${TEMPERATURE:-0}
SIMILARITY_LAYER=${SIMILARITY_LAYER:-10}
SIMILARITY_STEPS_TEXT=${SIMILARITY_STEPS:-"0 1 2 3 4 8 12"}
SIMILARITY_SAMPLES=${SIMILARITY_SAMPLES:-64}
MAX_MEMORY_PER_GPU=${MAX_MEMORY_PER_GPU:-38GiB}
FULL_LIFECYCLE=${FULL_LIFECYCLE:-0}
SKIP_SIMILARITY=${SKIP_SIMILARITY:-0}
EXTRA_ARGS=()
if [[ "${FULL_LIFECYCLE}" == "1" ]]; then EXTRA_ARGS+=(--full-lifecycle); fi
if [[ "${SKIP_SIMILARITY}" == "1" ]]; then EXTRA_ARGS+=(--skip-similarity); fi

read -r -a SIMILARITY_STEP_ARRAY <<< "${SIMILARITY_STEPS_TEXT}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)

echo "LLaDA2 token trajectory and expert-similarity experiment"
echo "  GPUs:              ${GPU_IDS}"
echo "  batch:             ${BATCH_SIZE}"
echo "  prompts:           ${NUM_PROMPTS}"
echo "  block/gen length:  ${BLOCK_LENGTH}/${GEN_LENGTH}"
echo "  denoising steps:   ${DENOISING_STEPS}"
echo "  similarity layer:  ${SIMILARITY_LAYER}"
echo "  similarity steps:  ${SIMILARITY_STEPS_TEXT}"
echo "  matched samples:   ${SIMILARITY_SAMPLES}"
echo "  output:            ${OUTPUT_DIR}"
echo "  full lifecycle:    ${FULL_LIFECYCLE}"
echo "  skip similarity:   ${SKIP_SIMILARITY}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python benchmark/profile_llada2_expert_trajectory.py \
    "${DATASET}" "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-input-len "${MAX_INPUT_LEN}" \
    --block-length "${BLOCK_LENGTH}" \
    --gen-length "${GEN_LENGTH}" \
    --denoising-steps "${DENOISING_STEPS}" \
    --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
    --temperature "${TEMPERATURE}" \
    --similarity-layer "${SIMILARITY_LAYER}" \
    --similarity-steps "${SIMILARITY_STEP_ARRAY[@]}" \
    --similarity-samples "${SIMILARITY_SAMPLES}" \
    --max-memory-per-gpu "${MAX_MEMORY_PER_GPU}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/trajectory_run.log"

if [[ "${FULL_LIFECYCLE}" == "1" ]]; then
    python benchmark/analyze_moe_lifecycle.py \
        "${OUTPUT_DIR}/token_trajectories_bs${BATCH_SIZE}.jsonl" \
        --output-dir "${OUTPUT_DIR}/lifecycle_analysis"
else
    python benchmark/analyze_llada2_expert_trajectory.py \
        "${OUTPUT_DIR}/token_trajectories_bs${BATCH_SIZE}.jsonl" \
        --output-dir "${OUTPUT_DIR}"
fi

echo "Experiment complete: ${OUTPUT_DIR}"
