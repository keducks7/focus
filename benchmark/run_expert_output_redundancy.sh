#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <trajectory_dir> <model_path> <new_output_dir>"
    exit 1
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
read -r -a STEPS_ARRAY <<< "${STEPS:-0 4 8 12}"
read -r -a REQUEST_SIZES_ARRAY <<< "${REQUEST_SIZES:-1 2 4 8}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -u "${SCRIPT_DIR}/analyze_expert_output_redundancy.py" \
    "$1" "$2" --output-dir "$3" \
    --batch-size "${BATCH_SIZE:-8}" --layer "${LAYER:-10}" \
    --steps "${STEPS_ARRAY[@]}" --request-sizes "${REQUEST_SIZES_ARRAY[@]}" \
    --matched-tokens "${MATCHED_TOKENS:-16}" --rank-cap "${RANK_CAP:-256}" \
    --neighbors "${NEIGHBORS:-4}" --repeats "${REPEATS:-3}" --seed "${SEED:-0}"
