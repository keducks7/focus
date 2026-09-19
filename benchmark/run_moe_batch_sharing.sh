#!/usr/bin/env bash
# Real physical batches; no microbatching and no synthetic aggregation of requests.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dataset> <model_path> [output_root]"
    exit 1
fi
DATASET=$1
MODEL=$2
OUTPUT_ROOT=${3:-results/expert_trajectory/batch_sharing_v1}
BATCH_SIZES_TEXT=${BATCH_SIZES:-"8 16 32"}
NUM_PROMPTS=${NUM_PROMPTS:-128}
read -r -a BATCHES <<< "$BATCH_SIZES_TEXT"
if [[ ! "$NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'NUM_PROMPTS must be a positive integer' >&2; exit 1
fi
for batch in "${BATCHES[@]}"; do
    if [[ ! "$batch" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid batch size: $batch" >&2; exit 1
    fi
    if (( NUM_PROMPTS % batch != 0 )); then
        echo "NUM_PROMPTS must be divisible by each physical batch size ($batch)." >&2; exit 1
    fi
done
TRACES=()
for batch in "${BATCHES[@]}"; do
    run_dir="$OUTPUT_ROOT/bs$batch"
    echo "Collecting REAL physical batch $batch, $NUM_PROMPTS total prompts"
    FULL_LIFECYCLE=1 SKIP_SIMILARITY=1 BATCH_SIZE="$batch" NUM_PROMPTS="$NUM_PROMPTS" \
        bash benchmark/run_llada2_expert_trajectory.sh "$DATASET" "$MODEL" "$run_dir"
    TRACES+=("$run_dir/token_trajectories_bs${batch}.jsonl")
    python benchmark/analyze_batch_expert_sharing.py "$run_dir/token_trajectories_bs${batch}.jsonl" \
        --output-dir "$run_dir/batch_sharing_analysis"
done
# Ensure batch-size comparison uses the exact same tokenized requests.
python - "$OUTPUT_ROOT" "$NUM_PROMPTS" "${BATCHES[@]}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected_prompts = int(sys.argv[2])
reference = None
for batch in sys.argv[3:]:
    snapshot = json.loads((root / f'bs{batch}' / 'sampled_prompt_token_ids.json').read_text())
    prompts = snapshot['prompt_token_ids']
    if len(prompts) != expected_prompts:
        raise SystemExit(f'B{batch} only collected {len(prompts)} prompts; expected {expected_prompts}.')
    if reference is None:
        reference = prompts
    elif prompts != reference:
        raise SystemExit(f'Prompt-token mismatch at B{batch}; do not interpret this as a controlled batch comparison.')
print(f'Exact prompt-token match verified: {len(reference)} requests across batches.')
PY
python benchmark/analyze_batch_expert_sharing.py "${TRACES[@]}" \
    --output-dir "$OUTPUT_ROOT/comparison"
echo "Complete: $OUTPUT_ROOT (physical batch and offline subset size are separate columns)"
