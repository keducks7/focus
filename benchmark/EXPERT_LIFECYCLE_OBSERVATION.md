# Observation-only expert roles, functional geometry, and batch opportunity

This is NOT the intervention experiment. Native HF Vanilla decoding is unchanged:
no route masking, branch deletion, replacement, cache replay or acceptance update
other than the original confidence/minimum-transfer rule. No training/OpenCompass.

## Three questions

1. Which expert IDs persist, leave and enter when a position becomes decoded?
   Role summaries cover ALL recorded generation positions in live requests, with
   expert counts and retained routing mass. Prefix/padding and finished requests
   are excluded. Existing complete all-state JSONL can answer this on CPU.
2. Do the routed-expert FUNCTIONS have different nearest neighbors across input
   lifecycles? Capture the actual target MLP INPUT, not final hidden states.
   After ALL normal batches finish, evaluate all routed experts off-path on a
   small cohort's saved inputs. Never feed side-probe outputs back into decoding.
3. Is a good functional neighbor already used in the same request, other requests,
   or the whole live batch? Compare these pools with all legally eligible experts.
   Nested subsets include the anchor request and use the same request ordering
   across that case's steps. They are OFFLINE pool sizes, not real B1/B2/... runs.

## Protocol and limits

- Real B32 input; target L10 (zero-based); one 32-token block; greedy; .95 confidence.
- Two A800 40G, native HF balanced layer placement (not TP), eager attention.
- Default 8 sampled token positions/group, request-diverse, chosen after generation
  without conditioning on expert persistence or nearest-neighbor quality. Earliest
  and final accepts can be selected: unavailable pre/post observations are censored.
- Window -2,-1,0,+1,+2,+3 relative to acceptance. Offset 0 still has MASK input.
- A first-forward hook/no-hook final-hidden equality check is performed per group.
  It is a check on that state, not a proof of byte-identical historical trajectories.
- ALL generation finishes before ALL side expert evaluations. Baseline data and
  inputs are streamed to disk; no whole-trajectory vocabulary probability cache.
- Expert outputs exclude the architectural shared expert and router weights.
  Relative L2 = ||E_e(h)-E_f(h)|| / ||E_e(h)||; source=0 is undefined, not distance 0.
  Self experts are excluded. Cosine is also recorded as a diagnostic.
  Nearest/runner-up distance gaps are logged: identity changes near a tie are not
  evidence of a large functional transition. Probes use native BF16 arithmetic;
  distance calculations use FP32, which does not turn the expert forward into FP32.
- Source rows cover the union of that token's original Top-8 experts over the
  observed window. A row states whether its source is ACTUALLY selected now.
  An absent expert's geometry is valid off-path data, not actual contribution;
  no invented source weight is used. Primary functional summaries use selected
  sources only. Fixed-pair geometry deliberately also reports absent sources.
- Candidate pools intersect reconstructed group eligibility from original logits
  and checkpoint bias. If this reconstruction excludes an original selected expert,
  fail closed (possible numerical ties); do not silently force it eligible.
- Same-cohort matrices use offsets -1,0,+1,+2 and ONLY cases observed at all four.
  Their expert IDs/parameters are within one layer. Do not merge IDs across layers.
- Whole-batch MASK count/context change with lifecycle; similarity is not causal
  proof of a token identity effect. Function closeness is not substitutability,
  task quality, or speedup. Pool expansion mechanically improves nearest distances.

## Server commands

Use the existing environment that ran trajectories (PyTorch, Transformers,
Accelerate, Safetensors). NumPy/Matplotlib are needed only for optional plots.
36GiB is a weight-placement budget, NOT a hard peak memory cap. No CPU/disk offload.
8 tokens * 6 offsets * 256 experts produces an output tensor of roughly
`256 * observed_events * hidden_size * 4` bytes per group. At H=2048 and 48 events
this is ~96MiB, plus temporary buffers. Native input snapshots use BF16 CPU tensors
and can occupy hundreds of MB/group; provide several GB of free disk. Start with 1
group; real GPU/host memory feasibility must be checked, not assumed guaranteed.

```bash
cd /root/lkd/FOCUS
conda activate focus-moe
python tests/test_lmdeploy/test_expert_lifecycle_observation.py

CUDA_VISIBLE_DEVICES=0,1 python -u benchmark/observe_expert_lifecycle.py \
  /root/lkd/Models/LLaDA2.0-mini \
  --prompt-snapshot results/expert_trajectory/sharing_b1_to_b32_v1/bs32/sampled_prompt_token_ids.json \
  --output-dir results/expert_lifecycle_observation/b32_l10_smoke_v1 \
  --batch-size 32 --num-groups 1 --layer 10 \
  --block-length 32 --denoising-steps 32 --confidence-threshold 0.95 \
  --tokens-per-group 8 --pre-steps 2 --post-steps 3 \
  --subset-sizes 1 2 4 8 16 32 \
  --logit-chunk-size 8 --expert-chunk-size 16 --max-memory-per-gpu 36GiB

python benchmark/analyze_expert_lifecycle_observation.py \
  results/expert_lifecycle_observation/b32_l10_smoke_v1 \
  --layer 10 --output-dir artifacts/expert_lifecycle_observation/b32_l10_smoke_v1 \
  --matrices --plot
```

Use a NEW/EMPTY capture directory. For all 128 prompts set `--num-groups 4` and a
new `b32_l10_full_v1` directory. B8 smoke requires `--batch-size 8`, a new directory,
and `--subset-sizes 1 2 4 8`; do not label it B32. The same snapshot token IDs are
reused; no dataset re-download. `--capture-only` omits all side probes, but saves
normal target-layer inputs and roles. No probe-only/resume entry point is currently
provided; do not overwrite an interrupted directory. `generation_complete.json`
means the normal decode finished; `complete.json` additionally covers side probes.

Existing-data role analysis (no GPU or Torch required):

```bash
python3 benchmark/analyze_expert_lifecycle_observation.py \
  results/expert_trajectory/sharing_b1_to_b32_v1/bs32/token_trajectories_bs32.jsonl \
  --layer 10 --output-dir artifacts/expert_lifecycle_observation/roles_existing_b32_l10
```

## Files to retain

- `metadata.json`, `routes.jsonl`, `group*.json`, generation/overall completion markers.
- `inputs_group*_step*.pt`: true target-layer inputs/logits, all generation routes.
- `functional_missing_group*.json`: random cohort and explicit censored observations.
- `functional_outputs_group*.pt`: expert outputs indexed by event/expert, inputs,
  actual selections, reconstructed eligibility and live request supports.
- `neighbors_group*.jsonl`: current-source and fixed-source function neighbors,
  source norm/weight, pool size, and nested-subset nearest errors.
- `matrices_group*.pt`: all-expert cosine/relative-L2 on the SAME complete cohort.
- Analyzer: role events/group summaries; selected-source function/pool summaries;
  nearest-identity changes; optional matrix agreement and fixed-pair geometry;
  optional candidate-pool regret plot and same-cohort expert similarity heatmaps
  (not quality or speed graphs). Pool curves use `paired_k*_regret`: identical
  source/event eligibility across ALL requested sizes. Raw `k*_regret` columns
  have size-specific coverage and must not be treated as a paired scaling curve.
  Late events with fewer than 32 live requests cannot provide a k=32 observation.

Primary function means: branches -> event -> token case/phase -> group -> group-equal
summary. Correlated tokens/experts/subsets are not independent replicates. A single
group and eight cases are a pilot. Follow up with more seeds/tasks, random pool-size
controls and independent calibration before designing a deployment rule.
