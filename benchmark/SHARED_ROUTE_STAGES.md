# Shared-route experiments: stages 1 and 2

These are research diagnostics, not a production acceleration implementation.
All selections use the original Top-k IDs. Only MASK generation positions are
compressed; decoded/prefix/padding positions retain original computation.
`epsilon` bounds omitted normalized routing mass, NOT prediction error.

## Stage 1 (CPU, existing trajectories)

```bash
cd /Users/keduck/Documents/nju/Code/FOCUS
python3 benchmark/simulate_shared_routes.py \
  results/expert_trajectory/sharing_b1_to_b32_v1 \
  --output-dir artifacts/shared_routes_stage1_rerun \
  --batches 1 2 4 8 16 32 --layers 2 10 18 --steps 4 \
  --epsilons 0 0.05 0.10 0.20
```

Output: streaming `groups.jsonl` and group-equal means in `summary.csv`.
`joint`: fixed decoded support, greedy residual coverage, minimal weighted prefix.
`independent`: each token independently retains its highest weights to coverage.
`joint_no_fixed`: same greedy selection but no initial decoded support.
Fixed rows are unchanged in all methods. Zero epsilon is exact identity.

The input does NOT contain prefix routes. Expert unions and assignments therefore
cover generation positions only. The real full-forward working set may have a
larger fixed support and less union reduction. No quality or latency is inferred.
Each raw record is released after processing; no full trajectory is held in RAM.
Only common fully-active steps and single-block full-lifecycle files are accepted.

## Stage 2 (server, two GPUs, functional test)

Use the stored prompt token IDs from the existing experiment, not a newly sampled
dataset. Verify model/tokenizer version and generation settings match that run.
The script regenerates an unmodified Vanilla prefix of denoising steps, saves the
exact intervention input, and replays every branch from that same state. It does
not claim byte-identical reproduction of the historical trajectory across hosts.

```bash
cd /root/lkd/FOCUS
conda activate focus-moe
CUDA_VISIBLE_DEVICES=0,1 python benchmark/intervene_shared_routes.py \
  /root/lkd/Models/LLaDA2.0-mini \
  --prompt-snapshot results/expert_trajectory/sharing_b1_to_b32_v1/bs32/sampled_prompt_token_ids.json \
  --output-dir results/shared_routes_stage2/b32_l10_s4_smoke \
  --batch-size 32 --num-groups 1 --layer 10 --step 4 \
  --block-length 32 --denoising-steps 32 --confidence-threshold 0.95 \
  --epsilons 0.05 0.10 0.20 --max-memory-per-gpu 38GiB
```

After the smoke run succeeds, use `--num-groups 4` for all 128 prompts and a NEW
output directory. Repeat with `--layer 2` and a different directory. Optional
`--no-renormalize` tests omitted mass without rescaling; do not mix these runs.

Model loading follows the existing HF balanced two-GPU path (not tensor parallel).
No OpenCompass or training is required. The target gate must return the native
`(topk_ids, topk_weight, router_logits)` tuple. Shape/type mismatches fail closed.
All actual full-sequence fixed routes, including padding executed by this HF
implementation, participate in the fixed support; route metrics say full-forward.

The hook zeros discarded weights but retains all original expert execution.
This isolates functional impact using the native MoE accumulation implementation;
it is deliberately NOT a sparse dispatch implementation and NOT a TPS/TPF test.
`assignments_after` means nonzero/retained branches, not actually avoided work.

Every group first runs epsilon=0 as a no-op check. Any candidate/acceptance change
or >1e-5 log-probability discrepancy aborts. All intervention branches leave the
input unchanged and never feed their predictions into later steps.

Outputs:
- `state_group*.pt`: exact input/attention/position tensors and request IDs.
- `interventions.jsonl`: full-forward structural counts, MASK candidate flips,
  KL(reference || intervention), accepted additions/losses, and per-request metrics.
- Shared experts are never modified; only the target gate's routed weights change.

Metrics are single-forward perturbations, NOT task accuracy. Greedy temperature
0 is required by this entry point. Full task evaluation, multiple blocks,
FOCUS integration and a genuinely sparse execution path are future stages.

## Tests

```bash
python -m pytest tests/test_lmdeploy/test_shared_route_selection.py -q
```

CPU selector tests include randomized coverage constraints, fixed routes,
identity and weight-mass conservation. Tensor tests require PyTorch; actual remote
HF model/two-GPU compatibility still needs the server smoke run.
