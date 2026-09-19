# Layer/state-dependent expert sharing in real batched MoE-dLLM decoding

Research question: where does cross-request expert sharing arise, and how does it change
with physical batch, layer, decoding step, and MASK/decoded composition? This experiment
does not implement pruning, caching, expert replacement, or scheduling acceleration.

## Two different batch quantities

- `physical_batch`: real simultaneous requests in an HF forward. New B16/B32 runs retain
  the full-state Vanilla profiler, 32 generation positions, all MoE layers and every executed step.
- `subset_size`: requests selected OFFLINE from one captured batch at one fixed layer/step.
  It never reruns generation. A subset of a B32 trace is not a measured B1/B2/B4 execution.

For B requests and an expert visited by m requests, its probability of appearing in a uniformly
chosen k-request subset is `1 - C(B-m,k)/C(B,k)`. Summing over experts gives the **exact mean**
union over all request subsets, avoiding enumeration of C(32,16)=601,080,390 combinations.
No sampled approximation or heuristic threshold is used. This does not produce subset variance.

## Observation and null control

Analyze MASK, decoded and all positions separately, including empty state sets for a request
(do not silently change the subset's request population). Output the expected token count so
state-size differences remain visible. For the combined state, also compute exact mean shared,
MASK-exclusive and decoded-exclusive expert counts using inclusion-exclusion.

Concave/sublinear set-union growth alone is not evidence for a new model mechanism: overlap
and the finite expert pool can produce it automatically. The independent expert-label null
preserves each request's number of distinct experts, then independently randomly permutes
expert IDs per request. Its expected subset union is computed exactly with elementary symmetric
polynomials. `alignment_gap = null mean union - observed mean union`; a positive gap means
more common expert preference than this size-matched uniform-label baseline. It may still arise
from global learned router preferences, not specifically diffusion or semantic redundancy.

`incidence_to_union_ratio` is the ratio of expected request-expert incidences to expected union,
**not** the average of per-subset ratios. At k=B it is exactly the average number of requests
visiting each active expert. It is not a speedup or removable-computation fraction.

## Server: real B16/B32 smoke tests

Run from `/root/lkd/FOCUS` in `focus-moe`, after syncing the code. Each run needs a new directory.

```bash
CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZES="16" NUM_PROMPTS=16 \
MAX_INPUT_LEN=128 BLOCK_LENGTH=32 GEN_LENGTH=32 DENOISING_STEPS=32 \
bash benchmark/run_moe_batch_sharing.sh \
  openai/gsm8k /root/lkd/Models/LLaDA2.0-mini \
  results/expert_trajectory/sharing_b16_smoke_v1

CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZES="32" NUM_PROMPTS=32 \
MAX_INPUT_LEN=128 BLOCK_LENGTH=32 GEN_LENGTH=32 DENOISING_STEPS=32 \
bash benchmark/run_moe_batch_sharing.sh \
  openai/gsm8k /root/lkd/Models/LLaDA2.0-mini \
  results/expert_trajectory/sharing_b32_smoke_v1
```

During full collection Q should remain 512 for B16 and 1024 for B32. Q here counts observed
generation positions, not remaining MASKs. All MoE layers are recorded despite the separate
legacy similarity-layer option; the costly all-expert similarity replay is disabled.

## Controlled comparison

```bash
CUDA_VISIBLE_DEVICES=0,1 BATCH_SIZES="8 16 32" NUM_PROMPTS=128 \
MAX_INPUT_LEN=128 BLOCK_LENGTH=32 GEN_LENGTH=32 DENOISING_STEPS=32 \
bash benchmark/run_moe_batch_sharing.sh \
  openai/gsm8k /root/lkd/Models/LLaDA2.0-mini \
  results/expert_trajectory/sharing_b8_b16_b32_v1
```

Runs serially, not concurrently. Each batch uses the same requested 128 prompts and seed 0.
The runner checks exact tokenized-prompt equality after collection. B8/B16/B32 yield 16/8/4 groups
if all prompts were obtained; inspect the saved snapshots and metadata. No tensor parallelism,
microbatch substitution, OOM fallback or sequential-request aggregation is used. BF16 numerical
differences can still alter acceptance trajectories across batch sizes: matching prompts does
not guarantee matching decoding states.

The existing HF profiler uses balanced layer placement on two visible GPUs, with 38GiB maximum
weight-placement budget per GPU by default. This is not a reservation for runtime activations.
B32 has not been GPU-tested locally, so fitting two A800 40G cards is not guaranteed. If OOM:
retain the log, stop, inspect memory; optionally retry shorter input length on **all compared
batches** in a new experiment. Do not silently replace B32 with microbatches. Trace JSONL and
event CSV files may be large; retain enough disk and host memory. Existing lifecycle analysis
buffers one batch group's trace; the new sharing analysis streams one step at a time.

## CPU-only analysis (including existing B8 data)

```bash
python benchmark/analyze_batch_expert_sharing.py \
  results/expert_trajectory/gsm8k_bs8_lifecycle_smoke_v2_retry/token_trajectories_bs8.jsonl \
  --output-dir artifacts/batch_sharing_b8 \
  --layers 2 10 18 --max-step 8
```

Multiple trace paths can be passed together. By default subset sizes are 1/2/4/8/16/32 up to
the physical batch size; the full-batch endpoint is always included. Default layers are 2/10/18,
steps 0–8. Change `--layers` to include all desired MoE layers; raw collection already has them.
These arguments specify observation slices, not a proposed method's tuning parameters.

## Outputs

| File | Meaning |
| --- | --- |
| `subset_group_curves.csv` | Exact offline subset curves for each source/group/step/layer/state |
| `subset_summary.csv` | Equal-group means at the same step, without pooling different physical batches or sources |
| `physical_batch_observations.csv` | k=B endpoints: actual recorded batch expert coverage |
| `expert_request_occupancy.csv` | For each expert, number of visiting requests and total token assignments, separated by state |
| `manifest.json` | Source hashes, included steps, excluded completed-request steps and metric definitions |

No group-step with an already-finished request enters the primary sharing analysis. This keeps
the physical batch meaningful but can introduce survivor selection at late steps. Compare the
same absolute steps and inspect group counts/inclusion lists; use a common window where all
groups of all configurations are active for the strongest controlled comparison. State composition
still changes; never present time trends alone as causal effects of token identity.

Neither many subset combinations nor many token events are independent experimental replicates.
Report uncertainty over actual prompt groups/requests in subsequent statistical analysis.
The data can locate shared access, not prove functional substitutability or safe pruning.
