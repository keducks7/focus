# Real batch HF FOCUS / Vanilla expert scaling

This replaces the LMDeploy experiment entry point. Load the original HF model
weights with Accelerate balanced layer placement, then execute a packed-token
PyTorch forward using those HF embedding, normalization, QKV, rotary, MoE and
output modules. No LMDeploy engine, NCCL executor or OpenCompass runtime is used.

The experiment is **physical batch execution**, not a union of independently
generated requests: QKV and MoE process all valid tokens together, and attention
uses one B-dimensional padded matrix computation. Padding never enters MoE.
Requests are isolated by the attention batch dimension and separate cache slots.

## Source alignment and limits

`hf_focus_batch.py` ports the algorithm in
`lmdeploy/pytorch/kernels/cuda/focus.py` and the fill-before-evict order in
`lmdeploy/pytorch/models/llada2.py`:

1. Compute masked-token attention importance in layers 0 and 1 with width-3
   max pooling, softmax, and head/query summation. FP32 accumulation is converted
   to query dtype, as in the kernel.
2. Target = min(mask count, ceil(max(average decoded, 1) * alpha)).
3. Use layer-1 minus layer-0 importance; mean + **population** standard deviation
   threshold, stable descending Top-k if too few threshold candidates survive.
4. Restore the immediate masked predecessor when its original position is
   adjacent to a selected position. Restore unprocessed positions before the
   rightmost retained position, based on per-request progress.
5. Write all incoming K/V BEFORE compacting Query and residual at layer 1.
6. Retain each deeper layer's own KV cache; positions never written there remain
   masked out. Limit visible KV to each request's CURRENT rightmost processing
   position, as in the official ragged metadata (not historical progress).
   Reuse accepted/cached positions according to the adjacent-token
   delayed-cache update also used by the repository HF reference.

This is a **semantic port**, not a claim of bitwise equivalence or the official
optimized implementation. FP32 reductions and BF16 matrix shapes can change
rounding, and near a threshold/tie even selected positions can change. The kernel
source hash and versions are stored in experiment.json. Kernel selection tests
and the --verify checkpoint checks must pass on the server before trusting a
scaling conclusion. Whole-engine scheduling and end-to-end LMDeploy parity are
not asserted by these checks.

## Fixed experimental protocol

- Generation block=32, generation slots=32, maximum 32 denoising steps.
- EOS does not terminate early. Remaining masks and generated IDs are recorded;
  incomplete generation is not silently treated as success at task solving.
- Modes: Vanilla; delayed-cache only; delayed-cache + FOCUS. The middle control
  separates delayed cache from additional Query eviction.
- Same sampled prompt IDs, same fixed left-padding position frame across all B,
  same confidence=.95 and greedy candidate selection. Prompts are prefetched
  into prefix caches; decode observations exclude prompt prefill.
- Padding positions are never projected or routed. Non-mask generation positions
  that still need processing DO count as actual Query work.
- MoE input has shape [1, sum(valid query lengths), H]. This leading 1 is the
  packed tensor convention, not physical request batch=1. q_seqlens records B
  per-request lengths. Attention remains [B, heads, padded_Q, KV_length].
- One group of B requests runs to completion before the next group. Completed
  requests have zero Query length; full-batch summaries exclude those tail steps
  and report coverage. Modes may complete at different steps, so pooled averages
  are descriptive, not a matched-step causal effect.
- Short prompts, one generation block and balanced two-GPU layer placement target
  two A800 40G devices. No tensor parallelism is used. CPU/disk offload is rejected.
- Tracing synchronizes CUDA; do not use its times to report throughput/speedup.

## Server commands

```bash
cd /root/lkd/FOCUS
git pull origin main
conda activate focus-moe

# Unit tests: CPU semantics plus an official Triton comparison if CUDA is available.
CUDA_VISIBLE_DEVICES=0 python tests/test_lmdeploy/test_hf_focus_batch.py

# Real-model smoke + verification. A NEW output directory is required.
CUDA_VISIBLE_DEVICES=0,1 python -u benchmark/profile_llada2_hf_focus_scaling.py \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/hf_focus_moe_scaling/smoke_v1 \
  --batch-sizes 1 2 --num-prompts 4 --max-input-len 128 --verify

# Main observation after verification succeeds.
CUDA_VISIBLE_DEVICES=0,1 python -u benchmark/profile_llada2_hf_focus_scaling.py \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/hf_focus_moe_scaling/gsm8k_32_v1 \
  --batch-sizes 1 2 4 8 --num-prompts 32 --max-input-len 128
```

--verify runs independent single-request references only for the first group at
the largest requested B. It checks selected coordinates, FOCUS progress, exact
route histograms and cache validity, plus tolerance-based K/V/logit comparisons.
Vanilla additionally compares with the checkpoint's unmodified HF full forward.
These extra forwards are excluded from experiment traces. Once a group loses full
occupancy, single-request comparisons stop; original-HF Vanilla checks continue.
Failures are errors, not swallowed warnings. Do not increase tolerances to hide
a routing/cache mismatch. Verification adds time and memory (extra small caches).

## Outputs

- experiment.json and prompt_token_ids.json: complete settings/provenance/prompts.
- verification.json: completed numerical checks; [] when verification is disabled.
- mode_bsB/routes.jsonl: per-layer real Query counts, expert histogram, prefill vs
  decode, group and forward ID, plus per-request generated IDs/remaining masks.
- scaling.csv: group-averaged E(B), Q(B), expert coverage, doubling ratio E(2B)/E(B),
  and Query ratios to Vanilla and delayed-only controls.
- group_layers.csv and coverage.csv: group statistics and full-batch coverage.

First confirm Query reduction versus delayed-only, then inspect E versus B and
E/total_experts. Saturation near the expert-count ceiling alone is weak evidence.
The unit tests isolate semantic correctness, not model quality or method benefit.

## Why sequential references can check batch execution

Request-wise attention and KV updates have no cross-request dependency; per-token
normalization, fixed-weight MLP and this model's uncapped Top-k MoE routing also
have none. Therefore, under identical token inputs, positions, state and sampling,
concatenating request-local outputs gives the same mathematical result as packed
execution. Batched floating-point reductions need not be bitwise identical.

Sequential requests are used ONLY as this verification reference. The experiment
itself performs packed batch projections, batched attention and packed batch MoE.
Expert counts combine assignments by expert identity, not by adding each request's
number of experts. No equivalence claim is made for wall-clock time, memory use,
load balancing, stochastic sampling streams, or models with batch-level capacity
limits. Success here also does not imply an optimized FOCUS speedup in this HF port.
