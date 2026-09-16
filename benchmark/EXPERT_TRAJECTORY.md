# LLaDA2 token trajectory and temporal expert-similarity experiment

## Full-state lifecycle experiment (v2, recommended for new collection)

This is an **observational research experiment**, not an acceleration method. Enable
`FULL_LIFECYCLE=1 SKIP_SIMILARITY=1` to collect every generation position at every MoE layer,
including already-decoded tokens. The old MASK-only mode remains the default for backward
compatibility. No expert pruning, routing replacement, output reuse or extra terminal forward
is introduced. It uses real request-batch parallelism and HF balanced **layer placement** across
two GPUs, not LMDeploy, tensor parallelism, or sequential-request aggregation.

Run from `/root/lkd/FOCUS` in the existing `focus-moe` environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
FULL_LIFECYCLE=1 SKIP_SIMILARITY=1 \
BATCH_SIZE=8 NUM_PROMPTS=8 MAX_INPUT_LEN=128 \
BLOCK_LENGTH=32 GEN_LENGTH=32 DENOISING_STEPS=32 \
bash benchmark/run_llada2_expert_trajectory.sh \
  openai/gsm8k /root/lkd/Models/LLaDA2.0-mini \
  results/expert_trajectory/gsm8k_bs8_lifecycle_smoke_v2
```

For the initial research run, change `NUM_PROMPTS=32` and use a new output directory, e.g.
`results/expert_trajectory/gsm8k_bs8_lifecycle_v2`. Use identical prompts, lengths and seeds
when later comparing B1/B4/B8. The prompt-token snapshot is saved for checking this.
The supported generation region is still **one block**, with `GEN_LENGTH == BLOCK_LENGTH`;
prompt positions are not traced. Longer multi-block generation is not implemented here.

### What is recorded

- All generation positions, every natural denoising forward, every MoE layer: selected expert
  IDs, selected router logits and reconstructed model routing weights.
- Pre-forward token ID and state (`mask`/`decoded`), acceptance event and acceptance time,
  whether the request was already finished before this forward, and fresh-execution status.
- Each layer's actual MLP output: online per-token cosine similarity and relative L2 change
  against its own previous-step output, with the previous output norm as denominator.
  These measure the **whole MLP output, including shared experts if present**, not each
  individual routed expert. Zero-norm undefined normalized metrics are null.
- Only the previous output tensor per layer is retained on CPU; full output tensors for all
  steps are not stored. Group boundaries reset the observer. Hooks do not modify outputs.
- Raw `query_tokens` counts all observed generation positions in v2;
  `unresolved_tokens_before` counts MASK positions. The legacy `layer_histograms` field stays
  MASK-only; use the new state-load analysis for decoded/all-state questions.

At accepting step t the input is still MASK. The t→t+1 comparison includes the replacement
of MASK by the accepted token. Later decoded comparisons are separate. A request that finished
early may still be physically forwarded with its batch; these observations are flagged, and
must not be mixed with active-request evidence. No extra steps are run to fabricate a
post-acceptance window. Coverage records flag tokens with no post-acceptance observation while
their request is still active; `accepted_by_end` distinguishes unaccepted tokens.

### Analysis outputs

The runner automatically writes `lifecycle_analysis/`:

| File | Purpose |
| --- | --- |
| `lifecycle_events.csv` | Same-token, same-layer adjacent-step route Jaccard, normalized routing-weight TV and MLP-output change; includes request and group IDs |
| `lifecycle_by_phase.csv` | Each layer: unresolved MASK, accepting MASK, first decoded forward, later decoded forwards |
| `lifecycle_by_relative_acceptance_step.csv` | Layer × time relative to eventual acceptance (negative=before, zero=accepting, positive=after) |
| `lifecycle_state_loads.csv` | Per group/step/layer/state active experts, effective experts and top-10 assignment share |
| `lifecycle_cross_layer.csv` | Adjacent-layer correlation of token-pair expert-sharing patterns, not expert-ID overlap across layers |
| `lifecycle_coverage.csv` | Acceptance and available decoded observations, including right-censored tokens |

Phase/time summaries are token-event-weighted, separately stratified by
`request_finished_before`. Use the False rows for natural active-request conclusions. Sample
sizes change with relative time; inspect counts before interpreting a trend. Individual events
within a request are not independent replicates. Use request/group IDs for later uncertainty
estimation. Step zero has no predecessor, hence no adjacent-step event; it still appears in
raw routes and state-load statistics.

Cross-layer analysis uses up to 512 uniformly sampled, fixed identity pairs per group (seed 0),
the same pairs for every layer and step. It compares within-layer `|TopK_i ∩ TopK_j| / K`
across adjacent layers and separates same/cross-request and mask/decoded pair types. Completed
requests are excluded. Empty strata produce no rows; constant or one-pair correlations are
null, not zero. This sampling cap is an offline analysis budget, not a method parameter.

Re-run CPU-only analysis without loading a model:

```bash
python benchmark/analyze_moe_lifecycle.py \
  results/expert_trajectory/gsm8k_bs8_lifecycle_v2/token_trajectories_bs8.jsonl \
  --output-dir results/expert_trajectory/gsm8k_bs8_lifecycle_v2/lifecycle_analysis
```

Old MASK-only traces cannot recover missing decoded routes and require recollection. The old
acceptance analyzer can read v2 traces but explicitly filters decoded records, preserving its
original interpretation. No timing/throughput claim should be based on this heavily instrumented
profiler. Local CPU tests do not replace a server GPU smoke run.

## Original MASK-only experiment

This experiment asks two questions before designing a method:

1. Does the expert route of one unresolved token become stable as that token approaches acceptance?
2. On the same surviving token cohort, is functional expert similarity stable across denoising steps?

It is an observational Vanilla experiment. It does not prune, merge, re-route, or reorder experts. SERE is a later
LLM baseline: this experiment first checks whether its static expert-equivalence assumption survives the evolving
hidden-state distribution of diffusion decoding.

## Scope of the first implementation

The first implementation runs one generated block, with `GEN_LENGTH == BLOCK_LENGTH`. The trace format nevertheless
records `block_id`, block-local position, global generation position, generation length, and number of blocks. A later
multi-block extension can therefore preserve token identity and analysis code.

Defaults target LLaDA2.0-mini on two 40 GB GPUs using HuggingFace Accelerate's balanced layer placement:

- GSM8K, 32 prompts, request batch 8;
- one 32-token block and at most 32 denoising steps;
- true greedy decoding, confidence threshold 0.95;
- token routes from every MoE layer at every executed step;
- functional similarity for layer 10 at steps 0, 1, 2, 3, 4, 8, and 12;
- at most 64 token identities that remain unresolved at every selected similarity step.

Layer-10 MoE inputs are copied to CPU during generation. After generation, all 256 routed experts process the same
matched token cohort one at a time. This uses the real expert weights already loaded with the model, not synthetic
weights. It does not run all experts inside the generation trajectory itself.

## Server commands

After pulling the GitHub commit:

```bash
cd /root/lkd/FOCUS
source /root/miniconda3/etc/profile.d/conda.sh
conda activate focus-moe
python -c "import torch, transformers, accelerate, datasets; print(torch.__version__, torch.cuda.device_count())"
```

Use a new output directory for every run. First run a small smoke experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BATCH_SIZE=8 NUM_PROMPTS=8 MAX_INPUT_LEN=64 \
BLOCK_LENGTH=16 GEN_LENGTH=16 DENOISING_STEPS=16 \
SIMILARITY_LAYER=10 SIMILARITY_STEPS="0 1 2 3 4 8" SIMILARITY_SAMPLES=16 \
bash benchmark/run_llada2_expert_trajectory.sh \
  openai/gsm8k \
  /root/lkd/Models/LLaDA2.0-mini \
  /root/lkd/FOCUS/results/expert_trajectory/gsm8k_smoke_v1
```

Then run the main pilot:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
BATCH_SIZE=8 NUM_PROMPTS=32 MAX_INPUT_LEN=128 \
BLOCK_LENGTH=32 GEN_LENGTH=32 DENOISING_STEPS=32 \
SIMILARITY_LAYER=10 SIMILARITY_STEPS="0 1 2 3 4 8 12" SIMILARITY_SAMPLES=64 \
bash benchmark/run_llada2_expert_trajectory.sh \
  openai/gsm8k \
  /root/lkd/Models/LLaDA2.0-mini \
  /root/lkd/FOCUS/results/expert_trajectory/gsm8k_bs8_layer10_v1
```

The first all-expert pass can take time because it invokes every expert at every selected similarity step. Progress
through generation is printed per group and step. Monitor both GPUs with `nvidia-smi`; the target layer and its expert
bank reside together on one of the two GPUs, while the full model remains layer-sharded.

## Outputs

- `token_trajectories_bs8.jsonl`: token identity, confidence, acceptance, every layer's Top-8 IDs, raw selected router
  logits, actual normalized router weights, and per-step expert histograms;
- `sampled_prompt_token_ids.json`: exact sampled prompt token IDs and dataset/split/seed metadata for replay;
- `layer10_hidden_step*_bs8.pt`: raw target-layer MoE inputs and token identities;
- `expert_similarity_layer10_step*_bs8.pt`: 256×256 functional-similarity matrices and nearest-expert maps;
- `expert_similarity_pairs_bs8.csv`: matrix Pearson correlation and nearest-substitute consistency for every selected
  step pair;
- `expert_similarity_steps_bs8.csv`: error from applying the first selected step's static nearest-expert mapping;
- `matched_tokens_bs8.json`: the exact cohort shared by all selected steps;
- `token_route_events.csv`: same-token adjacent-step route observations;
- `token_route_layer_summary.csv`: accepted/unresolved route stability and acceptance AUC per layer;
- `route_stability_by_acceptance_distance.csv`: route stability as tokens approach their acceptance step;
- `trajectory_run.log`: loading and progress log.

## Interpretation boundaries

The similarity analysis uses one common token cohort across all selected steps. It therefore measures representation
evolution rather than a changing mix of easy and hard tokens. It intentionally selects tokens that survive through the
latest requested step, so it describes hard unresolved tokens, not all generation tokens.

Step 0 is the static reference in this pilot; it is not SERE's generic calibration matrix. If similarity is stable, the
next experiment should compare one-step and repeatedly applied SERE-style re-routing. If adjacent steps are stable but
long-range pairs are not, temporal expert equivalence is supported. If route stability predicts acceptance, expert
trajectory becomes a candidate token-maturity signal.

Do not infer whole-model behavior from layer 10 alone. Only after the pilot shows a clear signal should the experiment
be repeated at one early and one late MoE layer, then on HumanEval, batch 16, FOCUS, and multiple generated blocks.

The multi-block extension should be a second implementation stage, not a change to this pilot's scientific variable.
It will loop over generated blocks, set `block_id = 0, 1, ...`, and define
`generation_position = block_id * block_length + block_position`. Similarity must first be compared within each block,
then across equal relative denoising progress between blocks; pooling all blocks without this distinction would mix
block position with denoising-step effects.
