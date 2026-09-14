# LLaDA2 token trajectory and temporal expert-similarity experiment

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
