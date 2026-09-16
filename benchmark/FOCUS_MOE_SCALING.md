# Actual FOCUS versus Vanilla MoE scaling

Question: after actual FOCUS Query eviction, does the number of active routed
experts still grow sublinearly with request batch size?

This runner uses the repository LMDeploy LLaDA2 implementation for **all** modes.
The previous HF profiler counted only unresolved masks; these traces count **all
tokens actually entering each MoE**. Do not directly combine those two metrics.
The OpenCompass HF implementation uses single-request indexing in eviction;
this experiment uses the ragged batch FOCUS implementation instead.

## Fixed protocol

- GSM8K; one block of size 32; maximum generated length 32 (EOS may stop earlier).
- Batch sizes 1, 2, 4, 8; 32 identical saved prompts across modes and sizes.
- Confidence .95, greedy selection (temperature 0, top-k 1), at most configured 32 steps.
- Three modes: vanilla, delayed-cache only, delayed-cache + FOCUS (alpha=1).
  FOCUS requires delayed cache in this repository, so the middle mode isolates
  the additional effect of FOCUS. These are experimental settings, not a claim
  about the paper's official evaluation settings.
- TP=1, executor=uni, eager mode, one visible GPU. Each mode/batch runs in its own
  process. Default cache fraction .1 reduces KV allocation; weights still occupy
  full precision BF16 memory, and fitting on A800 40G needs server verification.
- Freeze sampled rendered prompts and lengths in requests.json; reuse them.
- Submit one group of B requests at a time, wait for completion, then submit the
  next. The engine can still execute partially occupied batches at arrival/exit.
- Enable observation only after model initialization/warmup. For every actual
  layer execution save phase, group, forward ordinal, actual batch, per-request
  Query lengths and the full expert-load histogram. Assert sum(load)=Q*top_k.
- The ordinal is an actual execution index inside a group, not a synchronized
  per-request denoising step. Modes can follow different decoding trajectories.
- Main summaries retain only decode records with B nonempty requests. All other
  records remain in the raw trace. Coverage is mandatory; missing full batches
  cause a final error rather than a false scaling conclusion.

## Server

Use the existing FOCUS environment with this repository's LMDeploy dependencies.
The script puts the repository first on Python's import path. No OpenCompass is
needed. Wait for the existing dual-GPU trajectory experiment to release a GPU.

```bash
cd /root/lkd/FOCUS
git pull origin main
conda activate focus-moe

# Smoke: two batch sizes, two groups at B=2, all three modes.
CUDA_VISIBLE_DEVICES=0 python -u benchmark/profile_focus_moe_scaling.py \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/focus_moe_scaling/smoke_v1 \
  --batch-sizes 1 2 --num-prompts 4 --max-input-len 128 --timeout 600

# Main: fixed 32-token generation and 32-token block.
CUDA_VISIBLE_DEVICES=0 python -u benchmark/profile_focus_moe_scaling.py \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/focus_moe_scaling/gsm8k_32_v1 \
  --batch-sizes 1 2 4 8 --num-prompts 32 --max-input-len 128
```

Output directory must be new. Per-mode/batch logs are at `<mode>_bs<B>.log`;
use `tail -f` to inspect initialization and progress. Default per-process timeout
is 1800 seconds and may be increased explicitly. Timeouts and OOM stop the sweep;
existing traces are retained. This does not guarantee a fix for every LMDeploy
initialization problem, but avoids the previous TP=2 multi-process path.

## Results

- `<mode>_bs<B>/routes.jsonl`: actual per-layer measurements, prefill and decode.
- `coverage.csv`: total decode-layer records and full-batch records per run.
- `group_layers.csv`: average Q/E/coverage within each group and layer.
- `scaling.csv`: equal-weight group means per mode, batch and layer; group SD;
  expert growth ratio E(2B)/E(B); Q ratios to vanilla and delayed-only controls.
- `experiment.json`, `requests.json`: settings and identical rendered prompts.
- `<mode>_bs<B>/complete.json`: run finished marker.

First check coverage and whether FOCUS actually reduces Q relative to delayed
cache at affected layers. Then plot E versus B, separately by layer, and inspect
E(2B)/E(B) alongside expert coverage E/num_experts. A ratio below two near the
expert-count ceiling can simply reflect saturation. The SD describes between-group
variation, not a confidence interval or independent per-forward samples.

Cross-mode means average their own executed trajectories; they do not constitute
a matched-step causal comparison. EOS and partially occupied batches can select
different portions of a trajectory, especially for short generations. Report
coverage and inspect the raw records before claiming the relationship holds.
The per-layer GPU-to-CPU observations synchronize execution: do not use these
runs to report throughput or speedup. Subsequent integration and quality
experiments are needed before claiming FOCUS composes with a new method.
