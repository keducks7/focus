# Dataset evaluation: HF Vanilla and Vanilla + shared sparse routing

## What is preserved

Uses the vendored OpenCompass dataset configs, prompt templates, prediction
postprocessors, evaluators and summaries. New model config: `llada2_shared_pair`.
The original `llada2` config and wrapper are NOT changed (that config enables
FOCUS and its wrapper loops over requests sequentially).

Both new variants use the SAME real batch generator, BF16, eager attention,
two-GPU HF balanced layer placement (not tensor parallel), one process, and the
repository's block-diffusion confidence/minimum-transfer rule. Complete prefix
blocks and committed generation blocks use KV caching. Within-block KV reuse,
Query eviction and FOCUS are disabled. Partial prompt blocks are preserved:
padding adds only whole blocks, with per-request original rotary positions and
padding keys masked. Multi-block generation is supported; max_out_len need not
be a multiple of block_length. Final rounded-block extras are not counted as
output tokens. EOS requires all preceding generation positions to be resolved.

This is a static batch: rows finishing early stay allocated until batch completion.
Their remaining compute and all padding work are included in timing. This is not
a continuous-batching serving benchmark. Real GPU numerical equivalence to B1
is not promised; batched kernels can differ in floating-point rounding.

## Method execution

Vanilla (`epsilon=0`) directly calls the original `moe_infer` method. The method
variant selects a subset of original Top-k routes for currently MASK positions
at every routed layer. All other actually computed positions keep original routes.
The joint greedy coverage selector runs with torch operations on the device;
it includes synchronization/selection overhead and is a reference implementation,
not a performance-tuned CUDA kernel. Removed pairs are NOT sent to experts.
Shared experts are unchanged. Weights are rescaled to preserve original mixture
mass. No compression is applied to prefill or the complete-block KV commit pass.

Unlike stage 2's zero-weight intervention, this path actually reduces evaluated
token-expert pairs. It may still be slower due to selection/dispatch overhead;
do not infer speedup from pair counts alone. Counts exclude shared-expert work.
`ROUTE_METHOD=independent` and `joint_no_fixed` are optional ablation settings.

## Sampling is explicit

Default `SAMPLING=native` invokes the repository model's
`_sample_with_temperature_topk_topp`, matching its existing wrapper. The vendored
implementation calls multinomial EVEN at temperature=0. `SAMPLING=greedy` is an
explicit deterministic argmax alternative (same confidence threshold/rule), useful
for the trajectory research setting. Never compare runs using different sampling
modes. Seed is set independently on model load; changing the number of steps
changes RNG consumption, so native samples are not paired random draws forever.
Default confidence threshold 0.8 follows the repository OpenCompass config,
whereas earlier trajectory experiments used 0.95. Set 0.95 explicitly if desired.

## Environment

Use the existing `focus-moe` environment with the repository OpenCompass
dependencies installed according to `opencompass-0.5.1.post1/README.md`, plus the
HF Accelerate loading dependencies already used by trajectory experiments.
No new training/framework backend dependency is introduced. Do not install the
Mac test dependency versions into your server environment just to run this.
Loaded model code must expose `model.forward(..., store_kv=...)` and routed
`moe_infer(x, topk_ids, topk_weight)`; unsupported code fails with an explicit error.
CPU/disk offload is rejected for these GPU timing runs.

## Server smoke run (32 GSM8K examples, 64 output tokens)

```bash
cd /root/lkd/FOCUS
conda activate focus-moe
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/root/lkd/Models/LLaDA2.0-mini \
BATCH_SIZE=8 MAX_OUT_LEN=64 ROUTE_EPSILON=0.10 \
SAMPLING=native \
OUTPUT_DIR=/root/lkd/FOCUS/results/shared_route_eval/gsm8k_smoke_b8_e010 \
bash benchmark/run_opencompass_shared_eval.sh gsm8k_shared_smoke
```

The script runs BOTH models serially with `--max-num-workers 1`; each has
`run_cfg(num_gpus=2,num_procs=1)`. Do not set two workers for this two-GPU model.
The smoke subset uses the original `gsm8k_0shot_v2_gen_17d799` evaluator and prompt,
only restricting test_range to `[0:32]`. It is not the 128 filtered prompts used
by trajectory experiments, and 64 tokens is NOT sufficient for formal accuracy.
The script refuses an existing output directory to prevent stale result reuse.

## Full GSM8K evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/root/lkd/Models/LLaDA2.0-mini \
BATCH_SIZE=16 MAX_OUT_LEN=1024 ROUTE_EPSILON=0.10 \
SAMPLING=native \
OUTPUT_DIR=/root/lkd/FOCUS/results/shared_route_eval/gsm8k_full_b16_e010 \
bash benchmark/run_opencompass_shared_eval.sh gsm8k_gen
```

For B32 change BATCH_SIZE and output directory. Quality must be compared at the
same output budget, sampling mode, threshold, dataset and prompt configuration.
All positions (including prompt + rounded output) must fit MAX_SEQ_LEN (4096 by
default); there is no silent truncation/filtering. Input lengths are dataset
dependent: success in a short trajectory run does not guarantee B32 eval fits.
Reduce batch size if necessary and rerun BOTH variants at the matched size.

Other existing configs: `math500_gen`, `humaneval_gen`, `sanitized_mbpp_gen`.
Use a separate output directory per dataset. Code-task evaluators execute
generated code: run those in an isolated server environment, not on your Mac.
They may require optional evaluation dependencies/data per OpenCompass README.

To run only one variant use `SHARED_MODE=vanilla` or `SHARED_MODE=shared`.
For an identity smoke check set `ROUTE_EPSILON=0 SAMPLING=greedy`: both named
variants then use the same native routed execution and should produce matching
predictions under deterministic kernels. Check this before interpreting quality
differences if the server's model code differs from the vendored version.

## Results and definitions

- `<OUTPUT_DIR>/opencompass/<timestamp>/summary/`: task scores via original OC evaluator.
- Corresponding `predictions/`: per-example generations.
- `<OUTPUT_DIR>/metrics/<model>/metrics-*.jsonl`: each actual batch's timings,
  output length/EOS status, denoising steps, executed versus original route pairs.
- `<OUTPUT_DIR>/timing_summary.json`: aggregation across completed batches.
- `<OUTPUT_DIR>/run.log`: full OC logs (inspect for failed tasks even if OC exits zero).

`generation_tps = sum(actual output tokens) / sum(generate wall seconds)`.
Output token count includes EOS, excludes prompt, padding and rounded extras.
Wall includes tokenization, selection, decode loop, sampling, GPU work, and text
decoding; excludes model loading, dataset I/O, OpenCompass scoring and log writing.
This is generation-call end-to-end time, NOT the entire benchmark wall time.
`inference_tps` excludes tokenization/text decoding; `denoise_tpf_ms` is summed
synchronized denoising-forward time (including selection and LM head) / number
of batch forwards, excluding prefill, complete-block commits and token sampling.
Prefill/commit times and request denoising steps are reported separately.
Both methods use the same synchronization/timing boundaries.

First calls are flagged but included by default. A separate warm-state summary:

```bash
python benchmark/summarize_shared_eval.py \
  results/shared_route_eval/gsm8k_full_b16_e010/metrics \
  --exclude-first-call \
  --output results/shared_route_eval/gsm8k_full_b16_e010/timing_warm.json
```

If either model fails or has fewer evaluated samples, do not compare partial TPS
or accuracy as a matched result. Summaries do not automatically assert benchmark
completeness; the launcher checks presence of both variants and equal measured
request counts, but still check OC logs, prediction counts and the expected
dataset size (equally incomplete runs are not detected by that check).

## Validation

```bash
python -m pytest tests/test_lmdeploy/test_opencompass_shared_batch.py \
  tests/test_lmdeploy/test_shared_route_selection.py -q
```

Tests run CPU tensor selection versus the pure-Python reference, native vendored
MoE weight-intervention versus actual sparse execution (including shared experts),
zero-epsilon passthrough, true skipped expert calls, per-request block alignment,
multi-block generation, EOS, KV shape progression and weighted timing summaries.
No local real LLaDA2 model, CUDA or full OpenCompass task execution was available;
the server smoke run is still required.
