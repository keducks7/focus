# Benchmark

We provide several profiling tools to benchmark our models.

## profile with dataset

Download the dataset below or create your own dataset.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Profiling your model with `profile_throughput.py`

```bash
python profile_throughput.py \
 ShareGPT_V3_unfiltered_cleaned_split.json \
 /path/to/your/model \
 --concurrency 64
```

### ShareGPT (HuggingFace)

You can also pass the HuggingFace dataset ID directly (requires the `datasets` package):

```bash
python profile_throughput.py \
  anon8231489123/ShareGPT_Vicuna_unfiltered \
  /path/to/your/model \
  --dataset-format sharegpt \
  --hf-split train \
  --concurrency 64
```

If the dataset repo doesn't load via `datasets.load_dataset`, use `--hf-data-file` to point to the JSON/JSONL file.
By default, HuggingFace dataset IDs are loaded in non-streaming mode for accurate shuffling; use `--hf-streaming` to enable streaming.

### WildChat

`profile_throughput.py` also supports the WildChat dataset from HuggingFace:

```bash
python profile_throughput.py \
  allenai/WildChat \
  /path/to/your/model \
  --dataset-format wildchat \
  --hf-split train \
  --concurrency 64
```

Note: loading HuggingFace datasets requires the `datasets` package.
By default, HuggingFace dataset IDs are loaded in non-streaming mode for accurate shuffling; use `--hf-streaming` to enable streaming.

### GSM8K

`profile_throughput.py` also supports the GSM8K evaluation split from HuggingFace:

```bash
python profile_throughput.py \
  openai/gsm8k \
  /path/to/your/model \
  --dataset-format gsm8k \
  --hf-split test \
  --hf-config main \
  --concurrency 64
```

`openai/gsm8k` defaults to the `main` config and `test` split. If your local `datasets` metadata exposes the evaluation split as `validation`, the loader falls back automatically.

## LLaDA2 MoE expert-saturation experiment

Use the controlled full-denoising runner to test whether a small, concentrated
expert working set remains stable throughout generation. It uses one Python
process and HuggingFace Accelerate's balanced device map, avoiding LMDeploy's
multi-process executor. Run task datasets separately so task-dependent routing
remains visible:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_PROMPTS=32 MAX_INPUT_LEN=128 BATCH_SIZES="8" \
  benchmark/run_llada2_moe_saturation.sh \
  openai/gsm8k /root/lkd/Models/LLaDA2.0-mini \
  ./results/llada2_moe_saturation/gsm8k

CUDA_VISIBLE_DEVICES=0,1 NUM_PROMPTS=32 MAX_INPUT_LEN=128 BATCH_SIZES="8" \
  benchmark/run_llada2_moe_saturation.sh \
  google-research-datasets/mbpp /root/lkd/Models/LLaDA2.0-mini \
  ./results/llada2_moe_saturation/mbpp
```

The GSM8K run uses the `main` test split. The MBPP run uses the hand-verified
`sanitized` test split. Neither run executes a full benchmark or evaluates
answer correctness.

Defaults target two 40 GB GPUs, cap prompts at 128 tokens, fix request batch at
8, and run 32 denoising steps for one 32-token mask block. Before every model
forward, the profiler marks the unresolved mask positions. It reads LLaDA2's
official `output_router_logits` result and counts only the routes belonging to
those positions—not prompt, padding, or already-decoded tokens. Candidate tokens
are then accepted independently for every request using confidence > 0.95 or the
step's minimum transfer quota. Thus each record describes the queries that
actually entered that denoising step. The output directory contains:

- `routes_bs*.jsonl`: per-step, per-layer expert-load histograms;
- `moe_denoising_layers.csv`: raw layer/group/step metrics and hot-expert IDs;
- `moe_denoising_summary.csv`: step-wise aggregates across layers and prompt groups;
- `moe_denoising.svg`: query volume, working-set size, concentration, and stability curves;
- `hf_trace_run.log`: model placement and progress output.

The four primary statistics are active experts, inverse-Simpson effective
experts, Top-10 load share, and the Jaccard overlap between the Top-10 sets of
the same layer in adjacent steps. Expert IDs are never pooled across layers.
`query_tokens` and `query_fraction` are retained as controls for the naturally
shrinking diffusion workload. Step 0 is also the Vanilla all-mask observation;
later work can run the same trace under FOCUS for a binary comparison.

For a lower-memory smoke run:

```bash
MAX_INPUT_LEN=64 NUM_PROMPTS=16 BATCH_SIZES="8" \
  MASK_BLOCK_LENGTH=16 DENOISING_STEPS=16 \
  benchmark/run_llada2_moe_saturation.sh \
  /path/to/dataset.json /path/to/LLaDA2.0-mini
```

The profiler loads the model only once. An OOM at a larger batch does not
discard completed smaller-batch traces. `MAX_MEMORY_PER_GPU` defaults to
`38GiB`; override it if other processes reserve memory.
Set `MASK_BLOCK_LENGTH`, `DENOISING_STEPS`, `CONFIDENCE_THRESHOLD`, or
`TEMPERATURE` to change the controlled generation process. Temperature 0 uses
true greedy decoding. The older `MAX_NEW_TOKENS` environment name remains
accepted as a compatibility alias for `MASK_BLOCK_LENGTH`.

Dependencies for this runner are `torch`, `transformers`, `accelerate`, and
`datasets`. OpenCompass is not used: it is only needed later when checking that
an acceleration method preserves task accuracy.

## profile restful api

`profile_restful_api.py` is used to do benchmark on api server.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 profile_restful_api.py --backend lmdeploy --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json
```
