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

Use the controlled first-denoising-step runner to test whether the union of
active experts saturates at a small request batch. It uses one Python process
and HuggingFace Accelerate's balanced device map, avoiding LMDeploy's
multi-process executor. Run task datasets separately so task-dependent routing
remains visible:

```bash
NUM_PROMPTS=64 MAX_INPUT_LEN=128 \
  benchmark/run_llada2_moe_saturation.sh \
  openai/gsm8k inclusionAI/LLaDA2.0-mini \
  ./results/llada2_moe_saturation/gsm8k

NUM_PROMPTS=64 MAX_INPUT_LEN=128 \
  benchmark/run_llada2_moe_saturation.sh \
  google-research-datasets/mbpp inclusionAI/LLaDA2.0-mini \
  ./results/llada2_moe_saturation/mbpp
```

The GSM8K run uses the `main` test split. The MBPP run uses the hand-verified
`sanitized` test split. Neither run executes a full benchmark or evaluates
answer correctness.

Defaults target two 40 GB GPUs, cap prompts at 128 tokens, and append one
32-token all-mask block. For each batch, the profiler performs the initial
denoising forward only and reads LLaDA2's official `output_router_logits`
result. It counts routing for the mask block—not left-padding or prompt tokens.
This isolates the maximum-query step while keeping batch size as the only swept
variable. The output directory contains:

- `routes_bs*.jsonl`: expert-load histograms for the initial all-mask block;
- `moe_saturation_summary.csv`: overall and per-layer saturation statistics;
- `moe_saturation.svg`: measured active-expert ratio and the uniform-routing null;
- `hf_trace_run.log`: model placement and progress output.

For a lower-memory smoke run:

```bash
MAX_INPUT_LEN=64 NUM_PROMPTS=32 BATCH_SIZES="1 2 4 8" \
  benchmark/run_llada2_moe_saturation.sh \
  /path/to/dataset.json /path/to/LLaDA2.0-mini
```

The profiler loads the model only once. An OOM at a larger batch does not
discard completed smaller-batch traces. `MAX_MEMORY_PER_GPU` defaults to
`38GiB`; override it if other processes reserve memory.
Set `MASK_BLOCK_LENGTH` to change the observed all-mask block length. The older
`MAX_NEW_TOKENS` environment name remains accepted as a compatibility alias.

Dependencies for this runner are `torch`, `transformers`, `accelerate`, and
`datasets`. OpenCompass is not used: it is only needed later when checking that
an acceleration method preserves task accuracy.

## profile restful api

`profile_restful_api.py` is used to do benchmark on api server.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 profile_restful_api.py --backend lmdeploy --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json
```
