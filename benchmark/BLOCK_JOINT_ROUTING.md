# Training-free current-block joint routing: complete generation

## Scope and relationship to previous code

The previous `llada2_shared_pair` configuration is left unchanged. That earlier
method compressed MASK-only routes with a joint coverage heuristic and
renormalized weights. The NEW `llada2_joint_four` configuration implements the
current agreed method, not a relabeling of those earlier experiments.

Four configurations, run serially using the same real-batch HF generator:

| Mode | Routing decision |
| --- | --- |
| native | Original Top-8 and native `moe_infer`, no compression |
| independent | Highest-mass minimum-cardinality feasible subset per token |
| request | Jointly choose feasible subsets inside each request's current block |
| batch | Jointly choose across requests, with request/independent solutions as seeds |

These are comparison configurations, not four stacked components. No training,
history route retention, output caching, expert-output similarity, FOCUS eviction,
or isolated intervention experiment is involved. XShare/OEA/SERE are NOT yet
implemented in this entry: the four configurations are internal controls, not
an external-method comparison or a claim of novelty against those papers.

## Exact local constraints, approximate global objective

The original gate executes unchanged, including sigmoid scores, expert bias,
group-limited selection and routed scaling. On its ORIGINAL Top-8, normalize
mixing weights only for selection: `p = w / sum(w)`.

At `epsilon = 0.1`, find the smallest count `k_i` reaching 0.9 mass. Enumerate
ALL subsets with that count satisfying the same mass constraint. Independent,
request and batch selections use those identical feasible families on an
identical forward state. Exact enumeration supports native Top-1 to Top-8.
The temporary table has at most 255 nonempty combinations; a given minimum
count has at most 70 feasible candidate combinations. No route history is saved.

Joint planning tries independent/request support seeds and the union of all
feasible candidate supports. Experts are considered in ascending static token
owner count, expert-ID ties. Remove an expert only when every affected token
still has a feasible combination. Choose the solution with the smallest support,
then largest retained mass; remaining ties preserve the earlier seed. Local
family ties use lexicographic expert-ID order. This final global tie differs
from the offline planner's arbitrary global bit-mask tie; constraints and the
heuristic/seed structure are the same, not promised bit-for-bit route identity.
Within-request deletion rounds are processed in parallel, not B separate GPU
planning loops. This is a reference solver, NOT an exact global optimum or a
performance-tuned kernel.

Removed token-expert pairs genuinely do not execute the routed FFN. Retained
weights keep their ORIGINAL values, without renormalization. The architectural
shared expert and attention are unchanged. Each expert computes a fresh output
for every retained token; sharing expert identity does NOT copy token outputs.
`epsilon=0` explicitly calls the original native execution, even with zero
native weights. Mass retention is a routing proxy, not a functional/quality bound.

## Current block and KV contract

Both MASK AND previously decoded generation positions in live requests are
targets on every denoising forward. No special decoded skip or fixed hot pool.
Complete prefix blocks use KV cache and are not recomputed every denoising step.
Complete generated blocks are recomputed WITHOUT compression to commit KV for
subsequent blocks. This commit cost is measured, not hidden.

Partial prompt positions sharing the first block remain native and are NOT used
as free background support in the optimization. Finished rows remain allocated
and execute native routed work until the static batch completes, also not used
as free support. Rounded final-block extra positions actually executed are
included in routing accounting but excluded from reported output token count.
Whole-block left padding preserves per-request rotary positions/block boundaries.
Prefill padding keys are masked. The repository eager backend expects binary
4D masks; the existing main-repository binary-mask fix is preserved.

The model uses vendored repository HF code (`store_kv` support), with config and
weights from MODEL_PATH. This preserves the main-repository loader fix rather
than assuming checkpoint remote code has the same cache API. Default sampling
is explicit GREEDY; `SAMPLING=native` uses the existing repository multinomial
sampler (including its temperature=0 behavior). All compared modes must match.

## Two A800 40G GPUs

One process; Accelerate `device_map='balanced'`; complete decoder layers on GPU
0 and GPU 1. This is **layer-wise weight/model sharding, NOT tensor parallelism,
expert parallelism, data parallelism or sequential per-request simulation**.
Every decoder layer processes the same physical B-request batch. Hidden states
cross devices between the layer groups; KV stays with its decoder layer.

The loader verifies decoder layers actually use BOTH GPUs, rejects split layers
and CPU/disk offload. Default loading budget is `34GiB` per card, leaving room
for activations/KV; this is a weight-placement limit, NOT an inference memory cap
or an OOM guarantee. Increase prompt/output lengths and B gradually. No CPU
offload fallback, silent request serialization, silent truncation or OOM retry
at a different batch size. The model's Accelerate/Transformers dependencies and
the existing OpenCompass environment are required. Do not reinstall your whole
server environment from the Mac testing setup.

## Server commands

After copying/pulling the updated code:

```bash
cd /root/lkd/FOCUS
conda activate focus-moe

# B8 first: 32 GSM8K examples, 32 NEW tokens maximum, four modes.
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/root/lkd/Models/LLaDA2.0-mini \
BATCH_SIZE=8 MAX_OUT_LEN=32 ROUTE_EPSILON=0.1 \
OUTPUT_DIR=results/block_joint_eval/gsm8k_smoke_b8_e010_v1 \
bash benchmark/run_llada2_block_joint_eval.sh gsm8k_shared_smoke

# Actual B32, both GPUs still hold different layers of ONE model.
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/root/lkd/Models/LLaDA2.0-mini \
BATCH_SIZE=32 MAX_OUT_LEN=32 ROUTE_EPSILON=0.1 \
OUTPUT_DIR=results/block_joint_eval/gsm8k_smoke_b32_e010_v1 \
bash benchmark/run_llada2_block_joint_eval.sh gsm8k_shared_smoke

# Multi-block smoke: tests mechanics, NOT a valid full GSM8K quality result.
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/root/lkd/Models/LLaDA2.0-mini \
BATCH_SIZE=8 MAX_OUT_LEN=128 ROUTE_EPSILON=0.1 \
OUTPUT_DIR=results/block_joint_eval/gsm8k_multiblock_b8_e010_v1 \
bash benchmark/run_llada2_block_joint_eval.sh gsm8k_shared_smoke

# Same real task, full data and a longer output budget. Check B8 memory first.
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=/root/lkd/Models/LLaDA2.0-mini \
BATCH_SIZE=8 MAX_OUT_LEN=1024 ROUTE_EPSILON=0.1 \
EXPECTED_REQUESTS=1319 \
OUTPUT_DIR=results/block_joint_eval/gsm8k_full_b8_e010_v1 \
bash benchmark/run_llada2_block_joint_eval.sh gsm8k_gen
```

After memory/flow checks, run formal B1/2/4/8/16/32 with matching dataset,
output budget and sampler. B32 at 32 tokens does not ensure B32 at 1024 tokens
or long prompts fits. Inspect length-limit rate; 1024 is an initial task budget,
not a universal assurance of sufficient generation. Each output directory must
be NEW; the launcher refuses to reuse one. Use absolute OUTPUT_DIR if launching
from elsewhere. Inputs are not truncated to 64/128 tokens by this task entry.
Default confidence threshold 0.8 matches the existing task config, NOT the
earlier trajectory's 0.95; set CONFIDENCE_THRESHOLD=0.95 for all modes if desired.

Optional identity smoke: `ROUTE_EPSILON=0 SAMPLING=greedy`, compare all four
prediction files. They should match under deterministic execution. Optional
mode subset: `JOINT_METHODS=native,batch`. Use `MAX_MEMORY_PER_GPU=34GiB` to
make the default loading budget explicit. Launcher uses one worker and each
model has two GPUs / one process; do not request two workers for two cards.

For component timing run a SEPARATE diagnostic directory with
`DETAILED_TIMING=1`. This synchronizes the layer's GPU around planning and
routed expert execution, perturbing performance; do not present diagnostic TPS
as the ordinary fast-path TPS. With the default `0`, component seconds are
host/enqueue time (some planner transfers synchronize), not GPU kernel duration.

## Outputs / interpretation

* `opencompass/<timestamp>/predictions/`: full per-example text, original OC task evaluation.
* `opencompass/<timestamp>/summary/`: GSM8K / other task quality, not proxy route mass.
* `metrics/<mode>/metrics-*.jsonl`: real batch generation/phase times, pairs,
  expert supports, output lengths/EOS, memory peaks on both GPUs.
* `timing_summary.json`: aggregate throughput and counters.
* `run.log`: inspect for model/task/evaluation failures.

Generation-call wall time includes tokenization, planning, sparse dispatch, GPU
work, sampling and text decoding. It excludes model loading, dataset I/O,
evaluator, and measurement-file writing. Both GPUs synchronize at common forward
boundaries. `routed_expert_seconds` excludes router and shared expert; it is not
total MoE-layer duration. Complete denoising-forward time includes attention,
router, planner, routed/shared FFNs and LM head.
Diagnostic `moe_layer_seconds` includes router, planning, routed FFNs and the
architectural shared expert (no attention/LM head). It is `null` in normal
throughput runs, rather than reporting host enqueue time as total MoE GPU time.

`assignments_*` covers all actually computed routed pairs, including prefill,
commit, partial prompt and finished rows. `block_assignments_*` covers denoising
forwards only. `target_assignments_*` covers eligible current generation positions.
`mean_block_active_*` is the average expert union PER layer-forward over ALL
executed rows; `mean_target_active_*` is its eligible-generation-only counterpart.
Expert IDs belong to a particular layer: never union IDs across different layers
or label these means as the total expert count of the model.

Native task runs and method task runs can have different hidden states, per-token
budgets and completion histories. Identical local budget constraints on a fixed
state do NOT guarantee identical actual pair counts across online runs. Report
real measured work and task quality, not only expert-union reduction.

The launcher checks all selected methods have equal measured request counts;
smoke also requires exactly 32. EXPECTED_REQUESTS allows formal completeness
checking (1319 for the unmodified GSM8K test split). Still inspect evaluator
completion/score files; timing counts cannot prove successful task scoring.
First calls are flagged and included. Warm-state summaries can exclude them
using summarize_shared_eval.py --exclude-first-call; B32 smoke only has one
batch per mode, so cannot supply a warm-state estimate.

## Validation / limitations

```bash
python -m pytest tests/test_lmdeploy/test_block_joint_routing.py \
  tests/test_lmdeploy/test_opencompass_shared_batch.py \
  tests/test_lmdeploy/test_routing_competition.py -q
```

CPU tests verify exact local family constraints against the offline enumerator,
equal per-token counts, best-of-seed support monotonicity, native passthrough,
actual skipped expert rows, retained native weights and shared-expert behavior,
acceptance lifecycle mapping, per-request boundaries and multi-block KV shape
progression. No Mac CUDA/A800/real-checkpoint/OpenCompass task run is available;
these tests do not establish two-GPU runtime correctness, quality or speedup.
This is a complete-generation REFERENCE prototype: selection/dispatch overhead
may outweigh the savings. A measured slowdown is a valid result, not a reason
to omit the planner from timing or advertise offline sparsity as speedup.
