# Cross-request redundancy inside a routed expert

This post-processing experiment reuses the complete files produced by
`profile_llada2_expert_trajectory.py`. It does not regenerate tokens. Wait for the
trajectory run to finish: hidden-state snapshots are written after generation.

## Question and protocol

Does sharing an expert across requests create similar expert outputs, or merely
share the same expert weights? For each saved step, generation group, block, and
expert, join token identities in the JSONL with target-layer hidden states in the
PT snapshot. Replay **only tokens actually routed to that expert**, using its real
checkpoint gate/up/down weights and the LLaDA2 SwiGLU formula. Shared experts and
router weights are excluded: the measurement is the routed expert function itself.

Only one expert's weights are resident on the GPU at once. This needs neither
the full model nor LMDeploy, NCCL, or OpenCompass. A single A800 40G is sufficient
for the intended batch-8, block-32 snapshots; runtime also includes many small
eigendecompositions and should not be interpreted as an inference speed benchmark.

Three sampling repetitions are a fixed robustness protocol, not method tuning.

### Rank

For nested random request subsets of sizes 1, 2, 4, 8 from each original group:

- Measure the number of actual routed tokens and contributing requests.
- Compute rank on up to 256 uniformly sampled routed tokens, recording that cap.
- Independently compute rank on exactly 16 tokens when available.
- Record input and output ranks, both raw and after subtracting the token mean.

Rank is `exp(-sum(p * log(p)))`, where `p` is normalized squared singular-value
energy. A zero matrix has rank zero. Centered rank distinguishes variation between
tokens from a common mean direction. Input rank is the control for redundancy
already present before the expert computation.

The paired comparison includes only the same `(step, group, block, expert, repeat)`
eligible for 16-token sampling at **every** requested subset size. This avoids
comparing different expert populations, but selects high-load experts: consult
coverage and do not extrapolate to all experts. A comparison file can be absent
when no expert satisfies that condition. Do not lower the sample count merely to
obtain a favorable result; report coverage first.

These are nested subsets of a **fixed captured batch**, not new generation runs at
different batch sizes. They measure how request diversity changes the expert's
sampled input/output population. A genuine batch-size scaling claim subsequently
requires separate trajectory runs at different actual batch sizes.

### Within-request versus cross-request neighbors

Use the whole captured group for each expert. For every eligible anchor token,
sample exactly four candidates from its own request (excluding itself) and four
from other requests. Compare minimum cosine distance and minimum relative L2
distance to each set. Each distance chooses its own nearest candidate. Both sides
use the same anchor and candidate count. Zero-norm cases are excluded and counted.
Requests with fewer than four same-request alternatives cannot supply an anchor;
the output reports this coverage. Repetitions reuse anchors and are not independent
experimental observations.

Low rank or nearby outputs support further approximation experiments; they do not
prove interchangeable outputs, preserved task quality, or a measured speedup.
Same mask states and positional composition can still explain similarity; the
planned mask/position control experiment remains necessary.

## Server commands

After pulling the commit and activating `focus-moe`, use the completed trajectory
directory. Required packages: torch, safetensors. The checkpoint must be the same
unquantized LLaDA2.0-mini checkpoint used for the trajectory.

Smoke post-processing on just step 0 (no regeneration):

```bash
cd /root/lkd/FOCUS
git pull origin main
conda activate focus-moe
CUDA_VISIBLE_DEVICES=0 STEPS="0" REPEATS=1 \
bash benchmark/run_expert_output_redundancy.sh \
  /root/lkd/FOCUS/results/expert_trajectory/gsm8k_bs8_layer10_v1 \
  /root/lkd/Models/LLaDA2.0-mini \
  /root/lkd/FOCUS/results/expert_redundancy/gsm8k_smoke_v1
```

Main post-processing:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash benchmark/run_expert_output_redundancy.sh \
  /root/lkd/FOCUS/results/expert_trajectory/gsm8k_bs8_layer10_v1 \
  /root/lkd/Models/LLaDA2.0-mini \
  /root/lkd/FOCUS/results/expert_redundancy/gsm8k_bs8_layer10_v1
```

Use a new output directory each time. For the earlier block-16 smoke trajectory,
set `STEPS="0 4 8"`; step 12 was not captured. Missing requested snapshots cause a
clear error. If an entire group finished before a selected step, it has no cell
at that step. Do not interpret the surviving groups as all original requests.

## Outputs and first analysis

- `metadata.json`: protocol, source metadata, environment and definitions.
- `ranks.csv`: per-cell load, request coverage, raw/centered input/output ranks,
  variable-count and fixed-count observations, including ineligible cells.
- `paired_rank_changes.csv`: fixed-count rank changes relative to subset size 1,
  restricted to common eligible experts across all requested sizes.
- `neighbors.csv`: matched within/cross-request neighbor distances and coverage.
- `coverage.json`: number of eligible rank cells, paired comparisons and neighbor cells.

Start with coverage. Plot output rank against routed-token count to describe the
load relationship, then compare fixed-count rank changes for common experts.
Check raw versus centered ranks and input versus output ranks. Finally compare
within/cross-request neighbor distances using paired cells. Repeats, experts and
steps from one request group are correlated; aggregate per group before estimating
uncertainty. With only four original groups, treat results as a pilot.

Future multi-block capture must preserve unique token IDs and the snapshot schema.
This analyzer retains block IDs and does not pool blocks into a single cell.
