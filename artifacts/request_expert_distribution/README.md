# Request-level expert distributions — first plotting draft

Source: three token-trajectory runs in `results/expert_trajectory/`:
GSM8K B8, GSM8K B16, HumanEval B8. No new model inference.

## Figures

- `*_step{0,4,12}_heatmap.png/pdf`: actual batch group 0; rows of panels are
  layers 2, 10, 18 (0-based model indices), columns are accepted this step and
  unresolved after this step. Every row inside a panel is one request.
- `request_overlap_curves.png/pdf`: pair-weighted mean distribution overlap
  across all valid request pairs within the original batches, by step/layer/state.
- `request_overlap_support.png/pdf`: token counts and number of eligible pairs
  for those curves. Missing groups are not extrapolated.

Counts are Top-8 assignments, not router weights or latency. Each request-state
distribution is normalized by its own token count times eight. Colour range is
fixed at 0–12.5%, the maximum assignment share of one expert when Top-8 IDs are
unique. Grey denotes an empty state, not a zero distribution. Labels give the
number of tokens in each row. Prompt and previously accepted tokens are excluded.

All 256 experts are shown. Within each run and layer, experts are ranked once
by total assignment load pooled over all recorded groups, steps and both states;
ties are broken by expert ID. That order is reused at steps 0, 4 and 12 and for
both state columns. Rankings are layer-specific and run-specific: equal x-axis
positions across different layers/runs do not imply equal expert IDs. Ranking
is a descriptive display choice using the complete trace, not an online rule.

Overlap is `sum_e min(p_request_a(e), p_request_b(e))` on normalized 256-expert
distributions. It lies in [0,1]. Pairs never cross batch-group boundaries.
Empty states are excluded; nonempty states with only one token are retained
and may yield noisy overlap. Support changes over time. Lines do not represent
confidence intervals or control for sample-count differences. No request-label
permutation baseline has yet been applied, so high overlap alone does not prove
request-specific structure beyond shared global expert preferences.

`*_pair_details.csv` includes token counts for both requests;
`*_expert_sharing.csv` records per-expert request coverage and assignment load;
`request_overlap_summary.csv` records curve means, support and range of group
means (the latter is not a confidence interval).

## Reproduce

Requires Python, NumPy and Matplotlib:

```bash
python benchmark/plot_request_expert_distribution.py \
  --input-root results/expert_trajectory \
  --output-dir artifacts/request_expert_distribution
```

This is a first scientific plotting draft; styling can be adapted to a supplied
reference image while keeping the measurements and normalization unchanged.
