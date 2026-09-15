# Cross-layer route-stabilization frontier

This offline analysis asks whether the depth profile of one token's adjacent-step
Top-k route overlap changes as the token approaches acceptance. It reads the
existing token trajectory and does not load LLaDA2 or require a GPU.

For each `steps_until_acceptance = d`, it first computes mean route Jaccard at
every MoE layer. Every boundary between adjacent layers is then evaluated as a
two-segment description of that depth curve. The reported frontier is the split
with minimum within-segment squared error. This introduces neither a Jaccard
threshold nor a hand-selected early/middle/late layer range.

Run:

```bash
python benchmark/analyze_cross_layer_stabilization_frontier.py \
  results/expert_trajectory/gsm8k_bs8_layer10_v1/token_trajectories_bs8.jsonl \
  --output-dir results/expert_trajectory/gsm8k_bs8_layer10_v1
```

Outputs:

- `cross_layer_distance_profile.csv`: the full layer-by-acceptance-distance
  route-overlap surface;
- `cross_layer_frontier_summary.csv`: selected frontier, before/after means,
  direction, and variance explained for every acceptance distance;
- `cross_layer_direction_changes.json`: adjacent acceptance distances where the
  fitted depth direction changes sign.

## GSM8K batch-8 pilot

The fitted direction changes once, between two and three steps before
acceptance:

- distance 0: frontier 9|10, after-minus-before Jaccard `+0.0907`, R2 `0.6888`;
- distance 1: frontier 9|10, delta `+0.0932`, R2 `0.7859`;
- distance 2: frontier 12|13, delta `+0.0692`, R2 `0.3758`;
- distance 3: frontier 2|3, delta `-0.1501`, R2 `0.4463`;
- distances 4--18 retain the negative depth direction; distances above 18 have
  fewer than 40 token transitions per layer and are exploratory only.

Thus the data do not support a single static stabilization frontier throughout
decoding. Far from acceptance, the first few layers are unusually route-stable
and deeper routing is more dynamic. In the final two transitions before
acceptance, the direction reverses and a middle-depth boundary emerges. The
clean research hypothesis is a phase-conditioned cross-layer routing process,
not generic route stability.

The layer-10 expert replay provides an important boundary on interpretation.
Adjacent similarity matrices are strongly correlated after step 0, but the
step-0 nearest-expert replacement error is already about `1.51` and remains at
that level. Expert relations are temporally structured, yet cosine-nearest
experts are not functionally interchangeable under the current metric. This
pilot therefore supports using route dynamics as a decoding-state signal more
than static SERE-style expert substitution.

The next empirical requirement is replication, first with GSM8K batch 16 and
then HumanEval batch 8. Method design should wait until the direction flip near
acceptance survives both batch and task changes.
