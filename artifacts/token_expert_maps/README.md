# Per-token expert assignment maps

Reference-style first version: true expert ID on x, block token position on y,
stored router weight on colour. Model layers 2, 10, 18 are 0-based indices.

- Main step: 4. Each actual batch group of GSM8K B8, GSM8K B16 and HumanEval B8
  has three all-request atlas figures, one per layer. Every request in that batch
  has its own subplot. Requests from different batch groups are not pooled.
- Step 0, group 0: baseline atlases with all 32 positions initially masked.
- Step 4, group 0: an additional full-sized three-layer figure for every request,
  matching the supplied reference layout.
- Both PNG and PDF are provided. File names encode step, group, layer or request.

All figures share one colour scale, with its maximum equal to the maximum stored
weight across the selected steps/layers in the three runs. Weights retain the
model's routed scaling factor; do not assume each row sums to one. Experts are
never sorted: vertical alignment between requests at the same layer means the
same expert. An expert ID in different layers denotes different parameters.

Each observed row contains exactly eight selected expert cells. Light grey
background means an expert was not selected. Full darker-grey rows mean a token
was accepted earlier and its routing is absent from this mask-only trace, NOT
that the model skipped its computation. The right-hand status strip is orange
for acceptance at this step, blue for remaining unresolved, grey for previously
accepted. Both orange and blue routes were measured before the current update.

Overlap@8 is mean |Top8(token_a) intersection Top8(token_b)| / 8 over pairs of
observed tokens inside one request and layer. It is not Jaccard and not a
cross-request similarity. No pairs gives NA. Counts and values are saved in
token_map_statistics.csv; the colour range is in plot_metadata.json.

GSM8K B8/B16 prompt IDs agree. Compare R00–R07 across the two batch sizes for a
matched-request view; B16 group 0 also contains R08–R15 (B8 group 1). Do not pair
HumanEval and GSM8K requests by numerical ID as if they were the same prompt.
Different counts of surviving tokens can affect apparent homogeneity.

Reproduce with Python, NumPy and Matplotlib:

```bash
python benchmark/plot_token_expert_maps.py \
  --input-root results/expert_trajectory \
  --output-dir artifacts/token_expert_maps --step 4
```
