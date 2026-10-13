# Fixed-pair expert lifecycle intervention (minimal research experiment)

This is a **Vanilla HF functional diagnostic**, not a FOCUS run, not a speed test,
not training, and not OpenCompass evaluation. The default physical batch is 32;
the model is distributed across two GPUs by `device_map='balanced'`, **not TP**.
Requests use the existing snapshot token IDs, without downloading a dataset again.
The unmodified native HF Transformer/experts run with the existing block attention
and confidence-or-minimum-transfer decoder; no custom LMDeploy executor is used.

## Protocol

1. Generate one normal 32-token block, saving target-layer generation routes,
   predictions, acceptance times, input states, and final hidden states on CPU.
2. Randomly select up to four token positions from different requests, requiring
   at least two pre-acceptance observations and a still-live first decoded step.
   Tokens accepted at the last request step are censored, not declared stable.
3. At each token's accepting-MASK snapshot, pick an original branch with minimal
   request incidence (ties: lower weight, lower ID). Shortlist four experts used by
   other live requests and allowed by the native group rule at that anchor.
   Select the nearest expert output on this SAME token input from this shortlist.
   This is an **anchor-local output oracle**, not a deployable acceleration rule,
   not global SERE similarity, and not a comparison of all 256 candidates.
4. Fix this `(source, replacement)` pair for that token. For offsets -2,-1,0,+1,+2,+3,
   run a no-op baseline, a single-branch deletion, and a single-branch replacement.
   Offset 0 still has MASK input; offset +1 is first actual-token input. If the
   source disappears from original Top-8, record `source_not_selected` and do NOT
   inject it. Finished requests/ended trajectories are explicitly censored.
5. Each intervention restarts from the SAME Vanilla input; no intervened result is
   written back into later inputs. There is no continuation or accumulated-error
   experiment in this minimal entry point.

The native MLP runs before a hook adds `w_e*(E_f(h)-E_e(h))` at one position;
deletion sets `E_f(h)=0`. Original weights are not renormalized. The shared expert
is not separately modified. If the replacement is already in this token's Top-8,
the formula is algebraically the same as merging its weights. Router records
remain the ORIGINAL routes: this is a patch, not modified gate/dispatch output.
The source expert is evaluated on a one-row input, so BF16 rounding need not match
its original multi-row evaluation bit-for-bit; report this as a diagnostic delta,
not official native rerouting parity. Native group eligibility is logged at EACH
replay: a later ineligible candidate remains a labelled functional counterfactual,
not an admissible production route. Do not pool it silently with legal routes.

No-op checks compare hidden states, routes, candidates, acceptance, and full-vocab
MASK log probabilities at each eligible replay. Any mismatch stops the run.

## Server configuration

- GPUs: two A800 40G; expose exactly 0,1. BF16; eager attention; no CPU/disk offload.
- Existing `focus-moe` environment with PyTorch, Transformers, Accelerate,
  Safetensors and the checkpoint's remote-code dependencies. No new framework.
  The earlier traces used PyTorch 2.8.0+cu126 / Transformers 4.57.1; prefer the
  environment that successfully ran those traces rather than blindly upgrading.
- Default per-GPU weight-placement budget: 36GiB (not a total-memory cap).
- Logit chunks: 8 positions. Full vocabulary KL is calculated in chunks from
  small cached hidden vectors: no trajectory-sized probability tensors in RAM.
- One group/4 anchor tokens/6 offsets implies at most 72 intervention/no-op replay
  forwards, plus normal generation and 4 anchor selection forwards. Missing source
  observations reduce that count. Replays still use a REAL B32 input, not B1 unions.
- Host RAM and disk need room for model loading plus CPU state snapshots (typically
  hundreds of MB per group, dependent on hidden size); no guaranteed OOM-free claim.
  Start with one group. There is no reason to run all 128 prompts first.

```bash
cd /root/lkd/FOCUS
conda activate focus-moe

# No GPU needed; tensor tests run only if Torch is available.
python tests/test_lmdeploy/test_expert_lifecycle_intervention.py

# Native HF single-group smoke. A NEW/EMPTY output directory is mandatory.
CUDA_VISIBLE_DEVICES=0,1 python -u benchmark/intervene_expert_lifecycle.py \
  /root/lkd/Models/LLaDA2.0-mini \
  --prompt-snapshot results/expert_trajectory/sharing_b1_to_b32_v1/bs32/sampled_prompt_token_ids.json \
  --output-dir results/expert_lifecycle_intervention/b32_l10_smoke_v1 \
  --batch-size 32 --num-groups 1 --layer 10 \
  --block-length 32 --denoising-steps 32 --confidence-threshold 0.95 \
  --tokens-per-group 4 --candidate-count 4 --pre-steps 2 --post-steps 3 \
  --logit-chunk-size 8 --max-memory-per-gpu 36GiB --seed 0

# CPU analysis; --plot additionally requires numpy/matplotlib.
python benchmark/analyze_expert_lifecycle_interventions.py \
  results/expert_lifecycle_intervention/b32_l10_smoke_v1 --plot
```

After smoke: use a new output directory and `--num-groups 4` for 128 requests.
To reduce replay TIME without changing batch, set `--tokens-per-group 2` first.
For a B8 environment check, use `--batch-size 8 --tokens-per-group 2` in a new
directory; that is explicitly a B8 experiment, not equivalent to B32. Shortening
prompt inputs creates another protocol; do not silently truncate snapshot IDs.

## Outputs and interpretation

- `metadata.json`: settings, model/device map, prompt SHA256, candidate protocol.
- `baseline_trajectory.jsonl`: target-layer routes for MASK AND decoded, token IDs,
  decisions and confidence. Other layers are not traced by this entry point.
- `states_group*.pt`: exact replay inputs, attention/positions, hidden states and
  acceptance times; only load trusted locally generated Torch archives.
- `cases.jsonl`: cohort IDs, fixed expert pairs, shortlist distances, skipped cases.
- `interventions.jsonl`: no-op/delete/replace metrics and explicit missing events.
  Metrics: full-vocabulary KL(ref||intervention) on currently MASK positions,
  candidate flips, acceptance additions/losses, per-request effects, source weight,
  weighted source norm, expert output error, full-MLP output delta.
  Primary plots use `target_request_mean_kl` and the affected request's acceptance
  changes; full-batch means would dilute the effect with 31 untouched requests.
- `complete.json`: written ONLY when every requested group finishes diagnostics.
- `analysis/`: case-equal `phase_summary.csv`, `case_phase_metrics.csv`,
  `coverage.csv`, complete-pair `paired_phase_changes.csv`, `summary.json`;
  optional `lifecycle_perturbation.png/pdf`, `fixed_pair_phase_heatmap.png/pdf`.
  Main tables include labelled functional counterfactuals. The additional
  `legal_only_phase_summary.csv` and `legal_only_case_phase_metrics.csv` exclude
  later group-ineligible candidate events from BOTH delete and replace arms.

Inspect missing observations FIRST. The anchor-MASK oracle makes low anchor error
partly a selection effect; higher decoded error alone is not proof of a universal
lifecycle law. Require more anchors/seeds, control branch weight/output norm,
compare alternatives selected on independent calibration data, and validate final
quality before a method claim. Pre/post state, context, and MASK cohort all change.
Grey heatmap cells mean missing data, NEVER zero perturbation. This minimal run
does not establish safe pruning, task accuracy, actual speedup or novelty.
