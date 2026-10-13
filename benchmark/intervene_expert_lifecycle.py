#!/usr/bin/env python3
"""Single-token, single-branch lifecycle interventions on native HF LLaDA2.

Research counterfactuals, not a sparse execution implementation or speed test.
The original MLP runs, then w_e*(E_f(h)-E_e(h)) is added at exactly one position.
Original shared-expert outputs and every other position are left unchanged.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def phase_at(step, acceptance_step):
    if step < acceptance_step:
        return 'pre_accept_mask'
    if step == acceptance_step:
        return 'accepting_mask'
    if step == acceptance_step + 1:
        return 'first_decoded'
    return 'later_decoded'


def visits_by_request(routes, live_requests):
    """Binary request incidence, generation positions only, excluding finished requests."""
    visits = defaultdict(set)
    for request, positions in enumerate(routes):
        if not live_requests[request]:
            continue
        for ids in positions:
            for expert in ids:
                visits[int(expert)].add(request)
    return dict(visits)


def source_and_candidates(ids, weights, visits, request, eligible, limit):
    """Descriptive tail anchor; popularity shortlist, NOT a learned routing policy."""
    source = min(zip(ids, weights), key=lambda pair: (len(visits.get(pair[0], set())), pair[1], pair[0]))[0]
    candidates = [e for e, requests in visits.items()
                  if e != source and e in eligible and requests - {request}]
    candidates.sort(key=lambda e: (-len(visits[e] - {request}), -len(visits[e]), e))
    return int(source), candidates[:limit]


def select_cases(acceptance, snapshots, count, seed, pre_steps, post_steps):
    """Random request-diverse cohort, without requiring expert persistence."""
    eligible = [(request, position, step) for request, row in enumerate(acceptance)
                for position, step in enumerate(row)
                if step is not None and step >= pre_steps
                and step + 1 < len(snapshots) and snapshots[step + 1]['live_requests'][request]]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen, used = [], set()
    for item in eligible:
        if item[0] not in used:
            chosen.append(item)
            used.add(item[0])
            if len(chosen) == count:
                break
    if len(chosen) < count:
        chosen.extend(item for item in eligible if item not in chosen)
        chosen = chosen[:count]
    return [dict(case_id=i, batch_index=r, block_position=p, acceptance_step=s,
                 requested_start=s-pre_steps, requested_end=s+post_steps)
            for i, (r, p, s) in enumerate(chosen)]


def _tensor(output):
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not hasattr(value, 'ndim') or value.ndim != 3:
        raise RuntimeError('Expected native MLP tensor [batch, sequence, hidden].')
    return value


class LayerProbe:
    """Temporary hooks; no modifications to the checkpoint's remote Python code."""
    def __init__(self, moe, batch_size, block_length, intervention=None):
        self.moe, self.batch_size, self.block_length = moe, batch_size, block_length
        self.intervention = intervention
        self.calls = 0
        self.local_metrics = None

    def __enter__(self):
        self.handles = [self.moe.register_forward_pre_hook(self._pre),
                        self.moe.gate.register_forward_hook(self._gate),
                        self.moe.register_forward_hook(self._post)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def _pre(self, module, inputs):
        self.hidden = inputs[0]
        if self.hidden.ndim != 3 or self.hidden.shape[0] != self.batch_size:
            raise RuntimeError('Unexpected MLP input shape; refusing ambiguous position mapping.')

    def _gate(self, module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError('Expected native gate (topk_ids, topk_weights, logits).')
        ids, weights, logits = output
        sequence = self.hidden.shape[1]
        if ids.shape != weights.shape or ids.shape[0] != self.batch_size * sequence:
            raise RuntimeError('Gate rows do not match batch-major full-sequence positions.')
        self.ids = ids.view(self.batch_size, sequence, -1)[:, -self.block_length:].detach().cpu()
        self.weights = weights.view(self.batch_size, sequence, -1)[:, -self.block_length:].float().cpu()
        self.logits = logits.view(self.batch_size, sequence, -1)[:, -self.block_length:].float().cpu()

    def _post(self, module, inputs, output):
        import torch
        self.calls += 1
        original = _tensor(output)
        if self.intervention is None:
            self.block_hidden = self.hidden[:, -self.block_length:].detach().cpu()
            return output
        case = self.intervention
        r, p, source = case['batch_index'], case['block_position'], case['source_expert']
        slot = (self.ids[r, p] == source).nonzero().flatten()
        if slot.numel() != 1:
            raise RuntimeError('Source must be present exactly once; absent sources are skipped by the driver.')
        full_position = original.shape[1] - self.block_length + p
        h = self.hidden[r, full_position:full_position+1]
        weight = self.weights[r, p, slot.item()].to(device=h.device, dtype=torch.float32)
        left = module.experts[source](h).float()
        if case['method'] == 'delete':
            right = torch.zeros_like(left)
        elif case['method'] == 'replace':
            right = module.experts[case['replacement_expert']](h).float()
        else:
            raise ValueError('Unknown intervention method.')
        delta = weight * (right - left)
        norm = float(left.norm())
        base_norm = float(original[r, full_position].float().norm())
        self.local_metrics = dict(
            source_weight=float(weight), source_output_norm=norm,
            source_weighted_output_norm=float((weight * left).norm()),
            expert_output_relative_l2=float((right-left).norm())/norm if norm else None,
            weighted_delta_norm=float(delta.norm()),
            full_mlp_output_relative_delta=float(delta.norm())/base_norm if base_norm else None,
            expert_output_cosine=float(torch.nn.functional.cosine_similarity(left, right).item())
            if norm and float(right.norm()) else None)
        changed = original.clone()
        changed[r, full_position] = (original[r, full_position].float() + delta[0]).to(original.dtype)
        if isinstance(output, tuple):
            return (changed,) + output[1:]
        if isinstance(output, list):
            return [changed] + output[1:]
        return changed


def allowed_experts(gate, logits):
    """Respect the native score+bias group eligibility at the anchor snapshot."""
    import torch
    device = gate.expert_bias.device
    scores = logits.to(device).float().sigmoid() + gate.expert_bias.float()
    groups = scores.reshape(gate.n_group, -1)
    selected = groups.topk(2, dim=-1).values.sum(-1).topk(gate.topk_group).indices.cpu().tolist()
    width = groups.shape[1]
    return {e for group in selected for e in range(group*width, (group+1)*width)}


def predict(model, hidden, chunk_size):
    """Never materialize B*block*vocab logits or probabilities at once."""
    import torch
    shape = hidden.shape[:2]
    flat = hidden.reshape(-1, hidden.shape[-1])
    candidates, confidence = [], []
    device = model.lm_head.weight.device
    for start in range(0, len(flat), chunk_size):
        logits = model.lm_head(flat[start:start+chunk_size].to(device)).float()
        values, ids = logits.max(-1)
        candidates.append(ids.cpu())
        confidence.append((values-logits.logsumexp(-1)).exp().cpu())
    return dict(candidates=torch.cat(candidates).reshape(shape),
                confidence=torch.cat(confidence).reshape(shape))


def distribution_metrics(model, reference_hidden, changed_hidden, active, chunk_size):
    """Exact full-vocabulary KL(ref||changed) on MASK rows, streamed through lm_head."""
    import torch
    old = reference_hidden[active]
    new = changed_hidden[active]
    device = model.lm_head.weight.device
    divergences, max_delta = [], 0.
    for start in range(0, len(old), chunk_size):
        ref = torch.log_softmax(model.lm_head(old[start:start+chunk_size].to(device)).float(), -1)
        branch = torch.log_softmax(model.lm_head(new[start:start+chunk_size].to(device)).float(), -1)
        divergences.append((ref.exp()*(ref-branch)).sum(-1).cpu())
        max_delta = max(max_delta, float((ref-branch).abs().max()))
    kl = torch.cat(divergences) if divergences else torch.empty(0)
    return dict(mean_kl=float(kl.mean()) if len(kl) else None,
                max_token_kl=float(kl.max()) if len(kl) else None,
                max_abs_logprob_delta=max_delta), kl


def decision_metrics(reference, branch, active, accepted_ref, accepted_new, kl, target_request):
    import torch
    flips = (reference['candidates'] != branch['candidates']) & active
    added, lost = accepted_new & ~accepted_ref, accepted_ref & ~accepted_new
    total = int(active.sum())
    per_request, offset = [], 0
    for request in range(len(active)):
        n = int(active[request].sum())
        per_request.append(dict(batch_index=request, mask_tokens=n,
            candidate_flips=int(flips[request].sum()),
            mean_kl=float(kl[offset:offset+n].mean()) if n else None,
            accepted_added=int(added[request].sum()), accepted_lost=int(lost[request].sum())))
        offset += n
    other = flips.clone()
    other[target_request] = False
    target = per_request[target_request]
    target_count = target['mask_tokens']
    return dict(mask_tokens=total, candidate_flips=int(flips.sum()),
        candidate_flip_rate=float(flips.sum())/total if total else None,
        target_request_mean_kl=target['mean_kl'],
        target_request_mask_tokens=target_count,
        target_request_candidate_flip_rate=target['candidate_flips']/target_count if target_count else None,
        target_request_acceptance_difference_rate=(target['accepted_added']+target['accepted_lost'])/target_count
            if target_count else None,
        other_request_candidate_flips=int(other.sum()),
        accepted_added=int(added.sum()), accepted_lost=int(lost.sum()),
        acceptance_symmetric_difference_rate=float((added|lost).sum())/total if total else None,
        per_request=per_request)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_path')
    p.add_argument('--prompt-snapshot', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-groups', type=int, default=1)
    p.add_argument('--group-start', type=int, default=0)
    p.add_argument('--layer', type=int, default=10, help='Zero-based Transformer layer index, as in existing traces.')
    p.add_argument('--block-length', type=int, default=32)
    p.add_argument('--denoising-steps', type=int, default=32)
    p.add_argument('--confidence-threshold', type=float, default=.95)
    p.add_argument('--tokens-per-group', type=int, default=4)
    p.add_argument('--candidate-count', type=int, default=4)
    p.add_argument('--pre-steps', type=int, default=2)
    p.add_argument('--post-steps', type=int, default=3)
    p.add_argument('--logit-chunk-size', type=int, default=8)
    p.add_argument('--max-memory-per-gpu', default='36GiB')
    p.add_argument('--seed', type=int, default=0)
    return p


def main():
    p = parser()
    a = p.parse_args()
    sizes = [a.batch_size, a.num_groups, a.block_length, a.denoising_steps,
             a.tokens_per_group, a.candidate_count, a.logit_chunk_size]
    if min(sizes) < 1 or min(a.group_start, a.layer, a.pre_steps) < 0 or a.post_steps < 1:
        p.error('Sizes must be positive, indices/pre-steps nonnegative, post-steps >= 1.')
    if not 0 <= a.confidence_threshold <= 1:
        p.error('Confidence threshold must lie in [0,1].')
    # Fast validation and --help do not import Torch/Transformers or contact HF.
    payload = a.prompt_snapshot.read_bytes()
    snapshot = json.loads(payload)
    prompts = snapshot['prompt_token_ids']
    if (a.group_start+a.num_groups)*a.batch_size > len(prompts):
        p.error('Prompt snapshot has insufficient requests for the requested groups.')
    request_ids = snapshot.get('request_ids', list(range(len(prompts))))
    if len(request_ids) != len(prompts) or any(not row for row in prompts):
        p.error('Malformed prompt snapshot.')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise FileExistsError('Use an EMPTY output directory; partial runs are not overwritten.')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from profile_llada2_hf_moe_saturation import _build_batch_inputs, _transfer_schedule
    from profile_llada2_expert_trajectory import choose_candidate_mask
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Expose exactly two GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1.')
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if mask_id is None or pad_id is None:
        raise ValueError('Checkpoint tokenizer must define MASK and padding/EOS.')
    print('Loading native HF weights across two GPUs (balanced placement, NOT TP)...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='balanced',
        max_memory={0:a.max_memory_per_gpu, 1:a.max_memory_per_gpu},
        low_cpu_mem_usage=True, attn_implementation='eager').eval()
    mapping = getattr(model, 'hf_device_map', {})
    if not mapping or any(str(device) in ('cpu', 'disk', 'meta') for device in mapping.values()):
        raise RuntimeError('CPU/disk offload is unsupported; reduce batch/prompt size instead.')
    core = model.model
    if a.layer >= len(core.layers):
        raise ValueError('Layer index outside checkpoint.')
    moe = core.layers[a.layer].mlp
    if not hasattr(moe, 'gate') or not hasattr(moe, 'experts'):
        raise ValueError('Target layer does not expose native routed experts.')
    schedule = _transfer_schedule(a.block_length, a.denoising_steps)
    metadata = dict(record_type='metadata', format_version=1, backend='huggingface', mode='vanilla',
        intervention='post_mlp_single_branch_additive_patch', speed_measurement=False,
        fixed_pair=True, renormalize=False, candidate_selection='anchor_local_output_oracle_on_popularity_shortlist',
        prompt_sha256=hashlib.sha256(payload).hexdigest(), prompt_metadata={k:v for k,v in snapshot.items() if k!='prompt_token_ids'},
        torch_version=torch.__version__, device_map={k:str(v) for k,v in mapping.items()},
        args={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()})
    (a.output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    input_device = core.word_embeddings.weight.device
    baseline_file = a.output_dir/'baseline_trajectory.jsonl'
    cases_file = a.output_dir/'cases.jsonl'
    events_file = a.output_dir/'interventions.jsonl'

    def emit(stream, row):
        stream.write(json.dumps(row, allow_nan=False)+'\n')

    with baseline_file.open('x', buffering=1) as baseline_stream, cases_file.open('x', buffering=1) as case_stream, \
            events_file.open('x', buffering=1) as event_stream, torch.inference_mode():
        for group in range(a.group_start, a.group_start+a.num_groups):
            start = group*a.batch_size
            ids, attention, positions = _build_batch_inputs(prompts[start:start+a.batch_size],
                a.block_length, pad_id, mask_id, input_device, torch.bfloat16)
            acceptance = [[None]*a.block_length for _ in range(a.batch_size)]
            snapshots = []

            def forward(state_ids, intervention=None):
                with LayerProbe(moe, a.batch_size, a.block_length, intervention) as probe:
                    out = core(input_ids=state_ids, attention_mask=attention, position_ids=positions,
                               use_cache=False, output_router_logits=False, return_dict=True)
                    hidden = out.last_hidden_state[:, -a.block_length:].detach().cpu()
                    del out
                if probe.calls != 1:
                    raise RuntimeError(f'Expected one target MLP call, got {probe.calls}.')
                result = predict(model, hidden, a.logit_chunk_size)
                result['hidden'] = hidden
                return result, probe

            def accepted(result, active, step):
                return choose_candidate_mask(active, result['confidence'], schedule[step],
                    a.confidence_threshold) & result['candidates'].ne(mask_id)

            # The only trajectory advanced here is the unmodified Vanilla trajectory.
            for step in range(len(schedule)):
                active = ids[:, -a.block_length:].eq(mask_id).cpu()
                if not active.any():
                    break
                before = ids.detach().cpu().clone()
                ref, probe = forward(ids)
                chosen = accepted(ref, active, step)
                for r, pos in chosen.nonzero().tolist():
                    acceptance[r][pos] = step
                states = dict(input_ids=before, hidden=ref['hidden'], candidates=ref['candidates'],
                    confidence=ref['confidence'], active=active, accepted=chosen,
                    live_requests=active.any(-1).tolist(), routes=probe.ids.tolist(),
                    weights=probe.weights.tolist())
                snapshots.append(states)
                emit(baseline_stream, dict(record_type='baseline_step', group=group, step=step,
                    request_ids=request_ids[start:start+a.batch_size], live_requests=states['live_requests'],
                    block_input_ids=before[:, -a.block_length:].tolist(), candidates=ref['candidates'].tolist(),
                    confidence=ref['confidence'].tolist(), accepted=chosen.tolist(),
                    routes=states['routes'], weights=states['weights']))
                ids[:, -a.block_length:][chosen.to(ids.device)] = ref['candidates'].to(ids.device)[chosen.to(ids.device)]
                print(f'baseline group={group} step={step} masks={int(active.sum())}->{int(ids[:, -a.block_length:].eq(mask_id).sum())}', flush=True)
                del probe, ref
            torch.save(dict(group=group, layer=a.layer, acceptance_steps=acceptance,
                request_ids=request_ids[start:start+a.batch_size], attention_mask=attention.cpu(),
                position_ids=positions.cpu(), snapshots=snapshots), a.output_dir/f'states_group{group}.pt')
            chosen_cases = select_cases(acceptance, snapshots, a.tokens_per_group,
                a.seed+group, a.pre_steps, a.post_steps)
            emit(case_stream, dict(record_type='cohort_coverage', group=group,
                accepted_tokens=sum(s is not None for row in acceptance for s in row),
                no_first_decoded_observation=sum(s is not None and
                    (s+1 >= len(snapshots) or not snapshots[s+1]['live_requests'][r])
                    for r,row in enumerate(acceptance) for s in row),
                too_early_for_requested_pre_window=sum(s is not None and s < a.pre_steps
                    for row in acceptance for s in row), selected_cases=len(chosen_cases)))
            if not chosen_cases:
                emit(case_stream, dict(record_type='no_eligible_cases', group=group))
            for case in chosen_cases:
                r, pos, anchor = case['batch_index'], case['block_position'], case['acceptance_step']
                case.update(group=group, request_id=request_ids[start+r], layer=a.layer)
                state = snapshots[anchor]
                # Select once at accepting-MASK anchor; never optimize the pair independently at later steps.
                anchor_ref, probe = forward(state['input_ids'].to(input_device))
                if not torch.equal(anchor_ref['candidates'], state['candidates']):
                    raise RuntimeError('Anchor replay candidates changed; no pair selection permitted.')
                visits = visits_by_request(state['routes'], state['live_requests'])
                legal = allowed_experts(moe.gate, probe.logits[r, pos])
                source, candidates = source_and_candidates(state['routes'][r][pos], state['weights'][r][pos],
                    visits, r, legal, a.candidate_count)
                case.update(source_expert=source, source_request_visits=len(visits[source]),
                    candidate_experts=candidates, candidate_selection='anchor_local_output_oracle_on_popularity_shortlist')
                if not candidates:
                    emit(case_stream, dict(record_type='skipped_case', reason='no_legal_cross_request_candidate', **case))
                    del probe, anchor_ref
                    continue
                device = next(moe.experts[source].parameters()).device
                h = probe.block_hidden[r, pos:pos+1].to(device)
                left = moe.experts[source](h).float()
                distances = {e:float((moe.experts[e](h).float()-left).norm()) for e in candidates}
                replacement = min(candidates, key=lambda e: (distances[e], e))
                case.update(replacement_expert=replacement, anchor_candidate_distances=distances,
                            replacement_other_request_visits=len(visits[replacement]-{r}))
                emit(case_stream, dict(record_type='case', **case))
                print(f'group={group} case={case["case_id"]} token=R{r}/P{pos} accept={anchor} '
                      f'fixed_pair={source}->{replacement}', flush=True)
                del probe, anchor_ref, h, left
                for step in range(case['requested_start'], case['requested_end']+1):
                    common = dict(group=group, layer=a.layer, case_id=case['case_id'], batch_index=r,
                        request_id=case['request_id'], block_position=pos, acceptance_step=anchor,
                        step=step, relative_step=step-anchor, phase=phase_at(step, anchor),
                        source_expert=source, replacement_expert=replacement)
                    if step >= len(snapshots) or not snapshots[step]['live_requests'][r]:
                        emit(event_stream, dict(record_type='missing_observation', reason='request_finished_or_trajectory_ended', **common))
                        continue
                    state = snapshots[step]
                    if source not in state['routes'][r][pos]:
                        emit(event_stream, dict(record_type='missing_observation', reason='source_not_selected', **common))
                        continue
                    # No-op replay for EACH eligible snapshot, not merely for the first group.
                    ref, probe = forward(state['input_ids'].to(input_device))
                    noop_dist, noop_kl = distribution_metrics(model, state['hidden'], ref['hidden'], state['active'], a.logit_chunk_size)
                    no_accept = accepted(ref, state['active'], step)
                    if (not torch.equal(ref['candidates'], state['candidates']) or
                        not torch.equal(no_accept, state['accepted']) or
                        not torch.equal(probe.ids, torch.tensor(state['routes'])) or
                        noop_dist['max_abs_logprob_delta'] > 1e-5 or
                        not torch.equal(ref['hidden'], state['hidden'])):
                        raise RuntimeError('No-op replay mismatch; refusing intervention interpretation.')
                    legal_now = allowed_experts(moe.gate, probe.logits[r, pos])
                    visits_now = visits_by_request(state['routes'], state['live_requests'])
                    common.update(input_token_id=int(state['input_ids'][r, -a.block_length+pos]),
                        replacement_group_eligible=replacement in legal_now,
                        replacement_in_token_topk=replacement in state['routes'][r][pos],
                        replacement_other_request_visits=len(visits_now.get(replacement, set())-{r}),
                        source_request_visits=len(visits_now.get(source, set())))
                    emit(event_stream, dict(record_type='intervention', method='baseline_noop',
                        local_metrics=None, **common, **noop_dist,
                        **decision_metrics(state, ref, state['active'], state['accepted'], no_accept, noop_kl, r)))
                    del probe, ref
                    for method in ('delete', 'replace'):
                        branch, probe = forward(state['input_ids'].to(input_device), dict(case, method=method))
                        dist, kl = distribution_metrics(model, state['hidden'], branch['hidden'], state['active'], a.logit_chunk_size)
                        metrics = decision_metrics(state, branch, state['active'], state['accepted'],
                            accepted(branch, state['active'], step), kl, r)
                        emit(event_stream, dict(record_type='intervention', method=method,
                            local_metrics=probe.local_metrics, **common, **dist, **metrics))
                        print(f'group={group} case={case["case_id"]} offset={step-anchor:+d} {method} '
                              f'flip={metrics["candidate_flips"]} KL={dist["mean_kl"]:.6g}', flush=True)
                        del branch, probe
            emit(baseline_stream, dict(record_type='group_complete', group=group,
                remaining_masks=int(ids[:, -a.block_length:].eq(mask_id).sum()), cases_selected=len(chosen_cases)))
            del snapshots
    (a.output_dir/'complete.json').write_text(json.dumps(dict(status='complete',
        groups=a.num_groups, functional_interventions_only=True))+'\n')
    print(f'Complete: {a.output_dir}', flush=True)


if __name__ == '__main__':
    main()
