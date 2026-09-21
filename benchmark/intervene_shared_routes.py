#!/usr/bin/env python3
"""Stage 2: single-layer, single-step HF functional intervention, NOT a speed test.

Uses the native gate's (IDs, weights, logits) interface. Zeros discarded weights
before native expert aggregation, but computes all original branches. No remote
model source is edited. Each branch starts from the same unmodified input state.
"""
import argparse
import json
from pathlib import Path
from shared_route_selection import METHODS, select_routes, reweight


class GateIntervention:
    def __init__(self, compress, epsilon, method, renormalize=True):
        self.compress = compress
        self.epsilon = epsilon
        self.method = method
        self.renormalize = renormalize
        self.calls = 0
        self.metrics = None

    def __call__(self, module, inputs, output):
        import torch
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError('Expected official LLaDA2 gate (topk_ids, topk_weights, logits).')
        ids, weights, logits = output
        if ids.ndim != 2 or ids.shape != weights.shape or ids.shape[0] != self.compress.numel():
            raise RuntimeError('Unexpected gate shape; refusing ambiguous row mapping.')
        if ids.dtype not in (torch.int32, torch.int64) or not weights.is_floating_point():
            raise RuntimeError('Unexpected gate IDs/weights types.')
        compress = self.compress.flatten().cpu().tolist()
        id_rows, weight_rows = ids.cpu().tolist(), weights.float().cpu().tolist()
        keep, self.metrics = select_routes(id_rows, weight_rows, compress, self.epsilon, self.method)
        self.calls += 1
        if self.epsilon == 0:
            return output  # exact no-op control, not round-trip reweighting
        changed = weights.clone()
        indexes = [i for i, flags in enumerate(keep) if not all(flags)]
        if indexes:
            values = [reweight(weight_rows[i], keep[i], self.renormalize) for i in indexes]
            changed[indexes] = torch.tensor(values, device=weights.device, dtype=weights.dtype)
        return ids, changed, logits


def prediction_metrics(reference, branch, active, selected_ref, selected_new):
    """CPU metrics on all currently MASK positions, including per-request tails."""
    import torch
    flips = (reference['candidates'] != branch['candidates']) & active
    added = selected_new & ~selected_ref
    lost = selected_ref & ~selected_new
    # log probabilities are stored only for MASK positions, in batch-major order.
    kl_chunks = []
    for start in range(0, reference['logp'].shape[0], 16):
        old = reference['logp'][start:start+16]
        new = branch['logp'][start:start+16]
        kl_chunks.append((old.exp() * (old - new)).sum(-1))
    kl = torch.cat(kl_chunks)
    per_request = []
    offset = 0
    for i in range(active.shape[0]):
        count = int(active[i].sum())
        per_request.append(dict(batch_index=i, masks=count,
                                candidate_flip_rate=float(flips[i].sum())/max(1, count),
                                mean_kl=float(kl[offset:offset+count].mean()) if count else None,
                                accepted_added=int(added[i].sum()), accepted_lost=int(lost[i].sum())))
        offset += count
    return dict(mask_tokens=int(active.sum()), candidate_flip_rate=float(flips.sum())/int(active.sum()),
                baseline_accepted_candidate_flip_rate=float((flips & selected_ref).sum())/max(1, int(selected_ref.sum())),
                accepted_added=int(added.sum()), accepted_lost=int(lost.sum()),
                acceptance_symmetric_difference_rate=float((added | lost).sum())/int(active.sum()),
                mean_kl=float(kl.mean()), max_token_kl=float(kl.max()),
                max_abs_logprob_delta=float((reference['logp']-branch['logp']).abs().max()),
                per_request=per_request)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_path')
    p.add_argument('--prompt-snapshot', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-groups', type=int, default=1)
    p.add_argument('--group-start', type=int, default=0)
    p.add_argument('--layer', type=int, default=10)
    p.add_argument('--step', type=int, default=4)
    p.add_argument('--block-length', type=int, default=32)
    p.add_argument('--denoising-steps', type=int, default=32)
    p.add_argument('--confidence-threshold', type=float, default=.95)
    p.add_argument('--epsilons', nargs='+', type=float, default=[.05, .1, .2])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--max-memory-per-gpu', default='38GiB')
    p.add_argument('--no-renormalize', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if min(a.batch_size, a.num_groups, a.block_length, a.denoising_steps) < 1 or a.group_start < 0:
        p.error('Invalid sizes or group start.')
    if any(not 0 <= e < 1 for e in a.epsilons) or not 0 <= a.confidence_threshold <= 1:
        p.error('Invalid epsilon or confidence threshold.')
    from profile_llada2_hf_moe_saturation import _build_batch_inputs, _transfer_schedule, _sample_block
    from profile_llada2_expert_trajectory import choose_candidate_mask
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    schedule = _transfer_schedule(a.block_length, a.denoising_steps)
    if not 0 <= a.step < len(schedule):
        p.error('Step outside denoising schedule.')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Expose exactly two GPUs using CUDA_VISIBLE_DEVICES=0,1.')
    snapshot = json.loads(a.prompt_snapshot.read_text())
    prompts = snapshot['prompt_token_ids']
    if (a.group_start+a.num_groups)*a.batch_size > len(prompts):
        raise ValueError('Prompt snapshot has insufficient requests.')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    output = a.output_dir/'interventions.jsonl'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if mask_id is None or pad_id is None:
        raise ValueError('Missing MASK/padding token ID.')
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map='balanced', max_memory={0:a.max_memory_per_gpu, 1:a.max_memory_per_gpu},
        low_cpu_mem_usage=True, attn_implementation='eager').eval()
    core = model.model
    moe = core.layers[a.layer].mlp
    if not hasattr(moe, 'gate') or not hasattr(moe, 'experts'):
        raise ValueError('Target is not a supported routed MoE layer.')
    input_device = core.word_embeddings.weight.device
    with output.open('x', buffering=1) as stream, torch.inference_mode():
        stream.write(json.dumps(dict(record_type='metadata', backend='huggingface', mode='vanilla',
                                    intervention='gate_weight_zeroing_full_expert_execution',
                                    speed_measurement=False, temperature=0, model_class=type(model).__name__,
                                    args={k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}))+'\n')
        for group in range(a.group_start, a.group_start+a.num_groups):
            start = group*a.batch_size
            ids, attention, positions = _build_batch_inputs(prompts[start:start+a.batch_size],
                a.block_length, pad_id, mask_id, input_device, torch.bfloat16)
            block = ids[:, -a.block_length:]

            def forward(active, store_distribution=False):
                out = core(input_ids=ids, attention_mask=attention, position_ids=positions,
                           use_cache=False, output_router_logits=False, return_dict=True)
                hidden = out.last_hidden_state[:, -a.block_length:, :]
                logits = model.lm_head(hidden)
                candidates, confidence = _sample_block(logits, 0.)
                result = dict(candidates=candidates.cpu(), confidence=confidence.cpu())
                if store_distribution:
                    # Avoid constructing another full float32 distribution on the GPU.
                    selected = logits[active.to(logits.device)]
                    result['logp'] = torch.cat([
                        torch.log_softmax(selected[j:j+16].float(), -1).cpu()
                        for j in range(0, selected.shape[0], 16)])
                return result

            for step in range(a.step):
                active = block.eq(mask_id)
                if not active.any():
                    raise RuntimeError('Group finished before requested step.')
                result = forward(active)
                chosen = choose_candidate_mask(active.cpu(), result['confidence'], schedule[step], a.confidence_threshold)
                block[chosen.to(block.device)] = result['candidates'].to(block.device)[chosen.to(block.device)]
                del result
            active = block.eq(mask_id).cpu()
            if not active.any(dim=1).all():
                raise RuntimeError('Choose a step where every request remains active.')
            compress = torch.zeros_like(ids, dtype=torch.bool)
            compress[:, -a.block_length:] = active.to(ids.device)
            torch.save(dict(input_ids=ids.cpu(), attention_mask=attention.cpu(), position_ids=positions.cpu(),
                            request_ids=snapshot.get('request_ids', list(range(len(prompts))))[start:start+a.batch_size],
                            group=group, step=a.step, layer=a.layer), a.output_dir/f'state_group{group}.pt')
            reference = forward(active, True)

            def accepted(result):
                return choose_candidate_mask(active, result['confidence'], schedule[a.step],
                                             a.confidence_threshold) & result['candidates'].ne(mask_id)

            selected_ref = accepted(reference)
            experiments = [('noop', 0., 'joint')] + [(m, e, m) for e in a.epsilons if e > 0 for m in a.methods]
            for label, eps, method in experiments:
                hook = GateIntervention(compress, eps, method, not a.no_renormalize)
                handle = moe.gate.register_forward_hook(hook)
                try:
                    branch = forward(active, True)
                finally:
                    handle.remove()
                if hook.calls != 1:
                    raise RuntimeError(f'Expected one target gate call, got {hook.calls}.')
                metrics = prediction_metrics(reference, branch, active, selected_ref, accepted(branch))
                stream.write(json.dumps(dict(record_type='intervention', group=group, layer=a.layer, step=a.step,
                                            method=label, epsilon=eps, route_metrics_full_forward=hook.metrics,
                                            **metrics))+'\n')
                print(f'group={group} {label} eps={eps}: flip={metrics["candidate_flip_rate"]:.6f} '
                      f'KL={metrics["mean_kl"]:.6g}', flush=True)
                if label == 'noop' and (metrics['max_abs_logprob_delta'] > 1e-5 or
                                       metrics['candidate_flip_rate'] or metrics['accepted_added'] or metrics['accepted_lost']):
                    raise RuntimeError('No-op replay mismatch; stop and investigate before interpreting interventions.')
                del branch
            del reference
    print(f'Complete: {output}')


if __name__ == '__main__':
    main()
