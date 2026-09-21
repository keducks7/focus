"""Batched extension of the repository's Vanilla block/KV decode loop.

Left padding is in WHOLE blocks, preserving every request's original block
boundaries and rotary positions, including its partial prompt block.
No within-block KV reuse or FOCUS eviction. Static batch; finished rows remain
allocated until the batch completes, and their cost is included in timing.
"""
import math
import time

import torch


def repository_eager_mask(additive_mask, implementation):
    """Adapt decoder masks to the vendored eager model's binary-mask API.

    Its _prepare_4d_causal_attention_mask converts 1/0 into 0/dtype.min.
    Passing our additive 0/-inf mask directly would mask every position.
    SDPA uses a different contract and must not silently take this path.
    """
    if implementation != 'eager':
        raise ValueError('Shared batch decode requires repository eager attention.')
    return additive_mask.eq(0).to(dtype=additive_mask.dtype)


def prepare_prefix(prompts, block_length, pad_id, device, dtype):
    if not prompts or any(not row for row in prompts):
        raise ValueError('Nonempty prompts required.')
    lengths = [len(p)//block_length*block_length for p in prompts]
    width = max(lengths)
    padding = [width-n for n in lengths]
    ids = torch.full((len(prompts), width), pad_id, device=device, dtype=torch.long)
    positions = torch.zeros_like(ids)
    allow = torch.arange(width, device=device)//block_length
    base = allow[:, None] >= allow[None, :]
    attention = torch.zeros((len(prompts), 1, width, width), dtype=dtype, device=device)
    attention.masked_fill_(~base[None, None], float('-inf'))
    for i, (prompt, n, pad) in enumerate(zip(prompts, lengths, padding)):
        if n:
            ids[i, pad:] = torch.tensor(prompt[:n], device=device)
            positions[i, pad:] = torch.arange(n, device=device)
        if pad:
            attention[i, :, :, :pad] = float('-inf')
            diagonal = torch.arange(pad, device=device)
            attention[i, 0, diagonal, diagonal] = 0
    return ids, positions, attention, lengths, padding


def candidate_transfer(active, confidence, minimum, threshold):
    selected = torch.zeros_like(active)
    for i in range(active.shape[0]):
        n = int(active[i].sum())
        if not n:
            continue
        high = active[i] & (confidence[i] > threshold)
        if int(high.sum()) >= minimum:
            selected[i] = high
        else:
            ranks = confidence[i].masked_fill(~active[i], float('-inf')).topk(min(minimum, n)).indices
            selected[i, ranks] = True
    return selected


def synchronize_model(model):
    devices = {p.device for p in model.parameters() if p.device.type == 'cuda'}
    for device in devices:
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_batch(model, prompts, *, mask_id, pad_id, eos_id, gen_length,
                   block_length, steps, threshold, controller, cache_factory,
                   sampling='native', temperature=0., top_k=0, top_p=1.):
    if gen_length <= 0 or block_length <= 0 or not 1 <= steps <= block_length:
        raise ValueError('Positive lengths and 1 <= steps <= block_length required.')
    if sampling not in ('native', 'greedy') or (sampling == 'greedy' and temperature != 0):
        raise ValueError('Choose native sampling or greedy with temperature=0.')
    if any(mask_id in p for p in prompts):
        raise ValueError('Prompt contains MASK token; ambiguous generation-state classification.')
    core = model.model
    implementation = getattr(getattr(core, 'config', None), '_attn_implementation', None)
    if implementation != 'eager':
        raise ValueError('Shared batch decode requires repository eager attention.')
    device = core.word_embeddings.weight.device
    dtype = core.word_embeddings.weight.dtype
    ids, positions, attention, prefix_lengths, padding = prepare_prefix(
        prompts, block_length, pad_id, device, dtype)
    prefix_width = ids.shape[1]
    batch = len(prompts)
    remainders = [len(p)-n for p,n in zip(prompts, prefix_lengths)]
    blocks = [math.ceil((r+gen_length)/block_length) for r in remainders]
    cache = cache_factory()
    history = [[] for _ in prompts]
    outputs = [None for _ in prompts]
    finished = [False]*batch
    stats = {f'{phase}_{metric}': 0 for phase in ('prefill','denoise','commit')
             for metric in ('seconds','forwards')}
    stats['request_denoising_steps'] = [0]*batch
    controller.reset_stats()
    synchronize_model(model)
    started = time.perf_counter()

    def forward(x, attn, pos, store, phase, compress=None):
        controller.compress = compress
        synchronize_model(model)
        start = time.perf_counter()
        try:
            model_mask = repository_eager_mask(attn, implementation)
            out = core(input_ids=x, attention_mask=model_mask, position_ids=pos,
                       past_key_values=cache, use_cache=True, store_kv=store,
                       output_router_logits=False, return_dict=True)
            # Native LLaDA2ModelLM.forward exposes float32 logits to its sampler.
            logits = model.lm_head(out.last_hidden_state).float() if phase == 'denoise' else None
            synchronize_model(model)
        finally:
            controller.compress = None
        stats[f'{phase}_seconds'] += time.perf_counter()-start
        stats[f'{phase}_forwards'] += 1
        return logits

    if prefix_width:
        forward(ids, attention, positions, True, 'prefill')
    del ids, attention, positions
    schedule = model._get_num_transfer_tokens(block_length, steps).cpu().tolist()
    for block_idx in range(max(blocks)):
        current = torch.full((batch, block_length), mask_id, dtype=torch.long, device=device)
        if block_idx == 0:
            for i, (p,n,r) in enumerate(zip(prompts, prefix_lengths, remainders)):
                if r:
                    current[i,:r] = torch.tensor(p[n:], device=device)
        for i, done in enumerate(finished):
            if done:
                current[i] = pad_id
        pos = torch.tensor(prefix_lengths, device=device)[:,None] + block_idx*block_length + torch.arange(block_length,device=device)[None]
        attn = torch.zeros((batch,1,block_length,prefix_width+(block_idx+1)*block_length), device=device,dtype=dtype)
        for i,pad in enumerate(padding):
            attn[i,:,:, :pad] = float('-inf')
        for step in range(steps+1):
            alive = torch.tensor([not v for v in finished], device=device)[:,None]
            active = current.eq(mask_id) & alive
            if not active.any():
                break
            logits = forward(current, attn, pos, False, 'denoise', active)
            if sampling == 'native':
                candidates, confidence = model._sample_with_temperature_topk_topp(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p)
            else:
                prob = torch.softmax(logits.float(), -1)
                confidence, candidates = prob.max(-1)
                del prob
            candidates, confidence = candidates.to(device), confidence.to(device)
            selected = candidate_transfer(active, confidence, schedule[min(step,len(schedule)-1)], threshold)
            current[selected] = candidates[selected]
            for i in range(batch):
                if active[i].any():
                    stats['request_denoising_steps'][i] += 1
                if finished[i]:
                    continue
                begin = remainders[i] if block_idx == 0 else 0
                visible = history[i] + current[i,begin:].cpu().tolist()
                visible = visible[:gen_length]
                if eos_id is not None and eos_id in visible:
                    end = visible.index(eos_id)+1
                    if mask_id not in visible[:end]:
                        outputs[i] = visible[:end]
                        finished[i] = True
            del logits, candidates, confidence
            if all(finished):
                break
        if all(finished):
            break
        for i in range(batch):
            if finished[i]:
                continue
            begin = remainders[i] if block_idx == 0 else 0
            values = current[i,begin:].cpu().tolist()
            if mask_id in values:
                raise RuntimeError(f'Unresolved MASK after denoising budget, request={i}, block={block_idx}; no silent fallback.')
            history[i].extend(values)
            if len(history[i]) >= gen_length:
                outputs[i] = history[i][:gen_length]
                finished[i] = True
        if all(finished):
            break
        # Like the repository wrapper: recompute a complete block to store KV.
        # No approximation on this commit pass (all positions now contextual).
        forward(current, attn, pos, True, 'commit')
    if any(v is None for v in outputs):
        raise RuntimeError('Incomplete generated outputs.')
    synchronize_model(model)
    stats.update(inference_seconds=time.perf_counter()-started,
                 assignments_original=controller.assignments_original,
                 assignments_executed=controller.assignments_executed,
                 compressed_layer_calls=controller.compressed_layer_calls,
                 batch_size=batch, generated_tokens=sum(len(v) for v in outputs),
                 output_token_lengths=[len(v) for v in outputs])
    return outputs, stats
