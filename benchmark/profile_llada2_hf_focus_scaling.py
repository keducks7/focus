#!/usr/bin/env python3
"""Real batched HF LLaDA2 Vanilla/delayed/FOCUS expert observations; one block of 32."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from hf_focus_batch import PackedHF
from profile_focus_moe_scaling import summarize
from profile_llada2_expert_trajectory import choose_candidate_mask
from profile_llada2_hf_moe_saturation import _sample_block, _tokenize_prompts


def reference_logits(model, tokens, prompt_lengths, prefix, pad_id):
    """Unmodified HF full forward, used only as a Vanilla correctness check."""
    import torch
    batch, length = tokens.shape
    device = tokens.device
    pos = torch.arange(length, device=device)
    allow = (pos[:, None] // 32 >= pos[None, :] // 32)
    mask = torch.zeros((batch, 1, length, length), device=device, dtype=torch.bfloat16)
    mask.masked_fill_(~allow, float('-inf'))
    for request, prompt_length in enumerate(prompt_lengths):
        padding = prefix - prompt_length
        mask[request, :, :, :padding] = float('-inf')
        diagonal = torch.arange(padding, device=device)
        mask[request, 0, diagonal, diagonal] = 0
    result = model.model(input_ids=tokens, attention_mask=mask, position_ids=pos[None].expand(batch, -1),
                         use_cache=False, return_dict=True)
    return model.lm_head(result.last_hidden_state[:, prefix:]).float()


def check_forward(logits, coords, rows, shadow_outputs, prefix, atol, rtol):
    """Independent per-request state, same current tokens; test batching and cache isolation."""
    import torch
    report = []
    for request, (single_logits, single_coords, single_rows) in enumerate(shadow_outputs):
        selected = coords[:, 0] == request
        torch.testing.assert_close(coords[selected, 1].cpu(), single_coords[:, 1].cpu(), rtol=0, atol=0)
        expected = single_logits.to(logits.device)
        actual = logits[selected]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        report.append(float((actual - expected).abs().max()) if actual.numel() else 0.0)
    # The expert union must be the sum of request-local assignments, never the sum of expert counts.
    for row in rows:
        singles = [next(r for r in output[2] if r['layer_idx'] == row['layer_idx']) for output in shadow_outputs]
        expected = [sum(r['expert_load'][expert] for r in singles) for expert in range(row['num_experts'])]
        if row['expert_load'] != expected:
            raise AssertionError(f'Batch/single router discrepancy at layer {row["layer_idx"]}; inspect numerical routing boundaries')
    return max(report, default=0.0)


def run_group(model, prompts, mode, batch, group, args, stream):
    import torch
    device = model.model.word_embeddings.weight.device
    prefix = math.ceil(args.max_input_len / 32) * 32
    tokens = torch.full((batch, prefix + 32), args.pad_id, dtype=torch.long, device=device)
    tokens[:, prefix:] = args.mask_id
    prefix_coords = []
    for request, prompt in enumerate(prompts):
        tokens[request, prefix - len(prompt):prefix] = torch.tensor(prompt, device=device)
        prefix_coords.extend((request, position) for position in range(prefix - len(prompt), prefix))
    prefix_coords = torch.tensor(prefix_coords, device=device, dtype=torch.long)
    active = tokens[:, prefix:].eq(args.mask_id)
    meta = dict(mode=mode, configured_batch=batch, group_id=group, block_id=0, forward_id=-1)
    runner = PackedHF(model, batch, prefix, stream)
    runner.forward(tokens, prefix_coords, active, mode, [1.] * batch, args.focus_alpha, meta, prefill=True)
    verify = args.verify and group == 0 and batch == max(args.batch_sizes)
    shadows = []
    if verify:
        for request in range(batch):
            shadow = PackedHF(model, 1, prefix)
            selected = prefix_coords[prefix_coords[:, 0] == request].clone()
            selected[:, 0] = 0
            shadow.forward(tokens[request:request + 1], selected, active[request:request + 1], mode,
                           [1.], args.focus_alpha, meta, prefill=True)
            shadows.append(shadow)
    uncached = torch.ones((batch, 32), dtype=torch.bool, device=device)
    accepted_total = [0] * batch
    step_counts = [0] * batch
    checks = []
    for step in range(32):
        active = tokens[:, prefix:].eq(args.mask_id)
        alive = active.any(dim=1)
        if not bool(alive.any()):
            break
        process = (torch.ones_like(active) if mode == 'vanilla' else uncached) & alive[:, None]
        coords = process.nonzero(as_tuple=False)
        coords[:, 1] += prefix
        averages = [total / count if count else 1. for total, count in zip(accepted_total, step_counts)]
        meta['forward_id'] = step
        logits, final_coords, rows = runner.forward(tokens, coords, active, mode, averages, args.focus_alpha, meta)
        if verify:
            outputs = []
            # Verification uses B=1 forwards only as a reference, never as experimental data.
            for request, shadow in enumerate(shadows):
                selected = coords[coords[:, 0] == request].clone()
                selected[:, 0] = 0
                if not len(selected):
                    # This group is no longer full; cross-request test has already exercised its active prefix.
                    outputs = []
                    break
                outputs.append(shadow.forward(tokens[request:request + 1], selected, active[request:request + 1],
                                              mode, [averages[request]], args.focus_alpha, meta))
            if outputs:
                error = check_forward(logits, final_coords, rows, outputs, prefix, args.verify_atol, args.verify_rtol)
                if runner.progress != [shadow.progress[0] for shadow in shadows]:
                    raise AssertionError('Batch/single FOCUS progress mismatch')
                for layer_idx, (keys, values, valid) in runner.cache.items():
                    for request, shadow in enumerate(shadows):
                        single_k, single_v, single_valid = shadow.cache[layer_idx]
                        torch.testing.assert_close(valid[request:request + 1], single_valid, rtol=0, atol=0)
                        torch.testing.assert_close(keys[request:request + 1], single_k, atol=args.verify_atol, rtol=args.verify_rtol)
                        torch.testing.assert_close(values[request:request + 1], single_v, atol=args.verify_atol, rtol=args.verify_rtol)
                checks.append(dict(mode=mode, group=group, step=step, kind='batch_vs_single', max_abs_logit_error=error))
            if mode == 'vanilla':
                reference = reference_logits(model, tokens, list(map(len, prompts)), prefix, args.pad_id)
                expected = reference[final_coords[:, 0].to(reference.device),
                                     (final_coords[:, 1] - prefix).to(reference.device)].to(logits.device)
                torch.testing.assert_close(logits, expected, atol=args.verify_atol, rtol=args.verify_rtol)
                checks.append(dict(mode=mode, group=group, step=step, kind='packed_vs_original_hf',
                                   max_abs_logit_error=float((logits - expected).abs().max())))
        before = ~active
        right = torch.cat([before[:, 1:], torch.ones((batch, 1), device=device, dtype=torch.bool)], dim=1)
        uncached &= ~(before & right)
        candidate, confidence = _sample_block(logits, 0)
        final_coords = final_coords.to(device)
        selected_requests, selected_positions = final_coords[:, 0], final_coords[:, 1] - prefix
        eligible = torch.zeros_like(active)
        eligible[selected_requests, selected_positions] = active[selected_requests, selected_positions]
        full_confidence = torch.full((batch, 32), float('-inf'), device=device)
        full_candidates = tokens[:, prefix:].clone()
        full_confidence[selected_requests, selected_positions] = confidence.to(device)
        full_candidates[selected_requests, selected_positions] = candidate.to(device)
        transfer = choose_candidate_mask(eligible, full_confidence, 1, args.confidence)
        accepted = transfer & full_candidates.ne(args.mask_id)
        tokens[:, prefix:][transfer] = full_candidates[transfer]
        for request in range(batch):
            if alive[request]:
                accepted_total[request] += int(accepted[request].sum())
                step_counts[request] += 1
        print(f'{mode} B={batch} group={group} step={step} '
              f'Q={len(coords)}->{len(final_coords)} masks={int(tokens[:, prefix:].eq(args.mask_id).sum())}', flush=True)
    stream.write(json.dumps(dict(record_type='generation_result', mode=mode, group_id=group,
                                 generated_token_ids=tokens[:, prefix:].cpu().tolist(),
                                 remaining_masks=tokens[:, prefix:].eq(args.mask_id).sum(1).cpu().tolist())) + '\n')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_path')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--modes', nargs='+', choices=['vanilla', 'delayed', 'focus'], default=['vanilla', 'delayed', 'focus'])
    parser.add_argument('--dataset', default='openai/gsm8k')
    parser.add_argument('--num-prompts', type=int, default=32)
    parser.add_argument('--max-input-len', type=int, default=128)
    parser.add_argument('--max-scan-examples', type=int, default=20000)
    parser.add_argument('--focus-alpha', type=float, default=1.0)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-memory-per-gpu', default='38GiB')
    parser.add_argument('--verify', action='store_true', help='Check first group at largest B against single-request execution and original HF Vanilla')
    parser.add_argument('--verify-atol', type=float, default=0.15)
    parser.add_argument('--verify-rtol', type=float, default=0.01)
    args = parser.parse_args()
    if args.num_prompts <= 0 or any(b <= 0 or args.num_prompts % b for b in args.batch_sizes):
        parser.error('Batch sizes must be positive divisors of num-prompts')
    if args.focus_alpha < 1 or not 0 <= args.confidence <= 1:
        parser.error('Require alpha>=1 and confidence in [0,1]')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    args.dataset_format, args.hf_config, args.hf_split = 'gsm8k', 'main', 'test'
    prompts = _tokenize_prompts(args, tokenizer)
    if len(prompts) != args.num_prompts:
        raise RuntimeError('Too few prompts survived filtering; increase --max-input-len')
    (args.output_dir / 'prompt_token_ids.json').write_text(json.dumps(prompts))
    print('Loading HuggingFace weights with balanced layer placement...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                device_map='balanced', low_cpu_mem_usage=True, attn_implementation='eager',
                                                max_memory={i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())})
    if any(str(device) in ('cpu', 'disk') for device in model.hf_device_map.values()):
        raise RuntimeError('This runner requires all layers resident on GPUs; CPU/disk offload is unsupported')
    model.eval()
    torch.manual_seed(args.seed)
    args.mask_id = tokenizer.mask_token_id
    args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.mask_id is None or args.pad_id is None:
        raise RuntimeError('Tokenizer missing mask or pad/eos IDs')
    source = Path(__file__).resolve().parents[1] / 'lmdeploy/pytorch/kernels/cuda/focus.py'
    metadata = dict(vars(args), backend='huggingface', batching='physical packed QKV/MoE and batched padded attention',
                    focus_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    block_length=32, max_generated_length=32, eos_early_stop=False,
                    prefix_slots=math.ceil(args.max_input_len / 32) * 32,
                    torch=torch.__version__, transformers=__import__('transformers').__version__,
                    device_map=model.hf_device_map, status='experimental semantic port, server verification required')
    (args.output_dir / 'experiment.json').write_text(json.dumps(metadata, default=str, indent=2))
    verification = []
    with torch.inference_mode():
        for batch in args.batch_sizes:
            for mode in args.modes:
                directory = args.output_dir / f'{mode}_bs{batch}'
                directory.mkdir()
                with (directory / 'routes.jsonl').open('w', buffering=1) as stream:
                    stream.write(json.dumps(dict(record_type='metadata', mode=mode, batch=batch, backend='huggingface')) + '\n')
                    for group, start in enumerate(range(0, len(prompts), batch)):
                        verification.extend(run_group(model, prompts[start:start + batch], mode, batch, group, args, stream))
                        (args.output_dir / 'verification.json').write_text(json.dumps(verification, indent=2))
                (directory / 'complete.json').write_text(json.dumps({'groups': len(prompts) // batch}))
    summarize(args.output_dir, args.modes, args.batch_sizes)
    print(f'Complete: {args.output_dir}/scaling.csv', flush=True)


if __name__ == '__main__':
    main()
