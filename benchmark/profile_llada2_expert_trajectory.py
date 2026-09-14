#!/usr/bin/env python3
"""Collect token-level MoE trajectories and denoising-step expert similarity.

The first version intentionally profiles one generated block.  Every record
already carries ``block_id`` and global generation positions so the on-disk
format can be extended to multiple blocks without changing token identity.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from profile_llada2_hf_moe_saturation import (
    GSM8K_DATASET_ID,
    HUMANEVAL_DATASET_ID,
    MBPP_DATASET_ID,
    _build_batch_inputs,
    _router_layers,
    _sample_block,
    _tokenize_prompts,
    _transfer_schedule,
)


def choose_candidate_mask(active_mask, confidence, minimum_transfer, threshold):
    """Return the exact per-request token selection used for this step."""
    import torch

    selected = torch.zeros_like(active_mask)
    for batch_index in range(active_mask.shape[0]):
        active = active_mask[batch_index]
        active_count = int(active.sum().item())
        if active_count == 0:
            continue
        high_confidence = active & (confidence[batch_index] > threshold)
        if int(high_confidence.sum().item()) >= minimum_transfer:
            selected[batch_index] = high_confidence
        else:
            count = min(minimum_transfer, active_count)
            masked_confidence = confidence[batch_index].masked_fill(~active, float('-inf'))
            indices = torch.topk(masked_confidence, k=count).indices
            selected[batch_index, indices] = True
    return selected


def token_route_records(router_outputs, active_mask, first_moe_layer, routed_scaling_factor=1.0):
    """Return per-token routes in deterministic batch-major position order."""
    coordinates = active_mask.nonzero(as_tuple=False).to('cpu').tolist()
    records = [{'batch_index': int(batch), 'block_position': int(position), 'layers': []}
               for batch, position in coordinates]
    for offset, router_output in enumerate(router_outputs or ()):
        if not isinstance(router_output, (tuple, list)) or len(router_output) < 2:
            raise RuntimeError('Expected each router output to contain logits and top-k expert IDs.')
        router_logits, topk_ids = router_output[0], router_output[1]
        block_length = active_mask.shape[1]
        layer_mask = active_mask.to(topk_ids.device)
        selected_ids = topk_ids[:, -block_length:, :][layer_mask]
        selected_logits = router_logits[:, -block_length:, :][layer_mask]
        selected_logits = selected_logits.gather(1, selected_ids)
        selected_scores = selected_logits.sigmoid()
        if selected_scores.shape[1] > 1:
            selected_weights = selected_scores / selected_scores.sum(dim=1, keepdim=True).clamp_min(1e-20)
        else:
            selected_weights = selected_scores
        selected_weights = selected_weights * routed_scaling_factor
        ids_cpu = selected_ids.to('cpu').tolist()
        logits_cpu = selected_logits.float().to('cpu').tolist()
        weights_cpu = selected_weights.float().to('cpu').tolist()
        if len(ids_cpu) != len(records):
            raise RuntimeError('Router token order does not match active-mask token order.')
        for record, expert_ids, logits, weights in zip(records, ids_cpu, logits_cpu, weights_cpu):
            record['layers'].append({
                'layer_idx': first_moe_layer + offset,
                'expert_ids': [int(value) for value in expert_ids],
                'router_logits': [round(float(value), 7) for value in logits],
                'router_weights': [round(float(value), 7) for value in weights],
            })
    return records


class HiddenStateCollector:
    """Capture target-layer MoE inputs for selected denoising steps."""

    def __init__(self, block_length, selected_steps):
        self.block_length = block_length
        self.selected_steps = frozenset(selected_steps)
        self.pending = None
        self.hidden_by_step = defaultdict(dict)

    def prepare(self, step, active_mask, token_ids):
        self.pending = None
        if step in self.selected_steps:
            self.pending = (step, active_mask.detach(), list(token_ids))

    def hook(self, _module, inputs):
        if self.pending is None:
            return
        step, active_mask, token_ids = self.pending
        hidden = inputs[0][:, -self.block_length:, :]
        selected = hidden[active_mask.to(hidden.device)].detach().to('cpu')
        if selected.shape[0] != len(token_ids):
            raise RuntimeError('Captured hidden-state count does not match token identities.')
        destination = self.hidden_by_step[step]
        for token_id, vector in zip(token_ids, selected):
            if token_id in destination:
                raise RuntimeError(f'Duplicate hidden state for token {token_id} at step {step}.')
            destination[token_id] = vector
        self.pending = None

    def save_raw(self, output_dir, batch_size, layer_idx):
        import torch

        for step, values in sorted(self.hidden_by_step.items()):
            token_ids = sorted(values)
            payload = {
                'format_version': 1,
                'batch_size': batch_size,
                'layer_idx': layer_idx,
                'step': step,
                'token_ids': token_ids,
                'hidden_states': torch.stack([values[token_id] for token_id in token_ids]),
            }
            torch.save(payload, output_dir / f'layer{layer_idx}_hidden_step{step}_bs{batch_size}.pt')


def _off_diagonal_values(matrix):
    import torch

    count = matrix.shape[0]
    mask = ~torch.eye(count, dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def matrix_pearson(left, right):
    left_values = _off_diagonal_values(left).double()
    right_values = _off_diagonal_values(right).double()
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = left_values.norm() * right_values.norm()
    return float((left_values @ right_values / denominator).item()) if denominator.item() else 0.0


def nearest_experts(similarity):
    import torch

    without_self = similarity.clone()
    without_self.fill_diagonal_(float('-inf'))
    return torch.argmax(without_self, dim=1)


def evaluate_all_experts(expert_bank, hidden_states, progress_label=''):
    """Evaluate every routed expert on the same hidden-state cohort."""
    import torch
    import torch.nn.functional as functional

    device = next(expert_bank[0].parameters()).device
    inputs = hidden_states.to(device)
    outputs = []
    with torch.inference_mode():
        for index, expert in enumerate(expert_bank):
            outputs.append(expert(inputs).to(torch.bfloat16))
            if (index + 1) % 32 == 0 or index + 1 == len(expert_bank):
                print(f'{progress_label} experts={index + 1}/{len(expert_bank)}', flush=True)
    stacked = torch.stack(outputs)
    flattened = stacked.float().flatten(1)
    normalized = functional.normalize(flattened, dim=1)
    similarity = normalized @ normalized.T
    return stacked, similarity


def compute_similarity_outputs(collector, expert_bank, selected_steps, sample_limit,
                               seed, output_dir, batch_size, layer_idx):
    """Use one token cohort present at every requested step for fair comparison."""
    import torch

    available_steps = [step for step in selected_steps if collector.hidden_by_step.get(step)]
    if len(available_steps) < 2:
        raise RuntimeError('At least two selected similarity steps must contain captured tokens.')
    common_ids = set(collector.hidden_by_step[available_steps[0]])
    for step in available_steps[1:]:
        common_ids.intersection_update(collector.hidden_by_step[step])
    common_ids = sorted(common_ids)
    if not common_ids:
        raise RuntimeError('No token remains unresolved across all selected similarity steps.')
    generator = torch.Generator().manual_seed(seed)
    if len(common_ids) > sample_limit:
        order = torch.randperm(len(common_ids), generator=generator)[:sample_limit].tolist()
        common_ids = sorted(common_ids[index] for index in order)

    matrices = {}
    nearest = {}
    step_rows = []
    reference_nearest = None
    for step in available_steps:
        states = torch.stack([collector.hidden_by_step[step][token_id] for token_id in common_ids])
        outputs, similarity = evaluate_all_experts(
            expert_bank, states, f'similarity layer={layer_idx} step={step}')
        mapping = nearest_experts(similarity)
        if reference_nearest is None:
            reference_nearest = mapping
        difference = (outputs.float() - outputs[reference_nearest].float()).flatten(1).norm(dim=1)
        denominator = outputs.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        replacement_error = float((difference / denominator).mean().item())
        matrices[step] = similarity.detach().to('cpu')
        nearest[step] = mapping.detach().to('cpu')
        torch.save({
            'format_version': 1,
            'batch_size': batch_size,
            'layer_idx': layer_idx,
            'step': step,
            'matched_token_ids': common_ids,
            'similarity': matrices[step],
            'nearest_expert': nearest[step],
        }, output_dir / f'expert_similarity_layer{layer_idx}_step{step}_bs{batch_size}.pt')
        step_rows.append({
            'batch_size': batch_size,
            'layer_idx': layer_idx,
            'step': step,
            'matched_tokens': len(common_ids),
            'mean_step0_mapping_relative_error': replacement_error,
        })
        del outputs, similarity, states
        torch.cuda.empty_cache()

    pair_rows = []
    for index, step_a in enumerate(available_steps):
        for step_b in available_steps[index + 1:]:
            pair_rows.append({
                'batch_size': batch_size,
                'layer_idx': layer_idx,
                'step_a': step_a,
                'step_b': step_b,
                'step_distance': step_b - step_a,
                'matched_tokens': len(common_ids),
                'similarity_matrix_pearson': matrix_pearson(matrices[step_a], matrices[step_b]),
                'nearest_expert_consistency': float((nearest[step_a] == nearest[step_b]).float().mean().item()),
            })
    _write_csv(output_dir / f'expert_similarity_steps_bs{batch_size}.csv', step_rows)
    _write_csv(output_dir / f'expert_similarity_pairs_bs{batch_size}.csv', pair_rows)
    (output_dir / f'matched_tokens_bs{batch_size}.json').write_text(json.dumps({
        'format_version': 1,
        'layer_idx': layer_idx,
        'steps': available_steps,
        'matched_token_count': len(common_ids),
        'token_ids': common_ids,
    }, indent=2), encoding='utf-8')


def _write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _configure_dataset(args):
    if args.dataset == GSM8K_DATASET_ID:
        args.dataset_format = 'gsm8k'
        args.hf_config = args.hf_config or 'main'
        args.hf_split = args.hf_split or 'test'
    elif args.dataset == MBPP_DATASET_ID:
        args.dataset_format = 'mbpp'
        args.hf_config = args.hf_config or 'sanitized'
        args.hf_split = args.hf_split or 'test'
    elif args.dataset == HUMANEVAL_DATASET_ID:
        args.dataset_format = 'auto'
        args.hf_split = args.hf_split or 'test'
    else:
        args.hf_split = args.hf_split or 'train'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('model_path')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-prompts', type=int, default=32)
    parser.add_argument('--max-input-len', type=int, default=128)
    parser.add_argument('--max-scan-examples', type=int, default=20000)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--gen-length', type=int, default=32)
    parser.add_argument('--denoising-steps', type=int, default=32)
    parser.add_argument('--confidence-threshold', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--similarity-layer', type=int, default=10)
    parser.add_argument('--similarity-steps', nargs='+', type=int, default=[0, 1, 2, 3, 4, 8, 12])
    parser.add_argument('--similarity-samples', type=int, default=64)
    parser.add_argument('--dataset-format', choices=['auto', 'gsm8k', 'mbpp', 'math'], default='auto')
    parser.add_argument('--hf-split', default=None)
    parser.add_argument('--hf-config', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-memory-per-gpu', default='38GiB')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gen_length != args.block_length:
        raise ValueError('This first implementation profiles one block: --gen-length must equal --block-length. '
                         'The trace schema already includes block_id for the planned multi-block extension.')
    if min(args.batch_size, args.num_prompts, args.block_length, args.denoising_steps,
           args.similarity_samples) <= 0:
        raise ValueError('Batch, prompt, length, step, and sample counts must be positive.')
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError('confidence threshold must be in [0, 1].')
    if min(args.similarity_steps) < 0:
        raise ValueError('similarity steps must be non-negative.')
    _configure_dataset(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() < 2:
        raise RuntimeError(f'This profiler expects two visible GPUs, found {torch.cuda.device_count()}.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / f'token_trajectories_bs{args.batch_size}.jsonl'
    if trajectory_path.exists():
        raise FileExistsError(f'Use a new output directory; refusing to overwrite {trajectory_path}.')

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_ids = _tokenize_prompts(args, tokenizer)
    full_groups = len(prompt_ids) // args.batch_size
    if full_groups == 0:
        raise RuntimeError('Not enough prompts for one full batch.')
    prompt_ids = prompt_ids[:full_groups * args.batch_size]
    prompt_snapshot_path = args.output_dir / 'sampled_prompt_token_ids.json'
    prompt_snapshot_path.write_text(json.dumps({
        'format_version': 1,
        'dataset': args.dataset,
        'dataset_format': args.dataset_format,
        'hf_config': args.hf_config,
        'hf_split': args.hf_split,
        'seed': args.seed,
        'max_input_len': args.max_input_len,
        'request_ids': list(range(len(prompt_ids))),
        'prompt_token_ids': prompt_ids,
    }, indent=2), encoding='utf-8')
    max_memory = {0: args.max_memory_per_gpu, 1: args.max_memory_per_gpu}
    print(f'Sampled {len(prompt_ids)} prompts; loading LLaDA2 across two GPUs...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='balanced',
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        attn_implementation='eager',
    )
    model.eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = model.config
    core_model = model.model
    input_device = core_model.word_embeddings.weight.device
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError('Tokenizer has no mask_token_id.')
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError('Tokenizer has neither pad_token_id nor eos_token_id.')
    first_moe_layer = int(getattr(config, 'first_k_dense_replace', 0))
    target_layer = core_model.layers[args.similarity_layer]
    target_moe = target_layer.mlp
    if not hasattr(target_moe, 'experts'):
        raise RuntimeError(f'Layer {args.similarity_layer} is not an MoE layer with an expert bank.')
    collector = HiddenStateCollector(args.block_length, args.similarity_steps)
    hook_handle = target_moe.register_forward_pre_hook(collector.hook)
    schedule = _transfer_schedule(args.block_length, args.denoising_steps)

    metadata = {
        'record_type': 'metadata',
        'format_version': 1,
        'model_type': 'llada2_moe_token_trajectory',
        'model_path': args.model_path,
        'dataset': args.dataset,
        'dataset_format': args.dataset_format,
        'hf_config': args.hf_config,
        'hf_split': args.hf_split,
        'seed': args.seed,
        'max_input_len': args.max_input_len,
        'configured_batch_size': args.batch_size,
        'num_prompts': full_groups * args.batch_size,
        'num_experts': int(config.num_experts),
        'top_k': int(config.num_experts_per_tok),
        'first_moe_layer': first_moe_layer,
        'num_hidden_layers': int(config.num_hidden_layers),
        'block_length': args.block_length,
        'gen_length': args.gen_length,
        'num_generation_blocks': 1,
        'multi_block_schema': True,
        'similarity_layer': args.similarity_layer,
        'similarity_steps': args.similarity_steps,
        'confidence_threshold': args.confidence_threshold,
        'temperature': args.temperature,
        'observed_region': 'unresolved_mask_queries',
        'prompt_snapshot': prompt_snapshot_path.name,
        'similarity_cohort': 'same tokens unresolved at every available selected step',
    }

    try:
        with trajectory_path.open('w', encoding='utf-8', buffering=1) as stream:
            stream.write(json.dumps(metadata, separators=(',', ':')) + '\n')
            for group_id in range(full_groups):
                start = group_id * args.batch_size
                group = prompt_ids[start:start + args.batch_size]
                input_ids, attention_mask, position_ids = _build_batch_inputs(
                    group, args.block_length, pad_id, mask_id, input_device, torch.bfloat16)
                current_block = input_ids[:, -args.block_length:]
                for step, minimum_transfer in enumerate(schedule):
                    active_mask = current_block.eq(mask_id)
                    coordinates = active_mask.nonzero(as_tuple=False).to('cpu').tolist()
                    if not coordinates:
                        break
                    token_ids = [(start + int(batch), 0, int(position)) for batch, position in coordinates]
                    collector.prepare(step, active_mask, token_ids)
                    with torch.inference_mode():
                        outputs = core_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                            output_router_logits=True,
                            return_dict=True,
                        )
                        layer_histograms = _router_layers(
                            outputs.router_logits, active_mask, int(config.num_experts), first_moe_layer)
                        route_records = token_route_records(
                            outputs.router_logits,
                            active_mask,
                            first_moe_layer,
                            float(getattr(config, 'routed_scaling_factor', 1.0)),
                        )
                        block_hidden = outputs.last_hidden_state[:, -args.block_length:, :]
                        logits = model.lm_head(block_hidden)
                        candidates, confidence = _sample_block(logits, args.temperature)
                        confidence_local = confidence.to(current_block.device)
                        candidate_local = candidates.to(current_block.device)
                        selected = choose_candidate_mask(
                            active_mask, confidence_local, minimum_transfer, args.confidence_threshold)
                        accepted_mask = selected & candidate_local.ne(mask_id)
                        current_block[selected] = candidate_local[selected]

                    confidence_cpu = confidence_local.to('cpu')
                    candidates_cpu = candidate_local.to('cpu')
                    selected_cpu = selected.to('cpu')
                    accepted_cpu = accepted_mask.to('cpu')
                    for record in route_records:
                        batch_index = record.pop('batch_index')
                        position = record['block_position']
                        record.update({
                            'request_id': start + batch_index,
                            'block_id': 0,
                            'generation_position': position,
                            'candidate_token_id': int(candidates_cpu[batch_index, position]),
                            'confidence': round(float(confidence_cpu[batch_index, position]), 8),
                            'selected_for_transfer': bool(selected_cpu[batch_index, position]),
                            'accepted': bool(accepted_cpu[batch_index, position]),
                        })
                    remaining = current_block.eq(mask_id).sum(dim=1).to('cpu').tolist()
                    stream.write(json.dumps({
                        'record_type': 'token_trajectory_step',
                        'group_id': group_id,
                        'block_id': 0,
                        'step': step,
                        'query_tokens': len(route_records),
                        'minimum_transfer': minimum_transfer,
                        'remaining_after_per_sequence': remaining,
                        'tokens': route_records,
                        'layer_histograms': layer_histograms,
                    }, separators=(',', ':')) + '\n')
                    print(f'group={group_id + 1}/{full_groups} step={step} '
                          f'Q={len(route_records)}->{sum(remaining)}', flush=True)
                    del outputs, block_hidden, logits, candidates, confidence
                    del confidence_local, candidate_local, selected, accepted_mask
                    del route_records, layer_histograms
                stream.write(json.dumps({
                    'record_type': 'generation_result',
                    'group_id': group_id,
                    'block_id': 0,
                    'request_ids': list(range(start, start + args.batch_size)),
                    'generated_token_ids': current_block.to('cpu').tolist(),
                    'remaining_masks': current_block.eq(mask_id).sum(dim=1).to('cpu').tolist(),
                }, separators=(',', ':')) + '\n')
                del current_block, input_ids, attention_mask, position_ids
    finally:
        hook_handle.remove()

    collector.save_raw(args.output_dir, args.batch_size, args.similarity_layer)
    compute_similarity_outputs(
        collector,
        target_moe.experts,
        args.similarity_steps,
        args.similarity_samples,
        args.seed,
        args.output_dir,
        args.batch_size,
        args.similarity_layer,
    )
    print(f'Complete: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
