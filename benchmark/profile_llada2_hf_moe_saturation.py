#!/usr/bin/env python3
"""Profile full single-block denoising MoE routing with HF Accelerate.

This path intentionally avoids LMDeploy's multi-process executor.  The model is
loaded once with a balanced two-GPU device map, and LLaDA2's official
``output_router_logits`` result is used instead of a model-side tracing hook.
Only unresolved mask queries are counted at each denoising step.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional


GSM8K_DATASET_ID = 'openai/gsm8k'
MBPP_DATASET_ID = 'google-research-datasets/mbpp'
HUMANEVAL_DATASET_ID = 'openai/openai_humaneval'


def _extract_prompt(row: Dict, dataset_format: str) -> Optional[str]:
    """Extract one user prompt from a supported dataset row."""
    if dataset_format == 'gsm8k':
        value = row.get('question')
        return str(value).strip() if value is not None else None
    if dataset_format == 'mbpp':
        task = row.get('prompt', row.get('text'))
        if task is None:
            return None
        prompt = f'You are an expert Python programmer.\n{str(task).strip()}'
        tests = row.get('test_list')
        if isinstance(tests, (list, tuple)):
            tests_text = '\n'.join(str(item).strip() for item in tests if str(item).strip())
            if tests_text:
                prompt += f'\nYour code should pass these tests:\n\n{tests_text}'
        return prompt + '\nReturn only the completed Python code.'
    if dataset_format == 'math':
        value = row.get('problem', row.get('question'))
        return str(value).strip() if value is not None else None

    for key in ('question', 'problem', 'prompt', 'text'):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    turns = row.get('conversation', row.get('conversations', row.get('messages')))
    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get('role', turn.get('from', turn.get('speaker', '')))).lower()
            content = turn.get('content', turn.get('value', turn.get('text')))
            if role in ('user', 'human') and content is not None and str(content).strip():
                return str(content).strip()
    return None


def _iter_local_rows(path: Path) -> Iterable[Dict]:
    if path.suffix == '.jsonl':
        with path.open('r', encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return
    with path.open('r', encoding='utf-8') as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f'Expected a JSON array in {path}, got {type(rows).__name__}.')
    yield from rows


def _load_rows(args) -> Iterable[Dict]:
    path = Path(args.dataset)
    if path.is_file():
        rows = list(_iter_local_rows(path))
        random.Random(args.seed).shuffle(rows)
        return rows

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('HuggingFace datasets require `pip install datasets`.') from exc

    kwargs = {'split': args.hf_split}
    if args.hf_config:
        kwargs['name'] = args.hf_config
    dataset = load_dataset(args.dataset, **kwargs)
    try:
        dataset = dataset.shuffle(seed=args.seed)
    except Exception:
        pass
    return dataset


def _tokenize_prompts(args, tokenizer) -> List[List[int]]:
    tokenized: List[List[int]] = []
    scanned = 0
    for row in _load_rows(args):
        scanned += 1
        if scanned > args.max_scan_examples or len(tokenized) >= args.num_prompts:
            break
        if not isinstance(row, dict):
            continue
        prompt = _extract_prompt(row, args.dataset_format)
        if not prompt:
            continue
        try:
            ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        except (AttributeError, ValueError, TypeError):
            ids = tokenizer.encode(prompt, add_special_tokens=True)
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(token) for token in ids]
        if 4 <= len(ids) <= args.max_input_len:
            tokenized.append(ids)
    if not tokenized:
        raise RuntimeError('No prompts survived tokenization and the input-length filter.')
    return tokenized


def _build_batch_inputs(prompt_ids, block_length, pad_id, mask_id, device, dtype):
    """Left-pad prompts and append one aligned all-mask diffusion block."""
    import torch

    max_prompt = max(len(ids) for ids in prompt_ids)
    prefix_slots = math.ceil(max_prompt / block_length) * block_length
    total_length = prefix_slots + block_length
    batch_size = len(prompt_ids)
    input_ids = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=device)
    padding_lengths = []
    for index, ids in enumerate(prompt_ids):
        padding = prefix_slots - len(ids)
        padding_lengths.append(padding)
        input_ids[index, padding:prefix_slots] = torch.tensor(ids, dtype=torch.long, device=device)
    input_ids[:, prefix_slots:] = mask_id

    num_blocks = total_length // block_length
    block_allow = torch.tril(torch.ones((num_blocks, num_blocks), dtype=torch.bool, device=device))
    allow = block_allow.repeat_interleave(block_length, 0).repeat_interleave(block_length, 1)
    attention_mask = torch.zeros((batch_size, 1, total_length, total_length), dtype=dtype, device=device)
    attention_mask.masked_fill_(~allow.view(1, 1, total_length, total_length), float('-inf'))

    # Padding keys must not influence real prompt/mask queries.  A padding query
    # can attend to itself so its hidden state remains finite across layers.
    for batch_index, padding in enumerate(padding_lengths):
        if padding:
            attention_mask[batch_index, 0, :, :padding] = float('-inf')
            diag = torch.arange(padding, device=device)
            attention_mask[batch_index, 0, diag, diag] = 0

    position_ids = torch.arange(total_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    return input_ids, attention_mask, position_ids


def _router_layers(router_outputs, active_mask, num_experts, first_moe_layer):
    """Count expert assignments for unresolved mask positions only."""
    import torch

    layers = []
    for offset, router_output in enumerate(router_outputs or ()):
        if router_output is None:
            continue
        if isinstance(router_output, (tuple, list)) and len(router_output) >= 2:
            topk_ids = router_output[1]
        else:
            raise RuntimeError('Unexpected LLaDA2 router output; expected (router_logits, topk_ids).')
        block_topk = topk_ids[:, -active_mask.shape[1]:, :]
        selected_topk = block_topk[active_mask.to(block_topk.device)].reshape(-1)
        expert_load = torch.bincount(selected_topk, minlength=num_experts).to('cpu', dtype=torch.int64).tolist()
        layers.append({
            'layer_idx': first_moe_layer + offset,
            'active_experts': sum(load > 0 for load in expert_load),
            'assignments': sum(expert_load),
            'expert_load': expert_load,
        })
    return layers


def _transfer_schedule(block_length, denoising_steps):
    """Return the minimum number of tokens accepted per sequence and step."""
    denoising_steps = min(denoising_steps, block_length)
    base, remainder = divmod(block_length, denoising_steps)
    return [base + (step < remainder) for step in range(denoising_steps)]


def _sample_block(logits, temperature):
    """Sample candidates, using true greedy decoding at temperature zero."""
    import torch

    logits = logits.float()
    if temperature <= 0:
        probabilities = torch.softmax(logits, dim=-1)
        confidence, token_ids = probabilities.max(dim=-1)
        return token_ids, confidence
    probabilities = torch.softmax(logits / temperature, dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).reshape(probabilities.shape[:-1])
    confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    return sampled, confidence


def _accept_candidates(current_block, active_mask, candidates, confidence, minimum_transfer, threshold):
    """Apply LLaDA2's confidence-or-minimum-transfer update independently per request."""
    import torch

    transferred = []
    for batch_index in range(current_block.shape[0]):
        active = active_mask[batch_index]
        active_count = int(active.sum().item())
        if active_count == 0:
            transferred.append(0)
            continue
        high_confidence = active & (confidence[batch_index] > threshold)
        if int(high_confidence.sum().item()) >= minimum_transfer:
            selected = high_confidence
        else:
            selected = torch.zeros_like(active)
            masked_confidence = confidence[batch_index].masked_fill(~active, float('-inf'))
            count = min(minimum_transfer, active_count)
            indices = torch.topk(masked_confidence, k=count).indices
            selected[indices] = True
        current_block[batch_index, selected] = candidates[batch_index, selected]
        transferred.append(int(selected.sum().item()))
    return transferred


def _write_metadata(stream, args, batch_size, config):
    stream.write(json.dumps({
        'record_type': 'metadata',
        'format_version': 2,
        'model_type': 'llada2_moe_hf_full_denoising',
        'num_experts': int(config.num_experts),
        'top_k': int(config.num_experts_per_tok),
        'num_hidden_layers': int(config.num_hidden_layers),
        'configured_batch_size': batch_size,
        'observed_region': 'unresolved_mask_queries',
        'block_length': args.block_length,
        'requested_denoising_steps': args.denoising_steps,
        'effective_denoising_steps': min(args.denoising_steps, args.block_length),
        'confidence_threshold': args.confidence_threshold,
        'temperature': args.temperature,
    }, separators=(',', ':')) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='LLaDA2 HF/Accelerate MoE denoising-route profiler.')
    parser.add_argument('dataset')
    parser.add_argument('model_path')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[8])
    parser.add_argument('--num-prompts', type=int, default=32)
    parser.add_argument('--max-input-len', type=int, default=128)
    parser.add_argument('--max-scan-examples', type=int, default=20000)
    parser.add_argument('--block-length', type=int, default=32)
    parser.add_argument('--denoising-steps', type=int, default=32)
    parser.add_argument('--confidence-threshold', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--dataset-format', choices=['auto', 'gsm8k', 'mbpp', 'math'], default='auto')
    parser.add_argument('--hf-split', default=None)
    parser.add_argument('--hf-config', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-memory-per-gpu', default='38GiB')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.block_length <= 0 or args.denoising_steps <= 0:
        raise ValueError('block length and denoising steps must be positive.')
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError('confidence threshold must be in [0, 1].')
    if args.dataset == GSM8K_DATASET_ID:
        args.dataset_format = 'gsm8k'
        args.hf_config = args.hf_config or 'main'
        args.hf_split = args.hf_split or 'test'
    elif args.dataset == MBPP_DATASET_ID:
        args.dataset_format = 'mbpp'
        args.hf_config = args.hf_config or 'sanitized'
        args.hf_split = args.hf_split or 'test'
    elif args.dataset == HUMANEVAL_DATASET_ID:
        # HumanEval exposes code-completion prompts in its test split. The
        # generic prompt extractor reads the row's ``prompt`` field.
        args.dataset_format = 'auto'
        args.hf_split = args.hf_split or 'test'
    else:
        args.hf_split = args.hf_split or 'train'

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() < 2:
        raise RuntimeError(f'This runner expects two visible GPUs, found {torch.cuda.device_count()}.')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'successful_traces.txt'
    manifest_path.unlink(missing_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompt_ids = _tokenize_prompts(args, tokenizer)
    print(f'Sampled {len(prompt_ids)} prompts; loading model once across two GPUs...', flush=True)

    max_memory = {0: args.max_memory_per_gpu, 1: args.max_memory_per_gpu}
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
    print(f'Device map: {getattr(model, "hf_device_map", None)}', flush=True)

    config = model.config
    core_model = model.model
    input_device = core_model.word_embeddings.weight.device
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError('Tokenizer has no mask_token_id; LLaDA2 requires <|mask|>.')
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError('Tokenizer has neither pad_token_id nor eos_token_id.')
    first_moe_layer = int(getattr(config, 'first_k_dense_replace', 0))
    transfer_schedule = _transfer_schedule(args.block_length, args.denoising_steps)
    successful = []

    for batch_size in args.batch_sizes:
        trace_path = output_dir / f'routes_bs{batch_size}.jsonl'
        full_groups = len(prompt_ids) // batch_size
        if full_groups == 0:
            print(f'Skipping batch {batch_size}: only {len(prompt_ids)} prompts are available.', flush=True)
            continue
        print(f'Running batch {batch_size} ({full_groups} full groups)...', flush=True)
        try:
            with trace_path.open('w', encoding='utf-8', buffering=1) as stream:
                _write_metadata(stream, args, batch_size, config)
                for forward_index in range(full_groups):
                    start = forward_index * batch_size
                    group = prompt_ids[start:start + batch_size]
                    input_ids, attention_mask, position_ids = _build_batch_inputs(
                        group, args.block_length, pad_id, mask_id, input_device, torch.bfloat16)
                    current_block = input_ids[:, -args.block_length:]
                    for step, minimum_transfer in enumerate(transfer_schedule):
                        active_mask = current_block.eq(mask_id)
                        q_seqlens = active_mask.sum(dim=1).to('cpu', dtype=torch.int64).tolist()
                        query_tokens = int(sum(q_seqlens))
                        if query_tokens == 0:
                            break
                        with torch.inference_mode():
                            outputs = core_model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                use_cache=False,
                                output_router_logits=True,
                                return_dict=True,
                            )
                            layers = _router_layers(
                                outputs.router_logits,
                                active_mask,
                                int(config.num_experts),
                                first_moe_layer,
                            )
                            block_hidden = outputs.last_hidden_state[:, -args.block_length:, :]
                            logits = model.lm_head(block_hidden)
                            candidates, confidence = _sample_block(logits, args.temperature)
                            transferred = _accept_candidates(
                                current_block,
                                active_mask,
                                candidates.to(current_block.device),
                                confidence.to(current_block.device),
                                minimum_transfer,
                                args.confidence_threshold,
                            )
                        remaining_after = current_block.eq(mask_id).sum(dim=1).to('cpu', dtype=torch.int64).tolist()
                        print(
                            f'  batch={batch_size} group={forward_index + 1}/{full_groups} '
                            f'step={step + 1}/{len(transfer_schedule)} '
                            f'Q={query_tokens}->{sum(remaining_after)}',
                            flush=True,
                        )
                        stream.write(json.dumps({
                            'record_type': 'denoising_step',
                            'forward_index': forward_index * len(transfer_schedule) + step,
                            'group_id': forward_index,
                            'step': step,
                            'actual_batch_size': batch_size,
                            'query_tokens': query_tokens,
                            'q_seqlens': q_seqlens,
                            'minimum_transfer': minimum_transfer,
                            'transferred_per_sequence': transferred,
                            'remaining_after_per_sequence': remaining_after,
                            'layers': layers,
                        }, separators=(',', ':')) + '\n')
                        del outputs, block_hidden, logits, candidates, confidence, layers
                    del current_block, input_ids, attention_mask, position_ids
            successful.append(trace_path)
        except torch.OutOfMemoryError:
            print(f'Batch {batch_size} ran out of memory; keeping smaller completed traces.', flush=True)
            if trace_path.exists():
                trace_path.unlink()
            torch.cuda.empty_cache()

    if not successful:
        raise RuntimeError('No batch size completed successfully.')
    print('Successful traces:')
    for path in successful:
        print(path)
    manifest_path.write_text(''.join(f'{path}\n' for path in successful), encoding='utf-8')


if __name__ == '__main__':
    main()
