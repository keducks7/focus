#!/usr/bin/env python3
"""Synthetic single-layer replay of recorded MoE loads; requires CUDA/Triton."""
import argparse
import csv
import importlib
import json
import random
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np


def synthesize_routes(loads, top_k, seed=0):
    """Bipartite degree realization; exact loads and distinct experts per token.

    Havel-Hakimi with uniform token degree. This does not recover original routes.
    """
    remaining = np.asarray(loads, dtype=np.int64).copy()
    if remaining.ndim != 1 or np.any(remaining < 0) or top_k <= 0:
        raise ValueError('Invalid expert histogram/top-k')
    total = int(remaining.sum())
    if total == 0 or total % top_k or top_k > len(remaining):
        raise ValueError('Invalid assignment count')
    tokens = total // top_k
    if int(remaining.max()) > tokens:
        raise ValueError('An expert cannot occur twice in a token top-k')
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(remaining))
    routes = []
    for _ in range(tokens):
        ids = np.lexsort((tie, -remaining))[:top_k]
        if np.any(remaining[ids] <= 0):
            raise ValueError('Histogram cannot realize distinct top-k routes')
        remaining[ids] -= 1
        routes.append(rng.permutation(ids))
    assert not remaining.any()
    result = np.asarray(routes, dtype=np.int64)
    rng.shuffle(result)
    return result


def load_cases(path, layer, steps):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    metadata = records[0]
    if metadata.get('format_version') != 2:
        raise ValueError('Expected full single-block trajectory v2')
    observations = {}
    for record in records[1:]:
        if record.get('record_type') != 'denoising_step':
            continue
        match = [x for x in record['layers'] if x['layer_idx'] == layer]
        if len(match) != 1:
            raise ValueError(f'Layer {layer} missing/duplicated')
        loads = match[0]['expert_load']
        if len(loads) != metadata['num_experts'] or sum(loads) != record['query_tokens'] * metadata['top_k']:
            raise ValueError('Invalid load histogram')
        observations[(record['group_id'], record['step'])] = loads
    cases = []
    for (group, step), loads in sorted(observations.items()):
        if step in steps and (group, step - 1) in observations:
            cases.append((group, step, loads, observations[(group, step - 1)]))
    if not cases:
        raise ValueError('No selected step with a preceding step')
    return metadata, cases


def load_kernel():
    # Load only repository CUDA kernels, avoiding the LMDeploy engine and its
    # optional runtime dependencies. Relative kernel imports remain unchanged.
    name = '_focus_replay_cuda'
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'lmdeploy/pytorch/kernels/cuda')]
    sys.modules[name] = package
    return importlib.import_module(name + '.fused_moe')


def measure(torch, fn, iterations):
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    begin.record()
    for _ in range(iterations):
        result = fn()
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - wall_start) * 1000 / iterations
    return begin.elapsed_time(end) / iterations, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('model_path', type=Path, help='Only local config.json is read, not checkpoint weights')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--steps', nargs='+', type=int, default=[1, 4, 8, 12])
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.rounds) < 1:
        parser.error('warmup, iterations and rounds must be positive')
    metadata, cases = load_cases(args.trace, args.layer, args.steps)
    config = json.loads((args.model_path / 'config.json').read_text())
    hidden = int(config['hidden_size'])
    intermediate = int(config.get('moe_intermediate_size') or config['intermediate_size'])
    experts, topk = int(metadata['num_experts']), int(metadata['top_k'])
    if config['num_experts'] != experts or config['num_experts_per_tok'] != topk:
        raise ValueError('Trace and model configuration disagree')
    if config.get('hidden_act', 'silu') != 'silu':
        raise ValueError('Replay currently supports SwiGLU only')
    import torch
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; CPU timings would not test this hypothesis')
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    kernel = load_kernel()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'timings.csv').exists():
        raise FileExistsError('Use a new output directory to preserve previous timings')
    dtype = torch.bfloat16
    # Same persistent weights for all cases and all schedules; full expert bank.
    w1 = torch.randn(experts, 2 * intermediate, hidden, device='cuda', dtype=dtype) / hidden**0.5
    w2 = torch.randn(experts, hidden, intermediate, device='cuda', dtype=dtype) / intermediate**0.5
    identity = torch.arange(experts, device='cuda', dtype=torch.int64)
    info = dict(trace=str(args.trace.resolve()), layer=args.layer, steps=args.steps,
                hidden=hidden, intermediate=intermediate, experts=experts, top_k=topk,
                weight_bytes=w1.numel()*w1.element_size()+w2.numel()*w2.element_size(),
                torch=torch.__version__, triton=triton.__version__,
                gpu=torch.cuda.get_device_name(), seed=args.seed,
                warmup=args.warmup, iterations=args.iterations, rounds=args.rounds,
                synthetic=True, observed_region=metadata.get('observed_region'),
                cache_policy='repeated warm workload; no forced cache flush',
                backend='repository Triton fused_moe with optional expert grid permutation')
    (args.output_dir / 'metadata.json').write_text(json.dumps(info, indent=2))
    results = []
    rng = random.Random(args.seed)
    with torch.inference_mode():
        for group, step, loads, previous in cases:
            routes = synthesize_routes(loads, topk, args.seed + group * 100 + step)
            ids = torch.as_tensor(routes, device='cuda')
            x = torch.randn(len(routes), hidden, device='cuda', dtype=dtype)
            weights = torch.full((len(routes), topk), 1/topk, device='cuda', dtype=torch.float32)
            counts = torch.tensor(loads, device='cuda', dtype=torch.int64)
            prev = torch.tensor(previous, device='cuda', dtype=torch.int64)
            orders = {'native': None, 'identity': identity,
                      'previous': torch.argsort(prev, descending=True, stable=True),
                      'oracle': torch.argsort(counts, descending=True, stable=True)}
            # Prepared execution excludes routing dispatch, allocation and ranking.
            sorted_idx, start, end = kernel._get_sorted_idx(ids, experts)
            first = torch.empty(len(routes), topk, 2*intermediate, device='cuda', dtype=dtype)
            activated = torch.empty(len(routes)*topk, intermediate, device='cuda', dtype=dtype)
            second = torch.empty(len(routes), topk, hidden, device='cuda', dtype=dtype)

            def prepared(order):
                kernel.fused_moe_kernel_launcher(
                    x, w1, first, sorted_idx, start, end, top_k=topk,
                    num_tokens=len(routes), reindex_a=True, reindex_c=False, expert_order=order)
                kernel.silu_and_mul(first.flatten(0, 1), out=activated)
                kernel.fused_moe_kernel_launcher(
                    activated, w2, second, sorted_idx, start, end,
                    top_k=1, num_tokens=len(routes), reindex_a=False, reindex_c=True, expert_order=order)
                return second

            def total(name):
                # Previous order is available from the previous step. Include
                # current count+sort maintenance for the next step in measured work.
                order = orders[name]
                if name in ('previous', 'oracle'):
                    actual_counts = torch.bincount(ids.flatten(), minlength=experts)
                    next_order = torch.argsort(actual_counts, descending=True, stable=True)
                    if name == 'oracle':
                        order = next_order
                return kernel.fused_moe(x, w1, w2, weights, ids, topk, expert_order=order)

            baseline = total('native')
            # Independent PyTorch reference on two complete tokens, including
            # shared token activation across top-k and reduction.
            reference = []
            for token in range(min(2, len(routes))):
                accum = torch.zeros(hidden, device='cuda', dtype=torch.float32)
                for expert in routes[token]:
                    gateup = torch.nn.functional.linear(x[token], w1[int(expert)])
                    gate, up = gateup.chunk(2)
                    hidden_act = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)
                    y = torch.nn.functional.linear(hidden_act, w2[int(expert)])
                    accum += y.float() / topk
                reference.append(accum)
            torch.testing.assert_close(baseline[:len(reference)].float(),
                                       torch.stack(reference), atol=0.03, rtol=0.03)
            for name, order in orders.items():
                output = total(name)
                torch.testing.assert_close(output, baseline, atol=0.005, rtol=0.005)
                prepared_output = prepared(order)
                reduced = kernel.moe_reduce(prepared_output, weights)
                torch.testing.assert_close(reduced, baseline, atol=0.005, rtol=0.005)
            functions = {}
            for name in orders:
                functions[('prepared_experts', name)] = lambda n=name: prepared(orders[n])
                functions[('moe_total', name)] = lambda n=name: total(n)
            for fn in functions.values():
                for _ in range(args.warmup):
                    fn()
            torch.cuda.synchronize()
            for round_id in range(args.rounds):
                keys = list(functions)
                rng.shuffle(keys)
                for scope, name in keys:
                    gpu_ms, wall_ms = measure(torch, functions[(scope, name)], args.iterations)
                    results.append(dict(group=group, step=step, layer=args.layer,
                                        query_tokens=len(routes), scope=scope, schedule=name,
                                        round=round_id, cuda_ms=gpu_ms, wall_ms=wall_ms))
            # Save incrementally after each completed observation.
            with (args.output_dir / 'timings.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
            print(f'group={group} step={step} Q={len(routes)} validated and timed', flush=True)
            del functions, baseline, first, activated, second, x, weights, ids
    summary = []
    for group, step, _, _ in cases:
        for scope in ['prepared_experts', 'moe_total']:
            rows = [r for r in results if r['group']==group and r['step']==step and r['scope']==scope]
            native = statistics.median(r['cuda_ms'] for r in rows if r['schedule']=='native')
            for name in ['native', 'identity', 'previous', 'oracle']:
                values = [r['cuda_ms'] for r in rows if r['schedule']==name]
                median = statistics.median(values)
                summary.append(dict(group=group, step=step, scope=scope, schedule=name,
                                    median_cuda_ms=median, min_cuda_ms=min(values),
                                    max_cuda_ms=max(values), speedup_vs_native=native/median))
    with (args.output_dir / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'Complete: {args.output_dir}. Inspect per-case spread before claiming speedup.')


if __name__ == '__main__':
    main()

