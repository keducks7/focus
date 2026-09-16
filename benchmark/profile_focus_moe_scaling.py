#!/usr/bin/env python3
"""Actual Vanilla/delayed-cache/FOCUS expert coverage with one generated 32-token block."""
import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from queue import Queue

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(output, modes, batches):
    groups = defaultdict(list)
    coverage = []
    for mode in modes:
        for batch in batches:
            path = output / f'{mode}_bs{batch}' / 'routes.jsonl'
            if not (path.parent / 'complete.json').exists():
                raise RuntimeError(f'Incomplete run: {path.parent}')
            total = full = 0
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['record_type'] != 'moe_forward' or row['phase'] != 'decode':
                        continue
                    total += 1
                    if row['actual_batch'] != batch or row['nonempty_requests'] != batch:
                        continue
                    full += 1
                    groups[mode, batch, row['group_id'], row['layer_idx']].append(row)
            coverage.append(dict(mode=mode, batch=batch, decode_layer_records=total,
                                 full_batch_layer_records=full))
    group_rows = []
    for (mode, batch, group, layer), values in sorted(groups.items()):
        group_rows.append(dict(mode=mode, batch=batch, group=group, layer=layer, forwards=len(values),
                               mean_query_tokens=statistics.fmean(row['query_tokens'] for row in values),
                               mean_active_experts=statistics.fmean(row['active_experts'] for row in values),
                               mean_expert_coverage=statistics.fmean(row['active_experts'] / row['num_experts'] for row in values)))
    aggregate = defaultdict(list)
    for row in group_rows:
        aggregate[row['mode'], row['batch'], row['layer']].append(row)
    summary = []
    for (mode, batch, layer), values in sorted(aggregate.items()):
        summary.append(dict(mode=mode, batch=batch, layer=layer, groups=len(values),
                            mean_query_tokens=statistics.fmean(row['mean_query_tokens'] for row in values),
                            mean_active_experts=statistics.fmean(row['mean_active_experts'] for row in values),
                            group_sd_active_experts=statistics.stdev(row['mean_active_experts'] for row in values)
                            if len(values) > 1 else '',
                            mean_expert_coverage=statistics.fmean(row['mean_expert_coverage'] for row in values)))
    by_key = {(r['mode'], r['batch'], r['layer']): r for r in summary}
    for row in summary:
        previous = by_key.get((row['mode'], row['batch'] // 2, row['layer'])) if row['batch'] % 2 == 0 else None
        row['expert_doubling_ratio'] = (row['mean_active_experts'] / previous['mean_active_experts']
                                        if previous and previous['mean_active_experts'] else '')
        for control in ('vanilla', 'delayed'):
            baseline = by_key.get((control, row['batch'], row['layer']))
            row['query_ratio_vs_' + control] = (row['mean_query_tokens'] / baseline['mean_query_tokens']
                                                if baseline and baseline['mean_query_tokens'] else '')
    for name, rows in [('coverage.csv', coverage), ('group_layers.csv', group_rows), ('scaling.csv', summary)]:
        if rows:
            with (output / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if any(row['full_batch_layer_records'] == 0 for row in coverage):
        raise RuntimeError('At least one run has no full-batch decode observations; inspect coverage.csv and raw traces')


def worker(args):
    import torch
    from tqdm import tqdm
    from profile_throughput import Engine, Profiler, PytorchEngineConfig, sample_requests
    from lmdeploy.pytorch.models.moe_observation import MoEObserver, set_observer

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU: this runner uses TP=1 and executor=uni')
    torch.manual_seed(args.seed)
    run_dir = args.output_dir / f'{args.mode}_bs{args.batch_size}'
    run_dir.mkdir()
    config = PytorchEngineConfig(tp=1, distributed_executor_backend='uni', eager_mode=True,
                                 max_batch_size=args.batch_size, cache_max_entry_count=args.cache_fraction,
                                 max_prefill_token_num=4096, dtype='bfloat16',
                                 dllm_block_length=32, dllm_denoising_steps=32,
                                 dllm_unmasking_strategy='low_confidence_dynamic',
                                 dllm_confidence_threshold=args.confidence,
                                 dllm_enable_delayed_cache=args.mode != 'vanilla',
                                 dllm_enable_focus=args.mode == 'focus', dllm_focus_alpha=args.focus_alpha)
    print(f'Loading TP=1 eager engine: mode={args.mode}, batch={args.batch_size}', flush=True)
    engine = Engine(args.model_path, config)
    manifest = args.output_dir / 'requests.json'
    if manifest.exists():
        requests = json.loads(manifest.read_text())
    else:
        requests = sample_requests(dataset_path=args.dataset, num_requests=args.num_prompts,
                                   tokenizer=engine.tokenizer.model.model, chat_template=engine.chat_template,
                                   dataset_format='gsm8k', hf_split='test', hf_config='main',
                                   max_input_len=args.max_input_len, seed=args.seed)
        if len(requests) != args.num_prompts:
            raise RuntimeError(f'Only {len(requests)} prompts survived; increase --max-input-len')
        manifest.write_text(json.dumps(requests, ensure_ascii=False, indent=2))
    async def collect(stream):
        # One event loop for the entire engine lifetime; do not reuse it across asyncio.run calls.
        for group, start in enumerate(range(0, len(requests), args.batch_size)):
            observer = MoEObserver(stream, args.mode, args.batch_size, group)
            profiler = Profiler(False, [50])
            queue = Queue()
            batch = requests[start:start + args.batch_size]
            for prompt, input_len in batch:
                queue.put([prompt, input_len, 32, profiler.new_session(input_len, 0)])
            for _ in batch:
                queue.put(None)
            engine.pbar = tqdm(total=len(batch))
            set_observer(observer)
            profiler.start()
            try:
                await asyncio.gather(*(engine._inference(queue, start + i, 0, 1, 1, False,
                                                         False, True, args.batch_size, 32)
                                       for i in range(len(batch))))
            finally:
                profiler.finish()
                set_observer(None)
                engine.pbar.close()
            print(f'Completed mode={args.mode} batch={args.batch_size} group={group}', flush=True)

    with (run_dir / 'routes.jsonl').open('w', buffering=1) as stream:
        stream.write(json.dumps(dict(record_type='metadata', mode=args.mode, batch=args.batch_size,
                                     observed_region='all actually executed MoE queries, including non-mask tokens',
                                     gen_length=32, block_length=32, tp=1, eager=True,
                                     focus_alpha=args.focus_alpha, confidence=args.confidence)) + '\n')
        asyncio.run(collect(stream))
    (run_dir / 'complete.json').write_text(json.dumps(dict(groups=len(requests) // args.batch_size)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_path')
    parser.add_argument('--dataset', default='openai/gsm8k')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--modes', nargs='+', choices=['vanilla', 'delayed', 'focus'], default=['vanilla', 'delayed', 'focus'])
    parser.add_argument('--num-prompts', type=int, default=32)
    parser.add_argument('--max-input-len', type=int, default=128)
    parser.add_argument('--confidence', type=float, default=0.95)
    parser.add_argument('--focus-alpha', type=float, default=1.0)
    parser.add_argument('--cache-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=1800, help='Per-mode/batch subprocess timeout in seconds')
    parser.add_argument('--mode', choices=['vanilla', 'delayed', 'focus'], help=argparse.SUPPRESS)
    parser.add_argument('--batch-size', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.num_prompts <= 0 or any(b <= 0 or args.num_prompts % b for b in args.batch_sizes)
            or not 0 < args.cache_fraction < 1 or args.focus_alpha < 1 or not 0 <= args.confidence <= 1):
        parser.error('Require positive batch divisors of num-prompts, cache fraction in (0,1), alpha >=1, confidence in [0,1]')
    if args.mode:
        worker(args)
        return
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'experiment.json').write_text(json.dumps(vars(args), default=str, indent=2))
    for batch in args.batch_sizes:
        for mode in args.modes:
            command = [sys.executable, '-u', str(Path(__file__).resolve()), args.model_path,
                       '--output-dir', str(args.output_dir), '--dataset', args.dataset,
                       '--num-prompts', str(args.num_prompts), '--max-input-len', str(args.max_input_len),
                       '--batch-sizes', *map(str, args.batch_sizes), '--mode', mode, '--batch-size', str(batch),
                       '--seed', str(args.seed), '--confidence', str(args.confidence),
                       '--focus-alpha', str(args.focus_alpha), '--cache-fraction', str(args.cache_fraction)]
            print(f'Running {mode} batch={batch}; log: {args.output_dir}/{mode}_bs{batch}.log', flush=True)
            with (args.output_dir / f'{mode}_bs{batch}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
    summarize(args.output_dir, args.modes, args.batch_sizes)
    print(f'Complete: {args.output_dir}/scaling.csv', flush=True)


if __name__ == '__main__':
    # Preserve the old command name while switching execution to the HF backend.
    from profile_llada2_hf_focus_scaling import main as hf_main
    hf_main()
