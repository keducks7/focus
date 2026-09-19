#!/usr/bin/env python3
"""Exact request-subset expert coverage on fixed full-state decoding trajectories.

Offline subset size is NOT the physical batch size. Concave union growth alone
is not mechanistic evidence: a size-preserving independent expert-label null
controls for finite expert-pool saturation and per-request coverage.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def expected_union(request_sets, subset_size):
    """Average union over all C(B,k) subsets, without enumerating them."""
    batch = len(request_sets)
    if not 1 <= subset_size <= batch:
        raise ValueError('Subset size must be between 1 and physical batch size.')
    occupancy = Counter(e for experts in request_sets for e in experts)
    denominator = math.comb(batch, subset_size)
    return sum(1 - (math.comb(batch - count, subset_size) / denominator
                    if batch - count >= subset_size else 0) for count in occupancy.values())


def label_null_curve(request_sets, num_experts):
    """Exact mean union after independent uniform expert-label permutation per request.

    Elementary symmetric polynomials average products of exclusion probabilities.
    Preserves each request's expert-set size; does not model functional replacement.
    """
    batch = len(request_sets)
    elementary = [1.0] + [0.0] * batch
    for index, experts in enumerate(request_sets):
        if len(experts) > num_experts:
            raise ValueError('Request expert set larger than expert pool.')
        exclusion = 1 - len(experts) / num_experts
        for degree in range(index + 1, 0, -1):
            elementary[degree] += exclusion * elementary[degree - 1]
    return {size: num_experts * (1 - elementary[size] / math.comb(batch, size))
            for size in range(1, batch + 1)}


def rows_for_step(record, metadata, layers, subset_sizes):
    tokens = record['tokens']
    requests = sorted({t['request_id'] for t in tokens})
    batch = metadata['configured_batch_size']
    if len(requests) != batch or len(tokens) != batch * metadata['block_length']:
        raise ValueError('Incomplete batch token coverage.')
    positions = {(t['request_id'], t['block_position']) for t in tokens}
    if len(positions) != len(tokens) or any(not 0 <= p < metadata['block_length'] for _, p in positions):
        raise ValueError('Missing/duplicate token position.')
    if any(t['request_finished_before'] for t in tokens):
        return [], []  # Never relabel a shrunken cohort as the configured batch.
    for token in tokens:
        if token['state_before'] not in ('mask', 'decoded') or (
                token['state_before'] == 'mask') != token['masked_before']:
            raise ValueError('Inconsistent token state.')
        if not token['executed_this_step'] or token['skip_reason'] is not None:
            raise ValueError('Requires fresh executed Vanilla routes.')
    rows, occupancy_rows = [], []
    base = dict(physical_batch=batch, group_id=record['group_id'], step=record['step'])
    for layer_idx in layers:
        sets = {state: {r: set() for r in requests} for state in ['mask', 'decoded', 'all']}
        loads = {state: Counter() for state in sets}
        token_counts = Counter()
        for token in tokens:
            matches = [layer for layer in token['layers'] if layer['layer_idx'] == layer_idx]
            if len(matches) != 1:
                raise ValueError('Missing or repeated requested layer.')
            experts = matches[0]['expert_ids']
            if (len(experts) != metadata['top_k'] or len(set(experts)) != len(experts)
                    or any(e < 0 or e >= metadata['num_experts'] for e in experts)):
                raise ValueError('Invalid expert selection.')
            for state in [token['state_before'], 'all']:
                sets[state][token['request_id']].update(experts)
                loads[state].update(experts)
                token_counts[state] += 1
        curves = {state: {size: expected_union(list(mapping.values()), size) for size in subset_sizes}
                  for state, mapping in sets.items()}
        for state, mapping in sets.items():
            request_sets = list(mapping.values())
            null = label_null_curve(request_sets, metadata['num_experts'])
            total_incidence = sum(map(len, request_sets))
            occupancy = Counter(e for experts in request_sets for e in experts)
            for expert, count in sorted(occupancy.items()):
                occupancy_rows.append(dict(**base, layer_idx=layer_idx, state=state,
                                           expert_id=expert, request_occupancy=count,
                                           assignments=loads[state][expert]))
            for size in subset_sizes:
                union = curves[state][size]
                incidences = total_incidence * size / batch
                # Expectation is linear, so M+D-All gives exact mean state intersection.
                intersection = curves['mask'][size] + curves['decoded'][size] - curves['all'][size]
                rows.append(dict(**base, layer_idx=layer_idx, state=state, subset_size=size,
                    subset_combinations=math.comb(batch, size),
                    expected_tokens=token_counts[state] * size / batch,
                    mean_union_experts=union, mean_request_expert_incidences=incidences,
                    incidence_to_union_ratio=incidences / union if union else None,
                    mean_duplicate_request_expert_incidences=max(0.0, incidences - union),
                    label_null_mean_union=null[size],
                    alignment_gap=null[size] - union,
                    mean_mask_decoded_intersection=max(0.0, intersection) if state == 'all' else None,
                    mean_mask_exclusive_experts=max(0.0, curves['all'][size] - curves['decoded'][size])
                    if state == 'all' else None,
                    mean_decoded_exclusive_experts=max(0.0, curves['all'][size] - curves['mask'][size])
                    if state == 'all' else None))
    return rows, occupancy_rows


def analyze(paths, output_dir, layers=(2, 10, 18), subset_sizes=(1, 2, 4, 8, 16, 32), max_step=8):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'analysis': 'fixed-trajectory exact request-subset means', 'max_step': max_step,
                'layers': list(layers), 'files': [], 'summary_weighting': 'equal group within each source/step',
                'ratio_definition': 'ratio of expectations, NOT mean of per-subset ratios',
                'completed_requests': 'exclude entire group-step if any request finished before forward',
                'null': 'independent uniform expert-label permutation per request, preserving set sizes'}
    summaries = defaultdict(list)
    streams, writers = {}, {}
    def emit(name, row):
        if name not in writers:
            streams[name] = (output_dir / (name + '.csv')).open('w', newline='')
            writers[name] = csv.DictWriter(streams[name], fieldnames=list(row))
            writers[name].writeheader()
        writers[name].writerow(row)
    try:
        for path in paths:
            path = Path(path)
            source = str(path.resolve())
            included, excluded = [], []
            with path.open() as stream:
                metadata = json.loads(next(stream))
                if (metadata.get('format_version') != 2 or metadata.get('mode') != 'vanilla'
                        or metadata.get('observed_region') != 'all_generation_positions'):
                    raise ValueError('Requires Vanilla --full-lifecycle v2 traces.')
                if metadata['num_generation_blocks'] != 1:
                    raise ValueError('Only single-block trajectories currently supported.')
                if not set(layers) <= set(range(metadata['first_moe_layer'], metadata['num_hidden_layers'])):
                    raise ValueError('Requested layers not present.')
                sizes = sorted(set(k for k in subset_sizes if k <= metadata['configured_batch_size'])
                               | {metadata['configured_batch_size']})
                seen, completed = set(), set()
                last_steps = {}
                for line in stream:
                    record = json.loads(line)
                    if record['record_type'] == 'generation_result':
                        group = record['group_id']
                        if group not in last_steps or group in completed:
                            raise ValueError('Invalid generation completion record.')
                        completed.add(group)
                    if record['record_type'] != 'token_trajectory_step':
                        continue
                    group, step = record['group_id'], record['step']
                    if step != last_steps.get(group, -1) + 1 or group in completed:
                        raise ValueError('Missing, repeated, or out-of-order step.')
                    last_steps[group] = step
                    seen.add(group)
                    if step > max_step:
                        continue
                    rows, occupancy = rows_for_step(record, metadata, layers, sizes)
                    (included if rows else excluded).append([group, step])
                    for row in rows:
                        row = dict(source=source, **row)
                        emit('subset_group_curves', row)
                        key = (source, row['physical_batch'], step, row['layer_idx'], row['state'], row['subset_size'])
                        summaries[key].append(row)
                        if row['subset_size'] == row['physical_batch']:
                            emit('physical_batch_observations', row)
                    for row in occupancy:
                        emit('expert_request_occupancy', dict(source=source, **row))
                expected_groups = metadata['num_prompts'] // metadata['configured_batch_size']
                if seen != completed or len(seen) != expected_groups or not included:
                    raise ValueError('Incomplete trace or no eligible full-batch observations.')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            manifest['files'].append(dict(source=source, sha256=digest.hexdigest(),
                physical_batch=metadata['configured_batch_size'], included_group_steps=included,
                excluded_finished_group_steps=excluded))
        for key, rows in summaries.items():
            source, batch, step, layer, state, size = key
            row = dict(source=source, physical_batch=batch, step=step, layer_idx=layer,
                       state=state, subset_size=size, groups=len(rows))
            for metric in ['expected_tokens', 'mean_union_experts', 'mean_request_expert_incidences',
                           'label_null_mean_union', 'alignment_gap', 'mean_mask_decoded_intersection',
                           'mean_mask_exclusive_experts', 'mean_decoded_exclusive_experts']:
                values = [r[metric] for r in rows if r[metric] is not None]
                row[metric] = sum(values) / len(values) if values else None
            emit('subset_summary', row)
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    finally:
        for stream in streams.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectories', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 10, 18])
    parser.add_argument('--subset-sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--max-step', type=int, default=8)
    args = parser.parse_args()
    if min(args.subset_sizes) < 1 or args.max_step < 0:
        parser.error('Subset sizes must be positive; max-step must be nonnegative.')
    analyze(args.trajectories, args.output_dir, args.layers, args.subset_sizes, args.max_step)
    print(f'Exact batch-sharing analysis complete: {args.output_dir}')


if __name__ == '__main__':
    main()
