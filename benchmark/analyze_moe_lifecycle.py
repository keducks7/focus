#!/usr/bin/env python3
"""Analyze full-state trajectories by layer, acceptance time and token grouping.

Standard library only. Buffers one group, streams events, and never compares
expert IDs from different layers. Summary means are token-event weighted;
group/request identifiers in raw events permit request-level resampling.
"""
import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def overlap(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right)


def weight_distance(left, right):
    def distribution(layer):
        weights = layer['router_weights']
        total = sum(weights)
        if total <= 0:
            raise ValueError('Router weight sum must be positive.')
        return {e: w / total for e, w in zip(layer['expert_ids'], weights)}
    a, b = distribution(left), distribution(right)
    return sum(abs(a.get(e, 0) - b.get(e, 0)) for e in a.keys() | b.keys()) / 2


def pearson(left, right):
    if len(left) < 2:
        return None
    a, b = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - a) * (y - b) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - a)**2 for x in left) * sum((y - b)**2 for y in right))
    return numerator / denominator if denominator else None


def identity(token):
    return (token['request_id'], token['block_id'], token['block_position'])


def validate_group(records, metadata):
    layers = set(range(metadata['first_moe_layer'], metadata['num_hidden_layers']))
    expected_count = metadata['configured_batch_size'] * metadata['block_length']
    accepted = {}
    first_ids = None
    previous_step = -1
    for record in records:
        if record['step'] != previous_step + 1:
            raise ValueError('Missing, duplicated or out-of-order denoising step.')
        previous_step = record['step']
        tokens = record['tokens']
        ids = [identity(token) for token in tokens]
        if len(tokens) != expected_count or len(set(ids)) != expected_count:
            raise ValueError('Full lifecycle must contain every generation position exactly once.')
        if record['query_tokens'] != len(tokens):
            raise ValueError('query_tokens differs from token count.')
        if first_ids is None:
            first_ids = set(ids)
        elif set(ids) != first_ids:
            raise ValueError('Token cohort changed within a group.')
        for token in tokens:
            key = identity(token)
            masked = token['masked_before']
            if masked != (key not in accepted):
                raise ValueError('Token state does not agree with its acceptance history.')
            if not masked and token['input_token_id'] != accepted[key][1]:
                raise ValueError('An accepted token changed identity.')
            if token['accepted_this_step'] != token['accepted']:
                raise ValueError('Acceptance flags disagree.')
            if token['accepted']:
                if not masked:
                    raise ValueError('A decoded token was accepted twice.')
                accepted[key] = (record['step'], token['candidate_token_id'])
            if token['acceptance_step'] != (accepted[key][0] if key in accepted else None):
                raise ValueError('Invalid acceptance_step.')
            if not token['executed_this_step'] or token['skip_reason'] is not None:
                raise ValueError('This analyzer expects fresh Vanilla observations, not cached routes.')
            entries = token['layers']
            if len(entries) != len(layers) or {v['layer_idx'] for v in entries} != layers:
                raise ValueError('Incomplete layer trace.')
            for layer in entries:
                experts, weights = layer['expert_ids'], layer['router_weights']
                if (len(experts) != metadata['top_k'] or len(set(experts)) != len(experts)
                        or len(weights) != len(experts)
                        or any(e < 0 or e >= metadata['num_experts'] for e in experts)
                        or any(not math.isfinite(w) or w < 0 for w in weights)):
                    raise ValueError('Invalid routed expert IDs or weights.')
    return {key: value[0] for key, value in accepted.items()}


def events_for_group(records, metadata):
    acceptance = validate_group(records, metadata)
    previous = {}
    for record in records:
        step = record['step']
        for token in record['tokens']:
            key = identity(token)
            accepted_at = acceptance.get(key)
            phase = ('accepting_mask' if token['accepted'] else 'unresolved_mask') if token['masked_before'] else (
                'first_decoded' if step == accepted_at + 1 else 'later_decoded')
            for layer in token['layers']:
                route_key = key + (layer['layer_idx'],)
                prior = previous.get(route_key)
                if prior is not None:
                    yield {
                        'group_id': record['group_id'], 'request_id': key[0], 'block_id': key[1],
                        'block_position': key[2], 'layer_idx': layer['layer_idx'], 'step': step,
                        'acceptance_step': accepted_at,
                        'relative_acceptance_step': None if accepted_at is None else step - accepted_at,
                        'phase': phase, 'request_finished_before': token['request_finished_before'],
                        'route_jaccard': overlap(prior['expert_ids'], layer['expert_ids']),
                        'router_weight_tv': weight_distance(prior, layer),
                        'moe_output_relative_l2': layer.get('moe_output_relative_l2'),
                        'moe_output_cosine': layer.get('moe_output_cosine'),
                    }
                previous[route_key] = layer


def state_rows(record):
    buckets = defaultdict(list)
    for token in record['tokens']:
        for layer in token['layers']:
            buckets[(layer['layer_idx'], token['state_before'], token['request_finished_before'])].append(layer)
    for (layer_idx, state, finished), layers in sorted(buckets.items()):
        loads = Counter(e for layer in layers for e in layer['expert_ids'])
        total = sum(loads.values())
        yield dict(group_id=record['group_id'], step=record['step'], layer_idx=layer_idx,
                   state_before=state, request_finished_before=finished, tokens=len(layers),
                   active_experts=len(loads), effective_experts=total**2 / sum(c*c for c in loads.values()),
                   top10_load_share=sum(sorted(loads.values(), reverse=True)[:10]) / total)


def pair_sample(tokens, limit, seed):
    """Fixed identity pairs across layers/steps; no quadratic pair storage."""
    ids = sorted(identity(token) for token in tokens)
    n = len(ids)
    count = min(limit, n * (n - 1) // 2)
    if count == n * (n - 1) // 2:
        return list(itertools.combinations(ids, 2))
    rng, pairs = random.Random(seed), set()
    while len(pairs) < count:
        a, b = sorted(rng.sample(range(n), 2))
        pairs.add((ids[a], ids[b]))
    return sorted(pairs)


def cross_layer_rows(record, pairs):
    tokens = {identity(token): token for token in record['tokens']}
    routes = {key: {layer['layer_idx']: set(layer['expert_ids']) for layer in token['layers']}
              for key, token in tokens.items()}
    layer_ids = sorted(next(iter(routes.values())))
    for lower, upper in zip(layer_ids, layer_ids[1:]):
        buckets = defaultdict(lambda: ([], []))
        for a, b in pairs:
            left, right = tokens[a], tokens[b]
            # Completed requests kept only for auditing, not natural-decoding inference.
            if left['request_finished_before'] or right['request_finished_before']:
                continue
            state_pair = '-'.join(sorted([left['state_before'], right['state_before']]))
            key = (state_pair, 'same_request' if a[0] == b[0] else 'cross_request')
            values = buckets[key]
            for index, layer in enumerate([lower, upper]):
                values[index].append(len(routes[a][layer] & routes[b][layer]) / len(routes[a][layer]))
        for (state_pair, relation), (left, right) in sorted(buckets.items()):
            yield dict(group_id=record['group_id'], step=record['step'], lower_layer=lower,
                       upper_layer=upper, state_pair=state_pair, request_relation=relation,
                       sampled_pairs=len(left), sharing_pearson=pearson(left, right),
                       lower_mean_overlap=sum(left) / len(left), upper_mean_overlap=sum(right) / len(right))


def analyze(path, output_dir, pair_samples=512, seed=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    streams, writers = {}, {}
    summaries = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    def emit(name, row):
        if name not in writers:
            stream = (output_dir / (name + '.csv')).open('w', newline='', encoding='utf-8')
            streams[name] = stream
            writers[name] = csv.DictWriter(stream, fieldnames=list(row))
            writers[name].writeheader()
        writers[name].writerow(row)

    def process(records, metadata):
        if not records:
            return
        for event in events_for_group(records, metadata):
            emit('lifecycle_events', event)
            for axis, value in [('phase', event['phase']),
                                ('relative_acceptance_step', event['relative_acceptance_step'])]:
                if value is None:
                    continue
                key = (axis, event['layer_idx'], value, event['request_finished_before'])
                bucket = summaries[key]
                bucket['events'][0] += 1
                for metric in ['route_jaccard', 'router_weight_tv', 'moe_output_relative_l2', 'moe_output_cosine']:
                    if event[metric] is not None:
                        bucket[metric][0] += 1
                        bucket[metric][1] += event[metric]
        pairs = pair_sample(records[0]['tokens'], pair_samples, seed + records[0]['group_id'])
        for record in records:
            for row in state_rows(record):
                emit('lifecycle_state_loads', row)
            for row in cross_layer_rows(record, pairs):
                emit('lifecycle_cross_layer', row)

    try:
        with path.open(encoding='utf-8') as stream:
            metadata = json.loads(next(stream))
            if metadata.get('format_version') != 2 or metadata.get('observed_region') != 'all_generation_positions':
                raise ValueError('Requires --full-lifecycle v2 data; old MASK-only data cannot supply decoded routes.')
            records, group_id = [], None
            completed_groups, coverage_groups = set(), set()
            for line in stream:
                record = json.loads(line)
                if record['record_type'] == 'token_trajectory_step':
                    if ((group_id is not None and group_id != record['group_id'])
                            or record['group_id'] in completed_groups):
                        raise ValueError('Missing generation_result or repeated group.')
                    group_id = record['group_id']
                    records.append(record)
                elif record['record_type'] == 'lifecycle_coverage':
                    coverage_groups.add(record['group_id'])
                    for token in record['tokens']:
                        emit('lifecycle_coverage', dict(group_id=record['group_id'], **token))
                elif record['record_type'] == 'generation_result':
                    if not records or record['group_id'] != group_id or group_id not in coverage_groups:
                        raise ValueError('Missing steps/coverage before generation_result.')
                    process(records, metadata)
                    completed_groups.add(group_id)
                    records, group_id = [], None
            if records:
                raise ValueError('Incomplete run: last group has no generation_result.')
            expected_groups = metadata.get('num_prompts', 0) // metadata['configured_batch_size']
            if not completed_groups or (expected_groups and len(completed_groups) != expected_groups):
                raise ValueError('Incomplete run: missing generation groups.')
        for (axis, layer, value, finished), bucket in summaries.items():
            row = dict(layer_idx=layer, **{axis: value}, request_finished_before=finished,
                       events=bucket['events'][0])
            for metric in ['route_jaccard', 'router_weight_tv', 'moe_output_relative_l2', 'moe_output_cosine']:
                count, total = bucket[metric]
                row[metric + '_count'] = count
                row['mean_' + metric] = total / count if count else None
            emit('lifecycle_by_' + axis, row)
    finally:
        for stream in streams.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pair-samples', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.pair_samples < 1:
        parser.error('--pair-samples must be positive')
    analyze(args.trajectory, args.output_dir, args.pair_samples, args.seed)
    print(f'Full lifecycle analysis complete: {args.output_dir}')


if __name__ == '__main__':
    main()
