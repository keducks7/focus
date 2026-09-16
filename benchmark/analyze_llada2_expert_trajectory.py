#!/usr/bin/env python3
"""Analyze whether token-level expert routes predict denoising acceptance."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def binary_auc(scores, labels):
    """Tie-aware Mann-Whitney AUC without external dependencies."""
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(bool(label) for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def load_events(paths):
    """Create same-token, same-layer adjacent-step route observations."""
    events = []
    for path_value in paths:
        path = Path(path_value)
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not records or records[0].get('record_type') != 'metadata':
            raise ValueError(f'{path}: missing trajectory metadata')
        metadata = records[0]
        expected_layers = int(metadata['num_hidden_layers']) - int(metadata['first_moe_layer'])
        expected_layer_ids = set(range(int(metadata['first_moe_layer']),
                                       int(metadata['num_hidden_layers'])))
        top_k = int(metadata['top_k'])
        token_rows = []
        acceptance_step = {}
        accepted_tokens = set()
        for record in records[1:]:
            if record.get('record_type') != 'token_trajectory_step':
                continue
            step = int(record['step'])
            group_id = int(record['group_id'])
            tokens = record['tokens']
            if int(record['query_tokens']) != len(tokens):
                raise ValueError(f'{path}: query_tokens does not match the token record count')
            identities_this_step = set()
            for token in tokens:
                # Keep the original acceptance analysis strictly MASK-only.
                if not token.get('masked_before', True):
                    continue
                identity = (str(path), group_id, int(token['request_id']), int(token['block_id']),
                            int(token['block_position']))
                if identity in identities_this_step:
                    raise ValueError(f'{path}: duplicate token identity at group {group_id}, step {step}')
                if identity in accepted_tokens:
                    raise ValueError(f'{path}: accepted token reappears at group {group_id}, step {step}')
                identities_this_step.add(identity)
                layers = token['layers']
                layer_ids = {int(layer['layer_idx']) for layer in layers}
                if len(layers) != expected_layers or layer_ids != expected_layer_ids:
                    raise ValueError(f'{path}: token has an incomplete or unexpected MoE layer trace')
                for layer in layers:
                    expert_ids = [int(value) for value in layer['expert_ids']]
                    if len(expert_ids) != top_k or len(set(expert_ids)) != top_k:
                        raise ValueError(f'{path}: invalid Top-k route at layer {layer["layer_idx"]}')
                row = (identity, step, token)
                token_rows.append(row)
                if token['accepted']:
                    acceptance_step.setdefault(identity, step)
                    accepted_tokens.add(identity)

        previous = {}
        for identity, step, token in sorted(token_rows, key=lambda row: (row[0], row[1])):
            for layer in token['layers']:
                layer_idx = int(layer['layer_idx'])
                key = identity + (layer_idx,)
                prior = previous.get(key)
                if prior is not None and prior['step'] + 1 == step:
                    accepted = bool(token['accepted'])
                    final_step = acceptance_step.get(identity)
                    events.append({
                        'source': identity[0],
                        'group_id': identity[1],
                        'request_id': identity[2],
                        'block_id': identity[3],
                        'block_position': identity[4],
                        'layer_idx': layer_idx,
                        'previous_step': prior['step'],
                        'step': step,
                        'route_jaccard': jaccard(prior['expert_ids'], layer['expert_ids']),
                        'accepted': int(accepted),
                        'confidence': float(token['confidence']),
                        'acceptance_step': '' if final_step is None else final_step,
                        'steps_until_acceptance': '' if final_step is None else final_step - step,
                    })
                previous[key] = {'step': step, 'expert_ids': layer['expert_ids']}
        seen_layers = {event['layer_idx'] for event in events}
        if events and len(seen_layers) != expected_layers:
            raise ValueError(f'{path}: expected {expected_layers} MoE layers, observed {len(seen_layers)}')
    return events


def summarize(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event['layer_idx']].append(event)
    rows = []
    for layer_idx, values in sorted(grouped.items()):
        accepted = [value for value in values if value['accepted']]
        unresolved = [value for value in values if not value['accepted']]
        route_auc = binary_auc([value['route_jaccard'] for value in values],
                               [value['accepted'] for value in values])
        confidence_auc = binary_auc([value['confidence'] for value in values],
                                    [value['accepted'] for value in values])
        rows.append({
            'layer_idx': layer_idx,
            'observations': len(values),
            'accepted_observations': len(accepted),
            'unresolved_observations': len(unresolved),
            'mean_route_jaccard_accepted': statistics.fmean(
                value['route_jaccard'] for value in accepted) if accepted else '',
            'mean_route_jaccard_unresolved': statistics.fmean(
                value['route_jaccard'] for value in unresolved) if unresolved else '',
            'route_stability_acceptance_auc': '' if route_auc is None else route_auc,
            'confidence_acceptance_auc': '' if confidence_auc is None else confidence_auc,
        })
    return rows


def summarize_distance(events):
    grouped = defaultdict(list)
    for event in events:
        if event['steps_until_acceptance'] != '' and event['steps_until_acceptance'] >= 0:
            grouped[event['steps_until_acceptance']].append(event['route_jaccard'])
    return [{
        'steps_until_acceptance': distance,
        'observations': len(values),
        'mean_route_jaccard': statistics.fmean(values),
        'median_route_jaccard': statistics.median(values),
    } for distance, values in sorted(grouped.items())]


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f'Cannot infer CSV columns for empty output: {path}')
        fieldnames = list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectories', nargs='+')
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    events = load_events(args.trajectories)
    if not events:
        raise SystemExit('No adjacent token-route events found.')
    layer_rows = summarize(events)
    distance_rows = summarize_distance(events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'token_route_events.csv', events)
    write_csv(args.output_dir / 'token_route_layer_summary.csv', layer_rows)
    write_csv(args.output_dir / 'route_stability_by_acceptance_distance.csv', distance_rows,
              ['steps_until_acceptance', 'observations', 'mean_route_jaccard', 'median_route_jaccard'])
    print('layer\taccepted_J\tunresolved_J\troute_AUC\tconfidence_AUC')
    for row in layer_rows:
        values = [row['mean_route_jaccard_accepted'], row['mean_route_jaccard_unresolved'],
                  row['route_stability_acceptance_auc'], row['confidence_acceptance_auc']]
        formatted = ['-' if value == '' else f'{value:.4f}' for value in values]
        print(f'{row["layer_idx"]}\t' + '\t'.join(formatted))
    print(f'Wrote token-trajectory analysis to {args.output_dir}.')


if __name__ == '__main__':
    main()
