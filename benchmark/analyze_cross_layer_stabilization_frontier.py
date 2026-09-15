#!/usr/bin/env python3
"""Find a parameter-free cross-layer route-stability frontier.

For every distance to token acceptance, this script builds the mean route-
Jaccard curve over MoE depth.  It then tests every boundary between adjacent
MoE layers and selects the single split with minimum within-segment squared
error.  No Jaccard threshold or hand-selected layer range is introduced.
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_llada2_expert_trajectory import load_events


def best_single_frontier(layer_means):
    """Return the minimum-SSE split between adjacent ordered layers."""
    if len(layer_means) < 2:
        raise ValueError('At least two layer means are required.')
    values = [float(value) for _, value in layer_means]
    overall_mean = statistics.fmean(values)
    total_sse = sum((value - overall_mean) ** 2 for value in values)
    best = None
    for split_index in range(1, len(layer_means)):
        before = values[:split_index]
        after = values[split_index:]
        mean_before = statistics.fmean(before)
        mean_after = statistics.fmean(after)
        split_sse = (
            sum((value - mean_before) ** 2 for value in before)
            + sum((value - mean_after) ** 2 for value in after)
        )
        candidate = (split_sse, split_index, mean_before, mean_after)
        if best is None or candidate < best:
            best = candidate
    split_sse, split_index, mean_before, mean_after = best
    delta = mean_after - mean_before
    if delta > 0:
        direction = 'stabilizing_with_depth'
    elif delta < 0:
        direction = 'destabilizing_with_depth'
    else:
        direction = 'flat'
    return {
        'frontier_left_layer': int(layer_means[split_index - 1][0]),
        'frontier_right_layer': int(layer_means[split_index][0]),
        'mean_before_frontier': mean_before,
        'mean_after_frontier': mean_after,
        'delta_after_minus_before': delta,
        'direction': direction,
        'total_sse': total_sse,
        'split_sse': split_sse,
        'variance_explained': 1.0 - split_sse / total_sse if total_sse else 0.0,
    }


def analyze(events):
    """Build layer-distance profiles and frontier summaries."""
    grouped = defaultdict(lambda: defaultdict(list))
    for event in events:
        distance = event['steps_until_acceptance']
        if distance == '' or int(distance) < 0:
            continue
        grouped[int(distance)][int(event['layer_idx'])].append(float(event['route_jaccard']))

    profile_rows = []
    summary_rows = []
    for distance, layers in sorted(grouped.items()):
        sample_counts = {len(values) for values in layers.values()}
        if len(sample_counts) != 1:
            raise ValueError(f'Distance {distance} has unequal token counts across layers: {sample_counts}')
        layer_means = []
        for layer_idx, values in sorted(layers.items()):
            mean_value = statistics.fmean(values)
            layer_means.append((layer_idx, mean_value))
            profile_rows.append({
                'steps_until_acceptance': distance,
                'layer_idx': layer_idx,
                'token_transitions': len(values),
                'mean_route_jaccard': mean_value,
                'median_route_jaccard': statistics.median(values),
                'mean_route_change': 1.0 - mean_value,
            })
        frontier = best_single_frontier(layer_means)
        summary_rows.append({
            'steps_until_acceptance': distance,
            'token_transitions_per_layer': next(iter(sample_counts)),
            **frontier,
        })
    return profile_rows, summary_rows


def direction_changes(summary_rows):
    """Report adjacent acceptance distances at which frontier direction flips."""
    changes = []
    for left, right in zip(summary_rows, summary_rows[1:]):
        if left['direction'] != right['direction']:
            changes.append({
                'nearer_distance': left['steps_until_acceptance'],
                'nearer_direction': left['direction'],
                'farther_distance': right['steps_until_acceptance'],
                'farther_direction': right['direction'],
            })
    return changes


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory')
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    events = load_events([args.trajectory])
    profile_rows, summary_rows = analyze(events)
    if not summary_rows:
        raise SystemExit('No token transitions with a known acceptance step were found.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'cross_layer_distance_profile.csv', profile_rows)
    write_csv(args.output_dir / 'cross_layer_frontier_summary.csv', summary_rows)
    changes = direction_changes(summary_rows)
    (args.output_dir / 'cross_layer_direction_changes.json').write_text(
        json.dumps({'format_version': 1, 'direction_changes': changes}, indent=2),
        encoding='utf-8')

    print('distance\tfrontier\tdelta\tdirection\tR2\ttokens/layer')
    for row in summary_rows:
        print(
            f"{row['steps_until_acceptance']}\t"
            f"{row['frontier_left_layer']}|{row['frontier_right_layer']}\t"
            f"{row['delta_after_minus_before']:.4f}\t{row['direction']}\t"
            f"{row['variance_explained']:.4f}\t{row['token_transitions_per_layer']}"
        )
    print(f'Direction changes: {changes}')
    print(f'Wrote cross-layer frontier analysis to {args.output_dir}.')


if __name__ == '__main__':
    main()
