#!/usr/bin/env python3
"""Summarize LLaDA2 MoE route traces for the saturation experiment."""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _uniform_active_ratio(query_tokens: int, top_k: int, num_experts: int) -> float:
    """Uniform top-k-per-token routing null model with no fitted parameters."""
    if query_tokens <= 0 or top_k <= 0 or num_experts <= 0:
        return 0.0
    selected_fraction = min(top_k, num_experts) / num_experts
    return 1.0 - (1.0 - selected_fraction)**query_tokens


def load_observations(paths, include_partial_batches=False):
    """Load layer observations and their trace metadata."""
    observations = []
    skipped_partial = 0
    for path_value in paths:
        path = Path(path_value)
        metadata = None
        with path.open('r', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_type = record.get('record_type')
                if record_type == 'metadata':
                    metadata = record
                    continue
                if record_type != 'decode_forward':
                    continue
                if metadata is None:
                    raise ValueError(f'{path}:{line_number}: decode record appears before metadata')

                configured_batch = int(metadata['configured_batch_size'])
                actual_batch = int(record['actual_batch_size'])
                if not include_partial_batches and actual_batch != configured_batch:
                    skipped_partial += 1
                    continue

                query_tokens = int(record['query_tokens'])
                num_experts = int(metadata['num_experts'])
                top_k = int(metadata['top_k'])
                null_ratio = _uniform_active_ratio(query_tokens, top_k, num_experts)
                for layer in record['layers']:
                    active_experts = int(layer['active_experts'])
                    assignments = int(layer['assignments'])
                    observations.append({
                        'source': str(path),
                        'configured_batch_size': configured_batch,
                        'actual_batch_size': actual_batch,
                        'forward_index': int(record['forward_index']),
                        'layer_idx': int(layer['layer_idx']),
                        'query_tokens': query_tokens,
                        'num_experts': num_experts,
                        'top_k': top_k,
                        'active_experts': active_experts,
                        'active_ratio': active_experts / num_experts,
                        'assignments': assignments,
                        'assignments_per_active_expert': assignments / active_experts if active_experts else 0.0,
                        'uniform_null_active_ratio': null_ratio,
                    })
    return observations, skipped_partial


def _make_row(batch_size, layer_idx, values):
    active_experts = [value['active_experts'] for value in values]
    return {
        'configured_batch_size': batch_size,
        'layer_idx': layer_idx,
        'samples': len(values),
        'mean_actual_batch_size': statistics.fmean(value['actual_batch_size'] for value in values),
        'mean_query_tokens': statistics.fmean(value['query_tokens'] for value in values),
        'mean_active_experts': statistics.fmean(active_experts),
        'min_active_experts': min(active_experts),
        'max_active_experts': max(active_experts),
        'mean_active_ratio': statistics.fmean(value['active_ratio'] for value in values),
        'mean_assignments_per_active_expert': statistics.fmean(
            value['assignments_per_active_expert'] for value in values),
        'uniform_null_active_ratio': statistics.fmean(value['uniform_null_active_ratio'] for value in values),
    }


def summarize(observations):
    """Create per-layer and all-layer summary rows."""
    grouped = defaultdict(list)
    forward_groups = defaultdict(list)
    for observation in observations:
        batch_size = observation['configured_batch_size']
        grouped[(batch_size, observation['layer_idx'])].append(observation)
        forward_key = (observation['source'], batch_size, observation['forward_index'])
        forward_groups[forward_key].append(observation)

    overall = defaultdict(list)
    for (_, batch_size, _), values in forward_groups.items():
        template = values[0]
        overall[batch_size].append({
            **template,
            'active_experts': statistics.fmean(value['active_experts'] for value in values),
            'active_ratio': statistics.fmean(value['active_ratio'] for value in values),
            'assignments': statistics.fmean(value['assignments'] for value in values),
            'assignments_per_active_expert': statistics.fmean(
                value['assignments_per_active_expert'] for value in values),
        })

    rows = []
    for batch_size in sorted(overall):
        rows.append(_make_row(batch_size, 'all', overall[batch_size]))
        layer_keys = sorted(key for key in grouped if key[0] == batch_size)
        rows.extend(_make_row(batch_size, layer_idx, grouped[(batch_size, layer_idx)])
                    for _, layer_idx in layer_keys)
    return rows


def write_csv(rows, output_path):
    """Write summary rows."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(rows, output_path):
    """Write a dependency-free saturation plot for the all-layer rows."""
    points = [row for row in rows if row['layer_idx'] == 'all']
    if not points:
        return

    width, height = 760, 460
    left, right, top, bottom = 80, 30, 35, 65
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_coord(index):
        if len(points) == 1:
            return left + plot_width / 2
        return left + index * plot_width / (len(points) - 1)

    def y_coord(ratio):
        return top + (1.0 - max(0.0, min(1.0, ratio))) * plot_height

    actual_points = ' '.join(
        f'{x_coord(index):.1f},{y_coord(row["mean_active_ratio"]):.1f}' for index, row in enumerate(points))
    null_points = ' '.join(
        f'{x_coord(index):.1f},{y_coord(row["uniform_null_active_ratio"]):.1f}'
        for index, row in enumerate(points))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="22" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">LLaDA2 MoE Expert Saturation</text>',
    ]
    for percent in range(0, 101, 20):
        y = y_coord(percent / 100)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
                   f'font-family="sans-serif" font-size="12">{percent}%</text>')
    svg.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222222"/>')
    svg.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222222"/>')
    for index, row in enumerate(points):
        x = x_coord(index)
        svg.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" '
                   f'font-family="sans-serif" font-size="12">{row["configured_batch_size"]}</text>')
    svg.extend([
        f'<polyline points="{null_points}" fill="none" stroke="#999999" stroke-width="2" stroke-dasharray="6 4"/>',
        f'<polyline points="{actual_points}" fill="none" stroke="#1769aa" stroke-width="3"/>',
    ])
    for index, row in enumerate(points):
        x = x_coord(index)
        y = y_coord(row['mean_active_ratio'])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1769aa"/>')
    svg.extend([
        f'<text x="{left + plot_width/2:.1f}" y="{height-15}" text-anchor="middle" '
        'font-family="sans-serif" font-size="13">Configured request batch size</text>',
        f'<text x="18" y="{top + plot_height/2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_height/2:.1f})" font-family="sans-serif" '
        'font-size="13">Active expert ratio</text>',
        f'<line x1="{width-220}" y1="48" x2="{width-190}" y2="48" stroke="#1769aa" stroke-width="3"/>',
        f'<text x="{width-182}" y="52" font-family="sans-serif" font-size="12">Measured</text>',
        f'<line x1="{width-110}" y1="48" x2="{width-80}" y2="48" stroke="#999999" '
        'stroke-width="2" stroke-dasharray="6 4"/>',
        f'<text x="{width-72}" y="52" font-family="sans-serif" font-size="12">Uniform null</text>',
        '</svg>',
    ])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(svg), encoding='utf-8')


def print_overall(rows):
    """Print the compact table used for the first hypothesis check."""
    print('batch\tsamples\tmean_Q\tmean_active\tactive_ratio\tuniform_null\tassignments/active')
    for row in rows:
        if row['layer_idx'] != 'all':
            continue
        print(f'{row["configured_batch_size"]}\t{row["samples"]}\t'
              f'{row["mean_query_tokens"]:.1f}\t{row["mean_active_experts"]:.1f}\t'
              f'{100 * row["mean_active_ratio"]:.2f}%\t'
              f'{100 * row["uniform_null_active_ratio"]:.2f}%\t'
              f'{row["mean_assignments_per_active_expert"]:.2f}')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze LLaDA2 MoE saturation JSONL traces.')
    parser.add_argument('traces', nargs='+', help='One or more route trace JSONL files.')
    parser.add_argument('--output-csv', default='results/moe_saturation_summary.csv')
    parser.add_argument('--output-svg', default='results/moe_saturation.svg')
    parser.add_argument('--include-partial-batches', action='store_true',
                        help='Include forwards whose actual batch is smaller than the configured batch.')
    return parser.parse_args()


def main():
    args = parse_args()
    observations, skipped_partial = load_observations(args.traces, args.include_partial_batches)
    if not observations:
        raise SystemExit('No matching decode-forward observations found. '
                         'Use --include-partial-batches to inspect under-filled forwards.')
    rows = summarize(observations)
    write_csv(rows, args.output_csv)
    write_svg(rows, args.output_svg)
    print_overall(rows)
    print(f'\nWrote {args.output_csv} and {args.output_svg}.')
    if skipped_partial:
        print(f'Skipped {skipped_partial} under-filled decode forwards.', file=sys.stderr)


if __name__ == '__main__':
    main()
