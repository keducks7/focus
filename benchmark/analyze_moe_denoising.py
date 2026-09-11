#!/usr/bin/env python3
"""Analyze per-step LLaDA2 MoE routing during one-block denoising."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load_metrics(loads, hot_k):
    assignments = sum(loads)
    active = sum(value > 0 for value in loads)
    squared_sum = sum(value * value for value in loads)
    effective = assignments * assignments / squared_sum if squared_sum else 0.0
    ranked = sorted((index for index, value in enumerate(loads) if value > 0),
                    key=lambda index: (-loads[index], index))
    hot = frozenset(ranked[:hot_k])
    top_share = sum(loads[index] for index in hot) / assignments if assignments else 0.0
    return assignments, active, effective, hot, top_share


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_observations(paths, hot_k=10):
    """Load raw layer histograms and compute parameter-free concentration metrics."""
    observations = []
    previous_hot = {}
    for path_value in paths:
        path = Path(path_value)
        metadata = None
        with path.open('r', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get('record_type') == 'metadata':
                    metadata = record
                    continue
                if record.get('record_type') != 'denoising_step':
                    continue
                if metadata is None:
                    raise ValueError(f'{path}:{line_number}: denoising step appears before metadata')

                batch_size = int(metadata['configured_batch_size'])
                block_length = int(metadata['block_length'])
                num_experts = int(metadata['num_experts'])
                top_k = int(metadata['top_k'])
                group_id = int(record['group_id'])
                step = int(record['step'])
                query_tokens = int(record['query_tokens'])
                for layer in record['layers']:
                    layer_idx = int(layer['layer_idx'])
                    loads = [int(value) for value in layer['expert_load']]
                    if len(loads) != num_experts:
                        raise ValueError(f'{path}:{line_number}: expected {num_experts} expert loads, got {len(loads)}')
                    assignments, active, effective, hot, top_share = _load_metrics(loads, hot_k)
                    expected_assignments = query_tokens * top_k
                    if assignments != expected_assignments:
                        raise ValueError(f'{path}:{line_number}: layer {layer_idx} has {assignments} assignments; '
                                         f'expected query_tokens*top_k={expected_assignments}')
                    trajectory_key = (str(path), group_id, layer_idx)
                    previous = previous_hot.get(trajectory_key)
                    overlap = None
                    if previous is not None and previous[0] + 1 == step:
                        overlap = _jaccard(previous[1], hot)
                    previous_hot[trajectory_key] = (step, hot)
                    observations.append({
                        'source': str(path),
                        'configured_batch_size': batch_size,
                        'group_id': group_id,
                        'step': step,
                        'layer_idx': layer_idx,
                        'query_tokens': query_tokens,
                        'query_fraction': query_tokens / (batch_size * block_length),
                        'num_experts': num_experts,
                        'top_k': top_k,
                        'assignments': assignments,
                        'active_experts': active,
                        'active_ratio': active / num_experts,
                        'effective_experts': effective,
                        'effective_ratio': effective / num_experts,
                        'top10_load_share': top_share,
                        'adjacent_top10_jaccard': overlap,
                        'hot_experts': ' '.join(str(index) for index in sorted(hot)),
                    })
    return observations


def _percentile(values, fraction):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def summarize(observations, hot_k=10):
    grouped = defaultdict(list)
    for observation in observations:
        grouped[(observation['configured_batch_size'], observation['step'])].append(observation)

    rows = []
    for (batch_size, step), values in sorted(grouped.items()):
        group_queries = {}
        for value in values:
            group_queries[(value['source'], value['group_id'])] = value['query_tokens']
        effective_values = [value['effective_experts'] for value in values]
        overlaps = [value['adjacent_top10_jaccard'] for value in values
                    if value['adjacent_top10_jaccard'] is not None]
        num_experts = values[0]['num_experts']
        random_jaccard = hot_k / (2 * num_experts - hot_k)
        rows.append({
            'configured_batch_size': batch_size,
            'step': step,
            'groups': len(group_queries),
            'layer_observations': len(values),
            'mean_query_tokens': statistics.fmean(group_queries.values()),
            'mean_query_fraction': statistics.fmean(value['query_fraction'] for value in values),
            'mean_active_experts': statistics.fmean(value['active_experts'] for value in values),
            'mean_active_ratio': statistics.fmean(value['active_ratio'] for value in values),
            'mean_effective_experts': statistics.fmean(effective_values),
            'median_effective_experts': statistics.median(effective_values),
            'p25_effective_experts': _percentile(effective_values, 0.25),
            'p75_effective_experts': _percentile(effective_values, 0.75),
            'mean_effective_ratio': statistics.fmean(value['effective_ratio'] for value in values),
            'mean_top10_load_share': statistics.fmean(value['top10_load_share'] for value in values),
            'mean_adjacent_top10_jaccard': statistics.fmean(overlaps) if overlaps else '',
            'random_top10_jaccard': random_jaccard,
        })
    return rows


def write_csv(rows, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(rows, output_path):
    """Write three normalized trajectory panels without plotting dependencies."""
    if not rows:
        return
    batches = sorted({row['configured_batch_size'] for row in rows})
    by_batch = {batch: [row for row in rows if row['configured_batch_size'] == batch] for batch in batches}
    max_step = max(row['step'] for row in rows)
    width, height = 900, 860
    left, right, panel_width = 82, 32, width - 82 - 32
    panel_height, panel_gap = 190, 70
    panel_tops = [70, 70 + panel_height + panel_gap, 70 + 2 * (panel_height + panel_gap)]
    colors = ['#1769aa', '#d1495b', '#2a9d8f', '#7b2cbf', '#e76f51', '#5c677d']

    def x_coord(step):
        return left + (step / max_step if max_step else 0.5) * panel_width

    def y_coord(value, top):
        return top + (1 - max(0.0, min(1.0, float(value)))) * panel_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">'
        'LLaDA2 MoE Routing Across Denoising Steps</text>',
    ]
    panel_specs = [
        ('Remaining query fraction', [('mean_query_fraction', False)]),
        ('Expert working-set ratio', [('mean_active_ratio', False), ('mean_effective_ratio', True)]),
        ('Hot-expert concentration / stability',
         [('mean_top10_load_share', False), ('mean_adjacent_top10_jaccard', True)]),
    ]
    for panel_index, (title, metrics) in enumerate(panel_specs):
        top = panel_tops[panel_index]
        for tick in range(0, 6):
            value = tick / 5
            y = y_coord(value, top)
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
            svg.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" '
                       f'font-size="11">{value:.1f}</text>')
        svg.append(f'<text x="{left}" y="{top-14}" font-family="sans-serif" font-size="14">{title}</text>')
        svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_height}" stroke="#222"/>')
        svg.append(f'<line x1="{left}" y1="{top+panel_height}" x2="{width-right}" '
                   f'y2="{top+panel_height}" stroke="#222"/>')
        for batch_index, batch in enumerate(batches):
            points = by_batch[batch]
            color = colors[batch_index % len(colors)]
            for metric, dashed in metrics:
                usable = [row for row in points if row[metric] != '']
                coords = ' '.join(f'{x_coord(row["step"]):.1f},{y_coord(row[metric], top):.1f}' for row in usable)
                dash = ' stroke-dasharray="7 4"' if dashed else ''
                svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                           f'stroke-width="2.5"{dash}/>')
    bottom = panel_tops[-1] + panel_height
    for step in range(0, max_step + 1, max(1, math_ceil_div(max_step + 1, 8))):
        x = x_coord(step)
        svg.append(f'<text x="{x:.1f}" y="{bottom+22}" text-anchor="middle" font-family="sans-serif" '
                   f'font-size="11">{step}</text>')
    svg.append(f'<text x="{left+panel_width/2:.1f}" y="{height-25}" text-anchor="middle" '
               'font-family="sans-serif" font-size="13">Denoising step</text>')
    legend_x = width - 310
    for index, batch in enumerate(batches):
        y = 45 + index * 18
        color = colors[index % len(colors)]
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" '
                   f'stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x+30}" y="{y+4}" font-family="sans-serif" font-size="11">batch {batch}</text>')
    svg.extend([
        '<line x1="720" y1="45" x2="744" y2="45" stroke="#333" stroke-width="2.5"/>',
        '<text x="750" y="49" font-family="sans-serif" font-size="11">active / share</text>',
        '<line x1="720" y1="62" x2="744" y2="62" stroke="#333" stroke-width="2.5" stroke-dasharray="7 4"/>',
        '<text x="750" y="66" font-family="sans-serif" font-size="11">effective / overlap</text>',
        '</svg>',
    ])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(svg), encoding='utf-8')


def math_ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def print_summary(rows):
    print('batch\tstep\tQ\tactive\teffective\ttop10_share\tadjacent_overlap')
    for row in rows:
        overlap = row['mean_adjacent_top10_jaccard']
        overlap_text = '-' if overlap == '' else f'{100 * overlap:.2f}%'
        print(f'{row["configured_batch_size"]}\t{row["step"]}\t{row["mean_query_tokens"]:.1f}\t'
              f'{row["mean_active_experts"]:.1f}\t{row["mean_effective_experts"]:.1f}\t'
              f'{100 * row["mean_top10_load_share"]:.2f}%\t{overlap_text}')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze LLaDA2 per-step MoE route traces.')
    parser.add_argument('traces', nargs='+')
    parser.add_argument('--hot-k', type=int, default=10)
    parser.add_argument('--output-layer-csv', default='results/moe_denoising_layers.csv')
    parser.add_argument('--output-summary-csv', default='results/moe_denoising_summary.csv')
    parser.add_argument('--output-svg', default='results/moe_denoising.svg')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.hot_k <= 0:
        raise SystemExit('--hot-k must be positive.')
    observations = load_observations(args.traces, args.hot_k)
    if not observations:
        raise SystemExit('No denoising-step records found.')
    rows = summarize(observations, args.hot_k)
    write_csv(observations, args.output_layer_csv)
    write_csv(rows, args.output_summary_csv)
    write_svg(rows, args.output_svg)
    print_summary(rows)
    print(f'\nWrote {args.output_layer_csv}, {args.output_summary_csv}, and {args.output_svg}.')


if __name__ == '__main__':
    main()
