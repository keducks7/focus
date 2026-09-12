#!/usr/bin/env python3
"""Offline tests for temporal MoE predictability and query-count null effects."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_trace(path):
    """Load one full-denoising route trace into keyed layer observations."""
    metadata = None
    observations = {}
    with Path(path).open('r', encoding='utf-8') as stream:
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
                raise ValueError(f'{path}:{line_number}: record appears before metadata')
            group_id = int(record['group_id'])
            step = int(record['step'])
            query_tokens = int(record['query_tokens'])
            for layer in record['layers']:
                layer_idx = int(layer['layer_idx'])
                loads = np.asarray(layer['expert_load'], dtype=np.int64)
                if loads.size != int(metadata['num_experts']):
                    raise ValueError(f'{path}:{line_number}: invalid expert histogram length')
                expected = query_tokens * int(metadata['top_k'])
                if int(loads.sum()) != expected:
                    raise ValueError(
                        f'{path}:{line_number}: layer {layer_idx} has {loads.sum()} '
                        f'assignments, expected {expected}')
                observations[(group_id, step, layer_idx)] = {
                    'group_id': group_id,
                    'step': step,
                    'layer_idx': layer_idx,
                    'query_tokens': query_tokens,
                    'loads': loads,
                }
    if metadata is None or not observations:
        raise ValueError(f'No full-denoising observations found in {path}')
    return metadata, observations


def _rank(loads):
    return np.lexsort((np.arange(loads.size), -loads))


def _ranking_auc(current_loads, order):
    """Area under cumulative current-load coverage for a supplied expert order."""
    total = int(current_loads.sum())
    if total == 0:
        return 0.0
    coverage = np.cumsum(current_loads[order], dtype=np.float64) / total
    return float(coverage.mean())


def _cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def temporal_predictability(observations, num_experts, hot_k=10, max_lag=8):
    """Measure how well an earlier load ranking predicts a later step."""
    rows = []
    random_auc = (num_experts + 1) / (2 * num_experts)
    keys = set(observations)
    for group_id, target_step, layer_idx in sorted(keys):
        current = observations[(group_id, target_step, layer_idx)]
        current_loads = current['loads']
        current_order = _rank(current_loads)
        oracle_auc = _ranking_auc(current_loads, current_order)
        current_hot = frozenset(int(index) for index in current_order[:hot_k])
        for lag in range(1, min(max_lag, target_step) + 1):
            previous_key = (group_id, target_step - lag, layer_idx)
            if previous_key not in observations:
                continue
            previous_loads = observations[previous_key]['loads']
            previous_order = _rank(previous_loads)
            previous_auc = _ranking_auc(current_loads, previous_order)
            denominator = oracle_auc - random_auc
            recovered_gain = ((previous_auc - random_auc) / denominator
                              if denominator > 0 else 0.0)
            previous_hot = frozenset(int(index) for index in previous_order[:hot_k])
            rows.append({
                'group_id': group_id,
                'layer_idx': layer_idx,
                'source_step': target_step - lag,
                'target_step': target_step,
                'lag': lag,
                'query_tokens': current['query_tokens'],
                'oracle_auc': oracle_auc,
                'previous_auc': previous_auc,
                'random_auc': random_auc,
                'recovered_oracle_gain': recovered_gain,
                'load_cosine': _cosine(previous_loads, current_loads),
                'top10_jaccard': _jaccard(previous_hot, current_hot),
                'previous_top10_current_load_share': (
                    float(current_loads[list(previous_hot)].sum() / current_loads.sum())),
            })
    return rows


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def summarize_temporal(rows, main_max_step=12):
    grouped = defaultdict(list)
    for row in rows:
        grouped[('all', row['lag'])].append(row)
        if row['target_step'] <= main_max_step:
            grouped[('main', row['lag'])].append(row)
    summary = []
    for (phase, lag), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary.append({
            'phase': phase,
            'lag': lag,
            'observations': len(values),
            'mean_query_tokens': _mean([row['query_tokens'] for row in values]),
            'mean_oracle_auc': _mean([row['oracle_auc'] for row in values]),
            'mean_previous_auc': _mean([row['previous_auc'] for row in values]),
            'mean_random_auc': _mean([row['random_auc'] for row in values]),
            'mean_recovered_oracle_gain': _mean([row['recovered_oracle_gain'] for row in values]),
            'median_recovered_oracle_gain': statistics.median(
                row['recovered_oracle_gain'] for row in values),
            'mean_load_cosine': _mean([row['load_cosine'] for row in values]),
            'mean_top10_jaccard': _mean([row['top10_jaccard'] for row in values]),
            'mean_previous_top10_current_load_share': _mean(
                [row['previous_top10_current_load_share'] for row in values]),
        })
    return summary


def _load_metrics(loads, hot_k):
    total = loads.sum(axis=-1)
    active = np.count_nonzero(loads, axis=-1)
    squared = np.square(loads, dtype=np.float64).sum(axis=-1)
    effective = np.divide(np.square(total, dtype=np.float64), squared,
                          out=np.zeros_like(squared), where=squared > 0)
    top_count = min(hot_k, loads.shape[-1])
    top_load = np.partition(loads, -top_count, axis=-1)[..., -top_count:].sum(axis=-1)
    top_share = np.divide(top_load, total, out=np.zeros_like(top_load, dtype=np.float64), where=total > 0)
    return active, effective, top_share


def query_matched_null(observations, hot_k=10, repeats=256, seed=0):
    """Compare each step with step-0 load histograms thinned to equal assignments."""
    rng = np.random.default_rng(seed)
    rows = []
    for (group_id, step, layer_idx), current in sorted(observations.items()):
        if step == 0:
            continue
        base = observations.get((group_id, 0, layer_idx))
        if base is None:
            continue
        current_loads = current['loads']
        assignments = int(current_loads.sum())
        base_loads = base['loads']
        if assignments > int(base_loads.sum()):
            raise ValueError('Query-matched null cannot sample more assignments than step 0')
        samples = rng.multivariate_hypergeometric(
            base_loads, assignments, size=repeats, method='marginals')
        null_active, null_effective, null_share = _load_metrics(samples, hot_k)
        observed_active, observed_effective, observed_share = _load_metrics(
            current_loads[np.newaxis, :], hot_k)
        metrics = (
            ('active_experts', float(observed_active[0]), null_active),
            ('effective_experts', float(observed_effective[0]), null_effective),
            ('top10_load_share', float(observed_share[0]), null_share),
        )
        for metric, observed, null_values in metrics:
            lower, upper = np.quantile(null_values, [0.025, 0.975])
            null_mean = float(null_values.mean())
            rows.append({
                'group_id': group_id,
                'layer_idx': layer_idx,
                'step': step,
                'query_tokens': current['query_tokens'],
                'assignments': assignments,
                'metric': metric,
                'observed': observed,
                'null_mean': null_mean,
                'null_p025': float(lower),
                'null_p975': float(upper),
                'observed_minus_null': observed - null_mean,
                'outside_null_95': int(observed < lower or observed > upper),
            })
    return rows


def summarize_null(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['step'], row['metric'])].append(row)
    summary = []
    for (step, metric), values in sorted(grouped.items()):
        summary.append({
            'step': step,
            'metric': metric,
            'observations': len(values),
            'mean_query_tokens': _mean([row['query_tokens'] for row in values]),
            'mean_observed': _mean([row['observed'] for row in values]),
            'mean_null': _mean([row['null_mean'] for row in values]),
            'mean_observed_minus_null': _mean([row['observed_minus_null'] for row in values]),
            'fraction_outside_null_95': _mean([row['outside_null_95'] for row in values]),
        })
    return summary


def write_csv(rows, path):
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _polyline(points, color, dashed=False):
    coords = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
    dash = ' stroke-dasharray="7 4"' if dashed else ''
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>'


def write_temporal_svg(summary, path, num_experts, hot_k):
    rows = [row for row in summary if row['phase'] == 'main']
    if not rows:
        return
    width, height = 900, 720
    left, right, panel_height, gap = 78, 30, 150, 65
    panel_width = width - left - right
    tops = [70, 70 + panel_height + gap, 70 + 2 * (panel_height + gap)]
    max_lag = max(row['lag'] for row in rows)
    specs = [
        ('Recovered oracle ranking gain', 'mean_recovered_oracle_gain'),
        ('Full-load-vector cosine', 'mean_load_cosine'),
        ('Top-10 Jaccard', 'mean_top10_jaccard'),
    ]

    def x(value):
        return left + (value - 1) / max(1, max_lag - 1) * panel_width

    def y(value, top):
        return top + (1 - max(0.0, min(1.0, value))) * panel_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">'
        'Previous-Step Expert Ranking Predictability (Steps 0–12)</text>',
    ]
    for panel, (title, key) in enumerate(specs):
        top = tops[panel]
        for tick in range(6):
            value = tick / 5
            yy = y(value, top)
            svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e5e5"/>')
            svg.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
        svg.append(f'<text x="{left}" y="{top-12}" font-family="sans-serif" font-size="14">{title}</text>')
        svg.append(_polyline([(x(row['lag']), y(row[key], top)) for row in rows], '#1769aa'))
        if key == 'mean_top10_jaccard':
            baseline = hot_k / (2 * num_experts - hot_k)
            svg.append(f'<line x1="{left}" y1="{y(baseline, top):.1f}" x2="{width-right}" '
                       f'y2="{y(baseline, top):.1f}" stroke="#d1495b" stroke-dasharray="7 4"/>')
        svg.append(f'<line x1="{left}" y1="{top+panel_height}" x2="{width-right}" y2="{top+panel_height}" stroke="#222"/>')
    bottom = tops[-1] + panel_height
    for lag in range(1, max_lag + 1):
        svg.append(f'<text x="{x(lag):.1f}" y="{bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{lag}</text>')
    svg.append(f'<text x="{left+panel_width/2:.1f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">Step lag</text>')
    svg.append('</svg>')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(svg), encoding='utf-8')


def write_null_svg(summary, path):
    if not summary:
        return
    by_metric = defaultdict(list)
    for row in summary:
        by_metric[row['metric']].append(row)
    width, height = 900, 720
    left, right, panel_height, gap = 78, 30, 150, 65
    panel_width = width - left - right
    tops = [70, 70 + panel_height + gap, 70 + 2 * (panel_height + gap)]
    specs = [
        ('Active experts', 'active_experts'),
        ('Effective experts', 'effective_experts'),
        ('Top-10 load share', 'top10_load_share'),
    ]
    max_step = max(row['step'] for row in summary)

    def x(value):
        return left + (value - 1) / max(1, max_step - 1) * panel_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">'
        'Observed Routing vs Query-Matched Step-0 Null</text>',
    ]
    for panel, (title, metric) in enumerate(specs):
        top = tops[panel]
        rows = by_metric[metric]
        maximum = 1.0 if metric == 'top10_load_share' else max(
            max(row['mean_observed'], row['mean_null']) for row in rows) * 1.08

        def y(value):
            return top + (1 - max(0.0, min(maximum, value)) / maximum) * panel_height

        for tick in range(6):
            value = maximum * tick / 5
            yy = y(value)
            label = f'{value:.1f}' if maximum > 1 else f'{value:.1f}'
            svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e5e5"/>')
            svg.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{label}</text>')
        svg.append(f'<text x="{left}" y="{top-12}" font-family="sans-serif" font-size="14">{title}</text>')
        svg.append(_polyline([(x(row['step']), y(row['mean_observed'])) for row in rows], '#1769aa'))
        svg.append(_polyline([(x(row['step']), y(row['mean_null'])) for row in rows], '#d1495b', dashed=True))
        svg.append(f'<line x1="{left}" y1="{top+panel_height}" x2="{width-right}" y2="{top+panel_height}" stroke="#222"/>')
    bottom = tops[-1] + panel_height
    for step in range(1, max_step + 1, max(1, math.ceil(max_step / 8))):
        svg.append(f'<text x="{x(step):.1f}" y="{bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{step}</text>')
    svg.extend([
        '<line x1="630" y1="48" x2="654" y2="48" stroke="#1769aa" stroke-width="2.5"/>',
        '<text x="660" y="52" font-family="sans-serif" font-size="11">observed</text>',
        '<line x1="740" y1="48" x2="764" y2="48" stroke="#d1495b" stroke-width="2.5" stroke-dasharray="7 4"/>',
        '<text x="770" y="52" font-family="sans-serif" font-size="11">query-matched null</text>',
        f'<text x="{left+panel_width/2:.1f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">Target denoising step</text>',
        '</svg>',
    ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(svg), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hot-k', type=int, default=10)
    parser.add_argument('--max-lag', type=int, default=8)
    parser.add_argument('--main-max-step', type=int, default=12)
    parser.add_argument('--null-repeats', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    metadata, observations = load_trace(args.trace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal = temporal_predictability(
        observations, int(metadata['num_experts']), args.hot_k, args.max_lag)
    temporal_summary = summarize_temporal(temporal, args.main_max_step)
    null_rows = query_matched_null(observations, args.hot_k, args.null_repeats, args.seed)
    null_summary = summarize_null(null_rows)

    write_csv(temporal, output_dir / 'temporal_predictability_pairs.csv')
    write_csv(temporal_summary, output_dir / 'temporal_predictability_summary.csv')
    write_temporal_svg(
        temporal_summary,
        output_dir / 'temporal_predictability.svg',
        int(metadata['num_experts']),
        args.hot_k,
    )
    write_csv(null_rows, output_dir / 'query_matched_null_layers.csv')
    write_csv(null_summary, output_dir / 'query_matched_null_summary.csv')
    write_null_svg(null_summary, output_dir / 'query_matched_null.svg')

    print('Temporal predictability (main phase):')
    print('lag\trecovered_gain\tcosine\ttop10_jaccard\tprevious_top10_share')
    for row in temporal_summary:
        if row['phase'] == 'main':
            print(f'{row["lag"]}\t{row["mean_recovered_oracle_gain"]:.4f}\t'
                  f'{row["mean_load_cosine"]:.4f}\t{row["mean_top10_jaccard"]:.4f}\t'
                  f'{row["mean_previous_top10_current_load_share"]:.4f}')
    print(f'\nWrote offline hypothesis results to {output_dir}')


if __name__ == '__main__':
    main()
