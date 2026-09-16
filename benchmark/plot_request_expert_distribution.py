#!/usr/bin/env python3
"""Plot per-request expert assignments from existing token trajectories (CPU only)."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

LAYERS = (2, 10, 18)
STATES = ('accepted', 'unresolved')
COLORS = {'accepted': '#d86732', 'unresolved': '#256dad'}
RUNS = (('gsm8k_bs8_layer10_v1', 'GSM8K | batch 8', 8),
        ('gsm8k_bs16_layer10_v1', 'GSM8K | batch 16', 16),
        ('humaneval_bs8_layer10_v1', 'HumanEval | batch 8', 8))


def read_run(path):
    counts = {}
    sizes = {}
    with path.open() as stream:
        meta = json.loads(next(stream))
        batch, experts = meta['configured_batch_size'], meta['num_experts']
        for line in stream:
            record = json.loads(line)
            if record['record_type'] != 'token_trajectory_step':
                continue
            group, step = record['group_id'], record['step']
            for state in STATES:
                sizes[group, step, state] = np.zeros(batch, dtype=int)
                for layer in LAYERS:
                    counts[group, step, state, layer] = np.zeros((batch, experts), dtype=int)
            for token in record['tokens']:
                request = token['request_id'] - group * batch
                assert 0 <= request < batch
                state = 'accepted' if token['accepted'] else 'unresolved'
                sizes[group, step, state][request] += 1
                for layer in token['layers']:
                    if layer['layer_idx'] in LAYERS:
                        np.add.at(counts[group, step, state, layer['layer_idx']][request], layer['expert_ids'], 1)
            for state in STATES:
                for layer in LAYERS:
                    assert np.array_equal(counts[group, step, state, layer].sum(1), sizes[group, step, state] * meta['top_k'])
    return meta, counts, sizes


def distribution(count):
    total = count.sum(axis=1, keepdims=True)
    return np.divide(count, total, out=np.full(count.shape, np.nan), where=total > 0)


def save_figure(fig, output, name):
    for ext in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{ext}', dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def heatmap(data, title, output, name, step):
    meta, counts, sizes = data
    batch = meta['configured_batch_size']
    fig, axes = plt.subplots(3, 2, figsize=(17, 11), layout='constrained')
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#d8dee4')
    for row, layer in enumerate(LAYERS):
        # One fixed per-layer order across all plotted steps and states.
        total = sum((v.sum(0) for (g, t, s, l), v in counts.items() if l == layer), np.zeros(meta['num_experts'], dtype=int))
        order = np.lexsort((np.arange(meta['num_experts']), -total))
        for col, state in enumerate(STATES):
            ax = axes[row, col]
            n = sizes[0, step, state]
            im = ax.imshow(distribution(counts[0, step, state, layer])[:, order] * 100,
                           aspect='auto', interpolation='nearest', cmap=cmap, norm=Normalize(0, 12.5))
            ax.set_title(f'Layer {layer} | {state.title()} this step | tokens={n.sum()}, requests={(n > 0).sum()}/{batch}', fontsize=11)
            ax.set_yticks(np.arange(batch), [f'R{i:02d}  n={v}' for i, v in enumerate(n)], fontsize=8)
            ax.set_xticks([0, 31, 63, 127, 191, 255], ['1', '32', '64', '128', '192', '256'])
            ax.set_xlabel('Expert rank (fixed per layer; all 256 experts retained)', fontsize=9)
    fig.colorbar(im, ax=axes, shrink=.7, label='Share of request-state Top-8 assignments (%)')
    fig.suptitle(f'{title} | denoising step {step} | actual batch group 0\n'
                 'Tokens were masked before this forward; status is measured after its update. Grey = no tokens.', fontsize=15)
    save_figure(fig, output, f'{name}_step{step}_heatmap')


def summarize(data, run):
    meta, counts, sizes = data
    grouped = defaultdict(list)
    detail = []
    sharing = []
    for (group, step, state, layer), count in sorted(counts.items()):
        dist = distribution(count)
        valid = np.flatnonzero(sizes[group, step, state] > 0)
        values = [float(np.minimum(dist[a], dist[b]).sum()) for i, a in enumerate(valid) for b in valid[i+1:]]
        grouped[step, state, layer].append((values, len(valid), int(sizes[group, step, state].sum())))
        for i, a in enumerate(valid):
            for b in valid[i+1:]:
                detail.append(dict(run=run, group=group, step=step, state=state, layer=layer,
                                   request_a=group * meta['configured_batch_size'] + int(a),
                                   request_b=group * meta['configured_batch_size'] + int(b),
                                   tokens_a=int(sizes[group, step, state][a]), tokens_b=int(sizes[group, step, state][b]),
                                   overlap=float(np.minimum(dist[a], dist[b]).sum())))
        for expert in range(meta['num_experts']):
            sharing.append(dict(run=run, group=group, step=step, state=state, layer=layer, expert=expert,
                                requests_using=int((count[:, expert] > 0).sum()), assignments=int(count[:, expert].sum()),
                                eligible_requests=len(valid)))
    rows = []
    for (step, state, layer), groups in sorted(grouped.items()):
        values = [v for g, _, _ in groups for v in g]
        means = [np.mean(g) for g, _, _ in groups if g]
        rows.append(dict(run=run, step=step, state=state, layer=layer,
                         mean_overlap=float(np.mean(values)) if values else '',
                         group_min=float(min(means)) if means else '', group_max=float(max(means)) if means else '',
                         request_pairs=len(values), contributing_groups=len(means),
                         eligible_requests=sum(n for _, n, _ in groups), tokens=sum(n for _, _, n in groups)))
    return rows, detail, sharing


def curves(all_rows, output):
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True, sharey=True, layout='constrained')
    for col, (run, title, _) in enumerate(RUNS):
        for row, layer in enumerate(LAYERS):
            ax = axes[row, col]
            for state in STATES:
                vals = [r for r in all_rows if r['run'] == run and r['layer'] == layer and r['state'] == state]
                xs = [r['step'] for r in vals]
                ys = [r['mean_overlap'] if r['mean_overlap'] != '' else np.nan for r in vals]
                ax.plot(xs, ys, '.-', color=COLORS[state], label=state.title(), lw=1.6, markersize=4)
            ax.set_title(f'{title}\nLayer {layer}', fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(alpha=.18)
            if row == 2: ax.set_xlabel('Denoising step (0-based)')
            if col == 0: ax.set_ylabel('Mean request-pair overlap')
    axes[0, 0].legend(frameon=False)
    fig.suptitle('Within-batch expert-distribution overlap across requests\n'
                 'Pairs are formed only inside actual batches; pair-weighted means. Empty states excluded; no confidence intervals.', fontsize=15)
    save_figure(fig, output, 'request_overlap_curves')
    fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharex=True, layout='constrained')
    for col, (run, title, _) in enumerate(RUNS):
        for state in STATES:
            vals = [r for r in all_rows if r['run'] == run and r['layer'] == LAYERS[0] and r['state'] == state]
            for row, field in enumerate(('tokens', 'request_pairs')):
                axes[row,col].plot([r['step'] for r in vals], [r[field] for r in vals], '.-', color=COLORS[state], label=state.title())
                axes[row,col].grid(alpha=.18)
        axes[0,col].set_title(title)
        axes[1,col].set_xlabel('Denoising step')
    axes[0,0].set_ylabel('Tokens (all active groups)')
    axes[1,0].set_ylabel('Valid within-batch request pairs')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Sample support for the overlap curves | identical support at layers 2, 10, 18', fontsize=14)
    save_figure(fig, output, 'request_overlap_support')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    all_rows = []
    for run, title, batch in RUNS:
        data = read_run(args.input_root / run / f'token_trajectories_bs{batch}.jsonl')
        for step in (0, 4, 12):
            heatmap(data, title, args.output_dir, run, step)
        rows, detail, sharing = summarize(data, run)
        all_rows.extend(rows)
        write_csv(args.output_dir / f'{run}_pair_details.csv', detail)
        write_csv(args.output_dir / f'{run}_expert_sharing.csv', sharing)
        print(f'{run}: heatmaps and distribution statistics completed', flush=True)
    write_csv(args.output_dir / 'request_overlap_summary.csv', all_rows)
    curves(all_rows, args.output_dir)
    print(args.output_dir, flush=True)


if __name__ == '__main__':
    main()
