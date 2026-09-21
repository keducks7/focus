#!/usr/bin/env python3
"""Stage 1: streaming counterfactual selection on full-state generation traces.

No forward, no accuracy estimate, no speed measurement. Prefix routes are absent:
reported unions cover GENERATION ONLY, not the full-forward expert working set.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from shared_route_selection import METHODS, select_routes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--layers', nargs='+', type=int, default=[2, 10, 18])
    p.add_argument('--steps', nargs='+', type=int, default=[4])
    p.add_argument('--epsilons', nargs='+', type=float, default=[0, .05, .1, .2])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    a = p.parse_args()
    if any(not 0 <= e < 1 for e in a.epsilons):
        p.error('epsilon must be in [0,1).')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    path = a.output_dir/'groups.jsonl'
    totals = defaultdict(list)
    with path.open('x') as out:
        out.write(json.dumps(dict(record_type='metadata', scope='generation_only_no_prefix',
                                  quality_or_speed_measured=False, args={k: str(v) if isinstance(v, Path) else v
                                  for k, v in vars(a).items()}))+'\n')
        for b in a.batches:
            source = a.root/f'bs{b}'/f'token_trajectories_bs{b}.jsonl'
            seen = set()
            with source.open() as stream:
                meta = json.loads(next(stream))
                if meta.get('observed_region') != 'all_generation_positions' or meta['configured_batch_size'] != b:
                    raise ValueError(f'Wrong trajectory metadata: {source}')
                for line in stream:
                    row = json.loads(line)
                    if row['record_type'] != 'token_trajectory_step' or row['step'] not in a.steps:
                        continue
                    key = (row['group_id'], row['step'])
                    if key in seen:
                        raise ValueError('Duplicate group/step (single-block input required).')
                    seen.add(key)
                    tokens = row['tokens']
                    if any(t['request_finished_before'] for t in tokens):
                        raise ValueError('Choose a common fully-active step, e.g. step 4.')
                    if len(tokens) != b * meta['block_length']:
                        raise ValueError('Incomplete generation positions.')
                    compress = [t['state_before'] == 'mask' for t in tokens]
                    for layer in a.layers:
                        routes = [next(x for x in t['layers'] if x['layer_idx'] == layer) for t in tokens]
                        ids = [r['expert_ids'] for r in routes]
                        weights = [r['router_weights'] for r in routes]
                        for eps in a.epsilons:
                            for method in a.methods:
                                _, metrics = select_routes(ids, weights, compress, eps, method)
                                result = dict(batch=b, layer=layer, step=row['step'], group=row['group_id'],
                                              epsilon=eps, method=method, **metrics)
                                out.write(json.dumps(result)+'\n')
                                totals[(b, layer, row['step'], eps, method)].append(metrics)
            expected = meta['num_prompts']//b * len(set(a.steps))
            if len(seen) != expected:
                raise ValueError(f'Incomplete requested steps in {source}: {len(seen)} != {expected}')
            print(f'Finished B{b}', flush=True)
    summary = []
    for (b, l, s, e, m), values in sorted(totals.items()):
        means = {k: sum(v[k] for v in values)/len(values) for k in values[0]}
        summary.append(dict(batch=b, layer=l, step=s, epsilon=e, method=m, groups=len(values), **means))
    with (a.output_dir/'summary.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(a.output_dir/'summary.csv')


if __name__ == '__main__':
    main()
