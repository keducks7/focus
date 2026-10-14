#!/usr/bin/env python3
"""CPU/stdlib-only choice-space statistics and OFFLINE token/request/batch planning.

All model forwards remain native. Plans are never injected into a decoder.
"""
import argparse
import json
import math
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from statistics import mean

from routing_competition import (Means, StreamCSV, choice_family, file_hash, group_equal,
                                popcount, scope_plans, union, write_csv)

FAMILY_METRICS = ['minimum_branches', 'choice_count', 'has_choice', 'eligible_shared_choice_fraction',
                  'baseline_private_fraction', 'retained_mass_best']
PLAN_METRICS = ['native_union', 'token_union', 'request_union', 'batch_union',
                'assignments_native', 'assignments_planned', 'within_request_gain', 'cross_request_gain',
                'within_request_gain_fraction', 'cross_request_gain_fraction', 'total_joint_gain', 'total_joint_gain_fraction',
                'minimum_retained_mass_token', 'minimum_retained_mass_request', 'minimum_retained_mass_batch']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trajectory_root', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument('--layers', nargs='+', type=int, default=[2, 10, 18])
    p.add_argument('--steps', nargs='+', type=int, default=[4])
    p.add_argument('--groups', nargs='+', type=int, help='Optional physical group IDs for smoke runs')
    p.add_argument('--epsilon', type=float, default=.1)
    p.add_argument('--planning-states', nargs='+', choices=['mask', 'decoded'], default=['mask', 'decoded'])
    p.add_argument('--save-plans', action='store_true', help='Save OFFLINE expert IDs, never execute them')
    a = p.parse_args()
    batches, layers, steps = [sorted(set(values)) for values in [a.batches, a.layers, a.steps]]
    if min(batches) < 1 or min(layers + steps) < 0 or not 0 <= a.epsilon < 1:
        p.error('Invalid batch/layer/step or epsilon')
    if a.groups is not None and min(a.groups) < 0:
        p.error('Groups must be nonnegative')
    paths = {b: a.trajectory_root / f'bs{b}' / f'token_trajectories_bs{b}.jsonl' for b in batches}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError('Use a new/empty output directory')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(format_version=1, observation_only_model_path=True, offline_planning=True,
                    redecode=False, routes_modified=False, expert_forward=False,
                    source_sha256={}, source_metadata={},
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    constraints='same original TopK candidates, exact same minimum branch count per token, same retained mass threshold',
                    planner='deterministic expert elimination, broad and seeded starts; NOT a global optimum',
                    fixed_support='identical global fixed support in request and batch planning',
                    scope='actual physical batch traces; generation positions only; fully-live requested cells')
    (a.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    family_stats = Means(['batch', 'group', 'step', 'layer', 'state'], FAMILY_METRICS)
    plan_stats = []
    counts = dict(tokens=0, planning_cells=0)
    # Prefix-only header inspection avoids decoding huge non-target token records.
    step_pattern = re.compile(r'"step"\s*:\s*(\d+)')
    record_pattern = re.compile(r'"record_type"\s*:\s*"([^\"]+)"')
    with ExitStack() as stack:
        token_writer = StreamCSV(a.output_dir / 'choices.csv',
            ['batch', 'group', 'step', 'layer', 'request_id', 'position', 'state', 'fixed'] + FAMILY_METRICS)
        stack.callback(token_writer.close)
        plans_file = stack.enter_context((a.output_dir / 'plans.jsonl').open('x')) if a.save_plans else None
        for b, path in paths.items():
            seen, generation_groups = set(), set()
            with path.open() as stream:
                source = json.loads(next(stream))
                if (source.get('observed_region') != 'all_generation_positions' or source['configured_batch_size'] != b
                        or source.get('backend') != 'huggingface' or source.get('mode') != 'vanilla'
                        or source.get('num_generation_blocks') != 1 or source['num_prompts'] % b):
                    raise ValueError(f'Wrong/incomplete single-block HF trace: {path}')
                metadata['source_metadata'][str(b)] = source
                expected_groups = set(range(source['num_prompts'] // b))
                selected_groups = set(a.groups) if a.groups is not None else expected_groups
                if not selected_groups or not selected_groups <= expected_groups:
                    raise ValueError('Requested group outside physical trace')
                for line in stream:
                    header = line[:1024]
                    match = record_pattern.search(header)
                    if match is None:
                        raise ValueError('Unrecognizable streaming record header')
                    record_type = match.group(1)
                    if record_type == 'generation_result':
                        record = json.loads(line)
                        if record.get('block_id') != 0 or record['group_id'] in generation_groups:
                            raise ValueError('Duplicate or multi-block generation result')
                        generation_groups.add(record['group_id'])
                        continue
                    if record_type != 'token_trajectory_step':
                        continue
                    step_match = step_pattern.search(header)
                    if step_match is None:
                        raise ValueError('Missing step in native record header')
                    if int(step_match.group(1)) not in steps:
                        continue
                    record = json.loads(line)
                    group, step = record['group_id'], record['step']
                    if group not in selected_groups:
                        continue
                    if record.get('block_id') != 0 or (group, step) in seen:
                        raise ValueError('Duplicate/multi-block group-step')
                    seen.add((group, step))
                    tokens = record['tokens']
                    if len(tokens) != b * source['block_length'] or any(t['request_finished_before'] for t in tokens):
                        raise ValueError('Use a common fully-live window for physical batch comparisons (e.g. step 0-7)')
                    identities = [(t['request_id'], t['block_position']) for t in tokens]
                    if len(set(identities)) != len(tokens) or len({r for r, _ in identities}) != b:
                        raise ValueError('Invalid token identity mapping')
                    for layer in layers:
                        routes = []
                        for token in tokens:
                            matches = [row for row in token['layers'] if row['layer_idx'] == layer]
                            if len(matches) != 1:
                                raise ValueError(f'Missing/duplicate layer {layer}')
                            routes.append(matches[0])
                        ids = [row['expert_ids'] for row in routes]
                        weights = [row['router_weights'] for row in routes]
                        if any(len(row) != source['top_k'] or any(not 0 <= e < source['num_experts'] for e in row) for row in ids):
                            raise ValueError('Invalid original TopK IDs')
                        fixed = [t['state_before'] not in a.planning_states for t in tokens]
                        if any(t['state_before'] not in ['mask', 'decoded'] for t in tokens):
                            raise ValueError('Unknown pre-forward token state')
                        families = [choice_family(es, ws, a.epsilon, fixed=flag)
                                    for es, ws, flag in zip(ids, weights, fixed)]
                        requests = [t['request_id'] for t in tokens]
                        request_support = {r: {e for es, req in zip(ids, requests) if req == r for e in es}
                                           for r in set(requests)}
                        visits = Counter(e for support in request_support.values() for e in support)
                        for token, family, es, flag in zip(tokens, families, ids, fixed):
                            peer_mask = union(1 << e for req, support in request_support.items()
                                              if req != token['request_id'] for e in support)
                            # Fraction of admissible masks made entirely of experts active in OTHER requests.
                            shared_fraction = sum(mask & ~peer_mask == 0 for mask, _, _ in family['choices']) / len(family['choices'])
                            row = dict(batch=b, group=group, step=step, layer=layer,
                                       request_id=token['request_id'], position=token['block_position'],
                                       state=token['state_before'], fixed=flag,
                                       minimum_branches=family['count'], choice_count=len(family['choices']),
                                       has_choice=len(family['choices']) > 1,
                                       eligible_shared_choice_fraction=shared_fraction,
                                       baseline_private_fraction=sum(visits[e] == 1 for e in es)/len(es),
                                       retained_mass_best=family['choices'][0][1])
                            token_writer.write(row); family_stats.add(row); counts['tokens'] += 1
                        plans = scope_plans(families, requests, fixed)
                        native = union(1 << e for row in ids for e in row)
                        supports = {name: popcount(union(masks)) for name, masks in plans.items()}
                        result = dict(batch=b, group=group, step=step, layer=layer,
                            native_union=popcount(native), token_union=supports['token'],
                            request_union=supports['request'], batch_union=supports['batch'],
                            assignments_native=sum(map(len, ids)), assignments_planned=sum(r['count'] for r in families),
                            within_request_gain=supports['token']-supports['request'],
                            cross_request_gain=supports['request']-supports['batch'],
                            total_joint_gain=supports['token']-supports['batch'],
                            within_request_gain_fraction=(supports['token']-supports['request'])/supports['token'],
                            cross_request_gain_fraction=(supports['request']-supports['batch'])/supports['request'],
                            total_joint_gain_fraction=(supports['token']-supports['batch'])/supports['token'])
                        for name, masks in plans.items():
                            retained = [next(item[1] for item in row['choices'] if item[0] == mask)
                                        for row, mask in zip(families, masks)]
                            if min(retained) < 1-a.epsilon-1e-12 or sum(popcount(m) for m in masks) != result['assignments_planned']:
                                raise RuntimeError('Unequal per-token budget or mass constraint')
                            result['minimum_retained_mass_' + name] = min(retained)
                        plan_stats.append(result); counts['planning_cells'] += 1
                        if plans_file is not None:
                            serial = {name: [[e for e in es if mask & (1 << e)] for es, mask in zip(ids, masks)]
                                      for name, masks in plans.items()}
                            plans_file.write(json.dumps(dict(batch=b, group=group, step=step, layer=layer,
                                offline_only=True, token_ids=identities, retained_experts=serial)) + '\n')
                    write_csv(a.output_dir / 'planning_groups.csv', plan_stats)
                    print(f'B{b} group={group} step={step}: offline scopes verified at identical per-token budgets', flush=True)
            expected = {(g, s) for g in selected_groups for s in steps}
            if seen != expected or generation_groups != expected_groups:
                raise ValueError(f'Incomplete target steps or generation results in {path}')
            metadata['source_sha256'][str(b)] = file_hash(path)
    family_groups = family_stats.rows()
    write_csv(a.output_dir / 'choice_group_summary.csv', family_groups)
    write_csv(a.output_dir / 'choice_summary.csv', group_equal(family_groups, ['batch', 'step', 'layer', 'state'], FAMILY_METRICS))
    write_csv(a.output_dir / 'planning_summary.csv', group_equal(plan_stats, ['batch', 'step', 'layer'], PLAN_METRICS))
    (a.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report = dict(complete=True, offline_planning=True, observation_only_model_path=True,
        batches=batches, layers=layers, steps=steps, counts=counts, epsilon=a.epsilon,
        quality_or_speed_measured=False,
        caveats=['Plans are statistical scenarios, not executed routes or native observations.',
                 'Per-token cardinality and mass feasibility match exactly across all three scopes.',
                 'The deterministic heuristic need not attain a global optimum; a small gain does not prove no opportunity.',
                 'Best-of-seed monotonicity is a construction, not empirical evidence; report the magnitude of gains.',
                 'Request-only coordination can increase the global union; within-request gain may be negative.',
                 'Cross-request gain may restore sharing destroyed by request-only planning; also inspect total_joint_gain.',
                 'Route mass is not functional importance or a quality-error bound.',
                 'B1 can share experts across positions but has zero cross-request gain by construction.',
                 'Physical traces can differ numerically and in token-state composition across B.',
                 'Expert sets cover generation positions only, not prompt/shared-expert computation.'])
    (a.output_dir / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
