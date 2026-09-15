#!/usr/bin/env python3
"""Replay actual routed tokens from saved trajectory states, one real expert at a time."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_routes(path, layer):
    cases = defaultdict(dict)
    metadata = None
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['record_type'] == 'metadata':
                metadata = record
            elif record['record_type'] == 'token_trajectory_step':
                key = (record['step'], record['group_id'], record['block_id'])
                if key in cases:
                    raise ValueError(f'Duplicate trajectory step: {key}')
                for token in record['tokens']:
                    identity = (token['request_id'], token['block_id'], token['block_position'])
                    layers = [row for row in token['layers'] if row['layer_idx'] == layer]
                    if len(layers) != 1 or identity in cases[key]:
                        raise ValueError(f'Missing/duplicate layer or token: {identity}')
                    cases[key][identity] = layers[0]['expert_ids']
    if not metadata or not cases:
        raise ValueError('No usable trajectory records')
    for values in cases.values():
        for route in values.values():
            if (len(route) != metadata['top_k'] or len(set(route)) != len(route)
                    or any(expert < 0 or expert >= metadata['num_experts'] for expert in route)):
                raise ValueError('Invalid recorded Top-k expert IDs')
    return metadata, cases


def nested_request_sets(request_ids, sizes, seed):
    order = sorted(set(request_ids))
    random.Random(seed).shuffle(order)
    return {size: set(order[:size]) for size in sizes if size <= len(order)}


def matched_candidates(request_ids, count, seed):
    """Equal candidate counts, no self-match, only anchors eligible on BOTH sides."""
    rng = random.Random(seed)
    result = []
    for anchor, request in enumerate(request_ids):
        same = [i for i, other in enumerate(request_ids) if i != anchor and other == request]
        cross = [i for i, other in enumerate(request_ids) if other != request]
        if len(same) >= count and len(cross) >= count:
            result.append((anchor, rng.sample(same, count), rng.sample(cross, count)))
    return result


def effective_rank(values, centered=False):
    """Entropy rank of squared singular values; null matrix has rank zero."""
    import torch

    matrix = values.float()
    if centered:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    gram = matrix @ matrix.T
    energy = torch.linalg.eigvalsh(gram.double()).clamp_min(0)
    total = energy.sum()
    if total <= 1e-20:
        return 0.0
    probability = energy / total
    positive = probability[probability > 0]
    return float(torch.exp(-(positive * positive.log()).sum()).item())


class ExpertWeights:
    def __init__(self, model_path, layer, config, device):
        from safetensors import safe_open

        self.model_path, self.layer, self.device = Path(model_path), layer, device
        self.hidden, self.intermediate = config['hidden_size'], config['moe_intermediate_size']
        index = self.model_path / 'model.safetensors.index.json'
        if index.exists():
            self.weight_map = json.loads(index.read_text())['weight_map']
        else:
            single = self.model_path / 'model.safetensors'
            with safe_open(str(single), framework='pt', device='cpu') as stream:
                self.weight_map = {key: single.name for key in stream.keys()}

    def load(self, expert):
        import torch
        from safetensors import safe_open

        weights = {}
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            key = f'model.layers.{self.layer}.mlp.experts.{expert}.{projection}.weight'
            if key not in self.weight_map:
                raise ValueError(f'Checkpoint missing real expert weight: {key}')
            with safe_open(str(self.model_path / self.weight_map[key]), framework='pt', device='cpu') as stream:
                weight = stream.get_tensor(key)
            shape = ((self.hidden, self.intermediate) if projection == 'down_proj'
                     else (self.intermediate, self.hidden))
            if tuple(weight.shape) != shape:
                raise ValueError(f'Unexpected weight shape for {key}: {weight.shape}')
            weights[projection] = weight.to(device=self.device, dtype=torch.bfloat16)
        return weights


def replay(inputs, weights):
    import torch.nn.functional as F

    return F.linear(F.silu(F.linear(inputs, weights['gate_proj'])) *
                    F.linear(inputs, weights['up_proj']), weights['down_proj'])


def write_rows(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_rank_changes(rows, sizes):
    """Only compare experts eligible at every requested size, within the same cell."""
    grouped = defaultdict(dict)
    for row in rows:
        if row['matched_eligible']:
            key = tuple(row[name] for name in ('step', 'group_id', 'block_id', 'expert', 'repeat'))
            grouped[key][row['request_subset_size']] = row
    result = []
    for key, observations in sorted(grouped.items()):
        if not all(size in observations for size in sizes):
            continue
        baseline = observations[sizes[0]]
        for size in sizes[1:]:
            current = observations[size]
            row = dict(zip(('step', 'group_id', 'block_id', 'expert', 'repeat'), key))
            row.update(baseline_size=sizes[0], request_subset_size=size,
                       matched_tokens=baseline['matched_tokens'])
            for metric in ('input_rank', 'output_rank', 'input_centered_rank', 'output_centered_rank'):
                row[metric + '_delta'] = current['matched_' + metric] - baseline['matched_' + metric]
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory_dir', type=Path)
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=8, help='Actual batch of the saved run')
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--steps', nargs='+', type=int, default=[0, 4, 8, 12])
    parser.add_argument('--request-sizes', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--matched-tokens', type=int, default=16)
    parser.add_argument('--rank-cap', type=int, default=256)
    parser.add_argument('--neighbors', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    sizes = sorted(set(args.request_sizes))
    if min(sizes + [args.matched_tokens, args.rank_cap, args.neighbors, args.repeats]) < 1:
        parser.error('All counts must be positive')
    if max(sizes) > args.batch_size or args.matched_tokens > args.rank_cap:
        parser.error('Request sizes must fit saved batch; matched tokens must fit rank cap')
    if args.output_dir.exists():
        parser.error('Use a new output directory')
    metadata, cases = read_routes(args.trajectory_dir / f'token_trajectories_bs{args.batch_size}.jsonl', args.layer)
    if metadata['configured_batch_size'] != args.batch_size:
        raise ValueError('Saved batch size mismatch')
    group_requests = defaultdict(set)
    for (_, group, block), values in cases.items():
        group_requests[group, block].update(identity[0] for identity in values)
    config = json.loads((args.model_path / 'config.json').read_text())
    if config.get('hidden_act') != 'silu' or config.get('quantization_config'):
        raise ValueError('Only unquantized LLaDA2 SwiGLU expert checkpoints are supported')
    if config['num_experts'] != metadata['num_experts']:
        raise ValueError('Checkpoint/trace expert count mismatch')
    # Validate all inputs before loading weights. torch is imported lazily for CPU-only protocol tests.
    files = {step: args.trajectory_dir / f'layer{args.layer}_hidden_step{step}_bs{args.batch_size}.pt'
             for step in sorted(set(args.steps))}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f'{path}: wait for capture to finish or select saved steps')
    import torch
    import torch.nn.functional as F

    loader = ExpertWeights(args.model_path, args.layer, config, args.device)
    args.output_dir.mkdir(parents=True)
    info = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    info.update(format_version=1, source_metadata=metadata, torch_version=torch.__version__,
                interpretation='Nested request subsets of a fixed captured batch, NOT independent batch-size runs',
                output_definition='Raw routed expert output, excluding router weights and shared expert',
                rank_definition='exp(entropy(normalized squared singular values)); zero matrix = 0')
    (args.output_dir / 'metadata.json').write_text(json.dumps(info, indent=2))
    rank_rows, neighbor_rows = [], []
    with torch.inference_mode():
        for step, path in files.items():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if (payload['layer_idx'], payload['step'], payload['batch_size']) != (args.layer, step, args.batch_size):
                raise ValueError(f'Hidden-state metadata mismatch: {path}')
            token_ids = [tuple(identity) for identity in payload['token_ids']]
            states = payload['hidden_states']
            if len(set(token_ids)) != len(token_ids) or len(token_ids) != len(states):
                raise ValueError('Duplicate identity or invalid hidden-state count')
            lookup = dict(zip(token_ids, range(len(token_ids))))
            step_cases = [(key, values) for key, values in sorted(cases.items()) if key[0] == step]
            expected = {identity for _, values in step_cases for identity in values}
            if expected != set(lookup):
                raise ValueError('Trace and hidden-state token identities differ')
            for expert in range(config['num_experts']):
                selected_cases = []
                for key, values in step_cases:
                    identities = sorted(identity for identity, route in values.items() if expert in route)
                    if identities:
                        selected_cases.append((key, values, identities))
                if not selected_cases:
                    continue
                weights = loader.load(expert)
                for (step, group, block), _, identities in selected_cases:
                    inputs = states[[lookup[identity] for identity in identities]].to(args.device, torch.bfloat16)
                    outputs = replay(inputs, weights)
                    requests = [identity[0] for identity in identities]
                    base = dict(step=step, group_id=group, block_id=block, expert=expert)
                    rank_cache = {}
                    normalized = F.normalize(outputs.float(), dim=1)
                    norms = outputs.float().norm(dim=1)
                    for repeat in range(args.repeats):
                        seed = args.seed + 1000003 * step + 10007 * group + 101 * block + repeat
                        subsets = nested_request_sets(group_requests[group, block], sizes, seed)
                        rng = random.Random(seed + expert * 1009)
                        permutation = rng.sample(range(len(identities)), len(identities))
                        for size in sizes:
                            selected = [i for i in permutation if requests[i] in subsets.get(size, set())]
                            row = dict(base, repeat=repeat, request_subset_size=size, routed_tokens=len(selected),
                                       contributing_requests=len({requests[i] for i in selected}),
                                       rank_tokens=min(len(selected), args.rank_cap),
                                       matched_eligible=int(len(selected) >= args.matched_tokens),
                                       matched_tokens=args.matched_tokens if len(selected) >= args.matched_tokens else 0)
                            for prefix, indices in (('observed_', selected[:args.rank_cap]),
                                                    ('matched_', selected[:args.matched_tokens]
                                                     if row['matched_eligible'] else [])):
                                cache_key = tuple(sorted(indices))
                                if cache_key not in rank_cache:
                                    metrics = {}
                                    for name, tensor in (('input', inputs), ('output', outputs)):
                                        metrics[name + '_rank'] = effective_rank(tensor[list(cache_key)]) if indices else ''
                                        metrics[name + '_centered_rank'] = effective_rank(tensor[list(cache_key)], True) if indices else ''
                                    rank_cache[cache_key] = metrics
                                row.update({prefix + key: value for key, value in rank_cache[cache_key].items()})
                            rank_rows.append(row)
                        candidates = matched_candidates(requests, args.neighbors, seed + expert * 1009)
                        valid = 0
                        cosine_same, cosine_cross, relative_same, relative_cross = [], [], [], []
                        for anchor, same, cross in candidates:
                            if norms[anchor] <= 1e-12 or bool((norms[same + cross] <= 1e-12).any()):
                                continue
                            valid += 1
                            for ids, cosines, relatives in ((same, cosine_same, relative_same),
                                                          (cross, cosine_cross, relative_cross)):
                                cosines.append(float((1 - normalized[ids] @ normalized[anchor]).clamp(0, 2).min()))
                                distances = (outputs[ids].float() - outputs[anchor].float()).norm(dim=1)
                                relatives.append(float((distances / norms[anchor]).min()))
                        def mean(values):
                            return sum(values) / len(values) if values else ''
                        neighbor_rows.append(dict(base, repeat=repeat, routed_tokens=len(requests),
                                                  eligible_anchors=len(candidates), valid_anchors=valid,
                                                  candidates_per_side=args.neighbors,
                                                  same_cosine_distance=mean(cosine_same),
                                                  cross_cosine_distance=mean(cosine_cross),
                                                  same_relative_l2=mean(relative_same),
                                                  cross_relative_l2=mean(relative_cross)))
                    del inputs, outputs
                del weights
                if (expert + 1) % 32 == 0:
                    print(f'step={step} expert={expert + 1}/{config["num_experts"]}', flush=True)
            write_rows(args.output_dir / 'ranks.csv', rank_rows)
            write_rows(args.output_dir / 'neighbors.csv', neighbor_rows)
            write_rows(args.output_dir / 'paired_rank_changes.csv', paired_rank_changes(rank_rows, sizes))
            print(f'Completed step={step}', flush=True)
    report = dict(rank_rows=len(rank_rows), matched_eligible_rows=sum(row['matched_eligible'] for row in rank_rows),
                  paired_rows=len(paired_rank_changes(rank_rows, sizes)),
                  neighbor_cells=len(neighbor_rows), cells_with_valid_neighbors=sum(row['valid_anchors'] > 0 for row in neighbor_rows))
    (args.output_dir / 'coverage.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
