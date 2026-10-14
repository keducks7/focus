"""Read-only routing observations and explicitly OFFLINE route planning helpers.

No model forward, generation, expert evaluation, or route modification lives here.
The score postprocessing mirrors the HF LLaDA2MoeGate shipped with this repository.
"""
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def popcount(mask):
    return mask.bit_count() if hasattr(mask, 'bit_count') else bin(mask).count('1')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


class StreamCSV:
    def __init__(self, path, fields):
        self.stream = Path(path).open('x', newline='')
        self.writer = csv.DictWriter(self.stream, fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.stream.close()


class Means:
    """Bounded-memory summaries; missing denominators do not become zeros."""
    def __init__(self, keys, metrics):
        self.keys, self.metrics = keys, metrics
        self.cells = {}

    def add(self, row):
        key = tuple(row[k] for k in self.keys)
        cell = self.cells.setdefault(key, dict(rows=0, values={m: [0., 0] for m in self.metrics}))
        cell['rows'] += 1
        for metric in self.metrics:
            value = row.get(metric)
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite observation: {metric}')
                cell['values'][metric][0] += value
                cell['values'][metric][1] += 1

    def rows(self):
        result = []
        for key, cell in sorted(self.cells.items()):
            row = dict(zip(self.keys, key), observations=cell['rows'])
            for metric, (total, count) in cell['values'].items():
                row[metric] = total / count if count else None
                row[metric + '_count'] = count
            result.append(row)
        return result


def group_equal(rows, keys, metrics):
    cells = defaultdict(list)
    for row in rows:
        cells[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in sorted(cells.items()):
        row = dict(zip(keys, key), groups=len(values))
        for metric in metrics:
            available = [r[metric] for r in values if r.get(metric) is not None]
            row[metric] = mean(available) if available else None
            row[metric + '_groups'] = len(available)
        result.append(row)
    return result


def support_sets(routes, live, states=None, state=None):
    return {r: {e for p, ids in enumerate(positions)
                if state is None or states[r][p] == state for e in ids}
            for r, positions in enumerate(routes) if live[r]}


def incidence(supports):
    return Counter(e for experts in supports.values() for e in experts)


def request_pool(supports, anchor):
    return set().union(*(experts for r, experts in supports.items() if r != anchor))


def matched_popular_pool(legal, size, heldout_incidence):
    """Exactly equal LEGAL pool cardinality; popularity comes from OTHER groups."""
    if not 0 <= size <= len(legal):
        raise ValueError('Invalid matched pool size')
    return set(sorted(legal, key=lambda e: (-heldout_incidence.get(e, 0), e))[:size])


def read_router(model_path, layer):
    """Load one E-vector buffer, never the model/router weight or an expert."""
    from safetensors import safe_open
    model_path = Path(model_path)
    config = json.loads((model_path / 'config.json').read_text())
    if config.get('model_type') != 'llada2_moe':
        raise ValueError('This entry implements the HF LLaDA2 MoE gate only')
    key = f'model.layers.{layer}.mlp.gate.expert_bias'
    index = model_path / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
        if key not in mapping:
            raise ValueError(f'Missing native router buffer: {key}; do not assume a zero bias')
        shard = model_path / mapping[key]
    else:
        shard = model_path / 'model.safetensors'
    with safe_open(str(shard), framework='pt', device='cpu') as stream:
        bias = stream.get_tensor(key).float().tolist()
    result = dict(format_version=1, layer=layer, config=config, expert_bias=bias,
                  checkpoint_key=key, checkpoint_config_sha256=file_hash(model_path / 'config.json'),
                  algorithm='HF sigmoid + expert_bias; top2 group sum; group-limited TopK; normalized sigmoid weights')
    source = model_path / 'modeling_llada2_moe.py'
    if source.is_file():
        result['checkpoint_modeling_sha256'] = file_hash(source)
    return result


def reconstruct_gate(logits, router, candidates):
    """Vectorized native postprocessing on CAPTURED logits; no new gate forward."""
    import torch
    config = router['config']
    experts, k, groups, selected_groups = [config[n] for n in
        ['num_experts', 'num_experts_per_tok', 'n_group', 'topk_group']]
    if (groups <= 0 or experts % groups or experts // groups < 2
            or not 1 <= selected_groups <= groups or not k < candidates <= selected_groups * (experts // groups)):
        raise ValueError('Invalid HF group structure or candidate count')
    if logits.ndim != 2 or logits.shape[-1] != experts or logits.dtype != torch.float32:
        raise ValueError('Expected saved native FP32 router logits [tokens, experts]')
    bias = torch.tensor(router['expert_bias'], dtype=torch.float32)
    if bias.shape != (experts,) or not torch.isfinite(logits).all() or not torch.isfinite(bias).all():
        raise ValueError('Invalid logits or expert_bias')
    base = torch.sigmoid(logits.float()).type_as(logits)
    scores = base + bias
    group_scores = scores.view(-1, groups, experts // groups).topk(2, dim=-1).values.sum(-1)
    chosen = torch.topk(group_scores, k=selected_groups, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, chosen, 1)
    legal = group_mask.unsqueeze(-1).expand(-1, groups, experts // groups).reshape(-1, experts).bool()
    masked = scores.masked_fill(~legal, float('-inf'))
    expected = torch.topk(masked, k=k, dim=-1).indices
    extra = torch.topk(masked, k=candidates, dim=-1).indices
    ordered_groups = group_scores.sort(dim=-1, descending=True).values
    group_gap = (ordered_groups[:, selected_groups-1] - ordered_groups[:, selected_groups]
                 if selected_groups < groups else None)
    return dict(base=base, scores=scores, legal=legal, masked=masked, expected=expected,
                candidates=extra, group_gap=group_gap, top_k=k)


def native_row(gate, row, ids, weights, scale, candidates):
    """Audit actual IDs/weights; tolerate EXACT score ties, never silent mismatches."""
    import torch
    k = gate['top_k']
    if len(ids) != k or len(set(ids)) != k or len(weights) != k:
        raise ValueError('Invalid native TopK row')
    if any(not 0 <= e < gate['scores'].shape[1] or not gate['legal'][row, e] for e in ids):
        raise ValueError('Native IDs violate reconstructed official group eligibility')
    selected = set(ids)
    expected = set(gate['expected'][row].tolist())
    exact_tie = False
    if selected != expected:
        minimum = min(float(gate['scores'][row, e]) for e in selected)
        outside = max(float(gate['scores'][row, e]) for e in expected - selected)
        if minimum != outside:
            raise ValueError('Reconstructed native TopK mismatch; verify same checkpoint/router source')
        exact_tie = True
    probabilities = gate['base'][row, ids]
    if k > 1:
        probabilities = probabilities / (probabilities.sum() + 1e-20)
    probabilities = probabilities * scale
    if not torch.allclose(probabilities, torch.tensor(weights, dtype=probabilities.dtype), rtol=1e-5, atol=1e-6):
        raise ValueError('Native mixture weights do not match HF gate postprocessing')
    # Keep native TopK membership even when K and 2K topk calls resolve ties differently.
    prefix = sorted(ids, key=lambda e: (-float(gate['scores'][row, e]), e))
    extras = [e for e in gate['candidates'][row].tolist() if e not in selected]
    if len(extras) < candidates - k:
        legal = gate['legal'][row].nonzero().flatten().tolist()
        extras = [e for e in legal if e not in selected]
    extras = sorted(extras, key=lambda e: (-float(gate['scores'][row, e]), e))[:candidates-k]
    ranked = prefix + extras
    boundary_gap = float(gate['scores'][row, prefix[-1]] - gate['scores'][row, extras[0]])
    return dict(ids=ranked, scores=[float(gate['scores'][row, e]) for e in ranked],
                base_scores=[float(gate['base'][row, e]) for e in ranked],
                boundary_gap=boundary_gap,
                group_boundary_gap=float(gate['group_gap'][row]) if gate['group_gap'] is not None else None,
                exact_boundary_tie=exact_tie or boundary_gap == 0,
                normalized_weights={e: w / math.fsum(weights) for e, w in zip(ids, weights)})


def choice_family(ids, weights, epsilon, fixed=False):
    """ALL minimum-cardinality subsets of ORIGINAL TopK covering 1-epsilon mass."""
    if (not ids or len(ids) != len(weights) or len(set(ids)) != len(ids)
            or any(e < 0 for e in ids) or any(not math.isfinite(w) or w < 0 for w in weights)
            or not 0 <= epsilon < 1 or math.fsum(weights) <= 0):
        raise ValueError('Invalid original routes or mass tolerance')
    if len(ids) > 12:
        raise ValueError('Exact subset enumeration is capped at Top-12; this protocol targets Top-8')
    probs = [w / math.fsum(weights) for w in weights]
    if fixed:
        experts = tuple(sorted(ids))
        mask = sum(1 << e for e in experts)
        return dict(count=len(ids), choices=[(mask, math.fsum(probs), experts)], independent=mask)
    order = sorted(range(len(ids)), key=lambda j: (-probs[j], ids[j]))
    count = next(q for q in range(1, len(ids)+1)
                 if math.fsum(probs[j] for j in order[:q]) >= 1-epsilon-1e-12)
    families = []
    for indices in itertools.combinations(range(len(ids)), count):
        mass = math.fsum(probs[j] for j in indices)
        if mass >= 1-epsilon-1e-12:
            experts = tuple(sorted(ids[j] for j in indices))
            mask = sum(1 << e for e in experts)
            families.append((mask, mass, experts))
    families.sort(key=lambda x: (-x[1], x[2]))
    return dict(count=count, choices=families, independent=families[0][0])


def union(masks):
    result = 0
    for mask in masks:
        result |= mask
    return result


def _shrink(families, fixed, allowed):
    viable = [[item for item in row['choices'] if item[0] & ~allowed == 0] for row in families]
    if any(not options for options in viable):
        raise ValueError('Infeasible planning seed')
    owners = defaultdict(set)
    for i, options in enumerate(viable):
        for _, _, experts in options:
            for expert in experts:
                owners[expert].add(i)
    # Once an expert is unremovable, removing others cannot make it removable.
    for expert in sorted(owners, key=lambda e: (len(owners[e]), e)):
        bit = 1 << expert
        if fixed & bit:
            continue
        changes = {i: [item for item in viable[i] if not item[0] & bit] for i in owners[expert]}
        if all(changes.values()):
            for i, options in changes.items():
                viable[i] = options
    return [options[0][0] for options in viable]


def plan(families, fixed=0, seed=None, extra_seeds=None):
    """Deterministic heuristic, NOT a global optimum or a speed/quality estimate."""
    if not families:
        return []
    seed = seed if seed is not None else [row['independent'] for row in families]
    seeds = [list(seed)] + [list(s) for s in (extra_seeds or [])]
    broad = fixed | union(item[0] for row in families for item in row['choices'])
    solutions = seeds + [_shrink(families, fixed, fixed | union(s)) for s in seeds] + [_shrink(families, fixed, broad)]
    def key(masks):
        masses = [next(item[1] for item in row['choices'] if item[0] == mask)
                  for row, mask in zip(families, masks)]
        return (popcount(fixed | union(masks)), -math.fsum(masses), tuple(masks))
    return min(solutions, key=key)


def scope_plans(families, requests, fixed_flags):
    """Same per-token K and mass constraints, same GLOBAL fixed support for all scopes."""
    independent = [row['independent'] for row in families]
    fixed = union(mask for mask, flag in zip(independent, fixed_flags) if flag)
    per_request = independent.copy()
    for request in sorted(set(requests)):
        indices = [i for i, r in enumerate(requests) if r == request and not fixed_flags[i]]
        selected = plan([families[i] for i in indices], fixed, [independent[i] for i in indices])
        for i, mask in zip(indices, selected):
            per_request[i] = mask
    indices = [i for i, flag in enumerate(fixed_flags) if not flag]
    selected = plan([families[i] for i in indices], fixed, [per_request[i] for i in indices],
                    extra_seeds=[[independent[i] for i in indices]])
    batch = independent.copy()
    for i, mask in zip(indices, selected):
        batch[i] = mask
    result = dict(token=independent, request=per_request, batch=batch)
    for masks in result.values():
        for row, mask in zip(families, masks):
            if popcount(mask) != row['count'] or not any(item[0] == mask for item in row['choices']):
                raise RuntimeError('Planner changed per-token budget or violated mass feasibility')
    # Independent per-request minimization may DESTROY cross-request overlap.
    # Never force/request-impute monotonicity of its whole-batch union.
    if popcount(union(batch)) > min(popcount(union(per_request)), popcount(union(independent))):
        raise RuntimeError('Best-of-seed monotonicity broken')
    return result


def observed_transition(previous, current):
    """Same request/position, strictly adjacent original steps; no look-ahead routing."""
    if current['step'] != previous['step'] + 1:
        raise ValueError('Do not call a gap an adjacent denoising step')
    old, new = set(previous['actual_ids']), set(current['actual_ids'])
    extra = set(previous['candidate_ids']) - old
    shared_extra = set(previous['shared_extra_ids'])
    entered = new - old
    private = {e for e, count in previous['incidence'].items() if count == 1}
    return dict(phase=previous['state'] + '->' + current['state'],
                topk_jaccard=len(old & new)/len(old | new),
                candidate_jaccard=len(set(previous['candidate_ids']) & set(current['candidate_ids'])) /
                                  len(set(previous['candidate_ids']) | set(current['candidate_ids'])),
                entered_experts=len(entered),
                entered_from_previous_extra=len(entered & extra)/len(entered) if entered else None,
                shared_extra_adoption=len(new & shared_extra)/len(shared_extra) if shared_extra else None,
                private_persistence=len(new & private)/len(private) if private else None,
                identity_changed=previous['input_token_id'] != current['input_token_id'])


class EpisodeTracker:
    """Expert residence within an unchanged input identity; endings are labelled."""
    def __init__(self, emit):
        self.emit, self.open = emit, {}
        self.anchors = defaultdict(set)

    def close(self, key, reason, censored):
        episode = self.open.pop(key)
        self.anchors[key[:2]].discard(key)
        n = episode.pop('duration')
        episode.update(duration=n, end_reason=reason, right_censored=censored,
                       mean_rank=episode.pop('rank_sum')/n,
                       mean_weight=episode.pop('weight_sum')/n,
                       private_fraction=episode.pop('private_steps')/n)
        self.emit(episode)

    def update(self, record):
        anchor = (record['request_id'], record['position'])
        active = set(record['actual_ids'])
        for key in list(self.anchors.get(anchor, ())):
            old = self.open[key]
            if old['end_step'] + 1 != record['step']:
                self.close(key, 'observation_gap', True)
            elif old['input_token_id'] != record['input_token_id']:
                self.close(key, 'input_identity_change', False)
            elif key[2] not in active:
                self.close(key, 'native_topk_departure', False)
        for expert in active:
            key = anchor + (expert,)
            if key not in self.open:
                self.open[key] = dict(group=record['group'], request_id=record['request_id'],
                    position=record['position'], expert=expert, state=record['state'],
                    input_token_id=record['input_token_id'], start_step=record['step'], end_step=record['step'],
                    left_censored=record.get('first_observed', False) and record['step'] > 0,
                    duration=0, rank_sum=0., weight_sum=0., private_steps=0)
                self.anchors[anchor].add(key)
            episode = self.open[key]
            episode['end_step'] = record['step']
            episode['duration'] += 1
            episode['rank_sum'] += record['candidate_ids'].index(expert) + 1
            episode['weight_sum'] += record['normalized_weights'][expert]
            episode['private_steps'] += int(record['incidence'][expert] == 1)

    def finish_request(self, request, reason='request_finished'):
        for key in [k for k in self.open if k[0] == request]:
            self.close(key, reason, True)

    def finish(self):
        for key in list(self.open):
            self.close(key, 'trajectory_or_window_end', True)
