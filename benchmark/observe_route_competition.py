#!/usr/bin/env python3
"""CPU-only native candidate competition and lifecycle observation on saved HF logits.

Reads one router bias vector. NO decoder/model/expert forward and NO changed routes.
"""
import argparse
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from routing_competition import (EpisodeTracker, Means, StreamCSV, file_hash, group_equal,
    incidence, matched_popular_pool, native_row, observed_transition, read_router,
    reconstruct_gate, request_pool, support_sets, write_csv)

BASE = ['group', 'step', 'request_id', 'position', 'state', 'input_token_id',
        'active_requests', 'acceptance_step', 'relative_acceptance_step']
TOKEN_METRICS = ['boundary_gap', 'group_boundary_gap', 'private_branch_fraction',
                 'within_request_extra_share', 'cross_request_extra_share',
                 'matched_popular_extra_share', 'cross_minus_popular_extra_share',
                 'active_requests', 'heldout_request_count', 'matched_legal_pool_size']
BRANCH_METRICS = ['request_incidence', 'state_request_incidence', 'private',
                  'normalized_weight', 'gap_to_shared_extra', 'private_gap_to_shared_extra']
TRANSITION_METRICS = ['topk_jaccard', 'candidate_jaccard', 'entered_experts',
                     'entered_from_previous_extra', 'shared_extra_adoption', 'private_persistence']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observation_dir', type=Path)
    parser.add_argument('model_path', nargs='?', type=Path,
                        help='Only config and gate.expert_bias are read; no full model')
    parser.add_argument('--router-snapshot', type=Path,
                        help='Reuse router.json exported by a previous run, without Safetensors')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=10)
    parser.add_argument('--groups', type=int, nargs='+')
    parser.add_argument('--steps', type=int, nargs='+',
                        help='Default: every saved natural step; use consecutive steps for lifecycle')
    parser.add_argument('--candidates', type=int, default=16)
    a = parser.parse_args()
    if (a.model_path is None) == (a.router_snapshot is None):
        parser.error('Provide exactly one model_path or --router-snapshot')
    if a.layer < 0 or a.candidates < 2 or (a.steps is not None and min(a.steps) < 0):
        parser.error('Invalid layer/steps/candidate count')
    source = json.loads((a.observation_dir / 'metadata.json').read_text())
    completion = json.loads((a.observation_dir / 'generation_complete.json').read_text())
    if not source.get('observation_only') or source.get('mode') != 'vanilla' or source['args']['layer'] != a.layer:
        raise ValueError('Expected matching completed Vanilla HF observation inputs')
    all_groups = sorted(set(completion['groups']))
    groups = sorted(set(a.groups if a.groups is not None else all_groups))
    if not groups or not set(groups) <= set(all_groups):
        parser.error('Requested groups missing from generation completion')
    audits = {r['group']: r['first_forward_observer_matches_unobserved'] for r in completion.get('audits', [])}
    if any(audits.get(g) is not True for g in all_groups):
        raise ValueError('Missing/failed source observer audit')
    indices = {g: json.loads((a.observation_dir / f'group{g}.json').read_text()) for g in all_groups}
    requests = [r for g in all_groups for r in indices[g]['request_ids']]
    if len(set(requests)) != len(requests):
        raise ValueError('Held-out groups must have distinct request identities')
    steps = {g: sorted(set(a.steps)) if a.steps is not None else list(range(indices[g]['steps'])) for g in groups}
    for g in groups:
        if any(s >= indices[g]['steps'] for s in steps[g]):
            raise ValueError(f'Requested step outside saved group {g} trajectory')
        for s in steps[g]:
            path = a.observation_dir / f'inputs_group{g}_step{s}.pt'
            if not path.is_file():
                raise FileNotFoundError(path)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError('Use a new/empty output directory')
    router = (json.loads(a.router_snapshot.read_text()) if a.router_snapshot is not None
              else read_router(a.model_path, a.layer))
    if router['layer'] != a.layer:
        raise ValueError('Router snapshot layer mismatch')
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    batch, length = source['args']['batch_size'], source['args']['block_length']
    expert_count = router['config']['num_experts']
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / 'router.json').write_text(json.dumps(router, indent=2) + '\n')
    metadata = dict(format_version=1, observation_only=True, redecode=False, expert_forward=False,
                    routes_modified=False, native_gate_audited=False, source_metadata=source,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    resolved_groups=groups, resolved_steps=steps, torch_version=torch.__version__,
                    native_score_formula=router['algorithm'], source_snapshot_sha256={},
                    source_metadata_sha256=file_hash(a.observation_dir / 'metadata.json'),
                    router_snapshot_sha256=file_hash(a.output_dir / 'router.json'),
                    baseline='other-group request-incidence popularity; equal legal pool cardinality',
                    private_definition='visited by exactly one currently live request; all generation states',
                    scope='generation positions only; completed requests excluded; no cross-layer expert-ID merging')
    (a.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    token_stats = Means(['group', 'step', 'state'], TOKEN_METRICS)
    rank_stats = Means(['group', 'step', 'state', 'rank'], BRANCH_METRICS)
    transitions = Means(['group', 'step', 'phase'], TRANSITION_METRICS)
    survival = Means(['group', 'step', 'phase', 'previous_rank', 'previous_private'], ['persisted'])
    episode_hist = Means(['group', 'state', 'duration', 'left_censored', 'right_censored', 'all_private'],
                         ['mean_rank', 'mean_weight', 'private_fraction'])
    popularity_cache = {}
    counts = dict(tokens=0, branches=0, transitions=0, branch_transitions=0, episodes=0,
                  exact_boundary_tie_rows=0, skipped_finished_positions=0)

    def load(g, s):
        path = a.observation_dir / f'inputs_group{g}_step{s}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['logits'].shape != (batch, length, expert_count)
                or payload['block_input_ids'].shape != (batch, length)
                or len(payload['routes']) != batch or len(payload['weights']) != batch
                or len(payload['live_requests']) != batch
                or any(len(row) != length for row in payload['routes'])
                or any(len(row) != length for row in payload['weights'])):
            raise ValueError(f'Invalid native row mapping: {path}')
        return payload

    def heldout(g, s):
        if s not in popularity_cache:
            cells = {}
            for other in all_groups:
                if s >= indices[other]['steps']:
                    continue
                payload = load(other, s)
                support = support_sets(payload['routes'], payload['live_requests'])
                cells[other] = (incidence(support), len(support))
                del payload
            popularity_cache[s] = cells
        counter, total = Counter(), 0
        for other, (visits, n) in popularity_cache[s].items():
            if other != g:
                counter.update(visits)
                total += n
        return counter, total

    with torch.inference_mode(), ExitStack() as stack:
        tokens_file = stack.enter_context((a.output_dir / 'tokens.jsonl').open('x'))
        episodes_file = stack.enter_context((a.output_dir / 'episodes.jsonl').open('x'))
        branches = StreamCSV(a.output_dir / 'branches.csv', BASE + ['expert', 'rank'] + BRANCH_METRICS + ['shared_extra_candidate'])
        pair_rows = StreamCSV(a.output_dir / 'transitions.csv', BASE + ['phase'] + TRANSITION_METRICS + ['identity_changed'])
        branch_pairs = StreamCSV(a.output_dir / 'branch_transitions.csv',
            BASE + ['phase', 'expert', 'previous_rank', 'previous_private', 'persisted', 'current_rank'])
        for stream in [branches, pair_rows, branch_pairs]:
            stack.callback(stream.close)
        def emit_episode(row):
            counts['episodes'] += 1
            episodes_file.write(json.dumps(row, allow_nan=False) + '\n')
            episode_hist.add(dict(row, all_private=row['private_fraction'] == 1))
        for g in groups:
            tracker, previous = EpisodeTracker(emit_episode), {}
            for s in steps[g]:
                payload = load(g, s)
                ids_tensor = payload['block_input_ids']
                states = [['mask' if int(x) == source['mask_token_id'] else 'decoded' for x in row] for row in ids_tensor]
                supports = support_sets(payload['routes'], payload['live_requests'])
                visits = incidence(supports)
                by_state = {state: incidence(support_sets(payload['routes'], payload['live_requests'], states, state))
                            for state in ['mask', 'decoded']}
                popular, reference_requests = heldout(g, s)
                gate = reconstruct_gate(payload['logits'].reshape(-1, expert_count), router, a.candidates)
                for r in range(batch):
                    request_id = indices[g]['request_ids'][r]
                    if not payload['live_requests'][r]:
                        counts['skipped_finished_positions'] += length
                        tracker.finish_request(request_id)
                        for anchor in [key for key in previous if key[0] == request_id]:
                            del previous[anchor]
                        continue
                    peers = request_pool(supports, r)
                    for p in range(length):
                        ids, weights = payload['routes'][r][p], payload['weights'][r][p]
                        observed = native_row(gate, r * length + p, ids, weights,
                                              router['config']['routed_scaling_factor'], a.candidates)
                        counts['exact_boundary_tie_rows'] += int(observed['exact_boundary_tie'])
                        legal = set(gate['legal'][r * length + p].nonzero().flatten().tolist())
                        legal_peers = peers & legal
                        reference = matched_popular_pool(legal, len(legal_peers), popular) if reference_requests else None
                        extra = observed['ids'][len(ids):]
                        shared = [e for e in extra if e in peers]
                        accepting = indices[g]['acceptance_steps'][r][p]
                        state = states[r][p]
                        if accepting is not None and ((s <= accepting) != (state == 'mask')):
                            raise ValueError('Saved input state disagrees with natural acceptance history')
                        base = dict(group=g, step=s, request_id=request_id, position=p, state=state,
                                    input_token_id=int(ids_tensor[r, p]), active_requests=len(supports),
                                    acceptance_step=accepting, relative_acceptance_step=s-accepting if accepting is not None else None)
                        record = dict(**base, actual_ids=ids, candidate_ids=observed['ids'],
                            candidate_scores=observed['scores'], candidate_sigmoid_scores=observed['base_scores'],
                            original_weights=weights, normalized_weights=observed['normalized_weights'],
                            incidence={e: visits[e] for e in ids}, shared_extra_ids=shared,
                            boundary_gap=observed['boundary_gap'], group_boundary_gap=observed['group_boundary_gap'],
                            private_branch_fraction=sum(visits[e] == 1 for e in ids)/len(ids),
                            within_request_extra_share=len(set(extra) & supports[r])/len(extra),
                            cross_request_extra_share=len(shared)/len(extra),
                            matched_popular_extra_share=len(set(extra) & reference)/len(extra) if reference is not None else None,
                            matched_legal_pool_size=len(legal_peers), heldout_request_count=reference_requests,
                            exact_boundary_tie=observed['exact_boundary_tie'])
                        record['cross_minus_popular_extra_share'] = (record['cross_request_extra_share'] -
                            record['matched_popular_extra_share'] if reference is not None else None)
                        tokens_file.write(json.dumps(record, allow_nan=False) + '\n')
                        token_stats.add(record)
                        counts['tokens'] += 1
                        alternative = shared[0] if shared else None
                        for expert in ids:
                            rank = observed['ids'].index(expert) + 1
                            row = dict(**base, expert=expert, rank=rank, request_incidence=visits[expert],
                                       state_request_incidence=by_state[state][expert], private=visits[expert] == 1,
                                       normalized_weight=observed['normalized_weights'][expert],
                                       gap_to_shared_extra=(observed['scores'][rank-1] - observed['scores'][observed['ids'].index(alternative)]
                                                            if alternative is not None else None),
                                       shared_extra_candidate=alternative)
                            row['private_gap_to_shared_extra'] = row['gap_to_shared_extra'] if row['private'] else None
                            branches.write(row); rank_stats.add(row); counts['branches'] += 1
                        anchor = (request_id, p)
                        old = previous.get(anchor)
                        adjacent = old is not None and old['step'] + 1 == s
                        if adjacent:
                            pair = dict(**base, **observed_transition(old, record))
                            pair_rows.write(pair); transitions.add(pair); counts['transitions'] += 1
                            for expert in old['actual_ids']:
                                row = dict(**base, phase=pair['phase'], expert=expert,
                                    previous_rank=old['candidate_ids'].index(expert)+1,
                                    previous_private=old['incidence'][expert] == 1, persisted=expert in ids,
                                    current_rank=observed['ids'].index(expert)+1 if expert in observed['ids'] else None)
                                branch_pairs.write(row); survival.add(row); counts['branch_transitions'] += 1
                        tracker.update(dict(record, first_observed=not adjacent))
                        previous[anchor] = record
                metadata['source_snapshot_sha256'][f'inputs_group{g}_step{s}.pt'] = file_hash(
                    a.observation_dir / f'inputs_group{g}_step{s}.pt')
                print(f'group={g} step={s} live={len(supports)}/{batch} native gate audit passed', flush=True)
                tokens_file.flush(); episodes_file.flush()
                del payload, gate
            tracker.finish()
    token_groups, rank_groups, transition_groups = token_stats.rows(), rank_stats.rows(), transitions.rows()
    for name, rows in [('token_group_summary', token_groups), ('rank_group_summary', rank_groups),
                       ('transition_group_summary', transition_groups), ('rank_survival_group_summary', survival.rows()),
                       ('episode_duration_histogram', episode_hist.rows())]:
        write_csv(a.output_dir / f'{name}.csv', rows)
    for name, rows, keys, metrics in [
        ('token_summary', token_groups, ['step', 'state'], TOKEN_METRICS),
        ('rank_summary', rank_groups, ['step', 'state', 'rank'], BRANCH_METRICS),
        ('transition_summary', transition_groups, ['step', 'phase'], TRANSITION_METRICS)]:
        write_csv(a.output_dir / f'{name}.csv', group_equal(rows, keys, metrics))
    metadata['native_gate_audited'] = True
    (a.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    report = dict(complete=True, observation_only=True, physical_batch=batch, layer=a.layer,
                  groups=groups, counts=counts, expert_forward=False, redecode=False,
                  caveats=['Private means unique request incidence, not low weight or dispensability.',
                           'Ranking score includes expert_bias; mixture weight does not.',
                           'Candidate similarity and small score gaps do not prove functional equivalence.',
                           'Future acceptance is a retrospective label, not available to an online router.',
                           'Later steps have fewer live requests; use the common fully-live window for batch comparisons.',
                           'Episode ends at a window/request boundary are censored, not stable forever.',
                           'Matched popularity controls legal pool cardinality, not all graph marginals.',
                           'Only generation positions are observed; no quality or speed was measured.'])
    (a.output_dir / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
