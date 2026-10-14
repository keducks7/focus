"""Pure protocol tests plus CPU parity against the supplied OFFICIAL HF gate class."""
import ast
import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / 'benchmark'))
import routing_competition as H


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'benchmark' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PLANNER = module('analyze_route_choice_space')
OBSERVER = module('observe_route_competition')
try:
    import torch
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_minimum_budget_and_complete_choice_family(self):
        family = H.choice_family([0, 2, 4], [.89, .06, .05], .06)
        self.assertEqual(family['count'], 2)
        self.assertEqual({r[2] for r in family['choices']}, {(0, 2), (0, 4)})
        self.assertEqual(family['choices'][0][2], (0, 2))
        self.assertTrue(all(m >= .94 - 1e-12 for _, m, _ in family['choices']))

    def test_no_slack_retains_every_positive_original_branch(self):
        family = H.choice_family([1, 5, 7], [.4, .3, .3], 0)
        self.assertEqual(family['count'], 3)
        self.assertEqual(len(family['choices']), 1)

    def test_fixed_rows_retain_zero_weight_native_branches_too(self):
        family = H.choice_family([1, 2, 3], [1, 0, 0], .1, fixed=True)
        self.assertEqual(family['count'], 3)
        self.assertEqual(family['choices'][0][2], (1, 2, 3))

    def test_bad_weights_rejected(self):
        for ids, weights in [([1, 1], [1, 1]), ([1], [float('nan')]), ([1], [0]), ([1], [-1])]:
            with self.assertRaises(ValueError):
                H.choice_family(ids, weights, .1)

    def test_true_cross_request_gain_without_extra_branches(self):
        rows = [H.choice_family([0, 2, 4], [.89, .06, .05], .06),
                H.choice_family([1, 3, 4], [.89, .06, .05], .06)]
        plans = H.scope_plans(rows, [40, 41], [False, False])
        self.assertEqual(H.popcount(H.union(plans['token'])), 4)
        self.assertEqual(H.popcount(H.union(plans['request'])), 4)
        self.assertEqual(H.popcount(H.union(plans['batch'])), 3)
        self.assertTrue(all([H.popcount(mask) for mask in masks] == [2, 2] for masks in plans.values()))

    def test_single_request_has_no_cross_request_gain(self):
        rows = [H.choice_family([0, 2, 4], [.89, .06, .05], .06),
                H.choice_family([1, 3, 4], [.89, .06, .05], .06)]
        plans = H.scope_plans(rows, [40, 40], [False, False])
        self.assertEqual(H.popcount(H.union(plans['request'])), 3)
        self.assertEqual(plans['request'], plans['batch'])

    def test_fixed_support_identical_in_all_scopes(self):
        rows = [H.choice_family([0, 2, 4], [.89, .06, .05], .06),
                H.choice_family([4, 5, 6], [.8, .1, .1], 0)]
        plans = H.scope_plans(rows, [0, 1], [False, True])
        for masks in plans.values():
            self.assertEqual(masks[1], rows[1]['independent'])
            self.assertEqual(H.popcount(masks[1]), 3)

    def test_request_local_improvement_can_destroy_global_sharing(self):
        rows = [H.choice_family([0, 1, 9], [.89, .06, .05], .06),
                H.choice_family([0, 2, 9], [.89, .06, .05], .06),
                H.choice_family([1, 5, 6], [.96, .02, .02], .06),
                H.choice_family([2, 5, 6], [.96, .02, .02], .06)]
        plans = H.scope_plans(rows, [0, 0, 1, 1], [False]*4)
        self.assertEqual(H.popcount(H.union(plans['token'])), 3)
        self.assertEqual(H.popcount(H.union(plans['request'])), 4)
        self.assertEqual(H.popcount(H.union(plans['batch'])), 3)

    def test_popularity_pool_matches_legal_size(self):
        pool = H.matched_popular_pool({1, 3, 5, 7}, 2, {0: 100, 3: 20, 7: 10})
        self.assertEqual(pool, {3, 7})

    def record(self, step, ids, input_id=99):
        return dict(group=0, step=step, request_id=40, position=1, state='mask' if input_id == 99 else 'decoded',
                    input_token_id=input_id, actual_ids=ids, candidate_ids=ids + [e for e in [1, 2, 3, 4] if e not in ids],
                    shared_extra_ids=[3], incidence={e: 1 for e in ids}, normalized_weights={e: .5 for e in ids},
                    first_observed=step == 0)

    def test_transition_is_adjacent_and_uses_previous_candidates(self):
        old, new = self.record(0, [1, 2]), self.record(1, [1, 3])
        row = H.observed_transition(old, new)
        self.assertAlmostEqual(row['topk_jaccard'], 1/3)
        self.assertEqual(row['entered_from_previous_extra'], 1)
        self.assertEqual(row['shared_extra_adoption'], 1)
        self.assertEqual(row['private_persistence'], .5)
        new['step'] = 4
        with self.assertRaisesRegex(ValueError, 'gap'):
            H.observed_transition(old, new)

    def test_absent_denominators_remain_missing(self):
        record = self.record(0, [1, 2])
        record['incidence'] = {1: 2, 2: 2}
        record['shared_extra_ids'] = []
        row = H.observed_transition(record, self.record(1, [1, 2]))
        self.assertIsNone(row['entered_from_previous_extra'])
        self.assertIsNone(row['shared_extra_adoption'])
        self.assertIsNone(row['private_persistence'])

    def test_episodes_distinguish_departure_identity_change_and_censor(self):
        ended = []
        tracker = H.EpisodeTracker(ended.append)
        tracker.update(self.record(0, [1, 2]))
        tracker.update(self.record(1, [1, 3]))
        departed = next(r for r in ended if r['expert'] == 2)
        self.assertFalse(departed['right_censored'])
        self.assertEqual(departed['end_reason'], 'native_topk_departure')
        tracker.update(self.record(2, [1, 3], input_id=10))
        same = next(r for r in ended if r['expert'] == 1)
        self.assertEqual(same['duration'], 2)
        self.assertEqual(same['end_reason'], 'input_identity_change')
        tracker.finish()
        last = [r for r in ended if r['input_token_id'] == 10]
        self.assertTrue(all(r['right_censored'] for r in last))

    def test_missing_groups_not_imputed_as_zero(self):
        rows = [dict(group=0, state='mask', value=1.), dict(group=1, state='mask', value=None)]
        row = H.group_equal(rows, ['state'], ['value'])[0]
        self.assertEqual(row['value'], 1)
        self.assertEqual(row['groups'], 2)
        self.assertEqual(row['value_groups'], 1)

    def test_episode_gaps_are_censored_not_natural_departures(self):
        ended = []
        tracker = H.EpisodeTracker(ended.append)
        tracker.update(self.record(0, [1, 2]))
        later = self.record(4, [1, 2]); later['first_observed'] = True
        tracker.update(later)
        self.assertTrue(all(r['end_reason'] == 'observation_gap' and r['right_censored'] for r in ended))
        tracker.finish()
        self.assertTrue(all(r['left_censored'] for r in ended if r['start_step'] == 4))

    def test_planner_cli_end_to_end_on_physical_traces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for batch in [1, 2]:
                folder = root / f'bs{batch}'; folder.mkdir()
                meta = dict(record_type='metadata', observed_region='all_generation_positions',
                            configured_batch_size=batch, num_prompts=2, num_experts=5, top_k=3,
                            block_length=1, num_generation_blocks=1, backend='huggingface', mode='vanilla')
                rows = [meta]
                for group in range(2 // batch):
                    tokens = []
                    for r in range(group * batch, (group+1)*batch):
                        ids = [0, 2, 4] if r == 0 else [1, 3, 4]
                        tokens.append(dict(request_id=r, block_position=0, request_finished_before=False,
                            state_before='mask', layers=[dict(layer_idx=10, expert_ids=ids, router_weights=[.89, .06, .05])]))
                    rows.append(dict(record_type='token_trajectory_step', group_id=group, block_id=0, step=0, tokens=tokens))
                    rows.append(dict(record_type='generation_result', group_id=group, block_id=0))
                (folder / f'token_trajectories_bs{batch}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            out = root / 'out'
            argv = ['planner', str(root), '--output-dir', str(out), '--batches', '1', '2',
                    '--layers', '10', '--steps', '0', '--epsilon', '.06', '--save-plans']
            with patch.object(sys, 'argv', argv):
                PLANNER.main()
            with (out / 'planning_summary.csv').open() as stream:
                summaries = list(csv.DictReader(stream))
            self.assertEqual(float(summaries[0]['cross_request_gain']), 0)
            self.assertEqual(float(summaries[1]['cross_request_gain']), 1)
            self.assertFalse(json.loads((out / 'metadata.json').read_text())['routes_modified'])
            self.assertTrue(json.loads((out / 'complete.json').read_text())['complete'])


@unittest.skipIf(torch is None, 'torch unavailable')
class NativeGateTests(unittest.TestCase):
    def official_gate(self):
        path = ROOT / 'opencompass-0.5.1.post1/opencompass/models/LLaDA2.0-mini/modeling_llada2_moe.py'
        tree = ast.parse(path.read_text())
        gate_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'LLaDA2MoeGate')
        # Execute the actual supplied HF class, not a second handwritten reference.
        scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math)
        exec(compile(ast.Module(body=[gate_node], type_ignores=[]), str(path), 'exec'), scope)
        config = dict(model_type='llada2_moe', hidden_size=3, num_experts=8,
                      num_experts_per_tok=2, n_group=2, topk_group=1, routed_scaling_factor=2.5)
        torch.manual_seed(11)
        gate = scope['LLaDA2MoeGate'](SimpleNamespace(**config))
        gate.expert_bias.copy_(torch.tensor([.05, -.04, .02, -.01, .06, -.03, .01, -.02]))
        router = dict(format_version=1, layer=10, config=config, expert_bias=gate.expert_bias.tolist(), algorithm='official test gate')
        return gate, router

    def test_postprocessing_matches_official_hf_gate(self):
        gate, router = self.official_gate()
        inputs = torch.randn(12, 3)
        ids, weights, logits = gate(inputs)
        reconstructed = H.reconstruct_gate(logits.detach(), router, 4)
        for i in range(len(inputs)):
            row = H.native_row(reconstructed, i, ids[i].tolist(), weights[i].tolist(), 2.5, 4)
            self.assertEqual(set(row['ids'][:2]), set(ids[i].tolist()))
            self.assertEqual(len(row['ids']), 4)
            self.assertTrue(all(reconstructed['legal'][i, e] for e in row['ids']))
            self.assertGreaterEqual(row['boundary_gap'], 0)

    def test_top8_256_expert_protocol_matches_official_class(self):
        cls, _ = self.official_gate()
        config = dict(model_type='llada2_moe', hidden_size=3, num_experts=256,
                      num_experts_per_tok=8, n_group=8, topk_group=4, routed_scaling_factor=2.5)
        gate = type(cls)(SimpleNamespace(**config))
        gate.expert_bias.copy_(torch.linspace(-.1, .1, 256))
        router = dict(layer=10, config=config, expert_bias=gate.expert_bias.tolist())
        ids, weights, logits = gate(torch.randn(8, 3))
        result = H.reconstruct_gate(logits.detach(), router, 16)
        for i in range(8):
            row = H.native_row(result, i, ids[i].tolist(), weights[i].tolist(), 2.5, 16)
            self.assertEqual(set(row['ids'][:8]), set(ids[i].tolist()))
            self.assertEqual(len(set(row['ids'])), 16)

    def test_bias_affects_selection_not_mixture_weight(self):
        gate, router = self.official_gate()
        logits = torch.tensor([[1., .9, .8, .7, -1., -2., -3., -4.]])
        router['expert_bias'] = [0, 0, .4, 0, 0, 0, 0, 0]
        result = H.reconstruct_gate(logits, router, 4)
        ids = result['expected'][0].tolist()
        self.assertEqual(ids[0], 2)
        base = logits.sigmoid()[0, ids]
        weights = (base/base.sum()*2.5).tolist()
        row = H.native_row(result, 0, ids, weights, 2.5, 4)
        self.assertLess(row['normalized_weights'][2], row['normalized_weights'][0])

    def test_native_weight_mismatch_fails_closed(self):
        gate, router = self.official_gate()
        ids, weights, logits = gate(torch.randn(1, 3))
        result = H.reconstruct_gate(logits.detach(), router, 4)
        with self.assertRaisesRegex(ValueError, 'mixture weights'):
            H.native_row(result, 0, ids[0].tolist(), [1, 1], 2.5, 4)

    def test_invalid_group_and_candidates_rejected(self):
        _, router = self.official_gate()
        with self.assertRaises(ValueError):
            H.reconstruct_gate(torch.randn(1, 8), router, 8)
        router['config']['n_group'] = 3
        with self.assertRaises(ValueError):
            H.reconstruct_gate(torch.randn(1, 8), router, 4)

    def test_exact_topk_ties_keep_recorded_membership(self):
        _, router = self.official_gate()
        router['expert_bias'] = [0]*8
        logits = torch.tensor([[1., 1., 1., 1., -10., -10., -10., -10.]])
        result = H.reconstruct_gate(logits, router, 4)
        row = H.native_row(result, 0, [0, 1], [1.25, 1.25], 2.5, 4)
        self.assertEqual(row['ids'][:2], [0, 1])
        self.assertTrue(row['exact_boundary_tie'])
        self.assertEqual(row['boundary_gap'], 0)

    def test_observer_cli_end_to_end_without_model_or_experts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.mkdir(); out = root / 'out'
            gate, router = self.official_gate()
            router_path = root / 'router.json'; router_path.write_text(json.dumps(router))
            metadata = dict(observation_only=True, mode='vanilla', mask_token_id=99,
                            args=dict(layer=10, batch_size=2, block_length=3))
            (source / 'metadata.json').write_text(json.dumps(metadata))
            (source / 'generation_complete.json').write_text(json.dumps(dict(groups=[0, 1], audits=[
                dict(group=g, first_forward_observer_matches_unobserved=True) for g in [0, 1]])))
            acceptance = [[0, 1, 2], [1, 0, 2]]
            for group in [0, 1]:
                index = dict(steps=3, request_ids=[group*2, group*2+1], acceptance_steps=acceptance)
                (source / f'group{group}.json').write_text(json.dumps(index))
                for step in range(3):
                    inputs = torch.randn(2, 3, 3)
                    ids, weights, logits = gate(inputs)
                    identity = torch.tensor([[99 if step <= a else 20+r*3+p for p, a in enumerate(row)]
                                             for r, row in enumerate(acceptance)])
                    torch.save(dict(logits=logits.detach().reshape(2, 3, 8),
                        routes=ids.reshape(2, 3, 2).tolist(), weights=weights.reshape(2, 3, 2).tolist(),
                        block_input_ids=identity, live_requests=[True, True]), source / f'inputs_group{group}_step{step}.pt')
            argv = ['observer', str(source), '--router-snapshot', str(router_path), '--output-dir', str(out),
                    '--groups', '0', '--candidates', '4']
            with patch.object(sys, 'argv', argv):
                OBSERVER.main()
            report = json.loads((out / 'complete.json').read_text())
            self.assertEqual(report['counts']['tokens'], 18)
            self.assertEqual(report['counts']['branches'], 36)
            self.assertEqual(report['counts']['transitions'], 12)
            self.assertFalse(report['expert_forward'])
            records = [json.loads(line) for line in (out / 'tokens.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['heldout_request_count'] == 2 for r in records))
            self.assertTrue(all(len(r['candidate_ids']) == 4 for r in records))
            episodes = [json.loads(line) for line in (out / 'episodes.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['end_reason'] == 'input_identity_change' for r in episodes))
            self.assertTrue(any(r['right_censored'] for r in episodes))


if __name__ == '__main__':
    unittest.main()
