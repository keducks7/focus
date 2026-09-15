"""Protocol tests run without a GPU; numerical tests run when torch is installed."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / 'benchmark' / 'analyze_expert_output_redundancy.py'
SPEC = importlib.util.spec_from_file_location('redundancy', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
try:
    import torch
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_nested_requests_are_deterministic(self):
        subsets = MODULE.nested_request_sets(range(8), [1, 2, 4, 8], 7)
        self.assertEqual(subsets, MODULE.nested_request_sets(range(8), [1, 2, 4, 8], 7))
        for small, large in zip([1, 2, 4], [2, 4, 8]):
            self.assertEqual(len(subsets[small]), small)
            self.assertTrue(subsets[small] < subsets[large])

    def test_candidates_match_count_and_exclude_anchor(self):
        requests = [0] * 5 + [1] * 5 + [2]
        candidates = MODULE.matched_candidates(requests, 4, 0)
        self.assertEqual(len(candidates), 10)
        for anchor, same, cross in candidates:
            self.assertEqual(len(set(same)), 4)
            self.assertEqual(len(set(cross)), 4)
            self.assertNotIn(anchor, same + cross)
            self.assertTrue(all(requests[i] == requests[anchor] for i in same))
            self.assertTrue(all(requests[i] != requests[anchor] for i in cross))

    def test_pairing_excludes_experts_missing_a_size(self):
        rows = []
        for expert, sizes in [(0, [1, 2, 4]), (1, [1, 4])]:
            for size in sizes:
                row = dict(step=0, group_id=0, block_id=0, expert=expert, repeat=0,
                           matched_eligible=1, matched_tokens=16, request_subset_size=size)
                for metric in ('input_rank', 'output_rank', 'input_centered_rank', 'output_centered_rank'):
                    row['matched_' + metric] = size
                rows.append(row)
        paired = MODULE.paired_rank_changes(rows, [1, 2, 4])
        self.assertEqual(len(paired), 2)
        self.assertTrue(all(row['expert'] == 0 for row in paired))
        self.assertEqual(paired[-1]['output_rank_delta'], 3)


@unittest.skipIf(torch is None, 'torch unavailable')
class NumericalTests(unittest.TestCase):
    def test_rank_known_spectra_and_centering(self):
        self.assertAlmostEqual(MODULE.effective_rank(torch.eye(4)), 4, places=5)
        self.assertAlmostEqual(MODULE.effective_rank(torch.ones(4, 8)), 1, places=5)
        self.assertEqual(MODULE.effective_rank(torch.ones(4, 8), True), 0)
        self.assertEqual(MODULE.effective_rank(torch.zeros(4, 8)), 0)

    def test_expert_replay_matches_linear_modules(self):
        torch.manual_seed(0)
        gate, up, down = (torch.nn.Linear(4, 6, bias=False), torch.nn.Linear(4, 6, bias=False),
                          torch.nn.Linear(6, 4, bias=False))
        x = torch.randn(5, 4)
        expected = down(torch.nn.functional.silu(gate(x)) * up(x))
        actual = MODULE.replay(x, dict(gate_proj=gate.weight, up_proj=up.weight, down_proj=down.weight))
        torch.testing.assert_close(actual, expected)

    @unittest.skipIf(importlib.util.find_spec('safetensors') is None, 'safetensors unavailable')
    def test_end_to_end_real_weight_loading_and_token_join(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, trace, output = root / 'model', root / 'trace', root / 'output'
            model.mkdir()
            trace.mkdir()
            config = dict(hidden_size=4, moe_intermediate_size=6, num_experts=2, hidden_act='silu')
            (model / 'config.json').write_text(json.dumps(config))
            torch.manual_seed(2)
            weights = {f'model.layers.10.mlp.experts.0.{name}.weight': torch.randn(*shape)
                       for name, shape in [('gate_proj', (6, 4)), ('up_proj', (6, 4)), ('down_proj', (4, 6))]}
            save_file(weights, str(model / 'model.safetensors'))
            identities = [(request, 0, position) for request in range(2) for position in range(3)]
            metadata = dict(record_type='metadata', configured_batch_size=2, num_experts=2, top_k=1)
            tokens = [dict(request_id=request, block_id=block, block_position=position,
                           layers=[dict(layer_idx=10, expert_ids=[0])])
                      for request, block, position in identities]
            record = dict(record_type='token_trajectory_step', step=0, group_id=0, block_id=0, tokens=tokens)
            (trace / 'token_trajectories_bs2.jsonl').write_text(json.dumps(metadata) + '\n' + json.dumps(record) + '\n')
            # Deliberately reverse stored states to exercise identity lookup, rather than positional joining.
            torch.save(dict(layer_idx=10, step=0, batch_size=2, token_ids=list(reversed(identities)),
                            hidden_states=torch.randn(6, 4)), trace / 'layer10_hidden_step0_bs2.pt')
            subprocess.run([sys.executable, str(SCRIPT), str(trace), str(model), '--output-dir', str(output),
                            '--batch-size', '2', '--steps', '0', '--request-sizes', '1', '2',
                            '--matched-tokens', '2', '--neighbors', '1', '--repeats', '1', '--device', 'cpu'],
                           check=True, capture_output=True, text=True)
            coverage = json.loads((output / 'coverage.json').read_text())
            self.assertEqual(coverage['rank_rows'], 2)
            self.assertEqual(coverage['paired_rows'], 1)
            self.assertEqual(coverage['cells_with_valid_neighbors'], 1)


if __name__ == '__main__':
    unittest.main()
