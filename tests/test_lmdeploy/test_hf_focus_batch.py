"""CPU semantic tests and optional comparison against repository CUDA FOCUS kernels."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / 'benchmark'))
from hf_focus_batch import PackedHF, importance, select_positions

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    def rotate(x):
        return torch.stack([-x[..., 1], x[..., 0]], dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


if torch is not None:
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_qk_norm=False)
            self.num_heads, self.num_key_value_heads, self.head_dim, self.num_key_value_groups = 2, 1, 2, 2
            self.scaling = 2 ** -.5
            self.query_key_value = nn.Linear(4, 8)
            self.dense = nn.Linear(4, 4)

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 3)
            self.experts = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

        def forward(self, x):
            logits = self.gate(x)
            indices = logits.argmax(-1, keepdim=True)
            output = torch.zeros_like(x)
            for expert, module in enumerate(self.experts):
                mask = indices[..., 0] == expert
                output[mask] = module(x[mask])
            return output, (logits, indices)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(4)
            self.post_attention_layernorm = nn.LayerNorm(4)
            self.attention = Attention()
            self.mlp = MoE()

    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = nn.Embedding(16, 4)
            self.layers = nn.ModuleList([Layer() for _ in range(3)])
            self.norm = nn.LayerNorm(4)

        def rotary_emb(self, hidden, positions):
            angle = positions.to(hidden.dtype).unsqueeze(-1).expand(-1, -1, 2) * .17
            return angle.cos(), angle.sin()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()
            self.config = SimpleNamespace(num_experts=3)
            self.lm_head = nn.Linear(4, 16)


@unittest.skipIf(torch is None, 'torch unavailable')
class Semantics(unittest.TestCase):
    def test_population_std_threshold_and_progress(self):
        positions = torch.arange(4)
        # [0,0,0,3]: last token selected, its predecessor reinstated, unseen earlier positions protected.
        delta = torch.tensor([0., 0., 0., 3.])
        self.assertEqual(select_positions(delta, positions, 1, 1, -1).tolist(), [True] * 4)
        self.assertEqual(select_positions(delta, positions, 1, 1, 3).tolist(), [False, False, True, True])

    def test_batched_cache_query_positions_and_routes_match_single_requests(self):
        torch.manual_seed(19)
        model = Model().eval()
        tokens = torch.randint(0, 16, (2, 64))
        active = torch.ones((2, 32), dtype=torch.bool)
        for mode in ('vanilla', 'delayed', 'focus'):
            runners = [PackedHF(model, 2, 32), PackedHF(model, 1, 32), PackedHF(model, 1, 32)]
            prefix = torch.tensor([[0, 30], [0, 31], [1, 29], [1, 30], [1, 31]])
            with torch.inference_mode():
                runners[0].forward(tokens, prefix, active, mode, [1, 1], 1, {}, True)
                for request in range(2):
                    coords = prefix[prefix[:, 0] == request].clone()
                    coords[:, 0] = 0
                    runners[request + 1].forward(tokens[request:request + 1], coords, active[request:request + 1], mode, [1], 1, {}, True)
                for step in range(2):
                    coords = torch.tensor([(r, p) for r in range(2) for p in range(32, 64 - step * (r + 1))])
                    output, selected, rows = runners[0].forward(tokens, coords, active, mode, [1, 2], 1, {})
                    singles = []
                    for request in range(2):
                        single_coords = coords[coords[:, 0] == request].clone()
                        single_coords[:, 0] = 0
                        single_output, single_selected, single_rows = runners[request + 1].forward(
                            tokens[request:request + 1], single_coords, active[request:request + 1], mode, [request + 1], 1, {})
                        torch.testing.assert_close(selected[selected[:, 0] == request, 1], single_selected[:, 1])
                        torch.testing.assert_close(output[selected[:, 0] == request], single_output, atol=2e-5, rtol=2e-5)
                        singles.append(single_rows)
                        for layer, (_, _, valid) in runners[0].cache.items():
                            torch.testing.assert_close(valid[request:request + 1], runners[request + 1].cache[layer][2])
                    for layer, row in enumerate(rows):
                        self.assertEqual(row['expert_load'], [singles[0][layer]['expert_load'][e] + singles[1][layer]['expert_load'][e] for e in range(3)])

    def test_cached_tail_beyond_current_rightmost_is_not_visible(self):
        torch.manual_seed(8)
        model = Model().eval()
        runner = PackedHF(model, 1, 32)
        tokens = torch.randint(0, 16, (1, 64))
        active = torch.ones((1, 32), dtype=torch.bool)
        with torch.inference_mode():
            runner.forward(tokens, torch.tensor([[0, p] for p in range(30, 64)]), active, 'vanilla', [1], 1, {}, True)
            coords = torch.tensor([[0, p] for p in range(32, 36)])
            before, _, _ = runner.forward(tokens, coords, active, 'delayed', [1], 1, {})
            for key, value, _ in runner.cache.values():
                key[:, 36:] = 10000
                value[:, 36:] = 10000
            after, _, _ = runner.forward(tokens, coords, active, 'delayed', [1], 1, {})
            torch.testing.assert_close(before, after)

    def test_generation_driver_executes_true_batch_with_verification(self):
        import contextlib
        import io
        import json
        from profile_llada2_hf_focus_scaling import run_group
        torch.manual_seed(23)
        model = Model().eval()
        args = SimpleNamespace(max_input_len=32, pad_id=15, mask_id=0, verify=True, batch_sizes=[2],
                               focus_alpha=1., confidence=.95, verify_atol=.001, verify_rtol=.001)
        stream = io.StringIO()
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            checks = run_group(model, [[1, 2, 3, 4], [5, 6, 7, 8]], 'focus', 2, 0, args, stream)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertTrue(checks)
        self.assertEqual(records[-1]['record_type'], 'generation_result')
        self.assertEqual(len(records[-1]['generated_token_ids']), 2)
        self.assertTrue(all(row['actual_batch'] == 2 for row in records if row['record_type'] == 'moe_forward'))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available() and importlib.util.find_spec('triton'), 'CUDA/Triton unavailable')
    def test_selection_against_official_triton(self):
        path = ROOT / 'lmdeploy/pytorch/kernels/cuda/focus.py'
        spec = importlib.util.spec_from_file_location('official_focus_for_test', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.manual_seed(11)
        q = torch.randn(7, 4, 16, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(7, 2, 16, device='cuda', dtype=torch.bfloat16)
        positions = torch.tensor([0, 1, 2, 3, 0, 2, 3], device='cuda')
        globals_ = torch.arange(7, device='cuda')
        pointers = torch.tensor([0, 4, 7], device='cuda', dtype=torch.int32)
        actual_importance = module.focus_importance_ragged(q, k, globals_, pointers, 4, 2, 0.25)
        expected_importance = torch.cat([importance(q[:4], k[:4], 2, 0.25), importance(q[4:], k[4:], 2, 0.25)])
        torch.testing.assert_close(actual_importance, expected_importance, atol=0.04, rtol=0.01)
        current = torch.tensor([0., 0., 0., 3., 0., 2., -1.], device='cuda')
        previous = torch.zeros_like(current)
        lengths = torch.tensor([4, 3], device='cuda', dtype=torch.int32)
        average = torch.tensor([1., 1.], device='cuda')
        targets = module.focus_compute_targets(lengths, average, 1.)
        for progress in (-1, 3):
            official = module.focus_select_and_enforce_ragged(current, previous, globals_, positions, pointers,
                                                              targets, lengths > targets,
                                                              torch.tensor([progress, progress], device='cuda'), 4)
            expected = torch.cat([select_positions(current[:4], positions[:4], 1, 1, progress),
                                  select_positions(current[4:], positions[4:], 1, 1, progress)])
            torch.testing.assert_close(official, expected)


if __name__ == '__main__':
    unittest.main()
