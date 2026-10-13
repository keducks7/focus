"""Protocol tests need only stdlib; numerical hook tests require CPU PyTorch."""
import json
import subprocess
import sys
import tempfile
import unittest
import types
from unittest import mock
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2]/'benchmark'
sys.path.insert(0, str(BENCH))
from intervene_expert_lifecycle import phase_at, select_cases, source_and_candidates, visits_by_request, LayerProbe
from analyze_expert_lifecycle_interventions import aggregate
try:
    import torch
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_phase_timing(self):
        self.assertEqual([phase_at(t, 3) for t in [1, 3, 4, 5]],
                         ['pre_accept_mask', 'accepting_mask', 'first_decoded', 'later_decoded'])

    def test_live_generation_incidence_and_shortlist(self):
        routes = [[[1, 2], [1, 3]], [[1, 4], [1, 4]], [[2, 5], [3, 5]]]
        visits = visits_by_request(routes, [True, True, False])
        self.assertEqual(visits[1], {0, 1})
        self.assertNotIn(5, visits)
        source, candidates = source_and_candidates([1, 2], [.9, .1], visits, 0, {1,2,3,4}, 4)
        self.assertEqual(source, 2)
        self.assertEqual(candidates, [1,4])
        self.assertNotIn(3, candidates)

    def test_cohort_is_seeded_and_censored(self):
        acceptance = [[0,2], [3,4], [None,5]]
        states = [dict(live_requests=[True, True, True]) for _ in range(6)]
        states[5]['live_requests'][1] = False
        a = select_cases(acceptance, states, 3, 7, 2, 3)
        self.assertEqual(a, select_cases(acceptance, states, 3, 7, 2, 3))
        self.assertEqual({(r['batch_index'],r['block_position']) for r in a}, {(0,1),(1,0)})

    def test_case_equal_aggregation_ignores_missing(self):
        rows = [dict(record_type='intervention', group=0, case_id=case, phase='later_decoded',
                     method='replace', source_expert=2, replacement_expert=1,
                     replacement_group_eligible=True, mean_kl=value, local_metrics={})
                for case, value in [(0,0.),(0,2.),(1,9.)]]
        rows.append(dict(record_type='missing_observation', reason='source_not_selected'))
        cases, summary = aggregate(rows)
        self.assertEqual(len(cases), 2)
        self.assertEqual(summary[0]['mean_kl'], 5.)
        self.assertEqual(summary[0]['events'], 3)

    def test_help_without_model_dependencies(self):
        for script in ['intervene_expert_lifecycle.py', 'analyze_expert_lifecycle_interventions.py']:
            result = subprocess.run([sys.executable, str(BENCH/script), '--help'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_analysis_cli_missing_observations_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'complete.json').write_text('{}')
            (root/'cases.jsonl').write_text(json.dumps(dict(record_type='case'))+'\n')
            rows = []
            for phase, kl in [('accepting_mask', .01), ('first_decoded', .1)]:
                rows.append(dict(record_type='intervention', group=0, case_id=0,
                    phase=phase, method='replace', source_expert=2, replacement_expert=1,
                    replacement_group_eligible=True, mean_kl=kl, local_metrics={}))
            rows.append(dict(record_type='missing_observation', phase='later_decoded', reason='source_not_selected'))
            (root/'interventions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            proc = subprocess.run([sys.executable, str(BENCH/'analyze_expert_lifecycle_interventions.py'), str(root)],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads((root/'analysis/summary.json').read_text())
            self.assertEqual(report['paired_comparisons'], 1)
            self.assertEqual(report['missing_observations'], 1)


@unittest.skipIf(torch is None, 'CPU PyTorch not installed')
class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.guard = torch.no_grad()
        self.guard.__enter__()

    def tearDown(self):
        self.guard.__exit__(None, None, None)

    def make_moe(self):
        class Gate(torch.nn.Module):
            def forward(self, x):
                ids = torch.tensor([[0,1],[0,1],[0,1],[0,1]])
                weights = torch.tensor([[.75,.25]]*4)
                return ids, weights, torch.zeros(4,3)
        class Moe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Gate()
                self.experts = torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in range(3)])
                for i, expert in enumerate(self.experts):
                    expert.weight.data.copy_(torch.eye(2)*(i+1))
            def forward(self, x):
                ids, weights, logits = self.gate(x)
                y = .75*self.experts[0](x)+.25*self.experts[1](x)+5*x  # unchanged shared branch
                return y, (logits, ids)
        return Moe()

    def test_single_position_formula_shared_unchanged_and_noop(self):
        moe = self.make_moe()
        x = torch.arange(1,9,dtype=torch.float32).reshape(2,2,2)
        original = moe(x)[0]
        case = dict(batch_index=1, block_position=0, source_expert=0, replacement_expert=1)
        for method, expected_delta in [('delete',-.75*x[1,0]), ('replace',.75*x[1,0])]:
            with LayerProbe(moe, 2, 2, dict(case,method=method)) as probe:
                changed = moe(x)[0]
            expected = original.clone()
            expected[1,0] += expected_delta
            torch.testing.assert_close(changed, expected)
            self.assertEqual(probe.calls, 1)
            self.assertEqual(len(moe._forward_hooks), 0)
        with LayerProbe(moe,2,2) as probe:
            torch.testing.assert_close(moe(x)[0], original)
        self.assertEqual(probe.ids.shape, (2,2,2))

    def test_absent_source_fails_closed(self):
        moe = self.make_moe()
        with self.assertRaises(RuntimeError):
            with LayerProbe(moe,2,2,dict(batch_index=0,block_position=0,source_expert=2,
                                       replacement_expert=1,method='replace')):
                moe(torch.ones(2,2,2))
        self.assertEqual(len(moe._forward_hooks), 0)

    def test_chunked_predictions_and_kl_match_full_vocab(self):
        from intervene_expert_lifecycle import predict, distribution_metrics
        class Model:
            lm_head = torch.nn.Linear(3,7,bias=False)
        torch.manual_seed(4)
        old, new = torch.randn(2,3,3), torch.randn(2,3,3)
        active = torch.tensor([[True,False,True],[False,True,False]])
        model = Model()
        result = predict(model, old, 2)
        probabilities = model.lm_head(old).float().softmax(-1)
        self.assertTrue(torch.equal(result['candidates'], probabilities.argmax(-1)))
        torch.testing.assert_close(result['confidence'], probabilities.max(-1).values)
        metrics, kl = distribution_metrics(model, old, new, active, 2)
        ref = model.lm_head(old[active]).log_softmax(-1)
        branch = model.lm_head(new[active]).log_softmax(-1)
        torch.testing.assert_close(kl, (ref.exp()*(ref-branch)).sum(-1))

    def test_cpu_mock_end_to_end_driver(self):
        """Exercise generation, anchoring, three replays, files and analysis without CUDA/HF downloads."""
        import intervene_expert_lifecycle as experiment
        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('expert_bias', torch.zeros(3))
                self.n_group, self.topk_group = 1, 1
            def forward(self, x):
                n = x.shape[0]*x.shape[1]
                return (torch.tensor([[0,1]]).expand(n,-1),
                        torch.tensor([[.75,.25]]).expand(n,-1), torch.zeros(n,3))
        class Moe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = Gate()
                self.experts = torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in range(3)])
                for expert in self.experts:
                    expert.weight.data.copy_(torch.eye(2))
            def forward(self, x):
                ids, weights, logits = self.gate(x)
                return .75*self.experts[0](x)+.25*self.experts[1](x), (logits,ids)
        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.word_embeddings = torch.nn.Embedding(4,2)
                self.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.layers[0].mlp = Moe()
            def forward(self, input_ids, **kwargs):
                hidden = self.layers[0].mlp(self.word_embeddings(input_ids))[0]
                return types.SimpleNamespace(last_hidden_state=hidden)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Core()
                self.lm_head = torch.nn.Linear(2,4,bias=False)
                self.lm_head.weight.data.zero_()  # confidence .25: one transfer each step
                self.hf_device_map = {'model':0, 'lm_head':1}
        checkpoint = Model().eval()
        module = types.ModuleType('transformers')
        module.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a,**k:checkpoint)
        module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a,**k:
                                                     types.SimpleNamespace(mask_token_id=2,pad_token_id=3))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root/'prompts.json'
            prompt.write_text(json.dumps(dict(prompt_token_ids=[[1],[1]], request_ids=[7,8])))
            argv = ['experiment', 'mock', '--prompt-snapshot', str(prompt), '--output-dir', str(root/'run'),
                    '--batch-size','2', '--layer','0', '--block-length','4','--denoising-steps','4',
                    '--tokens-per-group','2', '--pre-steps','1', '--post-steps','2','--logit-chunk-size','2']
            with mock.patch.dict(sys.modules, {'transformers':module}), mock.patch.object(sys,'argv',argv), \
                    mock.patch.object(torch.cuda,'device_count',return_value=2):
                experiment.main()
            self.assertTrue((root/'run/complete.json').exists())
            with (root/'run/interventions.jsonl').open() as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual({r['method'] for r in records if r['record_type']=='intervention'},
                             {'baseline_noop','delete','replace'})
            self.assertTrue(any(r.get('phase')=='first_decoded' for r in records))
            proc = subprocess.run([sys.executable,str(BENCH/'analyze_expert_lifecycle_interventions.py'),str(root/'run')],
                                  capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertGreater(json.loads((root/'run/analysis/summary.json').read_text())['paired_comparisons'],0)


if __name__ == '__main__':
    unittest.main()
