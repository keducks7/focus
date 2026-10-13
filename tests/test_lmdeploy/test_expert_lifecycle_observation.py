"""Observation protocol tests; optional CPU Torch end-to-end without checkpoints/CUDA."""
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

BENCH=Path(__file__).resolve().parents[2]/'benchmark'
sys.path.insert(0,str(BENCH))
from observe_expert_lifecycle import sample_tokens,nested_request_sets,ObservationHooks,functional_distances,nearest
from analyze_expert_lifecycle_observation import role_event,analyze_roles
try:
    import torch
except ImportError:
    torch=None


class ProtocolTests(unittest.TestCase):
    def test_sampling_includes_final_acceptance_and_is_seeded(self):
        acceptance=[[0,2],[1,3]]
        rows=sample_tokens(acceptance,4,2)
        self.assertEqual(rows,sample_tokens(acceptance,4,2))
        self.assertEqual({r['acceptance_step'] for r in rows},{0,1,2,3})

    def test_nested_sets_include_anchor_and_censor(self):
        rows=nested_request_sets([0,2,3],2,[1,2,4],3)
        self.assertEqual(rows[1],[2])
        self.assertTrue(set(rows[1])<=set(rows[2]))
        self.assertIsNone(rows[4])

    def test_role_weights_and_transition(self):
        row=role_event(dict(masked=True,ids=[1,2],weights=[.8,.2]),
                       dict(masked=False,ids=[2,3],weights=[.6,.4]))
        self.assertEqual(row['transition'],'mask_to_decoded')
        self.assertAlmostEqual(row['jaccard'],1/3)
        self.assertEqual(row['persistent_experts'],1)
        self.assertAlmostEqual(row['old_persistent_mass'],.2)
        self.assertAlmostEqual(row['new_persistent_mass'],.6)

    def test_finished_request_excluded_and_groups_not_crossed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for group in [0,1]:
                for step in [0,1,2]:
                    rows.append(dict(record_type='route_step',group=group,block_id=0,step=step,
                        tokens=[dict(request_id=group,block_position=0,masked_before=step==0,
                            request_finished_before=step==2,expert_ids=[1,2],router_weights=[.5,.5])]))
            path=root/'routes.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(analyze_roles(path,10,root),2)

    def test_help_requires_no_model_dependencies(self):
        for name in ['observe_expert_lifecycle.py','analyze_expert_lifecycle_observation.py']:
            result=subprocess.run([sys.executable,str(BENCH/name),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipIf(torch is None,'CPU Torch unavailable')
class NumericalTests(unittest.TestCase):
    def test_distances_match_direct_formula_and_zero_source(self):
        x=torch.tensor([[1.,0.],[2.,0.],[0.,0.],[0.,1.]])
        relative,cosine,norms=functional_distances(x)
        self.assertAlmostEqual(float(relative[0,1]),1.)
        self.assertAlmostEqual(float(relative[1,0]),.5)
        self.assertEqual(nearest(relative,cosine,0,{0,1})['expert'],1)
        self.assertIsNone(nearest(relative,cosine,2,{0,1})['expert'])

    def test_end_to_end_unmodified_generation_then_probing(self):
        import observe_expert_lifecycle as experiment
        log=[]
        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer('expert_bias',torch.zeros(3))
                self.n_group,self.topk_group=1,1
            def forward(self,x):
                flat=x.reshape(-1,2)
                ids=torch.where((flat[:,0]>2)[:,None],torch.tensor([0,1]),torch.tensor([1,2]))
                return ids,torch.full_like(ids,.5,dtype=torch.float32),torch.zeros(len(flat),3)
        class Expert(torch.nn.Linear):
            def forward(self,x):
                log.append('expert')
                return super().forward(x)
        class Moe(torch.nn.Module):
            def __init__(self):
                super().__init__();self.gate=Gate()
                self.experts=torch.nn.ModuleList([Expert(2,2,bias=False) for _ in range(3)])
                for i,e in enumerate(self.experts):e.weight.data.copy_(torch.eye(2)*(i+1))
            def forward(self,x):
                ids,weights,logits=self.gate(x)
                flat=x.reshape(-1,2);y=torch.zeros_like(flat)
                for i,e in enumerate(self.experts):
                    selected=(ids==i).any(-1)
                    y[selected]+=e(flat[selected])*.5
                return y.reshape_as(x),(logits,ids)
        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__();self.word_embeddings=torch.nn.Embedding(4,2)
                self.word_embeddings.weight.data.copy_(torch.tensor([[1.,2.],[1.,1.],[3.,4.],[0.,0.]]))
                self.layers=torch.nn.ModuleList([torch.nn.Module()]);self.layers[0].mlp=Moe()
            def forward(self,input_ids,**kwargs):
                log.append('native')
                return types.SimpleNamespace(last_hidden_state=self.layers[0].mlp(self.word_embeddings(input_ids))[0])
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=Core();self.lm_head=torch.nn.Linear(2,4,bias=False)
                self.lm_head.weight.data.zero_();self.hf_device_map={'model':0,'lm_head':1}
        model=Model().eval()
        hf=types.ModuleType('transformers')
        hf.AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a,**k:model)
        hf.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:types.SimpleNamespace(mask_token_id=2,pad_token_id=3))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prompt=root/'prompts.json';output=root/'run'
            prompt.write_text(json.dumps(dict(prompt_token_ids=[[1]]*4)))
            args=['observe','mock','--prompt-snapshot',str(prompt),'--output-dir',str(output),
                  '--batch-size','2','--num-groups','2','--layer','0','--block-length','4',
                  '--denoising-steps','4','--tokens-per-group','4','--subset-sizes','1','2']
            with mock.patch.dict(sys.modules,{'transformers':hf}),mock.patch.object(sys,'argv',args), \
                    mock.patch.object(torch.cuda,'device_count',return_value=2):
                experiment.main()
            self.assertEqual(log.count('native'),10)  # 2*(1 audit + 4 normal forwards); NO replay forwards
            last_native=max(i for i,x in enumerate(log) if x=='native')
            self.assertTrue(all(x=='expert' for x in log[last_native+4:]))
            metadata=json.loads((output/'metadata.json').read_text())
            self.assertTrue(metadata['observation_only'])
            self.assertTrue((output/'generation_complete.json').exists())
            with (output/'neighbors_group0.jsonl').open() as stream:
                rows=[json.loads(l) for l in stream]
            self.assertTrue(any(r['relative_step']==1 and not r['source_selected'] for r in rows))
            self.assertTrue(any(r['relative_step']==1 and r['source_selected'] for r in rows))
            command=[sys.executable,str(BENCH/'analyze_expert_lifecycle_observation.py'),str(output),
                     '--layer','0','--output-dir',str(root/'analysis'),'--matrices']
            proc=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            self.assertTrue((root/'analysis/fixed_pair_geometry.csv').exists())
            self.assertTrue((root/'analysis/role_summary.csv').exists())


if __name__=='__main__':
    unittest.main()
