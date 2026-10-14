"""CPU tests for observation-only, balanced same-expert redundancy analysis."""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

SCRIPT=Path(__file__).parents[2]/'benchmark'/'analyze_intra_expert_batch_redundancy.py'
sys.path.insert(0,str(SCRIPT.parent))
SPEC=importlib.util.spec_from_file_location('intra_redundancy',SCRIPT)
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
try:
    import torch
except ImportError:
    torch=None


class ProtocolTests(unittest.TestCase):
    def test_balanced_nested_samples_without_replacement(self):
        by_request={r:list(range(r*20,(r+1)*20)) for r in range(10)}
        samples,sets=MODULE.balanced_samples(by_request,16,[1,2,4,8],13)
        self.assertEqual((samples,sets),MODULE.balanced_samples(by_request,16,[1,2,4,8],13))
        for k,indices in samples.items():
            self.assertEqual(len(indices),16)
            self.assertEqual(len(set(indices)),16)
            self.assertEqual(Counter(i//20 for i in indices),Counter({r:16//k for r in sets[k]}))
        for k,j in [(1,2),(2,4),(4,8)]:
            self.assertTrue(set(sets[k])<set(sets[j]))
            for request in sets[k]:
                self.assertTrue({i for i in samples[j] if i//20==request}<
                                {i for i in samples[k] if i//20==request})

    def test_expert_must_be_eligible_at_every_size(self):
        by_request={r:list(range(r*8,(r+1)*8)) for r in range(8)}
        self.assertEqual(MODULE.capacity(by_request,16,[1,2,4,8]),{1:0,2:8,4:8,8:8})
        self.assertIsNone(MODULE.balanced_samples(by_request,16,[1,2,4,8],0))

    def test_candidates_have_matched_pool_sizes(self):
        requests=[0]*5+[1]*5+[2]
        candidates,count=MODULE.neighbor_candidates(requests,4,6,0)
        self.assertEqual(count,10)
        self.assertEqual(len(candidates),6)
        self.assertEqual((candidates,count),MODULE.neighbor_candidates(requests,4,6,0))
        for anchor,same,cross in candidates:
            self.assertEqual(len(set(same)),4)
            self.assertEqual(len(set(cross)),4)
            self.assertNotIn(anchor,same+cross)
            self.assertTrue(all(requests[i]==requests[anchor] for i in same))
            self.assertTrue(all(requests[i]!=requests[anchor] for i in cross))

    def test_summaries_do_not_weight_experts_by_repeats(self):
        rows=[dict(group=0,step=0,state='mask',expert=0,metric=0),
              dict(group=0,step=0,state='mask',expert=0,metric=2),
              dict(group=0,step=0,state='mask',expert=1,metric=9)]
        grouped=MODULE.group_means(rows,['group','step','state'],['metric'])
        self.assertEqual(grouped[0]['metric'],5)
        self.assertEqual(grouped[0]['experts'],2)
        grouped.append(dict(group=1,step=0,state='mask',metric=3,experts=20))
        summary=MODULE.across_groups(grouped,['step','state'],['metric'])
        self.assertEqual(summary[0]['metric'],4)
        self.assertEqual(summary[0]['groups'],2)


@unittest.skipIf(torch is None,'torch unavailable')
class NumericalTests(unittest.TestCase):
    def state(self,batch=2,length=4):
        torch.manual_seed(3)
        return dict(hidden=torch.randn(batch,length,4).bfloat16(),
                    logits=torch.randn(batch,length,2),routes=[[[0] for _ in range(length)] for _ in range(batch)],
                    live_requests=[True]*batch,block_input_ids=torch.full((batch,length),99))

    def test_state_validation_rejects_finished_requests(self):
        state=self.state()
        self.assertEqual(MODULE.validate_state(state,2,4),2)
        state['live_requests'][1]=False
        with self.assertRaisesRegex(ValueError,'finished'):MODULE.validate_state(state,2,4)

    def test_state_validation_rejects_bad_gate_or_routes(self):
        state=self.state();state['logits']=torch.ones(2,4)
        with self.assertRaisesRegex(ValueError,'gate-logit'):MODULE.validate_state(state,2,4)
        state=self.state();state['routes'][0][0]=[0,0]
        with self.assertRaisesRegex(ValueError,'Top-k'):MODULE.validate_state(state,2,4)

    def test_known_ranks(self):
        self.assertAlmostEqual(MODULE.effective_rank(torch.eye(4)),4,places=5)
        self.assertAlmostEqual(MODULE.effective_rank(torch.ones(4,8)),1,places=5)
        self.assertEqual(MODULE.effective_rank(torch.ones(4,8),True),0)

    def test_neighbor_controls_use_same_samples_in_all_spaces(self):
        x=torch.tensor([[1.,1.],[2.,1.],[1.,2.],[2.,2.]])
        candidates,_=MODULE.neighbor_candidates([0,0,1,1],1,4,0)
        results=MODULE.neighbor_statistics(x,x.clone(),candidates)
        self.assertEqual(results['valid_anchors'],4)
        for side in ['same','cross']:
            for metric in ['cosine_distance','relative_l2']:
                for suffix in ['', '_centered']:
                    self.assertAlmostEqual(results[f'input{suffix}_{side}_{metric}'],
                                           results[f'output{suffix}_{side}_{metric}'])

    def test_joint_zero_norm_exclusion_is_not_fake_similarity(self):
        x=torch.ones(4,2)
        candidates,_=MODULE.neighbor_candidates([0,0,1,1],1,4,0)
        results=MODULE.neighbor_statistics(x,x,candidates)
        self.assertEqual(results['valid_anchors'],0)
        self.assertEqual(results['zero_norm_excluded'],4)
        self.assertIsNone(results['output_same_cosine_distance'])

    def fixture(self,root):
        source=root/'source';source.mkdir()
        metadata=dict(observation_only=True,mode='vanilla',mask_token_id=99,
                      args=dict(layer=10,batch_size=2,block_length=4))
        (source/'metadata.json').write_text(json.dumps(metadata))
        (source/'generation_complete.json').write_text(json.dumps(dict(groups=[0],audits=[
            dict(group=0,first_forward_observer_matches_unobserved=True)])))
        (source/'group0.json').write_text(json.dumps(dict(request_ids=[40,41])))
        torch.save(self.state(),source/'inputs_group0_step0.pt')
        return source

    def test_inventory_without_model_weights_and_missing_decoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);out=root/'out'
            argv=['experiment',str(source),'--output-dir',str(out),'--inventory-only',
                  '--steps','0','--request-sizes','1','2','--tokens','4']
            with patch.object(sys,'argv',argv):MODULE.main()
            report=json.loads((out/'complete.json').read_text())
            self.assertEqual(report['rank_rows'],0)
            self.assertEqual(report['rank_expert_cells'],1)
            with (out/'coverage.csv').open() as stream:rows=list(csv.DictReader(stream))
            decoded=[r for r in rows if r['kind']=='state_summary' and r['state']=='decoded'][0]
            self.assertEqual(decoded['active_experts'],'0')
            self.assertEqual(decoded['common_rank_assignment_share'],'')

    def test_end_to_end_cpu_replay_mock_loader(self):
        class Loader:
            weight_map={}
            def __init__(self,*_):pass
            def load(self,expert):
                assert expert==0  # Never replay an unselected expert.
                torch.manual_seed(17)
                return {name:torch.randn(*shape).bfloat16() for name,shape in
                        [('gate_proj',(6,4)),('up_proj',(6,4)),('down_proj',(4,6))]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root);out=root/'out';model=root/'model';model.mkdir()
            (model/'config.json').write_text(json.dumps(dict(hidden_size=4,moe_intermediate_size=6,
                                                           num_experts=2,hidden_act='silu')))
            argv=['experiment',str(source),str(model),'--output-dir',str(out),'--steps','0',
                  '--request-sizes','1','2','--tokens','4','--neighbors','1','--repeats','2','--device','cpu']
            with patch.object(sys,'argv',argv),patch.object(MODULE,'ExpertWeights',Loader):MODULE.main()
            report=json.loads((out/'complete.json').read_text())
            self.assertEqual(report['rank_rows'],4)
            self.assertEqual(report['neighbor_rows'],2)
            with (out/'ranks.csv').open() as stream:ranks=list(csv.DictReader(stream))
            for row in ranks:
                self.assertEqual(row['tokens'],'4')
                self.assertEqual(len(json.loads(row['token_ids'])),4)
                if row['request_count']=='1':self.assertEqual(float(row['output_rank_delta_vs_k1']),0)
                self.assertTrue(set(json.loads(row['request_ids']))<={40,41})
            metadata=json.loads((out/'metadata.json').read_text())
            self.assertFalse(metadata['redecode'])
            self.assertFalse(metadata['router_weights_included'])
            self.assertEqual(len(metadata['source_snapshot_sha256']),1)

    def test_failed_source_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=self.fixture(root)
            (source/'generation_complete.json').write_text(json.dumps(dict(groups=[0],audits=[])))
            argv=['experiment',str(source),'--output-dir',str(root/'out'),'--inventory-only',
                  '--steps','0','--request-sizes','1','2','--tokens','4']
            with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'audit'):MODULE.main()


if __name__=='__main__':unittest.main()
