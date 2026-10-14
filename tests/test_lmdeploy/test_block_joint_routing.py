"""CPU verification of online routing and the cached real-batch decode path."""
import ast
import importlib.util
import math
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT/'opencompass-0.5.1.post1/opencompass/models'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package = types.ModuleType('block_joint_cpu_test')
package.__path__ = [str(MODELS)]
sys.modules[package.__name__] = package
route = load(package.__name__+'.llada2_joint_routing', MODELS/'llada2_joint_routing.py')
decode = load('block_joint_decode_test', MODELS/'llada2_batch_decode.py')
offline = load('block_joint_offline_test', ROOT/'benchmark/routing_competition.py')
summary = load('block_joint_summary_test',ROOT/'benchmark/summarize_shared_eval.py')


def union_size(ids, flags):
    return len({e for row, keep in zip(ids.tolist(), flags.tolist()) for e,k in zip(row,keep) if k})


@pytest.mark.parametrize('method', route.METHODS)
def test_identity(method):
    ids = torch.tensor([[0,1,2],[2,3,4]])
    weights = torch.tensor([[.7,.2,.1],[.6,.3,.1]])
    assert route.select_joint_keep(ids,weights,torch.tensor([0,1]),0,5,method).all()


def test_b4_top8_example_and_budget_equality():
    ids = torch.tensor([[1,2,3,4,5,11,9,13], [1,2,3,4,5,12,9,14],
                        [1,2,3,4,5,15,9,16], [1,2,3,4,5,17,9,18]])
    weights = torch.tensor([[.40,.20,.10,.09,.08,.06,.04,.03]]*4, dtype=torch.float64)
    requests = torch.arange(4)
    independent = route.select_joint_keep(ids,weights,requests,.1,19,'independent')
    joint = route.select_joint_keep(ids,weights,requests,.1,19,'batch')
    assert independent.sum(-1).tolist() == joint.sum(-1).tolist() == [6]*4
    assert union_size(ids,independent) == 9 and union_size(ids,joint) == 6
    assert ((weights*joint).sum(-1) >= .9-1e-12).all()


@pytest.mark.parametrize('k', [1,2,4,8])
def test_local_families_match_offline_exact_constraints(k):
    g = torch.Generator().manual_seed(72+k)
    ids = torch.stack([torch.randperm(24,generator=g)[:k] for _ in range(8)])
    weights = torch.rand(8,k,generator=g,dtype=torch.float64)+.01
    requests = torch.arange(8)//2
    families = [offline.choice_family(es,ws,.15) for es,ws in zip(ids.tolist(),weights.tolist())]
    outcomes = {}
    for method in ('independent','request','batch'):
        keep = route.select_joint_keep(ids,weights,requests,.15,24,method)
        outcomes[method] = union_size(ids, keep)
        for i,row in enumerate(families):
            selected = set(ids[i,keep[i]].tolist())
            assert int(keep[i].sum()) == row['count']
            assert any(selected == set(choice[2]) for choice in row['choices'])
    assert outcomes['batch'] <= min(outcomes['request'],outcomes['independent'])


def test_permutation_and_no_other_request_support_for_request_mode():
    ids = torch.tensor([[1,2,3],[1,4,3],[5,6,3],[5,7,3]])
    weights = torch.tensor([[.7,.2,.1]]*4,dtype=torch.float64)
    req = torch.tensor([0,0,1,1])
    full = route.select_joint_keep(ids,weights,req,.2,8,'request')
    for r in (0,1):
        rows = req == r
        single = route.select_joint_keep(ids[rows],weights[rows],req[rows],.2,8,'batch')
        assert torch.equal(full[rows],single)


class Expert(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale, self.rows = scale, 0

    def forward(self,x):
        self.rows += len(x)
        return x*self.scale


class Moe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([Expert(i+1) for i in range(8)])
        self.native_calls = 0

    def moe_infer(self,x,ids,weights):
        self.native_calls += 1
        return route.sparse_moe_infer(self,x,ids,weights,torch.ones_like(ids,dtype=torch.bool),False)


def test_controller_targets_mask_and_decoded_no_renormalization_and_restores():
    moe = Moe()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]))
    ctl = route.BlockJointRoutingController(model,.15,'independent')
    x = torch.ones(4,3)
    ids = torch.tensor([[0,1],[2,3],[4,5],[6,7]])
    w = torch.tensor([[.9,.1]]*4)
    # Row0 MASK, row1 decoded: BOTH eligible. Row2 partial prompt, row3
    # finished request: unchanged, no free fixed/background expert seed.
    ctl.begin_block_forward(torch.tensor([0,0,0,1]), torch.tensor([True,True,False,False]))
    y = moe.moe_infer(x,ids,w)
    assert torch.allclose(y[:,0],torch.tensor([.9,2.7,5.1,7.1]))
    assert [e.rows for e in moe.experts] == [1,0,1,0,1,1,1,1]
    stats = ctl.extra_stats()
    assert stats['target_assignments_executed'] == 2
    assert stats['block_assignments_executed'] == 6
    ctl.end_block_forward()
    moe.moe_infer(x,ids,w)  # original prefill/commit path
    assert moe.native_calls == 1 and ctl.assignments_original == 16
    ctl.close()
    moe.moe_infer(x,ids,w)
    assert moe.native_calls == 2


def test_actual_vendored_moe_sparse_execution_equivalence():
    source = MODELS/'LLaDA2.0-mini/modeling_llada2_moe.py'
    tree = ast.parse(source.read_text())
    names = {'LLaDA2MoeMLP','LLaDA2MoeGate','LLaDA2MoeSparseMoeBlock'}
    subset = ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[])
    ns = dict(torch=torch,nn=torch.nn,F=torch.nn.functional,math=math,
              ACT2FN={'silu':torch.nn.functional.silu},LLaDA2MoeConfig=SimpleNamespace)
    exec(compile(subset,str(source),'exec'),ns)
    config = SimpleNamespace(hidden_size=4,moe_intermediate_size=6,hidden_act='silu',
        num_experts_per_tok=2,num_experts=4,n_group=1,topk_group=1,routed_scaling_factor=2.,num_shared_experts=1)
    torch.manual_seed(19)
    moe = ns['LLaDA2MoeSparseMoeBlock'](config).eval()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]))
    x = torch.randn(2,3,4)
    with torch.inference_mode():
        native,_ = moe(x)
        ctl = route.BlockJointRoutingController(model,0,'batch')
        ctl.begin_block_forward(torch.arange(2)[:,None].expand(2,3),torch.ones(2,3,dtype=torch.bool))
        unchanged,_ = moe(x)
        assert torch.equal(native,unchanged)
        ctl.close()
        ids,weights,_ = moe.gate(x)
        req = torch.arange(2)[:,None].expand(2,3).flatten()
        keep = route.select_joint_keep(ids,weights,req,.55,4,'batch')
        hook = moe.gate.register_forward_hook(lambda m,i,o: (o[0],o[1]*keep,o[2]))
        expected,_ = moe(x)
        hook.remove()
        ctl = route.BlockJointRoutingController(model,.55,'batch',detailed_timing=True)
        ctl.begin_block_forward(req,torch.ones(6,dtype=torch.bool))
        seen = []
        hooks = [e.register_forward_pre_hook(lambda m,args: seen.append(len(args[0]))) for e in moe.experts]
        actual,_ = moe(x)
        for hook in hooks:
            hook.remove()
        assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-6)
        assert sum(seen) == int(keep.sum()) < ids.numel()
        assert ctl.extra_stats()['moe_layer_seconds'] > 0
        ctl.close()


def test_generator_lifecycle_mapping_includes_previously_decoded_positions():
    class Cache:
        length = 0

    class Controller:
        compress = None
        valid = requests = None
        assignments_original = assignments_executed = compressed_layer_calls = 0
        records = []

        def reset_stats(self):
            self.records = []

        def begin_block_forward(self,requests,valid):
            self.requests,self.valid = requests.clone(),valid.clone()

        def end_block_forward(self):
            self.requests = self.valid = None

    ctl = Controller()

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation='eager')
            self.word_embeddings = torch.nn.Embedding(12,12)

        def forward(self,input_ids,attention_mask,position_ids,past_key_values,store_kv,**kwargs):
            b,q = input_ids.shape
            assert attention_mask.shape == (b,1,q,past_key_values.length+q)
            if store_kv:
                assert ctl.valid is None
                past_key_values.length += q
            else:
                ctl.records.append((input_ids.clone(),ctl.requests.clone(),ctl.valid.clone()))
            logits = torch.zeros(b,q,12)
            logits.scatter_(-1,(3+position_ids%4)[...,None],.1)  # accept one MASK at a time
            return SimpleNamespace(last_hidden_state=logits)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model,self.lm_head = Core(),torch.nn.Identity()

        def _get_num_transfer_tokens(self,length,steps):
            return torch.ones(steps,dtype=torch.long)

    output,stats = decode.generate_batch(Model(),[[1,2],[1]*5],mask_id=11,pad_id=0,eos_id=None,
        gen_length=9,block_length=4,steps=4,threshold=.8,controller=ctl,
        cache_factory=Cache,sampling='greedy')
    assert list(map(len,output)) == [9,9] and stats['commit_forwards'] >= 2
    first,_,v0 = ctl.records[0]
    second,req,v1 = ctl.records[1]
    assert v0.tolist() == [[False,False,True,True],[False,True,True,True]]
    assert torch.equal(v0,v1)
    assert ((second != 11) & v1).any()  # decoded positions remain targets
    assert req.tolist() == [[0]*4,[1]*4]
    assert ctl.valid is None


def test_invalid_routes_fail():
    with pytest.raises(ValueError,match='Duplicate'):
        route.select_joint_keep(torch.tensor([[1,1]]),torch.ones(1,2),torch.tensor([0]),.1,2)
    with pytest.raises(ValueError,match='Invalid native'):
        route.select_joint_keep(torch.tensor([[0,1]]),torch.zeros(1,2),torch.tensor([0]),.1,2)


def test_two_gpu_placement_requires_whole_layers_on_both_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda,'device_count',lambda: 2)

    def layer(*devices):
        return SimpleNamespace(parameters=lambda: [SimpleNamespace(device=torch.device(d)) for d in devices])

    model = SimpleNamespace(model=SimpleNamespace(layers=[layer('cuda:0'),layer('cuda:1')]),hf_device_map={})
    route.validate_two_gpu_placement(model)
    model.model.layers = [layer('cuda:0')]
    with pytest.raises(RuntimeError,match='BOTH'):
        route.validate_two_gpu_placement(model)
    model.model.layers = [layer('cuda:0','cuda:1')]
    with pytest.raises(RuntimeError,match='split'):
        route.validate_two_gpu_placement(model)
    model.hf_device_map = {'model.layers.0':'cpu'}
    with pytest.raises(RuntimeError,match='offload'):
        route.validate_two_gpu_placement(model)


def test_component_summary_is_call_weighted_and_memory_is_peak(tmp_path):
    folder = tmp_path/'batch'
    folder.mkdir()
    rows = []
    for calls,mean,peak in [(1,10,20),(3,20,22)]:
        rows.append(dict(record_type='batch_metrics',first_call=False,batch_size=2,
            generated_tokens=10,wall_seconds=1,inference_seconds=1,
            denoise_forwards=1,denoise_seconds=1,prefill_seconds=0,commit_seconds=0,
            assignments_original=100,assignments_executed=80,request_denoising_steps=[1,1],
            eos_finished=[False,False],block_layer_calls=calls,planning_seconds=.1,
            routed_expert_seconds=.2,block_assignments_original=100,block_assignments_executed=80,
            target_assignments_original=80,target_assignments_executed=60,
            mean_block_active_before=mean,mean_block_active_after=mean-1,
            mean_target_active_before=mean-2,mean_target_active_after=mean-3,
            component_timing_synchronized=False,component_timing_note='host time only',
            peak_allocated_gib_by_gpu={'cuda:0':peak,'cuda:1':peak-1}))
    (folder/'metrics-test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = summary.summarize(tmp_path)['models']['batch']
    assert result['mean_block_active_before'] == 17.5
    assert result['peak_allocated_gib_by_gpu']['cuda:0'] == 22
    assert result['planning_seconds'] == .2
    assert not result['component_timing_synchronized']


def test_wrapper_loader_uses_repository_code_checkpoint_weights_and_two_gpu_budget(monkeypatch,tmp_path):
    source = MODELS/'llada2_joint_batch.py'
    tree = ast.parse(source.read_text())
    subset = ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[])
    seen = {}

    class Base:
        def __init__(self,**kwargs):
            self.path = kwargs['path']

    class Parent(Base):
        pass

    class ModelClass:
        config_class = SimpleNamespace(from_pretrained=lambda path,**kw: {'path':path})

        @staticmethod
        def from_pretrained(path,**kw):
            seen['checkpoint'],seen['load_options'] = path,kw
            return SimpleNamespace(eval=lambda: SimpleNamespace(
                model=SimpleNamespace(forward=lambda store_kv: None)))

    def local_class(name,path,**kw):
        seen['code_reference'],seen['code_directory'] = name,path
        assert kw['local_files_only'] is True
        return ModelClass

    tokenizer = SimpleNamespace(mask_token_id=1,eos_token_id=2,pad_token_id=0)
    import inspect
    import uuid
    import os
    ns = dict(__file__=str(source),BaseModel=Base,LLaDA2SharedBatch=Parent,
        torch=torch,inspect=inspect,uuid=uuid,os=os,Path=Path,METHODS=route.METHODS,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw: tokenizer),
        get_class_from_dynamic_module=local_class,
        validate_two_gpu_placement=lambda m: seen.update(placement_checked=True),
        BlockJointRoutingController=lambda m,e,method,diagnostic: (e,method,diagnostic))
    monkeypatch.setattr(torch.cuda,'device_count',lambda: 2)
    exec(compile(subset,str(source),'exec'),ns)
    wrapper = ns['LLaDA2JointBatch']('/checkpoint',route_method='batch',metrics_dir=tmp_path)
    assert wrapper.controller == (.1,'batch',False)
    assert seen['checkpoint'] == '/checkpoint' and seen['placement_checked']
    assert seen['load_options']['device_map'] == 'balanced'
    assert seen['load_options']['max_memory'] == {0:'34GiB',1:'34GiB'}
    assert seen['load_options']['attn_implementation'] == 'eager'
    assert seen['code_directory'] == str(MODELS/'LLaDA2.0-mini')
    assert 'modeling_llada2_moe' in seen['code_reference']
