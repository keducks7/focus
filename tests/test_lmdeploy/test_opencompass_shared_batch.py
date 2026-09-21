"""CPU tests without importing the optional OpenCompass dependency tree."""
import importlib.util
import ast
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List

import pytest
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT/'opencompass-0.5.1.post1/opencompass/models'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


route = load('shared_routing_test', MODELS/'llada2_shared_routing.py')
decode = load('batch_decode_test', MODELS/'llada2_batch_decode.py')
select_cpu = load('route_cpu', ROOT/'benchmark/shared_route_selection.py')
summarizer = load('summary_test', ROOT/'benchmark/summarize_shared_eval.py')


@pytest.mark.parametrize('method',['joint','independent','joint_no_fixed'])
def test_selector_matches_cpu(method):
    g = torch.Generator().manual_seed(13)
    ids = torch.stack([torch.randperm(12, generator=g)[:4] for _ in range(12)])
    weights = torch.rand(12,4,generator=g)+.02
    mask = torch.tensor([i%3 != 0 for i in range(12)])
    for epsilon in (0.,.1,.2,.5):
        expected, _ = select_cpu.select_routes(ids.tolist(),weights.tolist(),mask.tolist(),epsilon,method)
        actual = route.select_keep(ids,weights,mask,epsilon,12,method)
        assert actual.tolist() == expected


class Expert(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.rows = 0

    def forward(self,x):
        self.rows += len(x)
        return x*self.scale


class Moe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([Expert(i+1) for i in range(5)])
        self.native_calls = 0

    def moe_infer(self,x,ids,weights):
        self.native_calls += 1
        return route.sparse_moe_infer(self,x,ids,weights,torch.ones_like(ids,dtype=torch.bool))


def test_sparse_execution_skips_real_branches_and_preserves_sum():
    moe = Moe()
    x = torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
    ids = torch.tensor([[0,1],[1,2],[3,4]])
    weights = torch.tensor([[.9,.1],[.8,.2],[.4,.6]])
    keep = torch.tensor([[True,False],[True,False],[True,True]])
    y = route.sparse_moe_infer(moe,x,ids,weights,keep)
    assert torch.allclose(y, torch.stack([x[0],2*x[1],4.6*x[2]]))
    assert [e.rows for e in moe.experts] == [1,1,0,1,1]


def test_controller_noop_and_restore():
    moe = Moe()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]))
    controller = route.SharedRoutingController(model,epsilon=0)
    controller.compress = torch.tensor([[True,False]])
    x = torch.ones(2,3)
    ids = torch.tensor([[0,1],[2,3]])
    w = torch.tensor([[.9,.1],[.8,.2]])
    actual = moe.moe_infer(x,ids,w)
    assert moe.native_calls == 1 and controller.assignments_executed == 4
    controller.close()
    assert torch.equal(moe.moe_infer(x,ids,w),actual)


@pytest.mark.parametrize('renormalize',[True,False])
def test_real_vendored_moe_sparse_matches_native_weight_intervention(renormalize):
    # Execute the repository's actual MoE classes without importing its optional
    # Transformers/FlashAttention/OpenCompass dependency tree.
    source = MODELS/'LLaDA2.0-mini/modeling_llada2_moe.py'
    tree = ast.parse(source.read_text())
    names = {'LLaDA2MoeMLP','LLaDA2MoeGate','LLaDA2MoeSparseMoeBlock'}
    subset = ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in names],type_ignores=[])
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math,
                     ACT2FN={'silu':torch.nn.functional.silu},LLaDA2MoeConfig=SimpleNamespace)
    exec(compile(subset,str(source),'exec'),namespace)
    config = SimpleNamespace(hidden_size=4,moe_intermediate_size=6,hidden_act='silu',
                             num_experts_per_tok=2,num_experts=4,n_group=1,topk_group=1,
                             routed_scaling_factor=2.,num_shared_experts=1)
    torch.manual_seed(4)
    moe = namespace['LLaDA2MoeSparseMoeBlock'](config).eval()
    x = torch.randn(2,3,4)
    compress = torch.tensor([[True,False,True],[True,True,False]])
    with torch.inference_mode():
        native,_ = moe(x)
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=moe)]))
        noop = route.SharedRoutingController(model,epsilon=0)
        noop.compress = compress
        unchanged,_ = moe(x)
        assert torch.equal(native,unchanged)
        noop.close()
        ids,weights,_ = moe.gate(x)
        keep = route.select_keep(ids,weights,compress,.55,4)
        adjusted = weights*keep
        if renormalize:
            adjusted = adjusted * (weights.sum(-1,keepdim=True)/adjusted.sum(-1,keepdim=True))
            adjusted = torch.where(keep.all(-1,keepdim=True),weights,adjusted)
        handle = moe.gate.register_forward_hook(lambda m,i,o: (o[0],adjusted,o[2]))
        expected,_ = moe(x)
        handle.remove()
        controller = route.SharedRoutingController(model,epsilon=.55,renormalize=renormalize)
        controller.compress = compress
        seen = []
        handles = [e.register_forward_pre_hook(lambda m,args: seen.append(len(args[0]))) for e in moe.experts]
        actual,_ = moe(x)
        for h in handles:
            h.remove()
        assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
        assert sum(seen) == controller.assignments_executed == int(keep.sum()) < ids.numel()
        controller.close()


class FakeCache:
    def __init__(self):
        self.length = 0


class FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(12,12)

    def forward(self,input_ids,attention_mask,position_ids,past_key_values,store_kv,**kwargs):
        b,q = input_ids.shape
        assert attention_mask.shape == (b,1,q,past_key_values.length+q)
        assert position_ids.shape == (b,q)
        logits = torch.full((b,q,12),-20.)
        logits.scatter_(-1, (3+position_ids%4)[...,None], 20.)
        if store_kv:
            past_key_values.length += q
        return SimpleNamespace(last_hidden_state=logits)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeCore()
        self.lm_head = torch.nn.Identity()

    @staticmethod
    def _get_num_transfer_tokens(length, steps):
        return torch.tensor([length//steps+(i<length%steps) for i in range(steps)])


class FakeController:
    compress = None

    def reset_stats(self):
        self.assignments_original = self.assignments_executed = self.compressed_layer_calls = 0


@pytest.mark.parametrize('length',[1,5,9])
def test_true_batch_preserves_individual_blocks_and_multiblock_budget(length):
    prompts = [[1,2], [1]*5, [2]*9, [1]*4]
    options = dict(mask_id=11,pad_id=0,eos_id=None,gen_length=length,block_length=4,
                   steps=4,threshold=.8,cache_factory=FakeCache,sampling='greedy')
    batch, stats = decode.generate_batch(FakeModel(),prompts,controller=FakeController(),**options)
    for prompt, output in zip(prompts,batch):
        single,_ = decode.generate_batch(FakeModel(),[prompt],controller=FakeController(),**options)
        assert output == single[0] == [3+(len(prompt)+j)%4 for j in range(length)]
    assert stats['generated_tokens'] == length*4
    assert stats['denoise_forwards'] > 0
    if length == 9:
        assert stats['commit_forwards'] >= 2


def test_individual_eos_and_partial_prompt():
    output,stats = decode.generate_batch(FakeModel(),[[1,2],[1]*5],mask_id=11,pad_id=0,eos_id=6,
        gen_length=9,block_length=4,steps=4,threshold=.8,cache_factory=FakeCache,
        sampling='greedy',controller=FakeController())
    assert output == [[5,6],[4,5,6]]
    assert stats['output_token_lengths'] == [2,3]


@pytest.mark.parametrize('eos_id',[None,6])
def test_b1_loop_matches_repository_vanilla_wrapper(eos_id):
    source = MODELS/'llada2.py'
    tree = ast.parse(source.read_text())
    subset = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                             and n.name == 'block_diffusion_generate'],type_ignores=[])
    namespace = dict(torch=torch,math=math,Optional=Optional,List=List,DynamicCache=FakeCache,
                     set_context=lambda **kwargs: None)
    exec(compile(subset,str(source),'exec'),namespace)

    class CompatibleFake(FakeModel):
        def forward(self,x,**kwargs):
            return SimpleNamespace(logits=self.model(input_ids=x,**kwargs).last_hidden_state)

        def _sample_with_temperature_topk_topp(self,logits,**kwargs):
            prob=logits.softmax(-1)
            confidence,candidates=prob.max(-1)
            return candidates,confidence

    for n in [1,2,4,5,9]:
        prompt=[1]*n
        expected=namespace['block_diffusion_generate'](
            CompatibleFake(),{'input_ids':torch.tensor([prompt])},mask_id=11,
            gen_length=9,block_length=4,denoising_steps=4,temperature=0.,top_k=0,
            top_p=1.,threshold=.8,eos_id=eos_id,eos_early_stop=True,use_block_cache=False,
            strategy='none')
        actual,_=decode.generate_batch(CompatibleFake(),[prompt],mask_id=11,pad_id=0,eos_id=eos_id,
            gen_length=9,block_length=4,steps=4,threshold=.8,cache_factory=FakeCache,
            sampling='native',controller=FakeController())
        assert actual[0] == expected[0,n:].tolist()


def test_transfer_only_mask_and_prefix_padding():
    active = torch.tensor([[False,True,True],[False,False,False]])
    confidence = torch.tensor([[.99,.8,.3],[.99,.99,.99]])
    assert decode.candidate_transfer(active,confidence,1,.7).tolist() == [[False,True,False],[False,False,False]]
    ids,pos,attn,lengths,pads = decode.prepare_prefix([[1,2],[1]*9],4,0,'cpu',torch.float32)
    assert lengths == [0,8] and pads == [8,0]
    assert torch.isfinite(attn[0,0].diagonal()).all()
    assert pos[1].tolist() == list(range(8))


def test_weighted_speed_summary(tmp_path):
    directory = tmp_path/'vanilla'
    directory.mkdir()
    rows = []
    for tokens,seconds,first in [(10,2,True),(30,3,False)]:
        rows.append(dict(record_type='batch_metrics',first_call=first,batch_size=2,
                         generated_tokens=tokens,wall_seconds=seconds,inference_seconds=seconds,
                         denoise_forwards=2,denoise_seconds=seconds,prefill_seconds=0,commit_seconds=0,
                         assignments_original=100,assignments_executed=80,
                         request_denoising_steps=[2,2],eos_finished=[True,False]))
    (directory/'metrics-test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result = summarizer.summarize(tmp_path)['models']['vanilla']
    assert result['generation_tps'] == 8
    assert summarizer.summarize(tmp_path,True)['models']['vanilla']['generation_tps'] == 10
