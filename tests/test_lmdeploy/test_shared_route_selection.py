import importlib.util
import random
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2]/'benchmark'
sys.path.insert(0, str(BENCH))
from shared_route_selection import METHODS, select_routes, reweight
from intervene_shared_routes import GateIntervention, prediction_metrics


def test_example_and_noop():
    ids = [[1, 2], [1, 3], [4, 1], [1, 5]]
    weights = [[.85,.15], [.82,.18], [.9,.1], [.5,.5]]
    keep, metrics = select_routes(ids, weights, [True,True,True,False], .2)
    assert keep == [[True,False], [True,False], [True,False], [True,True]]
    assert metrics['assignments_after'] == 5
    for method in METHODS:
        keep, _ = select_routes(ids, weights, [True]*4, 0, method)
        assert all(all(row) for row in keep)
    assert reweight([.85,.15], [True,False]) == [1.,0.]


def test_constraints_randomized():
    rng = random.Random(3)
    for _ in range(20):
        ids = [rng.sample(range(16), 8) for _ in range(10)]
        weights = [[rng.random()+.001 for _ in range(8)] for _ in ids]
        compress = [i % 3 != 0 for i in range(10)]
        for method in METHODS:
            keep, metrics = select_routes(ids, weights, compress, .2, method)
            assert metrics['active_after'] <= metrics['active_before']
            for ws, ks, c in zip(weights, keep, compress):
                assert sum(w for w,k in zip(ws,ks) if k)/sum(ws) >= .8-1e-12
                assert c or all(ks)
                assert abs(sum(reweight(ws,ks))-sum(ws)) < 1e-10


def test_native_gate_hook_preserves_fixed_rows_and_ids():
    import pytest
    torch = pytest.importorskip('torch')
    ids = torch.tensor([[1,2],[1,3],[4,1],[1,5]])
    weights = torch.tensor([[.85,.15],[.82,.18],[.9,.1],[.5,.5]])
    logits = torch.zeros(4,6)
    output = (ids, weights, logits)
    active = torch.tensor([[True,True,True,False]])
    noop = GateIntervention(active, 0, 'joint')
    assert noop(None,None,output) is output
    hook = GateIntervention(active, .2, 'joint')
    result = hook(None,None,output)
    assert result[0] is ids and result[2] is logits
    assert torch.equal(result[1][-1], weights[-1])
    assert torch.allclose(result[1].sum(-1), weights.sum(-1))
    assert torch.equal(weights, output[1])


def test_metrics_noop_and_acceptance_change():
    import pytest
    torch = pytest.importorskip('torch')
    ref = dict(candidates=torch.tensor([[1,2],[3,4]]),
               logp=torch.log_softmax(torch.tensor([[1.,2.],[3.,4.]]),-1))
    active = torch.tensor([[True,False],[True,False]])
    selected = active.clone()
    result = prediction_metrics(ref,ref,active,selected,selected)
    assert result['candidate_flip_rate'] == result['mean_kl'] == 0
    branch = dict(ref, candidates=torch.tensor([[2,2],[3,4]]))
    result = prediction_metrics(ref,branch,active,selected,torch.zeros_like(selected))
    assert result['candidate_flip_rate'] == .5
    assert result['accepted_lost'] == 2
