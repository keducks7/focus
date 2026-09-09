import json
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.models.moe_trace import MoERouteTrace


def test_moe_route_trace_writes_histograms(tmp_path):
    output_path = tmp_path / 'routes.jsonl'
    trace = MoERouteTrace(str(output_path), num_experts=4, top_k=2, num_hidden_layers=2, max_batch_size=2)
    context = SimpleNamespace(
        is_decoding=True,
        source_inputs=SimpleNamespace(is_dummy=False),
        q_seqlens=torch.tensor([2, 1]),
    )

    assert trace.begin_forward(context, query_tokens=3)
    trace.record(0, torch.tensor([[0, 1], [1, 2], [2, 3]]))
    trace.record(1, torch.tensor([[0, 0], [0, 1], [1, 1]]))
    trace.finish_forward()
    trace.close()

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert records[0]['record_type'] == 'metadata'
    assert records[0]['configured_batch_size'] == 2
    assert records[1]['actual_batch_size'] == 2
    assert records[1]['query_tokens'] == 3
    assert records[1]['q_seqlens'] == [2, 1]
    assert records[1]['layers'][0] == {
        'layer_idx': 0,
        'active_experts': 4,
        'assignments': 6,
        'expert_load': [1, 2, 2, 1],
    }
    assert records[1]['layers'][1]['active_experts'] == 2


def test_moe_route_trace_skips_dummy_and_prefill(tmp_path):
    trace = MoERouteTrace(str(tmp_path / 'routes.jsonl'),
                          num_experts=4,
                          top_k=2,
                          num_hidden_layers=2,
                          max_batch_size=2)
    dummy = SimpleNamespace(
        is_decoding=True,
        source_inputs=SimpleNamespace(is_dummy=True),
        q_seqlens=torch.tensor([2, 2]),
    )
    prefill = SimpleNamespace(
        is_decoding=False,
        source_inputs=SimpleNamespace(is_dummy=False),
        q_seqlens=torch.tensor([2, 2]),
    )

    assert not trace.begin_forward(dummy, query_tokens=4)
    assert not trace.begin_forward(prefill, query_tokens=4)
    trace.close()
