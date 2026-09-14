import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = Path(__file__).parents[2] / 'benchmark' / 'profile_llada2_expert_trajectory.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('profile_llada2_expert_trajectory', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_choose_candidate_mask_uses_confidence_or_minimum():
    active = torch.tensor([[True, True, False, True], [True, True, True, False]])
    confidence = torch.tensor([[0.99, 0.98, 1.0, 0.1], [0.2, 0.8, 0.1, 1.0]])
    selected = MODULE.choose_candidate_mask(active, confidence, minimum_transfer=2, threshold=0.95)
    assert selected.tolist() == [[True, True, False, False], [True, True, False, False]]


def test_token_route_records_keep_token_identity_order():
    active = torch.tensor([[False, True, True]])
    logits = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.9, 0.1], [0.8, 0.2, 0.7]]])
    topk = torch.tensor([[[0, 1], [1, 0], [0, 2]]])
    records = MODULE.token_route_records([(logits, topk)], active, first_moe_layer=4)
    assert [(record['batch_index'], record['block_position']) for record in records] == [(0, 1), (0, 2)]
    assert records[0]['layers'][0]['expert_ids'] == [1, 0]
    assert records[0]['layers'][0]['router_logits'] == [0.9, 0.2]
    assert abs(sum(records[0]['layers'][0]['router_weights']) - 1.0) < 1e-6
    assert records[1]['layers'][0]['expert_ids'] == [0, 2]


def test_nearest_experts_excludes_diagonal():
    similarity = torch.tensor([[1.0, 0.8, 0.2], [0.7, 1.0, 0.9], [0.3, 0.6, 1.0]])
    assert MODULE.nearest_experts(similarity).tolist() == [1, 2, 1]
