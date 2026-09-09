import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[2] / 'benchmark' / 'profile_llada2_hf_moe_saturation.py'
SPEC = importlib.util.spec_from_file_location('profile_llada2_hf_moe_saturation', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extract_supported_prompts():
    assert MODULE._extract_prompt({'question': '2+2?'}, 'gsm8k') == '2+2?'
    mbpp = MODULE._extract_prompt({'prompt': 'Write f.', 'test_list': ['assert f()']}, 'mbpp')
    assert 'Write f.' in mbpp
    assert 'assert f()' in mbpp
    assert MODULE._extract_prompt({'conversations': [{'from': 'human', 'value': 'hello'}]}, 'auto') == 'hello'


def test_build_batch_inputs_aligns_one_mask_block_and_masks_padding_keys():
    inputs, attention, positions = MODULE._build_batch_inputs(
        [[10, 11, 12], [20, 21, 22, 23, 24]],
        block_length=4,
        pad_id=0,
        mask_id=99,
        device='cpu',
        dtype=torch.float32,
    )
    assert inputs.shape == (2, 12)
    assert positions.shape == (2, 12)
    assert torch.equal(inputs[:, -4:], torch.full((2, 4), 99))
    # Five left-padding keys in row zero cannot be attended by mask queries.
    assert torch.isneginf(attention[0, 0, -1, :5]).all()
    # Padding query diagonals remain finite, preventing NaNs from all-masked rows.
    assert attention[0, 0, 0, 0] == 0


def test_router_layers_counts_only_observed_mask_block():
    # Shape: batch=1, sequence=3, top-k=2. Only the final two positions count.
    topk = torch.tensor([[[0, 1], [1, 2], [2, 2]]])
    records = MODULE._router_layers([(torch.empty(0), topk)], 2, 4, 1)
    assert records == [{
        'layer_idx': 1,
        'active_experts': 2,
        'assignments': 4,
        'expert_load': [0, 1, 3, 0],
    }]
