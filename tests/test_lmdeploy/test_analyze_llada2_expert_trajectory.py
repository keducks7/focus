import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / 'benchmark' / 'analyze_llada2_expert_trajectory.py'
SPEC = importlib.util.spec_from_file_location('analyze_llada2_expert_trajectory', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tie_aware_binary_auc():
    assert MODULE.binary_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert MODULE.binary_auc([0.5, 0.5], [0, 1]) == 0.5
    assert MODULE.binary_auc([0.1], [1]) is None


def test_load_same_token_adjacent_routes(tmp_path):
    path = tmp_path / 'trajectory.jsonl'
    metadata = {
        'record_type': 'metadata',
        'format_version': 1,
        'num_hidden_layers': 2,
        'first_moe_layer': 1,
        'top_k': 2,
    }
    token0 = {
        'request_id': 0,
        'block_id': 0,
        'block_position': 3,
        'confidence': 0.4,
        'accepted': False,
        'layers': [{'layer_idx': 1, 'expert_ids': [1, 2], 'router_weights': [0.8, 0.6]}],
    }
    token1 = {
        'request_id': 0,
        'block_id': 0,
        'block_position': 3,
        'confidence': 0.9,
        'accepted': True,
        'layers': [{'layer_idx': 1, 'expert_ids': [1, 3], 'router_weights': [0.9, 0.5]}],
    }
    records = [
        metadata,
        {'record_type': 'token_trajectory_step', 'group_id': 0, 'step': 0,
         'query_tokens': 1, 'tokens': [token0]},
        {'record_type': 'token_trajectory_step', 'group_id': 0, 'step': 1,
         'query_tokens': 1, 'tokens': [token1]},
    ]
    path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')
    events = MODULE.load_events([path])
    assert len(events) == 1
    assert events[0]['route_jaccard'] == 1 / 3
    assert events[0]['accepted'] == 1
    assert events[0]['steps_until_acceptance'] == 0


def test_layer_summary_separates_accepted_and_unresolved():
    events = [
        {'layer_idx': 1, 'accepted': 1, 'route_jaccard': 0.8, 'confidence': 0.9},
        {'layer_idx': 1, 'accepted': 0, 'route_jaccard': 0.2, 'confidence': 0.3},
    ]
    row = MODULE.summarize(events)[0]
    assert row['mean_route_jaccard_accepted'] == 0.8
    assert row['mean_route_jaccard_unresolved'] == 0.2
    assert row['route_stability_acceptance_auc'] == 1.0
