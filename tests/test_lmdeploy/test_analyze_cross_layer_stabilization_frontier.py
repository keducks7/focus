import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[2] / 'benchmark' / 'analyze_cross_layer_stabilization_frontier.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('analyze_cross_layer_stabilization_frontier', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_best_single_frontier_finds_rising_boundary():
    result = MODULE.best_single_frontier([(1, 0.2), (2, 0.3), (3, 0.8), (4, 0.9)])
    assert result['frontier_left_layer'] == 2
    assert result['frontier_right_layer'] == 3
    assert result['direction'] == 'stabilizing_with_depth'
    assert abs(result['delta_after_minus_before'] - 0.6) < 1e-12


def test_analyze_groups_by_acceptance_distance_and_layer():
    events = []
    for request_id in range(2):
        for layer_idx, jaccard in [(1, 0.2), (2, 0.3), (3, 0.8)]:
            events.append({
                'request_id': request_id,
                'layer_idx': layer_idx,
                'route_jaccard': jaccard,
                'steps_until_acceptance': 0,
            })
    profile, summary = MODULE.analyze(events)
    assert len(profile) == 3
    assert summary[0]['token_transitions_per_layer'] == 2
    assert summary[0]['frontier_left_layer'] == 2
    assert summary[0]['frontier_right_layer'] == 3


def test_direction_changes_reports_acceptance_phase_flip():
    rows = [
        {'steps_until_acceptance': 0, 'direction': 'stabilizing_with_depth'},
        {'steps_until_acceptance': 1, 'direction': 'stabilizing_with_depth'},
        {'steps_until_acceptance': 2, 'direction': 'destabilizing_with_depth'},
    ]
    assert MODULE.direction_changes(rows) == [{
        'nearer_distance': 1,
        'nearer_direction': 'stabilizing_with_depth',
        'farther_distance': 2,
        'farther_direction': 'destabilizing_with_depth',
    }]
