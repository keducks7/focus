import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[2] / 'benchmark' / 'analyze_moe_offline_hypotheses.py'
SPEC = importlib.util.spec_from_file_location('analyze_moe_offline_hypotheses', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_previous_ranking_recovers_oracle_order():
    observations = {
        (0, 0, 0): {
            'group_id': 0,
            'step': 0,
            'layer_idx': 0,
            'query_tokens': 5,
            'loads': np.asarray([10, 6, 3, 1]),
        },
        (0, 1, 0): {
            'group_id': 0,
            'step': 1,
            'layer_idx': 0,
            'query_tokens': 5,
            'loads': np.asarray([8, 5, 2, 1]),
        },
    }
    rows = MODULE.temporal_predictability(observations, 4, hot_k=2, max_lag=1)
    assert len(rows) == 1
    assert rows[0]['previous_auc'] == rows[0]['oracle_auc']
    assert rows[0]['recovered_oracle_gain'] == 1.0
    assert rows[0]['top10_jaccard'] == 1.0


def test_query_matched_null_preserves_shapes_and_metrics():
    observations = {
        (0, 0, 0): {
            'group_id': 0,
            'step': 0,
            'layer_idx': 0,
            'query_tokens': 4,
            'loads': np.asarray([8, 4, 2, 2]),
        },
        (0, 1, 0): {
            'group_id': 0,
            'step': 1,
            'layer_idx': 0,
            'query_tokens': 2,
            'loads': np.asarray([3, 2, 2, 1]),
        },
    }
    rows = MODULE.query_matched_null(observations, hot_k=2, repeats=32, seed=0)
    assert {row['metric'] for row in rows} == {
        'active_experts', 'effective_experts', 'top10_load_share'
    }
    assert all(row['assignments'] == 8 for row in rows)
    assert all(row['null_p025'] <= row['null_p975'] for row in rows)
