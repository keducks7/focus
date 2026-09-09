import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / 'benchmark' / 'analyze_moe_saturation.py'
SPEC = importlib.util.spec_from_file_location('analyze_moe_saturation', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


def test_summarize_trace_excludes_partial_batches(tmp_path):
    trace_path = tmp_path / 'routes.jsonl'
    records = [
        {
            'record_type': 'metadata',
            'format_version': 1,
            'num_experts': 4,
            'top_k': 2,
            'configured_batch_size': 2,
        },
        {
            'record_type': 'decode_forward',
            'forward_index': 0,
            'actual_batch_size': 2,
            'query_tokens': 3,
            'q_seqlens': [2, 1],
            'layers': [{
                'layer_idx': 1,
                'active_experts': 3,
                'assignments': 6,
                'expert_load': [2, 2, 2, 0],
            }],
        },
        {
            'record_type': 'decode_forward',
            'forward_index': 1,
            'actual_batch_size': 1,
            'query_tokens': 1,
            'q_seqlens': [1],
            'layers': [{
                'layer_idx': 1,
                'active_experts': 2,
                'assignments': 2,
                'expert_load': [1, 1, 0, 0],
            }],
        },
    ]
    trace_path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')

    observations, skipped = analyze.load_observations([trace_path])
    rows = analyze.summarize(observations)

    assert skipped == 1
    assert len(observations) == 1
    assert rows[0]['layer_idx'] == 'all'
    assert rows[0]['mean_active_experts'] == 3
    assert rows[0]['mean_active_ratio'] == 0.75
