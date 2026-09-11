import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / 'benchmark' / 'analyze_moe_denoising.py'
SPEC = importlib.util.spec_from_file_location('analyze_moe_denoising', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


def test_effective_experts_and_adjacent_hot_overlap(tmp_path):
    trace_path = tmp_path / 'routes_bs2.jsonl'
    records = [
        {
            'record_type': 'metadata',
            'format_version': 2,
            'num_experts': 4,
            'top_k': 1,
            'configured_batch_size': 2,
            'block_length': 2,
        },
        {
            'record_type': 'denoising_step',
            'group_id': 0,
            'step': 0,
            'query_tokens': 4,
            'layers': [{'layer_idx': 1, 'expert_load': [2, 1, 1, 0]}],
        },
        {
            'record_type': 'denoising_step',
            'group_id': 0,
            'step': 1,
            'query_tokens': 2,
            'layers': [{'layer_idx': 1, 'expert_load': [1, 1, 0, 0]}],
        },
    ]
    trace_path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')

    observations = analyze.load_observations([trace_path], hot_k=2)
    summary = analyze.summarize(observations, hot_k=2)

    assert observations[0]['active_experts'] == 3
    assert observations[0]['effective_experts'] == 16 / 6
    assert observations[0]['top10_load_share'] == 0.75
    assert observations[1]['adjacent_top10_jaccard'] == 1.0
    assert summary[1]['mean_query_fraction'] == 0.5
