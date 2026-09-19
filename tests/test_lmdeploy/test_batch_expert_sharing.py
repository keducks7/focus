import itertools
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'benchmark'))
from analyze_batch_expert_sharing import expected_union, label_null_curve, rows_for_step, analyze


def test_exact_expectations_match_enumeration():
    sets = [{0, 1}, {1, 2}, set(), {0, 3}, {3}]
    for size in range(1, 6):
        actual = [len(set().union(*subset)) for subset in itertools.combinations(sets, size)]
        assert expected_union(sets, size) == pytest.approx(sum(actual) / len(actual))


def test_null_matches_all_label_permutations_small_pool():
    # All size-2 and size-1 sets are equally likely under uniform label permutations.
    sets = [{0, 1}, {1}, set()]
    null = label_null_curve(sets, 4)
    for size in range(1, 4):
        means = []
        for subset in itertools.combinations(range(3), size):
            possibilities = [list(itertools.combinations(range(4), len(sets[r]))) for r in subset]
            values = [len(set().union(*map(set, choices))) for choices in itertools.product(*possibilities)]
            means.append(sum(values) / len(values))
        assert null[size] == pytest.approx(sum(means) / len(means))


def test_batch32_no_enumeration_needed_and_empty_state():
    sets = [{0, 1}] * 32
    assert expected_union(sets, 16) == 2
    assert math.comb(32, 16) == 601080390
    assert label_null_curve([set()] * 32, 256)[32] == 0
    assert expected_union([set()] * 32, 16) == 0
    assert label_null_curve(sets, 256)[32] > 2


def fixture():
    meta = dict(record_type='metadata', format_version=2, mode='vanilla',
                observed_region='all_generation_positions', num_generation_blocks=1,
                configured_batch_size=2, block_length=2, num_experts=4,
                first_moe_layer=2, num_hidden_layers=3, top_k=2, num_prompts=2)
    tokens = []
    for request in range(2):
        for position, state in enumerate(['mask', 'decoded']):
            experts = [0, 1] if state == 'mask' else [1, 2 + request]
            tokens.append(dict(request_id=request, block_position=position,
                               state_before=state, masked_before=state == 'mask',
                               request_finished_before=False, executed_this_step=True, skip_reason=None,
                               layers=[dict(layer_idx=2, expert_ids=experts)]))
    return meta, dict(record_type='token_trajectory_step', group_id=0, step=0, tokens=tokens)


def test_state_decomposition_and_true_batch_endpoint():
    meta, record = fixture()
    rows, occupancy = rows_for_step(record, meta, [2], [1, 2])
    all_row = next(r for r in rows if r['state'] == 'all' and r['subset_size'] == 2)
    assert all_row['mean_union_experts'] == 4
    assert all_row['mean_mask_decoded_intersection'] == 1
    assert all_row['mean_mask_exclusive_experts'] == 1
    assert all_row['mean_decoded_exclusive_experts'] == 2
    assert all_row['physical_batch'] == 2
    mask = next(r for r in occupancy if r['state'] == 'mask' and r['expert_id'] == 0)
    assert mask['request_occupancy'] == 2 and mask['assignments'] == 2
    record['tokens'][0]['request_finished_before'] = True
    assert rows_for_step(record, meta, [2], [1, 2]) == ([], [])


def test_cli_pipeline_and_incomplete_run(tmp_path):
    meta, record = fixture()
    trace = tmp_path / 'trace.jsonl'
    end = dict(record_type='generation_result', group_id=0)
    trace.write_text('\n'.join(map(json.dumps, [meta, record, end])))
    analyze([trace], tmp_path / 'out', layers=[2], subset_sizes=[1, 2, 32])
    manifest = json.loads((tmp_path / 'out/manifest.json').read_text())
    assert manifest['files'][0]['included_group_steps'] == [[0, 0]]
    assert (tmp_path / 'out/physical_batch_observations.csv').exists()
    trace.write_text('\n'.join(map(json.dumps, [meta, record])))
    with pytest.raises(ValueError, match='Incomplete trace'):
        analyze([trace], tmp_path / 'bad', layers=[2])
