import copy
import csv
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / 'benchmark'))
from moe_lifecycle import OutputDeltaCollector, annotate_lifecycle
from analyze_moe_lifecycle import analyze, events_for_group, cross_layer_rows, pair_sample, validate_group
from analyze_llada2_expert_trajectory import load_events


def fixture():
    metadata = dict(record_type='metadata', format_version=2, observed_region='all_generation_positions',
                    first_moe_layer=1, num_hidden_layers=3, configured_batch_size=1,
                    block_length=2, top_k=2, num_experts=4)
    records = []
    for step in range(4):
        tokens = []
        for position, accepted_at in [(0, 0), (1, 3)]:
            masked = step <= accepted_at
            tokens.append(dict(request_id=0, block_id=0, block_position=position,
                               input_token_id=99 if masked else 7, candidate_token_id=7,
                               masked_before=masked, state_before='mask' if masked else 'decoded',
                               accepted_this_step=step == accepted_at, accepted=step == accepted_at,
                               acceptance_step=accepted_at if step >= accepted_at else None,
                               executed_this_step=True, skip_reason=None, request_finished_before=False,
                               confidence=0.99 if step == accepted_at else 0.5,
                               layers=[dict(layer_idx=layer, expert_ids=[0, 1] if step == 0 else [1, 2],
                                            router_weights=[0.5, 0.5], moe_output_relative_l2=0.1,
                                            moe_output_cosine=0.99) for layer in [1, 2]]))
        records.append(dict(record_type='token_trajectory_step', group_id=0, step=step,
                            query_tokens=2, tokens=tokens))
    return metadata, records


def test_events_cover_acceptance_jump_and_post_acceptance():
    metadata, records = fixture()
    events = list(events_for_group(records, metadata))
    token0 = [e for e in events if e['block_position'] == 0 and e['layer_idx'] == 1]
    assert [e['phase'] for e in token0] == ['first_decoded', 'later_decoded', 'later_decoded']
    assert token0[0]['relative_acceptance_step'] == 1
    assert token0[0]['route_jaccard'] == pytest.approx(1 / 3)
    assert token0[0]['router_weight_tv'] == 0.5
    assert len(events) == 12
    assert [e for e in events if e['phase'] == 'accepting_mask'][0]['relative_acceptance_step'] == 0


def test_repeated_acceptance_and_missing_layer_rejected():
    metadata, records = fixture()
    broken = copy.deepcopy(records)
    broken[1]['tokens'][0]['accepted'] = True
    broken[1]['tokens'][0]['accepted_this_step'] = True
    with pytest.raises(ValueError, match='twice'):
        validate_group(broken, metadata)
    broken = copy.deepcopy(records)
    broken[0]['tokens'][0]['layers'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        validate_group(broken, metadata)


def test_analyzer_and_legacy_mask_only_compatibility(tmp_path):
    metadata, records = fixture()
    trace = tmp_path / 'trace.jsonl'
    ending = [dict(record_type='lifecycle_coverage', group_id=0, tokens=[]),
              dict(record_type='generation_result', group_id=0)]
    trace.write_text('\n'.join(json.dumps(v) for v in [metadata] + records + ending))
    analyze(trace, tmp_path / 'analysis', pair_samples=5)
    with (tmp_path / 'analysis/lifecycle_events.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 12
    legacy = load_events([trace])
    assert len(legacy) == 6  # only the still-MASK second token, two layers
    assert all(event['block_position'] == 1 for event in legacy)


def test_incomplete_trace_is_not_reported_as_complete(tmp_path):
    metadata, records = fixture()
    trace = tmp_path / 'trace.jsonl'
    trace.write_text('\n'.join(json.dumps(v) for v in [metadata] + records))
    with pytest.raises(ValueError, match='Incomplete run'):
        analyze(trace, tmp_path / 'analysis')


def test_finished_request_is_excluded_from_cross_layer_pairs():
    _, records = fixture()
    pairs = pair_sample(records[0]['tokens'], 10, 0)
    records[0]['tokens'][0]['request_finished_before'] = True
    assert list(cross_layer_rows(records[0], pairs)) == []


def test_cross_layer_invariant_to_expert_relabeling():
    _, records = fixture()
    pairs = pair_sample(records[0]['tokens'], 10, 0)
    before = list(cross_layer_rows(records[0], pairs))
    for token in records[0]['tokens']:
        layer = token['layers'][1]
        layer['expert_ids'] = [3 - expert for expert in layer['expert_ids']]
    after = list(cross_layer_rows(records[0], pairs))
    assert before == after
    assert before[0]['sharing_pearson'] is None  # constant/one-pair structure is undefined


def test_output_observer_does_not_mutate_and_resets():
    observer = OutputDeltaCollector(2)
    hook = observer.hook(1)
    first = torch.tensor([[[9., 9.], [1., 0.], [0., 0.]]])
    assert hook(None, (), first) is None
    first[0, 1, 0] = 99  # collector owns a copy, not an alias
    observer.begin_step()
    hook(None, (), (torch.tensor([[[0., 0.], [2., 0.], [0., 0.]]]), None))
    assert observer.metrics[1][0][0]['moe_output_relative_l2'] == 1
    assert observer.metrics[1][0][0]['moe_output_cosine'] == 1
    assert observer.metrics[1][0][1]['moe_output_cosine'] is None
    observer.reset()
    assert observer.previous == {} and observer.metrics == {}


def test_annotation_captures_pre_forward_token_not_candidate():
    tokens = [dict(batch_index=0, block_position=p) for p in range(2)]
    history = {}
    annotate_lifecycle(tokens, torch.tensor([[99, 99]]), torch.tensor([[True, True]]),
                       torch.tensor([[True, False]]), history, 0)
    assert tokens[0]['input_token_id'] == 99 and tokens[0]['accepted_this_step']
    assert tokens[0]['state_before'] == 'mask'
    annotate_lifecycle(tokens, torch.tensor([[7, 99]]), torch.tensor([[False, True]]),
                       torch.tensor([[False, False]]), history, 1)
    assert tokens[0]['state_before'] == 'decoded' and tokens[0]['steps_since_acceptance'] == 1
    assert not tokens[0]['request_finished_before']


def test_full_profiler_cpu_integration_preserves_generation(tmp_path, monkeypatch):
    """Run both collection modes through main with a small deterministic CPU model."""
    import types
    import profile_llada2_expert_trajectory as profiler

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])

        def forward(self, hidden):
            return hidden + 0.1

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(100, 4)
            self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
            for layer in self.layers:
                layer.mlp = MLP()

        def forward(self, input_ids, **kwargs):
            hidden = self.word_embeddings(input_ids)
            routes = []
            for index, layer in enumerate(self.layers):
                hidden = layer.mlp(hidden)
                if index >= 1:
                    routes.append((hidden, hidden.topk(2, dim=-1).indices))
            return types.SimpleNamespace(last_hidden_state=hidden, router_logits=routes)

    model = torch.nn.Module()
    model.model = Core()
    model.lm_head = torch.nn.Linear(4, 10)
    model.config = types.SimpleNamespace(first_k_dense_replace=1, num_experts=4,
                                        num_experts_per_tok=2, num_hidden_layers=3)
    tokenizer = types.SimpleNamespace(mask_token_id=99, pad_token_id=0, eos_token_id=0)
    fake_transformers = types.ModuleType('transformers')
    fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **kw: model)
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(profiler, '_tokenize_prompts', lambda *a: [[1], [2]])

    def sample(logits, temperature):
        confidence = torch.full(logits.shape[:2], 0.1)
        confidence[0, :] = 0.99  # first request finishes before second
        return torch.full(logits.shape[:2], 7, dtype=torch.long), confidence

    monkeypatch.setattr(profiler, '_sample_block', sample)
    results = []
    for full in [False, True]:
        destination = tmp_path / str(full)
        monkeypatch.setattr(sys, 'argv', ['profiler', 'fake_dataset', 'fake_model',
            '--output-dir', str(destination), '--batch-size', '2', '--num-prompts', '2',
            '--block-length', '3', '--gen-length', '3', '--denoising-steps', '3',
            '--similarity-layer', '1', '--skip-similarity'] + (['--full-lifecycle'] if full else []))
        profiler.main()
        trace = destination / 'token_trajectories_bs2.jsonl'
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        results.append(next(r['generated_token_ids'] for r in rows if r['record_type'] == 'generation_result'))
        if full:
            analyze(trace, destination / 'analysis')
            steps = [r for r in rows if r['record_type'] == 'token_trajectory_step']
            assert all(len(r['tokens']) == 6 for r in steps)
            assert steps[1]['tokens'][0]['request_finished_before']
            assert steps[1]['tokens'][0]['layers'][0]['moe_output_relative_l2'] is not None
            coverage = next(r for r in rows if r['record_type'] == 'lifecycle_coverage')
            assert all(t['post_acceptance_censored'] for t in coverage['tokens'][:3])
    assert results[0] == results[1] == [[7, 7, 7], [7, 7, 7]]
