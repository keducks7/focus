#!/usr/bin/env python3
"""Aggregate measured OpenCompass generation latency; task scores remain in OC."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(root, exclude_first=False):
    grouped = defaultdict(list)
    for path in sorted(root.rglob('metrics-*.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('record_type') == 'batch_metrics' and not (exclude_first and row['first_call']):
                    grouped[str(path.parent.relative_to(root))].append(row)
    results = {}
    for name, rows in grouped.items():
        total = lambda key: sum(r[key] for r in rows)
        wall, inference, tokens = total('wall_seconds'), total('inference_seconds'), total('generated_tokens')
        denoise_n, denoise_t = total('denoise_forwards'), total('denoise_seconds')
        original = total('assignments_original')
        results[name] = dict(batches=len(rows), requests=total('batch_size'), generated_tokens=tokens,
                            wall_seconds=wall, inference_seconds=inference,
                            generation_tps=tokens/wall if wall else None,
                            inference_tps=tokens/inference if inference else None,
                            denoise_tpf_ms=1000*denoise_t/denoise_n if denoise_n else None,
                            denoise_forwards=denoise_n,
                            prefill_seconds=total('prefill_seconds'), commit_seconds=total('commit_seconds'),
                            mean_request_denoising_steps=sum(sum(r['request_denoising_steps']) for r in rows)/total('batch_size'),
                            assignments_original=original, assignments_executed=total('assignments_executed'),
                            assignment_reduction=1-total('assignments_executed')/original if original else None,
                            reached_length_limit=sum(not e for r in rows for e in r['eos_finished']))
    return dict(exclude_first_call=exclude_first, models=results,
                scope='generate() wall excludes model load, dataset I/O and evaluation; tokens include EOS; static batches',
                accuracy='Use OpenCompass summary. This file does not compute task accuracy.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--exclude-first-call', action='store_true')
    p.add_argument('--expected-models', type=int)
    a = p.parse_args()
    result = summarize(a.root, a.exclude_first_call)
    if not result['models']:
        raise SystemExit('No eligible measurements found.')
    if a.expected_models is not None:
        if len(result['models']) != a.expected_models:
            raise SystemExit('Missing model measurements; inspect OpenCompass failure logs.')
        if len({v['requests'] for v in result['models'].values()}) != 1:
            raise SystemExit('Models processed different request counts; not a matched comparison.')
    content = json.dumps(result, indent=2)
    print(content)
    if a.output:
        with a.output.open('x') as f:
            f.write(content+'\n')
