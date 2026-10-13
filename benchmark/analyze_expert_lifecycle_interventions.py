#!/usr/bin/env python3
"""Summarize paired single-branch interventions; standard-library analysis, optional plots."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


PHASES = ['pre_accept_mask', 'accepting_mask', 'first_decoded', 'later_decoded']
METRICS = ['target_request_mean_kl', 'target_request_candidate_flip_rate',
           'target_request_acceptance_difference_rate', 'target_request_mask_tokens',
           'mean_kl', 'max_token_kl', 'candidate_flip_rate', 'acceptance_symmetric_difference_rate',
           'mask_tokens', 'source_weight', 'source_weighted_output_norm',
           'expert_output_relative_l2', 'full_mlp_output_relative_delta']


def value(row, key):
    return row.get(key, row.get('local_metrics', {}).get(key)) if row.get('local_metrics') else row.get(key)


def aggregate(rows):
    """Average within case/phase first; cases then have equal weight, not token-step events."""
    cells = defaultdict(list)
    for row in rows:
        if row.get('record_type') == 'intervention' and row['method'] != 'baseline_noop':
            cells[(row['group'], row['case_id'], row['phase'], row['method'])].append(row)
    case_rows = []
    for (group, case, phase, method), events in sorted(cells.items()):
        result = dict(group=group, case_id=case, phase=phase, method=method, events=len(events),
            source_expert=events[0]['source_expert'], replacement_expert=events[0]['replacement_expert'],
            replacement_group_eligible_events=sum(row['replacement_group_eligible'] for row in events))
        for key in METRICS:
            numbers = [value(row, key) for row in events if value(row, key) is not None]
            result[key] = mean(numbers) if numbers else None
        case_rows.append(result)
    grouped = defaultdict(list)
    for row in case_rows:
        grouped[(row['phase'], row['method'])].append(row)
    summary = []
    for (phase, method), cases in sorted(grouped.items()):
        result = dict(phase=phase, method=method, cases=len(cases), events=sum(r['events'] for r in cases))
        for key in METRICS:
            numbers = [r[key] for r in cases if r[key] is not None]
            result[key] = mean(numbers) if numbers else None
        summary.append(result)
    return case_rows, summary


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plots(output_dir, rows, cases):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = [r for r in rows if r.get('record_type') == 'intervention' and r['method'] != 'baseline_noop']
    if not valid:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for method, color in [('delete', '#d55e00'), ('replace', '#0072b2')]:
        grouped = defaultdict(list)
        for row in valid:
            if row['method'] == method:
                grouped[row['relative_step']].append(row)
        offsets = sorted(grouped)
        for ax, metric in zip(axes, ['target_request_mean_kl', 'target_request_acceptance_difference_rate']):
            ax.plot(offsets, [mean(r[metric] for r in grouped[offset]) for offset in offsets],
                    'o-', color=color, label=method)
            ax.set(xlabel='Step minus acceptance step', ylabel=metric)
            ax.axvline(0, color='grey', ls='--', lw=.8)
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle('Single-branch counterfactuals | offset 0 is still MASK\n'
                 'Eligible cases only; sample composition changes with expert presence')
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir/f'lifecycle_perturbation.{ext}', dpi=170)
    plt.close(fig)
    replacement = [r for r in cases if r['method'] == 'replace']
    identities = sorted({(r['group'], r['case_id'], r['source_expert'], r['replacement_expert']) for r in replacement})
    grid = np.full((len(identities), len(PHASES)), np.nan)
    for i, (group, case, source, dest) in enumerate(identities):
        for row in replacement:
            if row['group'] == group and row['case_id'] == case:
                grid[i, PHASES.index(row['phase'])] = row['target_request_mean_kl']
    fig, ax = plt.subplots(figsize=(9, max(3, len(identities)*.4+1.5)), layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#cccccc')
    im = ax.imshow(grid, aspect='auto', cmap=cmap)
    ax.set_xticks(range(len(PHASES)), ['Pre-MASK', 'Accepting MASK', 'First decoded', 'Later decoded'])
    ax.set_yticks(range(len(identities)), [f'G{g}/C{c}: {s}->{d}' for g,c,s,d in identities])
    ax.set_title('Fixed expert pair | affected-request MASK prediction KL\nGrey = no eligible observation, NOT zero error')
    fig.colorbar(im, ax=ax, label='KL(reference || replacement)')
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir/f'fixed_pair_phase_heatmap.{ext}', dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--plot', action='store_true', help='Requires NumPy and Matplotlib.')
    p.add_argument('--allow-incomplete', action='store_true', help='Analyze partial files, clearly labelled.')
    a = p.parse_args()
    complete = (a.input_dir/'complete.json').exists()
    if not complete and not a.allow_incomplete:
        p.error('No complete.json; use --allow-incomplete only for explicitly partial diagnostics.')
    rows = [json.loads(line) for line in (a.input_dir/'interventions.jsonl').open()]
    anchors = [json.loads(line) for line in (a.input_dir/'cases.jsonl').open()]
    case_rows, summary = aggregate(rows)
    output = a.output_dir or a.input_dir/'analysis'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'case_phase_metrics.csv', case_rows)
    write_csv(output/'phase_summary.csv', summary)
    legal_rows = [r for r in rows if r.get('record_type') == 'intervention'
                  and r.get('replacement_group_eligible', False)]
    legal_cases, legal_summary = aggregate(legal_rows)
    write_csv(output/'legal_only_case_phase_metrics.csv', legal_cases)
    write_csv(output/'legal_only_phase_summary.csv', legal_summary)
    counts = Counter((r['phase'], r.get('reason', 'observed')) for r in rows
                     if r.get('record_type') == 'missing_observation' or r.get('method') == 'replace')
    write_csv(output/'coverage.csv', [dict(phase=phase, status=status, observations=n)
                                   for (phase, status), n in sorted(counts.items())])
    # Complete phase pairs only; never substitute zero for an absent source branch.
    paired = []
    by_case = defaultdict(dict)
    for row in case_rows:
        if row['method'] == 'replace':
            by_case[(row['group'], row['case_id'])][row['phase']] = row
    for (group, case), phases in sorted(by_case.items()):
        for left, right in [('accepting_mask', 'first_decoded'), ('first_decoded', 'later_decoded')]:
            if left in phases and right in phases:
                paired.append(dict(group=group, case_id=case, source_expert=phases[left]['source_expert'],
                    replacement_expert=phases[left]['replacement_expert'], left_phase=left, right_phase=right,
                    left_mean_kl=phases[left]['mean_kl'], right_mean_kl=phases[right]['mean_kl'],
                    mean_kl_delta=phases[right]['mean_kl']-phases[left]['mean_kl'],
                    left_target_request_mean_kl=phases[left]['target_request_mean_kl'],
                    right_target_request_mean_kl=phases[right]['target_request_mean_kl'],
                    target_request_mean_kl_delta=(phases[right]['target_request_mean_kl']-
                        phases[left]['target_request_mean_kl'])
                        if phases[left]['target_request_mean_kl'] is not None and
                           phases[right]['target_request_mean_kl'] is not None else None))
    write_csv(output/'paired_phase_changes.csv', paired)
    report = dict(complete_input=complete, usable_anchor_cases=sum(r['record_type']=='case' for r in anchors),
        skipped_anchor_cases=sum(r['record_type']=='skipped_case' for r in anchors),
        no_eligible_groups=sum(r['record_type']=='no_eligible_cases' for r in anchors),
        paired_comparisons=len(paired), missing_observations=sum(r.get('record_type')=='missing_observation' for r in rows),
        cohort_coverage=[r for r in anchors if r['record_type']=='cohort_coverage'],
        replacement_group_ineligible_events=sum(r.get('method')=='replace' and
            not r.get('replacement_group_eligible', False) for r in rows),
        caveats=['Selected branch/pair anchored at accepting-MASK; not an unbiased sample of all expert pairs.',
                 'Eligibility depends on original source presence and live request; inspect coverage.csv.',
                 'Primary plots use affected-request MASK KL; full-batch means dilute a one-request intervention.',
                 'Lifecycle association is not an isolated causal effect of MASK identity.',
                 'Post-MLP additive patches compute original experts; no speed or task accuracy claim.',
                 'Main tables include labelled counterfactuals; legal_only tables exclude group-ineligible events in BOTH arms.',
                 'Case-level averages are not independent group repeats; no significance tests are reported.'])
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if a.plot:
        plots(output, rows, case_rows)
    print(json.dumps(report, indent=2))
    print(f'Analysis: {output}')


if __name__ == '__main__':
    main()
