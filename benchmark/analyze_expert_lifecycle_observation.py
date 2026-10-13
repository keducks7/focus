#!/usr/bin/env python3
"""Observe route turnover and off-path expert geometry; never modify model/data."""
import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean


def write_csv(path, rows):
    rows = list(rows)
    with path.open('w',newline='') as stream:
        if rows:
            fields = list(dict.fromkeys(name for row in rows for name in row))
            writer = csv.DictWriter(stream,fields)
            writer.writeheader()
            writer.writerows(rows)


def route_steps(path, layer):
    """Normalize both new observation logs and existing full-lifecycle JSONL."""
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get('record_type')
            if kind=='metadata':
                if row.get('observed_region') not in (None,'all_generation_positions'):
                    raise ValueError('MASK-only legacy traces cannot answer full-lifecycle questions.')
                continue
            if kind not in ('route_step','token_trajectory_step'):
                continue
            tokens = {}
            for token in row['tokens']:
                if token.get('executed_this_step',True) is False or token.get('request_finished_before',False):
                    continue
                if kind=='token_trajectory_step':
                    routes = next((l for l in token['layers'] if l['layer_idx']==layer),None)
                    if routes is None:
                        raise ValueError('Requested layer missing from legacy trajectory.')
                    ids,weights = routes['expert_ids'],routes['router_weights']
                else:
                    ids,weights = token['expert_ids'],token['router_weights']
                if 'masked_before' not in token:
                    raise ValueError('Missing explicit token input state; refusing MASK-only interpretation.')
                tokens[(token['request_id'],token['block_position'])] = dict(
                    masked=token['masked_before'],ids=ids,weights=weights)
            yield (row.get('group',row.get('group_id')),row.get('block_id',0),row['step']),tokens


def role_event(old, new):
    before,after = set(old['ids']),set(new['ids'])
    persist,exit_,enter = before&after,before-after,after-before
    old_weights,new_weights = dict(zip(old['ids'],old['weights'])),dict(zip(new['ids'],new['weights']))
    a,b = sum(old_weights.values()),sum(new_weights.values())
    state = ('mask' if old['masked'] else 'decoded')+'_to_'+('mask' if new['masked'] else 'decoded')
    return dict(transition=state,jaccard=len(persist)/len(before|after),
        persistent_experts=len(persist),exiting_experts=len(exit_),entering_experts=len(enter),
        old_persistent_mass=sum(old_weights[e] for e in persist)/a if a else None,
        new_persistent_mass=sum(new_weights[e] for e in persist)/b if b else None)


def analyze_roles(path, layer, output):
    previous_key,previous = None,{}
    totals = defaultdict(lambda:defaultdict(list))
    count = 0
    with (output/'role_events.csv').open('w',newline='') as stream:
        writer = None
        for key,tokens in route_steps(path,layer):
            if previous_key is not None and key[:2]==previous_key[:2] and key[2]==previous_key[2]+1:
                incidence = defaultdict(set)
                for (request,_),token in tokens.items():
                    for expert in token['ids']:
                        incidence[expert].add(request)
                for identity in previous.keys()&tokens.keys():
                    old,new = previous[identity],tokens[identity]
                    metrics = role_event(old,new)
                    gained = set(new['ids'])-set(old['ids'])
                    metrics['entering_mean_other_request_visits'] = mean(len(incidence[e]-{identity[0]}) for e in gained) if gained else None
                    row = dict(group=key[0],block_id=key[1],step=key[2],request_id=identity[0],position=identity[1],**metrics)
                    if writer is None:
                        writer = csv.DictWriter(stream,list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    count += 1
                    for name,value in metrics.items():
                        if name!='transition' and value is not None:
                            totals[(key[0],key[1],metrics['transition'])][name].append(value)
            previous_key,previous = key,tokens
    # Group-equal summary, not pooling hundreds of token events as replicates.
    group_rows = [dict(group=g,block_id=b,transition=t,events=len(values['jaccard']),
                       **{name:mean(xs) for name,xs in values.items()}) for (g,b,t),values in sorted(totals.items())]
    write_csv(output/'role_group_summary.csv',group_rows)
    grouped = defaultdict(list)
    for row in group_rows:
        grouped[row['transition']].append(row)
    summary = []
    names = ['jaccard','persistent_experts','exiting_experts','entering_experts',
             'old_persistent_mass','new_persistent_mass','entering_mean_other_request_visits']
    for transition,rows in sorted(grouped.items()):
        summary.append(dict(transition=transition,groups=len(rows),events=sum(r['events'] for r in rows),
            **{name:mean(r[name] for r in rows if name in r) if any(name in r for r in rows) else None for name in names}))
    write_csv(output/'role_summary.csv',summary)
    return count


def summarize_functional(directory, output):
    # event averages across original Top-k branches -> case/phase -> group/phase.
    events = defaultdict(lambda:defaultdict(list))
    fixed = defaultdict(dict)
    source_rows = 0
    zero_sources,selected_sources,empty_batch_pool = 0,0,0
    paired_pool_sources = 0
    for path in sorted(directory.glob('neighbors_group*.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                source_rows += 1
                zero_sources += row['source_output_norm']==0
                fixed[(row['group'],row['case_id'],row['source_expert'])][row['relative_step']] = row
                if not row['source_selected']:
                    continue
                selected_sources += 1
                empty_batch_pool += row['batch_active']['expert'] is None
                key = (row['group'],row['case_id'],row['phase'],row['relative_step'])
                for pool in ['all_legal','own_request','other_requests','batch_active']:
                    v = row[pool]['relative_l2']
                    if v is not None:
                        events[key][pool].append(v)
                full = row['all_legal']['relative_l2']
                batch = row['batch_active']['relative_l2']
                if full is not None and batch is not None:
                    events[key]['batch_regret'].append(batch-full)
                    events[key]['batch_matches_global'].append(int(row['all_legal']['expert']==row['batch_active']['expert']))
                for subset in row['nested_subsets']:
                    if subset['available'] and subset['relative_l2'] is not None and full is not None:
                        events[key][f'k{subset["request_count"]}_regret'].append(subset['relative_l2']-full)
                subsets=row['nested_subsets']
                if full is not None and subsets and all(s['available'] and s['relative_l2'] is not None for s in subsets):
                    paired_pool_sources += 1
                    for subset in subsets:
                        events[key][f'paired_k{subset["request_count"]}_regret'].append(subset['relative_l2']-full)
    names = sorted({name for values in events.values() for name in values})
    cases = defaultdict(lambda:defaultdict(list))
    for (group,case,phase,offset),metrics in events.items():
        for name,numbers in metrics.items():
            cases[(group,case,phase)][name].append(mean(numbers))
    groups = defaultdict(lambda:defaultdict(list))
    for (group,case,phase),metrics in cases.items():
        for name,numbers in metrics.items():
            groups[(group,phase)][name].append(mean(numbers))
    group_rows = [dict(group=group,phase=phase,
        **{name:mean(metrics[name]) if name in metrics else None for name in names})
        for (group,phase),metrics in sorted(groups.items())]
    write_csv(output/'functional_group_summary.csv',group_rows)
    phases = defaultdict(list)
    for row in group_rows:
        phases[row['phase']].append(row)
    summary = [dict(phase=phase,groups=len(rows),
        **{name:mean(r[name] for r in rows if r[name] is not None) if any(r[name] is not None for r in rows) else None for name in names})
        for phase,rows in sorted(phases.items())]
    write_csv(output/'functional_summary.csv',summary)
    # Exact same e->f across canonical offsets, even if source no longer selected.
    paired = []
    for (group,case,source),offsets in sorted(fixed.items()):
        if 0 not in offsets or offsets[0]['all_legal']['expert'] is None:
            continue
        anchor = offsets[0]['all_legal']['expert']
        # Distances to a fixed anchor are recovered from saved outputs, not from
        # independently chosen later nearest-neighbor distances (see --matrices).
        for left,right in [(0,1),(1,2)]:
            if left in offsets and right in offsets:
                a,b = offsets[left],offsets[right]
                paired.append(dict(group=group,case_id=case,source_expert=source,
                    left_offset=left,right_offset=right,source_selected_left=a['source_selected'],
                    source_selected_right=b['source_selected'],
                    left_nearest=a['all_legal']['expert'],right_nearest=b['all_legal']['expert'],
                    nearest_changed=a['all_legal']['expert']!=b['all_legal']['expert']))
    write_csv(output/'nearest_identity_changes.csv',paired)
    return dict(functional_source_rows=source_rows,functional_events=len(events),
                functional_cases=len({(g,c) for g,c,_ in cases}),nearest_identity_comparisons=len(paired),
                selected_source_rows=selected_sources,zero_norm_source_rows=zero_sources,
                selected_sources_without_batch_candidate=empty_batch_pool,
                paired_pool_source_rows=paired_pool_sources)


def analyze_matrices(directory,output):
    import torch
    from observe_expert_lifecycle import functional_distances
    rows,fixed = [],[]
    for path in sorted(directory.glob('matrices_group*.pt')):
        data = torch.load(path,map_location='cpu',weights_only=True)
        group = int(path.stem.split('group')[-1])
        matrices = data['matrices']
        for left,right in combinations(sorted(matrices),2):
            a,b = matrices[left]['relative_l2'],matrices[right]['relative_l2']
            valid = torch.isfinite(a).any(-1)&torch.isfinite(b).any(-1)
            aa = a.nan_to_num(nan=float('inf')).argmin(-1)
            bb = b.nan_to_num(nan=float('inf')).argmin(-1)
            rows.append(dict(group=group,common_token_cases=len(data['common_case_ids']),left_offset=left,right_offset=right,
                valid_source_experts=int(valid.sum()),nearest_agreement=float((aa[valid]==bb[valid]).float().mean()) if valid.any() else None))
        source = torch.load(directory/f'functional_outputs_group{group}.pt',map_location='cpu',weights_only=True)
        events,outs = source['events'],source['expert_outputs']
        lookup = {(e['case_id'],e['relative_step']):j for j,e in enumerate(events)}
        for case in {e['case_id'] for e in events}:
            if (case,0) not in lookup:
                continue
            anchor = lookup[case,0]
            distances,cos,_ = functional_distances(outs[:,anchor])
            legal = events[anchor]['legal']
            for expert in events[anchor]['source_ids']:
                candidates = [e for e in legal if e!=expert and torch.isfinite(distances[expert,e])]
                if not candidates:
                    continue
                dest = min(candidates,key=lambda e:float(distances[expert,e]))
                for offset in [-2,-1,0,1,2,3]:
                    if (case,offset) not in lookup:
                        continue
                    j = lookup[case,offset]
                    a,b = outs[expert,j],outs[dest,j]
                    norm=float(a.norm())
                    fixed.append(dict(group=group,case_id=case,source_expert=expert,replacement_expert=dest,
                        relative_step=offset,source_selected=expert in events[j]['source_ids'],
                        candidate_group_eligible=dest in events[j]['legal'],
                        relative_l2=float((a-b).norm())/norm if norm else None))
    write_csv(output/'matrix_nearest_agreement.csv',rows)
    write_csv(output/'fixed_pair_geometry.csv',fixed)
    return len(rows)


def plots(output,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (output/'functional_summary.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    if not rows:
        return
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for row in rows:
        columns=sorted((int(k[8:-7]),float(v)) for k,v in row.items() if k.startswith('paired_k') and k.endswith('_regret') and v)
        if columns:
            ax.plot([k for k,v in columns],[v for k,v in columns],'o-',label=row['phase'])
    ax.set(xlabel='Offline live-request subset size',ylabel='Best-candidate relative-L2 regret',
           title='Same eligible source cohort within each curve; not real batch speedup')
    if ax.lines:
        ax.legend()
    else:
        ax.text(.5,.5,'No common eligible subset cohort',ha='center',transform=ax.transAxes)
    ax.grid(alpha=.2)
    for ext in ['png','pdf']:
        fig.savefig(output/f'candidate_pool_regret.{ext}',dpi=170)
    plt.close(fig)
    matrix_paths=sorted(directory.glob('matrices_group*.pt'))
    if matrix_paths:
        import torch
        data=torch.load(matrix_paths[0],map_location='cpu',weights_only=True)
        matrices=data['matrices']
        if matrices:
            offsets=sorted(matrices)
            fig,axes=plt.subplots(1,len(offsets),figsize=(4*len(offsets),4.8),layout='constrained',squeeze=False)
            for ax,offset in zip(axes[0],offsets):
                image=ax.imshow(matrices[offset]['cosine'].numpy(),vmin=-1,vmax=1,cmap='coolwarm')
                ax.set(title=f'Offset {offset:+d}',xlabel='Expert ID',ylabel='Expert ID')
            fig.suptitle(f'Off-path function cosine | {matrix_paths[0].stem} | same {len(data["common_case_ids"])} cases\n'
                         'Offset 0 is still MASK; these are not routing or quality matrices')
            fig.colorbar(image,ax=list(axes[0]),label='Cosine')
            for ext in ['png','pdf']:
                fig.savefig(output/f'functional_similarity_phases.{ext}',dpi=170)
            plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_path',type=Path,help='New observation directory OR existing full-lifecycle JSONL.')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--layer',type=int,default=10)
    p.add_argument('--matrices',action='store_true',help='CPU Torch; load trusted side-probe archives only.')
    p.add_argument('--plot',action='store_true')
    a=p.parse_args()
    directory=a.input_path if a.input_path.is_dir() else None
    if directory:
        metadata=json.loads((directory/'metadata.json').read_text())
        if metadata['args']['layer']!=a.layer:
            p.error('Layer flag differs from capture metadata.')
        if not (directory/'generation_complete.json').exists():
            p.error('Normal generation did not complete.')
        if not (directory/'complete.json').exists():
            p.error('Observation/probe run is incomplete; analyze routes.jsonl directly for role-only diagnostics.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    report=dict(observation_only=True,layer=a.layer,role_events=analyze_roles(directory/'routes.jsonl' if directory else a.input_path,a.layer,a.output_dir))
    if directory:
        report['capture_only']=metadata['args']['capture_only']
        report.update(summarize_functional(directory,a.output_dir))
        if a.matrices:
            report['matrix_comparisons']=analyze_matrices(directory,a.output_dir)
        if a.plot:
            plots(a.output_dir,directory)
    elif a.matrices or a.plot:
        p.error('Functional matrices/plots require a new observation directory, not route-only legacy data.')
    report['caveats']=['No quality, substitutability or speed guarantee follows from output geometry.',
        'Subset size is offline pool size, not a physical batch experiment.',
        'Best distance decreases mechanically when candidates expand; this is opportunity, not causal collaboration.',
        'Source-selected metrics and off-path unselected source geometry are different quantities.',
        'Common matrix cohorts exclude censored tokens; inspect functional_missing_group*.json.']
    (a.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
