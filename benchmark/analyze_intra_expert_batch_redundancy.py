#!/usr/bin/env python3
"""Observation-only replay of ACTUALLY routed tokens from saved LLaDA2 layer inputs.

One expert resident at a time; no model/decoder, no route change, no LMDeploy.
Fixed N balanced across K requests, with identical expert eligibility at every K.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_expert_output_redundancy import ExpertWeights,effective_rank,replay


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)


def capacity(by_request, total, sizes):
    return {k:sum(len(indices)>=total//k for indices in by_request.values()) for k in sizes}


def balanced_samples(by_request,total,sizes,seed):
    """Nested request sets; exactly N/K distinct tokens per request, no replacement."""
    capacities=capacity(by_request,total,sizes)
    if any(capacities[k]<k for k in sizes):
        return None
    rng=random.Random(seed);chosen=[];samples={};sets={}
    orders={r:random.Random(seed+1009*(r+1)).sample(indices,len(indices)) for r,indices in sorted(by_request.items())}
    for k in sizes:
        take=total//k
        options=[r for r,indices in sorted(by_request.items()) if r not in chosen and len(indices)>=take]
        chosen.extend(rng.sample(options,k-len(chosen)))
        sets[k]=list(chosen)
        samples[k]=[i for r in chosen for i in orders[r][:take]]
        assert len(samples[k])==total and len(set(samples[k]))==total
    return samples,sets


def neighbor_candidates(requests,count,cap,seed):
    rng=random.Random(seed);eligible=[]
    for anchor,r in enumerate(requests):
        same=[i for i,s in enumerate(requests) if i!=anchor and s==r]
        cross=[i for i,s in enumerate(requests) if s!=r]
        if len(same)>=count and len(cross)>=count:
            eligible.append((anchor,same,cross))
    rng.shuffle(eligible)
    return [(i,rng.sample(s,count),rng.sample(c,count)) for i,s,c in eligible[:cap]],len(eligible)


def neighbor_statistics(inputs,outputs,candidates):
    """Same anchors/candidate draws in four spaces; zero-norm exclusions are joint."""
    import torch
    spaces=dict(input=inputs.float(),output=outputs.float())
    spaces['input_centered']=spaces['input']-spaces['input'].mean(0,keepdim=True)
    spaces['output_centered']=spaces['output']-spaces['output'].mean(0,keepdim=True)
    norms={k:x.norm(dim=-1) for k,x in spaces.items()}
    values=defaultdict(list);valid=0
    for anchor,same,cross in candidates:
        ids=[anchor]+same+cross
        if any(bool((n[ids]<=1e-12).any()) for n in norms.values()):
            continue
        valid+=1
        for name,x in spaces.items():
            for label,neighbors in [('same',same),('cross',cross)]:
                others=x[neighbors];a=x[anchor]
                cos=(others@a)/(norms[name][neighbors]*norms[name][anchor])
                values[f'{name}_{label}_cosine_distance'].append(float((1-cos).clamp(0,2).min()))
                values[f'{name}_{label}_relative_l2'].append(float((others-a).norm(dim=-1).min()/norms[name][anchor]))
    result=dict(valid_anchors=valid,zero_norm_excluded=len(candidates)-valid)
    for name in spaces:
        for side in ['same','cross']:
            for metric in ['cosine_distance','relative_l2']:
                key=f'{name}_{side}_{metric}';result[key]=mean(values[key]) if values[key] else None
    return result


def group_means(rows,key_names,metrics):
    """Collapse sampling repeats within each expert, then average experts equally."""
    experts=defaultdict(list)
    for row in rows:experts[tuple(row[k] for k in key_names)+ (row['expert'],)].append(row)
    cells=defaultdict(list)
    for key,observations in experts.items():
        cells[key[:-1]].append({m:mean(r[m] for r in observations if r.get(m) is not None)
            if any(r.get(m) is not None for r in observations) else None for m in metrics})
    return [dict(zip(key_names,key),experts=len(observations),
        **{m:mean(r[m] for r in observations if r[m] is not None)
           if any(r[m] is not None for r in observations) else None for m in metrics})
        for key,observations in sorted(cells.items())]


def across_groups(rows,key_names,metrics):
    cells=defaultdict(list)
    for row in rows:cells[tuple(row[k] for k in key_names)].append(row)
    return [dict(zip(key_names,key),groups=len(observations),
        **{m:mean(r[m] for r in observations if r.get(m) is not None)
           if any(r.get(m) is not None for r in observations) else None for m in metrics})
        for key,observations in sorted(cells.items())]


def validate_state(state,batch,length):
    import torch
    h=state['hidden'];routes=state['routes']
    if h.ndim!=3 or h.shape[:2]!=(batch,length) or state['block_input_ids'].shape!=(batch,length):
        raise ValueError('Layer input/identity shape mismatch.')
    if not torch.isfinite(h).all() or len(state['live_requests'])!=batch:
        raise ValueError('Invalid input or live-request flags.')
    if not all(state['live_requests']):
        raise ValueError('Requested cell has finished requests; choose a common fully-live window.')
    logits=state['logits']
    if logits.ndim!=3 or logits.shape[:2]!=(batch,length) or not torch.isfinite(logits).all():
        raise ValueError('Invalid gate-logit shape.')
    experts=logits.shape[-1]
    if len(routes)!=batch or any(len(positions)!=length for positions in routes):
        raise ValueError('Route identities do not align with inputs.')
    topk=len(routes[0][0])
    if topk<1:raise ValueError('Empty actual Top-k IDs.')
    for positions in routes:
        for ids in positions:
            if len(ids)!=topk or len(set(ids))!=topk or any(not 0<=e<experts for e in ids):
                raise ValueError('Invalid actual Top-k IDs.')
    return experts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('observation_dir',type=Path)
    p.add_argument('model_path',type=Path,nargs='?',help='Same unquantized checkpoint; optional for --inventory-only.')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--layer',type=int,default=10)
    p.add_argument('--steps',nargs='+',type=int,default=[0,4,7])
    p.add_argument('--groups',nargs='+',type=int)
    p.add_argument('--states',nargs='+',choices=['mask','decoded','all'],default=['mask','decoded'])
    p.add_argument('--request-sizes',nargs='+',type=int,default=[1,2,4,8])
    p.add_argument('--tokens',type=int,default=16)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--neighbors',type=int,default=4)
    p.add_argument('--max-anchors',type=int,default=64)
    p.add_argument('--replay-chunk',type=int,default=64)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();sizes=sorted(set(a.request_sizes));steps=sorted(set(a.steps));states=list(dict.fromkeys(a.states))
    if min(a.tokens,a.repeats,a.neighbors,a.max_anchors,a.replay_chunk,*sizes)<1 or min(a.layer,*steps)<0:
        p.error('Counts must be positive and layer/steps nonnegative.')
    if sizes[0]!=1 or any(a.tokens%k for k in sizes):
        p.error('Sizes must include 1 and divide the fixed token count exactly.')
    if not a.inventory_only and a.model_path is None:p.error('model_path is required for expert replay.')
    source=json.loads((a.observation_dir/'metadata.json').read_text())
    if not source.get('observation_only') or source['mode']!='vanilla' or source['args']['layer']!=a.layer:
        raise ValueError('Expected matching Vanilla observation layer inputs.')
    completion=json.loads((a.observation_dir/'generation_complete.json').read_text())
    groups=sorted(set(a.groups if a.groups is not None else completion['groups']))
    if not groups or not set(groups)<=set(completion['groups']):p.error('Requested groups missing from completed generation.')
    audits={row['group']:row['first_forward_observer_matches_unobserved'] for row in completion.get('audits',[])}
    if any(audits.get(g) is not True for g in groups):
        raise ValueError('Missing/failed source observer-versus-native first-forward audit.')
    batch,length=source['args']['batch_size'],source['args']['block_length']
    if max(sizes)>batch:p.error('Subset request count exceeds the physical source batch.')
    paths={(g,s):a.observation_dir/f'inputs_group{g}_step{s}.pt' for g in groups for s in steps}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):raise FileExistsError('Use a NEW/EMPTY output directory.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(min(torch.get_num_threads(),4))
    loader=None;config=None
    if not a.inventory_only:
        config=json.loads((a.model_path/'config.json').read_text())
        if config.get('hidden_act')!='silu' or config.get('quantization_config'):
            raise ValueError('Only unquantized LLaDA2 SwiGLU experts are supported.')
        if str(a.device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
        loader=ExpertWeights(a.model_path,a.layer,config,a.device)
        if any(key.endswith('.bias') and f'model.layers.{a.layer}.mlp.experts.' in key for key in loader.weight_map):
            raise ValueError('Biased experts are unsupported by this replay formula.')
    metadata=dict(format_version=1,observation_only=True,redecode=False,inventory_only=a.inventory_only,
        source_metadata=source,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        request_subset_scope='offline balanced composition at a fixed actual batch/step/expert/state',
        replay_dtype='bfloat16',metrics_dtype='float32 with float64 Gram eigenspectrum',
        torch_version=torch.__version__,shared_expert_included=False,router_weights_included=False)
    metadata['source_metadata_sha256']=hashlib.sha256((a.observation_dir/'metadata.json').read_bytes()).hexdigest()
    metadata['source_snapshot_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    if config is not None:metadata['checkpoint_config_sha256']=hashlib.sha256((a.model_path/'config.json').read_bytes()).hexdigest()
    (a.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    coverage=[];rank_rows=[];neighbor_rows=[]
    with torch.inference_mode():
        for (group,step),path in paths.items():
            payload=torch.load(path,map_location='cpu',weights_only=True)
            expert_count=validate_state(payload,batch,length)
            if config and (config['num_experts']!=expert_count or config['hidden_size']!=payload['hidden'].shape[-1]):
                raise ValueError('Checkpoint dimensions differ from captured model.')
            request_ids=json.loads((a.observation_dir/f'group{group}.json').read_text())['request_ids']
            if len(request_ids)!=batch or len(set(request_ids))!=batch:raise ValueError('Invalid request identities.')
            masked=payload['block_input_ids'].eq(source['mask_token_id'])
            for state in states:
                selected=masked if state=='mask' else ~masked if state=='decoded' else torch.ones_like(masked)
                allocations=defaultdict(list)
                for r,pos in selected.nonzero().tolist():
                    for expert in payload['routes'][r][pos]:allocations[expert].append((r,pos))
                assignments=sum(len(v) for v in allocations.values())
                common_count=common_load=neighbor_count=0
                if not a.inventory_only:
                    print(f'group={group} step={step} state={state}: '
                          f'checking/replaying {len(allocations)} actually visited experts',flush=True)
                replayed=0
                for expert,coords in sorted(allocations.items()):
                    by_request=defaultdict(list)
                    for i,(r,pos) in enumerate(coords):by_request[r].append(i)
                    capacities=capacity(by_request,a.tokens,sizes)
                    common=all(capacities[k]>=k for k in sizes)
                    _,eligible_neighbors=neighbor_candidates([r for r,pos in coords],a.neighbors,a.max_anchors,a.seed)
                    base=dict(group=group,step=step,state=state,expert=expert,routed_tokens=len(coords),
                        contributing_requests=len(by_request))
                    coverage.append(dict(kind='expert',**base,common_rank_eligible=common,
                        request_capacity_counts=json.dumps(capacities),eligible_neighbor_anchors=eligible_neighbors,
                        state_tokens=int(selected.sum()),state_assignments=assignments))
                    common_count+=int(common);common_load+=len(coords) if common else 0
                    neighbor_count+=int(eligible_neighbors>0)
                    if a.inventory_only or not (common or eligible_neighbors):continue
                    r_indices=[r for r,pos in coords];p_indices=[pos for r,pos in coords]
                    # Measure exactly the inputs used for BF16 replay, not an FP32 alternative.
                    inputs=payload['hidden'][r_indices,p_indices].to(torch.bfloat16)
                    weights=loader.load(expert);chunks=[]
                    for start in range(0,len(inputs),a.replay_chunk):
                        chunks.append(replay(inputs[start:start+a.replay_chunk].to(a.device),weights).float().cpu())
                    outputs=torch.cat(chunks);del weights,chunks
                    replayed+=1
                    if replayed%16==0:print(f'  replayed={replayed} current_expert={expert}',flush=True)
                    if not torch.isfinite(outputs).all():raise RuntimeError('Nonfinite expert replay output.')
                    for repeat in range(a.repeats):
                        seed=a.seed+1000003*step+10007*group+1009*expert+101*states.index(state)+repeat
                        if common:
                            samples,sets=balanced_samples(by_request,a.tokens,sizes,seed)
                            baseline=None
                            for k,indices in samples.items():
                                x,y=inputs[indices],outputs[indices]
                                row=dict(**base,repeat=repeat,request_count=k,tokens=a.tokens,tokens_per_request=a.tokens//k,
                                    request_ids=json.dumps([request_ids[r] for r in sets[k]]),
                                    token_ids=json.dumps([[request_ids[coords[i][0]],coords[i][1]] for i in indices]),
                                    input_rank=effective_rank(x),output_rank=effective_rank(y),
                                    input_centered_rank=effective_rank(x,True),output_centered_rank=effective_rank(y,True))
                                row['rank_difference']=row['output_rank']-row['input_rank']
                                row['centered_rank_difference']=row['output_centered_rank']-row['input_centered_rank']
                                if baseline is None:baseline=row.copy()
                                for metric in ['input_rank','output_rank','input_centered_rank','output_centered_rank']:
                                    row[metric+'_delta_vs_k1']=row[metric]-baseline[metric]
                                rank_rows.append(row)
                        if eligible_neighbors:
                            candidates,_=neighbor_candidates([r for r,pos in coords],a.neighbors,a.max_anchors,seed)
                            metrics=neighbor_statistics(inputs,outputs,candidates)
                            neighbor_rows.append(dict(**base,repeat=repeat,eligible_anchors=eligible_neighbors,
                                sampled_anchors=len(candidates),candidates_per_side=a.neighbors,**metrics))
                    del inputs,outputs
                coverage.append(dict(kind='state_summary',group=group,step=step,state=state,expert=None,
                    routed_tokens=assignments,contributing_requests=int(selected.any(-1).sum()),
                    common_rank_eligible=None,request_capacity_counts=None,eligible_neighbor_anchors=None,
                    state_tokens=int(selected.sum()),state_assignments=assignments,active_experts=len(allocations),
                    common_rank_experts=common_count,common_rank_assignment_share=common_load/assignments if assignments else None,
                    neighbor_eligible_experts=neighbor_count))
                print(f'group={group} step={step} state={state} experts={len(allocations)} '
                      f'balanced-common={common_count} neighbor-eligible={neighbor_count}',flush=True)
            del payload
            # Persist completed cell data so a killed process does not lose all statistics.
            write_csv(a.output_dir/'coverage.csv',coverage);write_csv(a.output_dir/'ranks.csv',rank_rows)
            write_csv(a.output_dir/'neighbors.csv',neighbor_rows)
    rank_metrics=['input_rank','output_rank','input_centered_rank','output_centered_rank',
                  'rank_difference','centered_rank_difference']+[
                  m+'_delta_vs_k1' for m in ['input_rank','output_rank','input_centered_rank','output_centered_rank']]
    rank_groups=group_means(rank_rows,['group','step','state','request_count'],rank_metrics)
    write_csv(a.output_dir/'rank_group_summary.csv',rank_groups)
    write_csv(a.output_dir/'rank_summary.csv',across_groups(rank_groups,['step','state','request_count'],rank_metrics))
    neighbor_metrics=[k for k in (neighbor_rows[0] if neighbor_rows else {}) if k.endswith('_distance') or k.endswith('_relative_l2')]
    valid=[row for row in neighbor_rows if row['valid_anchors']>0]
    neighbor_groups=group_means(valid,['group','step','state'],neighbor_metrics)
    write_csv(a.output_dir/'neighbor_group_summary.csv',neighbor_groups)
    write_csv(a.output_dir/'neighbor_summary.csv',across_groups(neighbor_groups,['step','state'],neighbor_metrics))
    report=dict(complete=True,inventory_only=a.inventory_only,physical_batch=batch,groups=groups,
        rank_rows=len(rank_rows),neighbor_rows=len(neighbor_rows),
        rank_expert_cells=sum(r['kind']=='expert' and r['common_rank_eligible'] for r in coverage),
        valid_neighbor_rows=sum(r['valid_anchors']>0 for r in neighbor_rows),
        caveats=['Request count is offline composition, not new actual batch runs.',
                 'Balanced rank cohorts select high-load experts; inspect assignment coverage.',
                 'Neighbor cohorts have separate eligibility; not the same population as rank cohorts.',
                 'Raw/centered rank differences do not prove safe merging/caching or speedup.',
                 'Sampling repeats and expert cells within a group are correlated.'])
    (a.output_dir/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
