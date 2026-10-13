#!/usr/bin/env python3
"""Unmodified HF decoding, then OFF-PATH all-expert functional observations.

All requested native batches finish BEFORE any side expert evaluation. No output,
route or acceptance is patched. Layer inputs (not final hidden states) are saved.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from intervene_expert_lifecycle import phase_at, predict, visits_by_request


class ObservationHooks:
    def __init__(self, moe, batch, length):
        self.moe, self.batch, self.length = moe, batch, length
        self.calls = 0

    def __enter__(self):
        self.handles = [self.moe.register_forward_pre_hook(self.pre),
                        self.moe.gate.register_forward_hook(self.gate)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()

    def pre(self, _module, inputs):
        h = inputs[0]
        if h.ndim != 3 or h.shape[0] != self.batch:
            raise RuntimeError('Unexpected native layer input shape.')
        self.sequence = h.shape[1]
        self.hidden = h[:, -self.length:].detach().to('cpu', copy=True)
        # Returning None is deliberate: never replace module inputs.

    def gate(self, _module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError('Expected native (IDs, weights, logits) gate.')
        ids, weights, logits = output
        if ids.shape != weights.shape or ids.shape[0] != self.batch*self.sequence:
            raise RuntimeError('Ambiguous native row mapping.')
        self.calls += 1
        self.ids = ids.reshape(self.batch, self.sequence, -1)[:, -self.length:].cpu()
        self.weights = weights.reshape(self.batch, self.sequence, -1)[:, -self.length:].float().cpu()
        self.logits = logits.reshape(self.batch, self.sequence, -1)[:, -self.length:].float().cpu()
        # Returning None is deliberate: never replace gate outputs.


def sample_tokens(acceptance, count, seed):
    """No conditioning on source persistence or future expert similarity."""
    eligible = [(r,p,s) for r,row in enumerate(acceptance) for p,s in enumerate(row) if s is not None]
    random.Random(seed).shuffle(eligible)
    chosen, requests = [], set()
    for item in eligible:
        if item[0] not in requests:
            chosen.append(item)
            requests.add(item[0])
        if len(chosen) == count:
            break
    if len(chosen) < count:
        chosen.extend(item for item in eligible if item not in chosen)
    return [dict(case_id=i, batch_index=r, block_position=p, acceptance_step=s)
            for i,(r,p,s) in enumerate(chosen[:count])]


def functional_distances(outputs):
    """Rows=source, columns=candidate; relative L2 is asymmetric; no self substitute."""
    import torch
    x = outputs.float()
    norms = x.norm(dim=-1)
    squared = (norms[:,None].square()+norms[None,:].square()-2*x@x.T).clamp_min(0)
    relative = squared.sqrt()/norms[:,None].clamp_min(1e-12)
    cosine = (x@x.T)/(norms[:,None]*norms[None,:]).clamp_min(1e-12)
    relative[norms == 0] = float('nan')
    cosine[(norms[:,None] == 0)|(norms[None,:] == 0)] = float('nan')
    relative.fill_diagonal_(float('inf'))
    return relative, cosine, norms


def nearest(relative, cosine, source, candidates):
    import math
    eligible = sorted(e for e in candidates if e != source and math.isfinite(float(relative[source,e])))
    if not eligible:
        return dict(candidate_count=0, expert=None, relative_l2=None, cosine=None,distance_gap=None)
    ranked=sorted(eligible,key=lambda e:(float(relative[source,e]),e))
    dest=ranked[0]
    c = float(cosine[source,dest])
    return dict(candidate_count=len(eligible), expert=dest, relative_l2=float(relative[source,dest]),
                cosine=c if math.isfinite(c) else None,
                distance_gap=float(relative[source,ranked[1]]-relative[source,dest]) if len(ranked)>1 else None)


def legal_experts(logits, bias, n_group, topk_group):
    groups = (logits.float().sigmoid()+bias.float()).reshape(n_group,-1)
    selected = groups.topk(2, dim=-1).values.sum(-1).topk(topk_group).indices.tolist()
    width = groups.shape[1]
    return {e for group in selected for e in range(group*width,(group+1)*width)}


def request_supports(routes, live):
    return {r:{e for row in positions for e in row} for r,positions in enumerate(routes) if live[r]}


def nested_request_sets(live, anchor, sizes, seed):
    """Each nested subset contains the anchor, and uses only currently live requests."""
    others = sorted(r for r in live if r != anchor)
    random.Random(seed).shuffle(others)
    return {k:([anchor]+others[:k-1] if k <= len(others)+1 else None) for k in sizes}


def probe_group(moe, output, group, pre, post, count, sizes, seed, chunk):
    import torch
    index = json.loads((output/f'group{group}.json').read_text())
    cases = sample_tokens(index['acceptance_steps'], count, seed+group)
    events, hidden = [], []
    missing = []
    for case in cases:
        r,p,anchor = case['batch_index'],case['block_position'],case['acceptance_step']
        for offset in range(-pre,post+1):
            step = anchor+offset
            common = dict(group=group, **case, request_id=index['request_ids'][r],
                          step=step, relative_step=offset, phase=phase_at(step,anchor))
            if step < 0 or step >= index['steps']:
                missing.append(dict(record_type='missing_observation',reason='outside_trajectory',**common))
                continue
            state = torch.load(output/f'inputs_group{group}_step{step}.pt',map_location='cpu',weights_only=True)
            if not state['live_requests'][r]:
                missing.append(dict(record_type='missing_observation',reason='request_finished',**common))
                continue
            events.append(dict(**common, input_token_id=int(state['block_input_ids'][r,p]),
                source_ids=state['routes'][r][p], source_weights=state['weights'][r][p],
                legal=sorted(legal_experts(state['logits'][r,p],moe.gate.expert_bias.cpu(),
                                          moe.gate.n_group,moe.gate.topk_group)),
                supports=request_supports(state['routes'],state['live_requests'])))
            if not set(events[-1]['source_ids']) <= set(events[-1]['legal']):
                raise RuntimeError('Reconstructed group eligibility excludes native selections; inspect numerical ties.')
            hidden.append(state['hidden'][r,p].clone())
            del state
    missing_path = output/f'functional_missing_group{group}.json'
    missing_path.write_text(json.dumps(dict(cases=cases,missing=missing),indent=2)+'\n')
    if not events:
        return dict(group=group,cases=len(cases),events=0,missing=len(missing),common_matrix_cases=0)
    inputs = torch.stack(hidden)
    evaluated = []
    for expert_id,expert in enumerate(moe.experts):
        device = next(expert.parameters()).device
        rows = []
        for start in range(0,len(inputs),chunk):
            rows.append(expert(inputs[start:start+chunk].to(device)).detach().float().cpu())
        combined = torch.cat(rows)
        if not torch.isfinite(combined).all():
            raise RuntimeError(f'Nonfinite off-path expert output: expert {expert_id}.')
        evaluated.append(combined)
        if (expert_id+1)%32 == 0:
            print(f'off-path group={group} experts={expert_id+1}/{len(moe.experts)}',flush=True)
    outputs = torch.stack(evaluated)  # [experts, events, hidden], CPU only
    del evaluated
    serial_events = [dict(event,supports={r:sorted(s) for r,s in event['supports'].items()}) for event in events]
    torch.save(dict(events=serial_events,inputs=inputs,expert_outputs=outputs,includes_shared_expert=False),
               output/f'functional_outputs_group{group}.pt')
    fixed_sources = {c['case_id']:sorted({e for event in events if event['case_id']==c['case_id']
                                         for e in event['source_ids']}) for c in cases}
    with (output/f'neighbors_group{group}.jsonl').open('x',buffering=1) as stream:
        for j,event in enumerate(events):
            relative,cosine,norms = functional_distances(outputs[:,j])
            r = event['batch_index']
            supports = event['supports']
            own = supports[r]
            other = set().union(*(support for request,support in supports.items() if request != r))
            batch = own|other
            legal = set(event['legal'])
            subsets = nested_request_sets(supports,r,sizes,seed+group*1000+event['case_id'])
            for source in fixed_sources[event['case_id']]:
                selected = source in event['source_ids']
                weights = dict(zip(event['source_ids'],event['source_weights']))
                row = {k:v for k,v in event.items() if k not in ('supports','legal','source_ids','source_weights')}
                row.update(source_expert=source,source_selected=selected,
                    source_weight=weights.get(source),source_output_norm=float(norms[source]),
                    all_legal=nearest(relative,cosine,source,legal),
                    own_request=nearest(relative,cosine,source,own&legal),
                    other_requests=nearest(relative,cosine,source,other&legal),
                    batch_active=nearest(relative,cosine,source,batch&legal),
                    live_requests=len(supports),nested_subsets=[])
                for k,requests in subsets.items():
                    support = set().union(*(supports[request] for request in requests)) if requests else set()
                    row['nested_subsets'].append(dict(request_count=k,available=requests is not None,
                        **nearest(relative,cosine,source,support&legal)))
                stream.write(json.dumps(row,allow_nan=False)+'\n')
    # SAME token cohort at the four canonical offsets. Experts may be unselected:
    # this is function geometry, not current contribution or valid replacement.
    canonical = [-1,0,1,2]
    lookup = {(e['case_id'],e['relative_step']):j for j,e in enumerate(events)}
    common = [c['case_id'] for c in cases if all((c['case_id'],off) in lookup for off in canonical)]
    matrices = {}
    for off in canonical:
        if common:
            vectors = outputs[:,[lookup[c,off] for c in common]].flatten(1)
            rel,cos,norms = functional_distances(vectors)
            matrices[off] = dict(relative_l2=rel,cosine=cos,output_norms=norms)
    torch.save(dict(common_case_ids=common,canonical_offsets=canonical,matrices=matrices,
        scope='all_expert_functions_on_same_token_cohort_not_routing_eligibility'),output/f'matrices_group{group}.pt')
    return dict(group=group,cases=len(cases),events=len(events),missing=len(missing),common_matrix_cases=len(common))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_path')
    p.add_argument('--prompt-snapshot',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--num-groups',type=int,default=1)
    p.add_argument('--group-start',type=int,default=0)
    p.add_argument('--layer',type=int,default=10)
    p.add_argument('--block-length',type=int,default=32)
    p.add_argument('--denoising-steps',type=int,default=32)
    p.add_argument('--confidence-threshold',type=float,default=.95)
    p.add_argument('--tokens-per-group',type=int,default=8)
    p.add_argument('--pre-steps',type=int,default=2)
    p.add_argument('--post-steps',type=int,default=3)
    p.add_argument('--subset-sizes',nargs='+',type=int,default=[1,2,4,8,16,32])
    p.add_argument('--logit-chunk-size',type=int,default=8)
    p.add_argument('--expert-chunk-size',type=int,default=16)
    p.add_argument('--max-memory-per-gpu',default='36GiB')
    p.add_argument('--capture-only',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    a = p.parse_args()
    if min(a.batch_size,a.num_groups,a.block_length,a.denoising_steps,a.tokens_per_group,
           a.logit_chunk_size,a.expert_chunk_size,*a.subset_sizes)<1 or min(a.group_start,a.layer,a.pre_steps)<0 or a.post_steps<2:
        p.error('Invalid sizes/indices; post-steps must be at least 2.')
    if not 0<=a.confidence_threshold<=1:
        p.error('Invalid confidence threshold.')
    payload = a.prompt_snapshot.read_bytes()
    snapshot = json.loads(payload)
    prompts = snapshot['prompt_token_ids']
    if len(prompts)<(a.group_start+a.num_groups)*a.batch_size or any(not row for row in prompts):
        p.error('Insufficient/nonempty prompt IDs required.')
    requests = snapshot.get('request_ids',list(range(len(prompts))))
    if len(requests)!=len(prompts):
        p.error('Request identity count differs from prompts.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise FileExistsError('Use a NEW/EMPTY output directory.')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from profile_llada2_hf_moe_saturation import _build_batch_inputs,_transfer_schedule
    from profile_llada2_expert_trajectory import choose_candidate_mask
    if torch.cuda.device_count()!=2:
        raise RuntimeError('Expose exactly two GPUs with CUDA_VISIBLE_DEVICES=0,1.')
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model_path,trust_remote_code=True)
    mask = tokenizer.mask_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if mask is None or pad is None:
        raise ValueError('Tokenizer must define MASK and PAD/EOS.')
    print('Loading native HF model; no route/output intervention.',flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_path,trust_remote_code=True,
        torch_dtype=torch.bfloat16,device_map='balanced',max_memory={0:a.max_memory_per_gpu,1:a.max_memory_per_gpu},
        low_cpu_mem_usage=True,attn_implementation='eager').eval()
    mapping = getattr(model,'hf_device_map',{})
    if not mapping or any(str(v) in ('cpu','disk','meta') for v in mapping.values()):
        raise RuntimeError('CPU/disk offload is unsupported.')
    core = model.model
    if a.layer>=len(core.layers) or not hasattr(core.layers[a.layer].mlp,'experts'):
        raise ValueError('Layer is not a supported routed MoE.')
    moe = core.layers[a.layer].mlp
    device = core.word_embeddings.weight.device
    schedule = _transfer_schedule(a.block_length,a.denoising_steps)
    sizes = sorted({k for k in a.subset_sizes if k<=a.batch_size})
    metadata = dict(format_version=1,mode='vanilla',backend='huggingface',observation_only=True,
        mask_token_id=mask,temperature=0,side_probes_after_all_native_generation=True,
        prompt_sha256=hashlib.sha256(payload).hexdigest(),torch_version=torch.__version__,
        selected_subset_sizes=sizes,subset_scope='offline_nested_live_request_subsets_not_physical_batch_runs',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    (a.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    groups = list(range(a.group_start,a.group_start+a.num_groups))
    audits = []
    with torch.inference_mode(),(a.output_dir/'routes.jsonl').open('x',buffering=1) as stream:
        for group in groups:
            start = group*a.batch_size
            ids,attention,positions = _build_batch_inputs(prompts[start:start+a.batch_size],a.block_length,pad,mask,device,torch.bfloat16)
            acceptance = [[None]*a.block_length for _ in range(a.batch_size)]
            def forward():
                out = core(input_ids=ids,attention_mask=attention,position_ids=positions,
                           use_cache=False,output_router_logits=False,return_dict=True)
                return out.last_hidden_state[:,-a.block_length:].detach().cpu()
            unobserved = forward()
            steps = 0
            for step in range(len(schedule)):
                before = ids[:,-a.block_length:].cpu().clone()
                active = before.eq(mask)
                if not active.any():
                    break
                with ObservationHooks(moe,a.batch_size,a.block_length) as hook:
                    final_hidden = forward()
                if hook.calls!=1:
                    raise RuntimeError('Expected one native gate call.')
                if step==0:
                    if not torch.equal(unobserved,final_hidden):
                        raise RuntimeError('Observation hooks changed first-forward result; stop.')
                    audits.append(dict(group=group,first_forward_observer_matches_unobserved=True))
                    del unobserved
                prediction = predict(model,final_hidden,a.logit_chunk_size)
                accepted = choose_candidate_mask(active,prediction['confidence'],schedule[step],a.confidence_threshold)&prediction['candidates'].ne(mask)
                for r,pos in accepted.nonzero().tolist():
                    acceptance[r][pos]=step
                live = active.any(-1).tolist()
                routes,weights = hook.ids.tolist(),hook.weights.tolist()
                torch.save(dict(hidden=hook.hidden,logits=hook.logits,routes=routes,weights=weights,
                    live_requests=live,block_input_ids=before),a.output_dir/f'inputs_group{group}_step{step}.pt')
                tokens = [dict(request_id=requests[start+r],block_position=pos,
                    input_token_id=int(before[r,pos]),masked_before=bool(active[r,pos]),
                    request_finished_before=not live[r],accepted_this_step=bool(accepted[r,pos]),
                    expert_ids=routes[r][pos],router_weights=weights[r][pos])
                    for r in range(a.batch_size) for pos in range(a.block_length)]
                stream.write(json.dumps(dict(record_type='route_step',group=group,block_id=0,step=step,tokens=tokens))+'\n')
                ids[:,-a.block_length:][accepted.to(device)] = prediction['candidates'].to(device)[accepted.to(device)]
                steps += 1
                print(f'normal group={group} step={step} masks={int(active.sum())}->{int(ids[:,-a.block_length:].eq(mask).sum())}',flush=True)
                del hook,final_hidden,prediction
            index = dict(group=group,steps=steps,acceptance_steps=acceptance,request_ids=requests[start:start+a.batch_size],
                         remaining_masks=int(ids[:,-a.block_length:].eq(mask).sum()))
            (a.output_dir/f'group{group}.json').write_text(json.dumps(index)+'\n')
        (a.output_dir/'generation_complete.json').write_text(json.dumps(dict(groups=groups,audits=audits))+'\n')
        coverage = []
        if not a.capture_only:
            for group in groups:
                coverage.append(probe_group(moe,a.output_dir,group,a.pre_steps,a.post_steps,a.tokens_per_group,
                                            sizes,a.seed,a.expert_chunk_size))
        (a.output_dir/'complete.json').write_text(json.dumps(dict(groups=groups,capture_only=a.capture_only,
                                                                  functional_coverage=coverage),indent=2)+'\n')
    print(f'Complete (observation only): {a.output_dir}',flush=True)


if __name__=='__main__':
    main()
