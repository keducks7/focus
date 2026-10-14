"""Current-block, minimum-cardinality joint routing; no persistent route cache.

Only ORIGINAL gate-selected experts are eligible. Exact local feasible families,
but heuristic global selection. GPU operations and host synchronization are part
of the measured implementation, not an inferred speedup from offline counts.
"""
import itertools
import time
import types

import torch

from .llada2_shared_routing import sparse_moe_infer


METHODS = ('native', 'independent', 'request', 'batch')


def _families(ids, weights, epsilon):
    n, k = ids.shape
    if not 1 <= k <= 8:
        raise ValueError('Exact online enumeration supports native Top-1 through Top-8.')
    # ID ordering supplies deterministic lexicographic tie-breaking locally.
    order = ids.argsort(dim=-1, stable=True)
    es = ids.gather(1, order)
    p = weights.double().gather(1, order)
    if not torch.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError('Invalid native mixing weights.')
    p = p / p.sum(-1, keepdim=True)
    subsets = sorted(c for q in range(1, k+1) for c in itertools.combinations(range(k), q))
    bits = torch.tensor([[j in c for j in range(k)] for c in subsets], device=ids.device)
    mass = p @ bits.double().T
    target = 1-epsilon-1e-12
    prefix = p.sort(-1, descending=True).values.cumsum(-1)
    count = (prefix < target).sum(-1).clamp_max(k-1)+1
    viable = (mass >= target) & (bits.sum(-1)[None] == count[:, None])
    independent = bits[mass.masked_fill(~viable, -1).argmax(-1)]
    return es, order, bits, mass, viable, independent


def _support(es, keep, experts, groups, scopes):
    support = torch.zeros((scopes, experts), dtype=torch.bool, device=es.device)
    linear = groups[:, None]*experts+es
    support.flatten()[linear[keep]] = True
    return support


def _shrink(es, bits, mass, feasible, allowed, experts, groups, scopes):
    """Remove an expert iff every affected token keeps a feasible combination.

    Static owner ordering matches the offline deletion heuristic. No per-expert
    device-to-host feasibility checks: the removal decision stays on the device.
    """
    # Compact local bit masks avoid allocating [tokens, families, native K].
    powers = 2**torch.arange(es.shape[1], device=es.device)
    combo = (bits.long()*powers).sum(-1)
    local_allowed = (allowed[groups[:, None], es].long()*powers).sum(-1)
    viable = feasible & ((combo[None] & ~local_allowed[:, None]) == 0)
    if not viable.any(-1).all():
        raise RuntimeError('Infeasible joint planning seed.')
    # Each native route can own many combinations, but only one expert slot.
    usable_slot = (viable.to(torch.float32) @ bits.float()) > 0
    row_ids, slot_ids = usable_slot.nonzero(as_tuple=True)
    e_ids = es[row_ids, slot_ids]
    owner_groups = groups[row_ids]
    owners = torch.bincount(owner_groups*experts+e_ids, minlength=scopes*experts)
    # One transfer per shrink, rather than one transfer per expert. Expert-ID
    # ties are deterministic. The native route IDs are NOT recomputed/sorted by
    # raw router logits here (official bias/group restrictions stay untouched).
    triples = torch.stack((owner_groups, e_ids, row_ids, slot_ids), -1).cpu().tolist()
    adjacency = {}
    scope_experts = {}
    for group, e, row, slot in triples:
        adjacency.setdefault((group,e), []).append((row, slot))
        scope_experts.setdefault(group,set()).add(e)
    counts = owners.cpu().tolist()
    schedules = {g: sorted(es, key=lambda e: (counts[g*experts+e],e))
                 for g,es in scope_experts.items()}
    # Process one deletion per request IN PARALLEL. Do not execute B separate
    # planners/kernel-launch loops. Each row belongs to exactly one scope.
    for index in range(max(map(len,schedules.values()), default=0)):
        affected = [pair for g, schedule in schedules.items() if index < len(schedule)
                    for pair in adjacency[(g,schedule[index])]]
        rows, slots = zip(*affected)
        rows = torch.tensor(rows, device=es.device)
        slots = torch.tensor(slots, device=es.device)
        contains = (combo[None] & powers[slots, None]) != 0
        current = viable[rows]
        violations = torch.zeros(scopes, device=es.device, dtype=torch.long)
        violations.scatter_add_(0, groups[rows], (~(current & ~contains).any(-1)).long())
        removable = violations[groups[rows]] == 0
        viable[rows] = current & ~(contains & removable[:, None])
    return bits[mass.masked_fill(~viable, -1).argmax(-1)]


def _best(es, p, solutions, experts, groups, scopes):
    # Support first, retained mass second. Equal keys retain the earlier seed;
    # unlike the offline planner's arbitrary global bit-mask final tie-break.
    result = solutions[0]
    def key(candidate):
        sizes = _support(es,candidate,experts,groups,scopes).sum(-1)
        masses = torch.zeros(scopes, device=es.device, dtype=p.dtype)
        masses.scatter_add_(0, groups, (p*candidate).sum(-1))
        return sizes, masses

    size, retained = key(result)
    for candidate in solutions[1:]:
        new_size,new_retained = key(candidate)
        better = (new_size < size) | ((new_size == size) & (new_retained > retained))
        result = torch.where(better[groups,None],candidate,result)
        size = torch.where(better,new_size,size)
        retained = torch.where(better,new_retained,retained)
    return result


def _plan(es, bits, mass, feasible, seeds, experts, groups, scopes):
    p = torch.zeros_like(es, dtype=mass.dtype)
    # Recover normalized probabilities from singleton combination masses.
    for j in range(es.shape[1]):
        singleton = (bits.sum(-1) == 1) & bits[:, j]
        p[:, j] = mass[:, singleton].flatten()
    broad = _support(es, (feasible.float() @ bits.float()) > 0, experts,groups,scopes)
    candidates = list(seeds)
    for seed in seeds:
        allowed = _support(es,seed,experts,groups,scopes)
        candidates.append(_shrink(es,bits,mass,feasible,allowed,experts,groups,scopes))
    candidates.append(_shrink(es,bits,mass,feasible,broad,experts,groups,scopes))
    return _best(es,p,candidates,experts,groups,scopes)


@torch.no_grad()
def select_joint_keep(ids, weights, requests, epsilon, num_experts, method='batch'):
    if method not in METHODS or not 0 <= epsilon < 1:
        raise ValueError('Unknown method or invalid epsilon.')
    if ids.ndim != 2 or weights.shape != ids.shape or requests.numel() != ids.shape[0]:
        raise ValueError('Incompatible native route/request shapes.')
    requests = requests.to(ids.device).flatten()
    if method == 'native' or epsilon == 0:
        return torch.ones_like(ids, dtype=torch.bool)
    if ids.numel() == 0:
        return torch.ones_like(ids, dtype=torch.bool)
    if (ids < 0).any() or (ids >= num_experts).any():
        raise ValueError('Expert ID out of range.')
    sorted_ids = ids.sort(-1).values
    if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise ValueError('Duplicate native expert ID.')
    es, order, bits, mass, feasible, independent = _families(ids, weights, epsilon)
    result = independent
    if method in ('request', 'batch'):
        labels,groups = requests.unique(sorted=True,return_inverse=True)
        per_request = _plan(es,bits,mass,feasible,[independent],num_experts,groups,labels.numel())
        result = per_request
        if method == 'batch':
            batch_group = torch.zeros(ids.shape[0], device=ids.device,dtype=torch.long)
            result = _plan(es,bits,mass,feasible,[per_request,independent],num_experts,batch_group,1)
    # Return flags in native gate order; keep the native expert reduction order.
    return torch.zeros_like(result).scatter(1, order, result)


def validate_two_gpu_placement(model):
    """Require complete decoder layers on GPU, and actual use of both devices."""
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Exactly two visible GPUs required; use CUDA_VISIBLE_DEVICES=0,1.')
    placements = getattr(model, 'hf_device_map', {})
    if any(str(v) in ('cpu', 'disk', 'meta') for v in placements.values()):
        raise RuntimeError('CPU/disk offload is not allowed for GPU timing.')
    used = set()
    for i, layer in enumerate(model.model.layers):
        devices = {p.device for p in layer.parameters()}
        if len(devices) != 1 or next(iter(devices)).type != 'cuda':
            raise RuntimeError(f'Decoder layer {i} is split/offloaded; require whole-layer placement.')
        used.update(d.index for d in devices)
    if used != {0, 1}:
        raise RuntimeError('Decoder layers must actually use BOTH visible GPUs.')


class BlockJointRoutingController:
    """Wrap only native routed FFNs. Prefill/commit remain original execution."""
    def __init__(self, model, epsilon=.1, method='batch', detailed_timing=False):
        if method not in METHODS or not 0 <= epsilon < 1:
            raise ValueError('Unknown method or invalid epsilon.')
        self.epsilon, self.method = epsilon, method
        self.detailed_timing = detailed_timing
        self.device_map = {str(k): str(v) for k,v in getattr(model,'hf_device_map',{}).items()}
        self.compress = None  # legacy generator compatibility; NOT routing policy
        self.block_requests = None
        self.block_valid = None
        self.originals = []
        self.forward_originals = []
        self.reset_stats()
        layers = [layer.mlp for layer in model.model.layers if hasattr(layer.mlp, 'experts')]
        if not layers or any(not callable(getattr(m, 'moe_infer', None)) for m in layers):
            raise TypeError('Requires native LLaDA2 moe_infer(x, ids, weights).')
        for moe in layers:
            native = moe.moe_infer
            self.originals.append((moe, native))

            def wrapped(module, x, ids, weights, native=native):
                if self.block_requests is None:
                    self.other_original.append(ids.numel())
                    return native(x, ids, weights)
                if self.block_requests.numel() != ids.shape[0]:
                    raise RuntimeError('Flattened block/request mapping does not match gate output.')
                self.block_original.append(ids.numel())
                if self.detailed_timing and x.is_cuda:
                    torch.cuda.synchronize(x.device)
                started = time.perf_counter()
                if self.method == 'native' or self.epsilon == 0:
                    keep = torch.ones_like(ids, dtype=torch.bool)
                else:
                    # Finished requests/partial-prompt positions are not method
                    # targets, and are NOT used as free background support.
                    valid = self.block_valid.to(ids.device).flatten()
                    keep = torch.ones_like(ids, dtype=torch.bool)
                    keep[valid] = select_joint_keep(ids[valid], weights[valid],
                        self.block_requests.to(ids.device).flatten()[valid], self.epsilon,
                        len(module.experts), self.method)
                if self.detailed_timing and x.is_cuda:
                    torch.cuda.synchronize(x.device)
                self.planning_seconds += time.perf_counter()-started
                if self.detailed_timing and x.is_cuda:
                    torch.cuda.synchronize(x.device)
                executed_start = time.perf_counter()
                if self.method == 'native' or self.epsilon == 0:
                    out = native(x, ids, weights)
                else:
                    self.compressed_layer_calls += 1
                    out = sparse_moe_infer(module, x, ids, weights, keep, renormalize=False)
                if self.detailed_timing and x.is_cuda:
                    torch.cuda.synchronize(x.device)
                self.routed_seconds += time.perf_counter()-executed_start
                # Deferred small counters: no trajectory or hidden-state capture,
                # and no per-layer .item() synchronization just for accounting.
                self.block_executed.append(keep.sum().detach())
                self.active_before.append(ids.unique().numel())
                self.active_after.append(ids[keep].unique().numel())
                valid = self.block_valid.to(ids.device).flatten()
                self.target_original.append(self.block_target_rows*ids.shape[1])
                self.target_executed.append(keep[valid].sum().detach())
                self.target_active_before.append(ids[valid].unique().numel())
                self.target_active_after.append(ids[valid][keep[valid]].unique().numel())
                return out

            moe.moe_infer = types.MethodType(wrapped, moe)
            if detailed_timing:
                native_forward = moe.forward
                self.forward_originals.append((moe,native_forward))

                def timed_forward(module, hidden_states, *args, native_forward=native_forward, **kwargs):
                    if self.block_requests is None:
                        return native_forward(hidden_states,*args,**kwargs)
                    if hidden_states.is_cuda:
                        torch.cuda.synchronize(hidden_states.device)
                    started = time.perf_counter()
                    result = native_forward(hidden_states,*args,**kwargs)
                    if hidden_states.is_cuda:
                        torch.cuda.synchronize(hidden_states.device)
                    self.moe_layer_seconds += time.perf_counter()-started
                    return result

                moe.forward = types.MethodType(timed_forward,moe)

    def begin_block_forward(self, requests, valid):
        self.block_requests, self.block_valid = requests, valid
        self.block_target_rows = int(valid.sum())

    def end_block_forward(self):
        self.block_requests = self.block_valid = None

    def reset_stats(self):
        self.other_original, self.block_original, self.block_executed = [], [], []
        self.active_before, self.active_after = [], []
        self.target_original, self.target_executed = [], []
        self.target_active_before, self.target_active_after = [], []
        self.planning_seconds = self.routed_seconds = 0.
        self.moe_layer_seconds = 0.
        self.compressed_layer_calls = 0

    @property
    def assignments_original(self):
        return sum(self.other_original)+sum(self.block_original)

    @property
    def assignments_executed(self):
        return sum(self.other_original)+sum(int(v) for v in self.block_executed)

    def extra_stats(self):
        calls = len(self.block_original)
        return dict(block_layer_calls=calls,
            execution_backend='hf_eager_balanced_layers', hf_device_map=self.device_map,
            routing_scope='live_current_block_generation_mask_and_decoded',
            block_assignments_original=sum(self.block_original),
            block_assignments_executed=sum(int(v) for v in self.block_executed),
            mean_block_active_before=sum(self.active_before)/calls if calls else None,
            mean_block_active_after=sum(self.active_after)/calls if calls else None,
            target_assignments_original=sum(self.target_original),
            target_assignments_executed=sum(int(v) for v in self.target_executed),
            mean_target_active_before=sum(self.target_active_before)/calls if calls else None,
            mean_target_active_after=sum(self.target_active_after)/calls if calls else None,
            planning_seconds=self.planning_seconds, routed_expert_seconds=self.routed_seconds,
            moe_layer_seconds=self.moe_layer_seconds if self.detailed_timing else None,
            component_timing_synchronized=self.detailed_timing,
            component_timing_note='Diagnostic synchronized timing only' if self.detailed_timing else
                'Host/enqueue time only; not GPU kernel duration. Use diagnostic timing for components.')

    def close(self):
        for moe, native in self.originals:
            moe.moe_infer = native
        self.originals.clear()
        for moe,native_forward in self.forward_originals:
            moe.forward = native_forward
        self.forward_originals.clear()
        self.end_block_forward()
