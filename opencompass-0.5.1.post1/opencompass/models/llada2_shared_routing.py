"""Inference-only shared route selection and genuinely sparse expert execution.

No parameters are modified. The native MLP still executes its own gate/shared
expert/residual path; only its routed ``moe_infer`` method is temporarily wrapped.
This reference implementation includes selection overhead, not a tuned kernel.
"""
import types

import torch


def select_keep(ids, weights, compress, epsilon, num_experts, method='joint'):
    if method not in ('joint', 'independent', 'joint_no_fixed'):
        raise ValueError(f'Unknown routing method: {method}')
    if not 0 <= epsilon < 1:
        raise ValueError('epsilon must be in [0,1).')
    if ids.ndim != 2 or weights.shape != ids.shape or compress.numel() != ids.shape[0]:
        raise ValueError('Incompatible native route shapes.')
    keep = torch.ones_like(ids, dtype=torch.bool)
    if epsilon == 0 or not compress.any():
        return keep
    compress = compress.to(ids.device).flatten()
    selected_ids = ids[compress]
    probs = weights[compress].float()
    if not torch.isfinite(probs).all() or (probs < 0).any() or (probs.sum(-1) <= 0).any():
        raise ValueError('Nonfinite, negative or zero-mass router weights.')
    probs = probs / probs.sum(-1, keepdim=True)
    tolerance = min(1e-6, (1-epsilon)*1e-3)
    if method == 'independent':
        eligible = torch.ones_like(selected_ids, dtype=torch.bool)
    else:
        support = torch.zeros(num_experts, dtype=torch.bool, device=ids.device)
        if method == 'joint':
            support[ids[~compress].flatten()] = True
        for _ in range(num_experts + 1):
            eligible = support[selected_ids]
            deficits = ((1-epsilon) - (probs*eligible).sum(-1)).clamp_min(0)
            if (deficits <= tolerance).all():
                break
            gains = torch.zeros(num_experts, device=ids.device, dtype=torch.float32)
            contributions = torch.minimum(probs, deficits[:, None]) * (~eligible)
            gains.scatter_add_(0, selected_ids.flatten(), contributions.flatten())
            gains[support] = -1
            chosen = gains.argmax()
            if gains[chosen] <= 0:
                raise RuntimeError('Unable to satisfy routing coverage.')
            support[chosen] = True
        else:
            raise RuntimeError('Coverage iteration did not converge.')
    # ID-first stable sorting makes equal-weight ties deterministic.
    by_id = selected_ids.argsort(dim=-1, stable=True)
    scores = probs.masked_fill(~eligible, -1).gather(1, by_id)
    order = by_id.gather(1, scores.argsort(dim=-1, descending=True, stable=True))
    ordered_p = probs.gather(1, order)
    ordered_eligible = eligible.gather(1, order)
    previous = (ordered_p * ordered_eligible).cumsum(-1) - ordered_p * ordered_eligible
    ordered_keep = ordered_eligible & (previous < (1-epsilon)-tolerance)
    flags = torch.zeros_like(eligible).scatter(1, order, ordered_keep)
    if ((flags*probs).sum(-1) < (1-epsilon)-2*tolerance).any() or not flags.any(-1).all():
        raise RuntimeError('Selected routes violate coverage.')
    keep[compress] = flags
    return keep


def sparse_moe_infer(moe, x, ids, weights, keep, renormalize=True):
    """Evaluate only retained token-expert pairs, retaining native sum order."""
    kept_weights = weights * keep
    if renormalize:
        kept_weights = kept_weights * (weights.sum(-1, keepdim=True) /
                                       kept_weights.sum(-1, keepdim=True).clamp_min(1e-20))
        # Preserve fixed rows bit-for-bit, avoiding roundoff from rescaling by 1.
        kept_weights = torch.where(keep.all(-1, keepdim=True), weights, kept_weights)
    n, k = ids.shape
    branch = x.new_zeros((n*k, x.shape[-1]))
    flat_slots = keep.flatten().nonzero(as_tuple=False).flatten()
    expert_ids = ids.flatten()[flat_slots]
    order = expert_ids.argsort()
    slots = flat_slots[order]
    sorted_tokens = x[slots // k]
    counts = torch.bincount(expert_ids, minlength=len(moe.experts)).cpu().tolist()
    offset = 0
    for expert_id, count in enumerate(counts):
        if count:
            outputs = moe.experts[expert_id](sorted_tokens[offset:offset+count]).to(x.device)
            branch[slots[offset:offset+count]] = outputs.to(branch.dtype)
            offset += count
    return (branch.view(n, k, -1).to(weights.dtype) * kept_weights[..., None]).sum(1).to(x.dtype)


class SharedRoutingController:
    def __init__(self, model, epsilon=0., method='joint', renormalize=True):
        if not 0 <= epsilon < 1:
            raise ValueError('epsilon must be in [0,1).')
        self.epsilon, self.method, self.renormalize = epsilon, method, renormalize
        self.compress = None
        self.originals = []
        self.reset_stats()
        for layer in model.model.layers:
            moe = layer.mlp
            if not hasattr(moe, 'experts'):
                continue
            if not callable(getattr(moe, 'moe_infer', None)):
                raise TypeError('Requires native LLaDA2 MoE with moe_infer(x, ids, weights).')
        for layer in model.model.layers:
            moe = layer.mlp
            if not hasattr(moe, 'experts'):
                continue
            native = moe.moe_infer
            self.originals.append((moe, native))

            def wrapped(module, x, ids, weights, native=native):
                self.assignments_original += ids.numel()
                if self.compress is None or self.epsilon == 0:
                    self.assignments_executed += ids.numel()
                    return native(x, ids, weights)
                keep = select_keep(ids, weights, self.compress.to(ids.device), self.epsilon,
                                   len(module.experts), self.method)
                count = int(keep.sum().item())
                self.assignments_executed += count
                if count == ids.numel():
                    return native(x, ids, weights)
                self.compressed_layer_calls += 1
                return sparse_moe_infer(module, x, ids, weights, keep, self.renormalize)

            moe.moe_infer = types.MethodType(wrapped, moe)
        if not self.originals:
            raise ValueError('No supported routed MoE layers found.')

    def reset_stats(self):
        self.assignments_original = 0
        self.assignments_executed = 0
        self.compressed_layer_calls = 0

    def close(self):
        for moe, native in self.originals:
            moe.moe_infer = native
        self.originals.clear()
        self.compress = None
