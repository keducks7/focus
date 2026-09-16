"""Packed-token HF LLaDA2 execution with batched padded attention and real packed MoE.

FOCUS selection follows lmdeploy/pytorch/kernels/cuda/focus.py. This is a
PyTorch semantic port, not the LMDeploy engine or a throughput implementation.
"""
import math
import sys


def importance(q, k, groups, scale):
    import torch.nn.functional as F
    # q/k: selected MASK tokens, heads, head_dim. Kernel accumulates in FP32.
    keys = k.repeat_interleave(groups, dim=1)
    scores = (q.float().transpose(0, 1) @ keys.float().permute(1, 2, 0)) * scale
    pooled = F.max_pool1d(scores, kernel_size=3, stride=1, padding=1)
    return pooled.softmax(-1).sum(dim=(0, 1)).to(q.dtype)


def select_positions(delta, positions, average, alpha, progress):
    """Dynamic mean+population-std, stable fallback, predecessor and progress protection."""
    import torch
    count = len(positions)
    target = min(count, max(1, math.ceil(max(float(average), 1.0) * alpha)))
    if count <= target:
        return torch.ones_like(positions, dtype=torch.bool)
    scores = delta.float()
    candidate = scores >= scores.mean() + scores.std(correction=0)
    if int(candidate.sum()) >= target:
        keep = candidate
    else:
        keep = torch.zeros_like(candidate)
        keep[torch.argsort(scores, descending=True, stable=True)[:target]] = True
    original = keep.clone()
    keep[:-1] |= (positions[1:] == positions[:-1] + 1) & original[1:] & ~original[:-1]
    if not bool(keep.any()):
        keep[:] = True
    rightmost = positions[keep].max()
    keep |= (positions < rightmost) & (positions > progress)
    return keep


class PackedHF:
    def __init__(self, model, batch_size, prefix_slots, stream=None):
        self.model = model
        self.batch = batch_size
        self.prefix = prefix_slots
        self.length = prefix_slots + 32
        self.cache = {}
        self.stream = stream
        self.progress = [-1] * batch_size

    def forward(self, tokens, coordinates, active, mode, averages, alpha, metadata, prefill=False):
        import torch
        import json
        core = self.model.model
        input_device = core.word_embeddings.weight.device
        coords = coordinates.to(input_device)
        hidden = core.word_embeddings(tokens.to(input_device)[coords[:, 0], coords[:, 1]])
        first_scores = None
        rows = []
        for layer_idx, layer in enumerate(core.layers):
            device = next(layer.parameters()).device
            hidden, coords = hidden.to(device), coords.to(device)
            attention = layer.attention
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            qkv = attention.query_key_value(normalized).view(-1, attention.num_heads + 2 * attention.num_key_value_heads,
                                                           attention.head_dim)
            q, k, v = qkv.split([attention.num_heads, attention.num_key_value_heads, attention.num_key_value_heads], dim=1)
            if attention.config.use_qk_norm:
                q, k = attention.query_layernorm(q), attention.key_layernorm(k)
            cos, sin = core.rotary_emb(hidden.unsqueeze(0), coords[:, 1].unsqueeze(0))
            rotate = sys.modules[type(attention).__module__].apply_rotary_pos_emb
            q, k = rotate(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                          cos.to(device), sin.to(device))
            q, k = q.squeeze(0).transpose(0, 1), k.squeeze(0).transpose(0, 1)
            if layer_idx not in self.cache:
                shape = (self.batch, self.length, attention.num_key_value_heads, attention.head_dim)
                self.cache[layer_idx] = (torch.zeros(shape, device=device, dtype=k.dtype),
                                         torch.zeros(shape, device=device, dtype=v.dtype),
                                         torch.zeros((self.batch, self.length), device=device, dtype=torch.bool))
            cached_k, cached_v, valid = self.cache[layer_idx]
            # Official ordering: fill ALL incoming K/V before selecting/compacting Q at layer 1.
            cached_k[coords[:, 0], coords[:, 1]] = k
            cached_v[coords[:, 0], coords[:, 1]] = v
            valid[coords[:, 0], coords[:, 1]] = True
            if mode == 'focus' and not prefill and layer_idx in (0, 1):
                active_device = active.to(device)
                is_mask = active_device[coords[:, 0], (coords[:, 1] - self.prefix).clamp_min(0)]
                keep = torch.ones(len(coords), device=device, dtype=torch.bool)
                current_scores = torch.zeros(len(coords), device=device, dtype=q.dtype)
                for request in range(self.batch):
                    indices = ((coords[:, 0] == request) & is_mask).nonzero(as_tuple=True)[0]
                    if not len(indices):
                        continue
                    score = importance(q[indices], k[indices], attention.num_key_value_groups, attention.scaling)
                    current_scores[indices] = score
                    if layer_idx == 1:
                        delta = score.float() - first_scores.to(device)[indices].float()
                        keep[indices] = select_positions(delta, coords[indices, 1] - self.prefix,
                                                         averages[request], alpha, self.progress[request])
                if layer_idx == 0:
                    first_scores = current_scores
                else:
                    coords, q, residual = coords[keep], q[keep], residual[keep]
                    for request in range(self.batch):
                        pos = coords[coords[:, 0] == request, 1] - self.prefix
                        if len(pos):
                            self.progress[request] = max(self.progress[request], int(pos.max()))
            # One B-dimensional attention operation. Padding is discarded BEFORE any MoE.
            counts = torch.bincount(coords[:, 0], minlength=self.batch)
            offsets = counts.cumsum(0) - counts
            slots = torch.arange(len(coords), device=device) - offsets[coords[:, 0]]
            width = int(counts.max())
            padded_q = torch.zeros((self.batch, width, attention.num_heads, attention.head_dim), device=device, dtype=q.dtype)
            padded_pos = torch.full((self.batch, width), self.length - 1, device=device, dtype=torch.long)
            padded_q[coords[:, 0], slots] = q
            padded_pos[coords[:, 0], slots] = coords[:, 1]
            allowed = (torch.arange(self.length, device=device)[None, None, :] // 32 <= padded_pos[:, :, None] // 32)
            allowed &= valid[:, None, :]
            # Official ragged metadata uses history + CURRENT rightmost Query + 1,
            # not all KV entries ever written, nor the historical progress maximum.
            rightmost = torch.full((self.batch,), self.prefix - 1, device=device, dtype=torch.long)
            rightmost.scatter_reduce_(0, coords[:, 0], coords[:, 1], reduce='amax', include_self=True)
            allowed &= torch.arange(self.length, device=device)[None, None, :] <= rightmost[:, None, None]
            # Completed/empty requests have valid prompt cache. No request attends to another request.
            keys = cached_k.repeat_interleave(attention.num_key_value_groups, dim=2).transpose(1, 2)
            values = cached_v.repeat_interleave(attention.num_key_value_groups, dim=2).transpose(1, 2)
            scores = (padded_q.transpose(1, 2) @ keys.transpose(-1, -2)) * attention.scaling
            scores = scores.masked_fill(~allowed[:, None], float('-inf'))
            weights = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            result = (weights @ values).transpose(1, 2)[coords[:, 0], slots].reshape(len(coords), -1)
            hidden = residual + attention.dense(result)
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden).unsqueeze(0)
            mlp_output = layer.mlp(mlp_input)
            if isinstance(mlp_output, tuple):
                mlp_hidden, router = mlp_output
                topk = router[1].reshape(len(coords), -1)
                load = torch.bincount(topk.flatten(), minlength=self.model.config.num_experts).cpu().tolist()
                if sum(load) != len(coords) * topk.shape[-1]:
                    raise RuntimeError('Actual packed Query/route assignment count mismatch')
                row = dict(metadata, record_type='moe_forward', layer_idx=layer_idx,
                           phase='prefill' if prefill else 'decode', actual_batch=self.batch,
                           nonempty_requests=int((counts > 0).sum()), query_tokens=len(coords),
                           q_seqlens=counts.cpu().tolist(), active_experts=sum(value > 0 for value in load),
                           num_experts=len(load), top_k=topk.shape[-1], expert_load=load)
                rows.append(row)
                if self.stream:
                    self.stream.write(json.dumps(row) + '\n')
            else:
                mlp_hidden = mlp_output
            hidden = residual + mlp_hidden.squeeze(0).to(device)
        hidden = core.norm(hidden)
        logits = self.model.lm_head(hidden).float()
        return logits, coords, rows
