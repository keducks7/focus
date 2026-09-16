"""Opt-in, eager-only MoE observations for the single-process research runner."""
import json

_observer = None


def set_observer(observer):
    global _observer
    _observer = observer


def observe(layer_idx, topk_ids, num_experts, context):
    if _observer is not None:
        _observer.record(layer_idx, topk_ids, num_experts, context)


class MoEObserver:
    def __init__(self, stream, mode, configured_batch, group_id):
        self.stream = stream
        self.mode = mode
        self.batch = configured_batch
        self.group = group_id
        self.forward_id = -1
        self.last_layer = None

    def record(self, layer_idx, topk_ids, num_experts, context):
        import torch

        if context is None or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('MoE observation requires a real context and eager execution')
        if self.last_layer is None or layer_idx <= self.last_layer:
            self.forward_id += 1
        self.last_layer = layer_idx
        ids = topk_ids.detach().reshape(-1, topk_ids.shape[-1])
        lengths = context.q_seqlens.detach().cpu().tolist()
        if sum(lengths) != ids.shape[0]:
            raise RuntimeError(f'Actual MoE tokens {ids.shape[0]} != ragged Query sum {sum(lengths)}')
        load = torch.bincount(ids.flatten(), minlength=num_experts).cpu().tolist()
        if len(load) != num_experts or sum(load) != ids.numel():
            raise RuntimeError('Invalid expert assignment histogram')
        row = dict(record_type='moe_forward', mode=self.mode, configured_batch=self.batch,
                   group_id=self.group, forward_id=self.forward_id, layer_idx=layer_idx,
                   phase='decode' if context.is_decoding else 'prefill',
                   actual_batch=len(lengths), nonempty_requests=sum(length > 0 for length in lengths),
                   query_tokens=sum(lengths), q_seqlens=lengths, top_k=ids.shape[-1],
                   num_experts=num_experts, active_experts=sum(value > 0 for value in load), expert_load=load)
        self.stream.write(json.dumps(row) + '\n')
