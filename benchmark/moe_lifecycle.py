"""Observation-only helpers for full generation-token MoE lifecycles."""


class OutputDeltaCollector:
    """Keep just the previous step on CPU; never change a module's output.

    Measures the actual MLP module output (including shared experts if present),
    not individual routed-expert outputs. All generation positions are observed.
    """

    def __init__(self, block_length):
        self.block_length = block_length
        self.previous = {}
        self.metrics = {}

    def reset(self):
        self.previous.clear()
        self.metrics.clear()

    def begin_step(self):
        self.metrics = {}

    def hook(self, layer_idx):
        def observe(_module, _inputs, output):
            import torch

            value = output[0] if isinstance(output, (tuple, list)) else output
            if value.ndim != 3:
                raise RuntimeError('Expected MoE output [batch, sequence, hidden].')
            current = value[:, -self.block_length:].detach().to(device='cpu', copy=True)
            prior = self.previous.get(layer_idx)
            if prior is not None:
                left, right = prior.float(), current.float()
                left_norm, right_norm = left.norm(dim=-1), right.norm(dim=-1)
                difference = (right - left).norm(dim=-1)
                relative = difference / left_norm.clamp_min(1e-12)
                cosine = (left * right).sum(-1) / (left_norm * right_norm).clamp_min(1e-12)
                # Undefined normalized metrics are missing, not artificial zeros.
                self.metrics[layer_idx] = [
                    [dict(moe_output_relative_l2=(float(relative[b, p]) if left_norm[b, p] > 0 else None),
                          moe_output_cosine=(float(cosine[b, p])
                                             if left_norm[b, p] * right_norm[b, p] > 0 else None))
                     for p in range(current.shape[1])]
                    for b in range(current.shape[0])]
            self.previous[layer_idx] = current
        return observe

    def attach(self, records):
        for token in records:
            batch, position = token['batch_index'], token['block_position']
            for layer in token['layers']:
                metrics = self.metrics.get(layer['layer_idx'])
                layer.update(metrics[batch][position] if metrics is not None else {
                    'moe_output_relative_l2': None, 'moe_output_cosine': None})


def annotate_lifecycle(records, before, active, accepted, acceptance_steps, step):
    """Annotate pre-forward state, then update acceptance time exactly once."""
    before, active, accepted = before.cpu(), active.cpu(), accepted.cpu()
    finished = ~active.any(dim=1)
    for token in records:
        batch, position = token['batch_index'], token['block_position']
        key = (batch, position)
        if bool(accepted[batch, position]):
            if key in acceptance_steps:
                raise RuntimeError('A decoded token cannot be accepted twice.')
            acceptance_steps[key] = step
        accepted_at = acceptance_steps.get(key)
        token.update({
            'input_token_id': int(before[batch, position]),
            'masked_before': bool(active[batch, position]),
            'state_before': 'mask' if active[batch, position] else 'decoded',
            'accepted_this_step': bool(accepted[batch, position]),
            'acceptance_step': accepted_at,
            'steps_since_acceptance': None if accepted_at is None else step - accepted_at,
            'request_finished_before': bool(finished[batch]),
            'executed_this_step': True,
            'skip_reason': None,
        })
