# Copyright (c) OpenMMLab. All rights reserved.
"""Low-overhead-enough MoE route tracing for controlled experiments.

The collector intentionally stores expert-load histograms instead of every
token's routed expert ids. Route tracing is expected to run separately from
latency measurement because ``finish_forward`` transfers the accumulated
histograms to the host once per model forward.
"""

import json
from pathlib import Path
from typing import Optional

import torch


class MoERouteTrace:
    """Write per-layer expert-load histograms as JSONL records."""

    def __init__(
        self,
        output_path: Optional[str],
        num_experts: int,
        top_k: int,
        num_hidden_layers: int,
        max_batch_size: int,
    ) -> None:
        self.output_path = output_path
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.num_hidden_layers = int(num_hidden_layers)
        self.max_batch_size = int(max_batch_size)
        self._stream = None
        self._active = False
        self._forward_index = 0
        self._q_seqlens = None
        self._query_tokens = 0
        self._layer_loads = []

        if output_path is None:
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open('w', encoding='utf-8', buffering=1)
        self._write({
            'record_type': 'metadata',
            'format_version': 1,
            'model_type': 'llada2_moe',
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'num_hidden_layers': self.num_hidden_layers,
            'configured_batch_size': self.max_batch_size,
        })

    @property
    def enabled(self) -> bool:
        """Whether this process owns an output stream."""
        return self._stream is not None

    def _write(self, record: dict) -> None:
        self._stream.write(json.dumps(record, separators=(',', ':')) + '\n')

    def begin_forward(self, context, query_tokens: int) -> bool:
        """Start a real (non-warmup) decode-forward record."""
        if not self.enabled or context is None or not context.is_decoding:
            return False
        source_inputs = getattr(context, 'source_inputs', None)
        if source_inputs is not None and getattr(source_inputs, 'is_dummy', False):
            return False
        if self._active:
            raise RuntimeError('MoE route trace already has an active forward')

        self._active = True
        self._q_seqlens = context.q_seqlens.detach()
        self._query_tokens = int(query_tokens)
        self._layer_loads = []
        return True

    def record(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Accumulate one layer's expert loads on the current device."""
        if not self._active:
            return
        expert_load = torch.bincount(topk_ids.reshape(-1), minlength=self.num_experts)
        self._layer_loads.append((int(layer_idx), expert_load))

    def finish_forward(self) -> None:
        """Transfer one forward's histograms once and append its JSON record."""
        if not self._active:
            return

        q_seqlens = self._q_seqlens.to(device='cpu', dtype=torch.int64).tolist()
        if self._layer_loads:
            layer_indices = [layer_idx for layer_idx, _ in self._layer_loads]
            load_tensor = torch.stack([loads for _, loads in self._layer_loads])
            load_rows = load_tensor.to(device='cpu', dtype=torch.int64).tolist()
        else:
            layer_indices = []
            load_rows = []

        layers = []
        for layer_idx, loads in zip(layer_indices, load_rows):
            active_experts = sum(load > 0 for load in loads)
            layers.append({
                'layer_idx': layer_idx,
                'active_experts': active_experts,
                'assignments': sum(loads),
                'expert_load': loads,
            })

        self._write({
            'record_type': 'decode_forward',
            'forward_index': self._forward_index,
            'actual_batch_size': len(q_seqlens),
            'query_tokens': self._query_tokens,
            'q_seqlens': q_seqlens,
            'layers': layers,
        })
        self._forward_index += 1
        self._active = False
        self._q_seqlens = None
        self._layer_loads = []

    def abort_forward(self) -> None:
        """Discard an incomplete record after a model-forward failure."""
        self._active = False
        self._q_seqlens = None
        self._layer_loads = []

    def close(self) -> None:
        """Flush and close the output stream."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
