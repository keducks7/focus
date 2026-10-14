"""Matched complete generation for four training-free routing configurations."""
import inspect
import os
import uuid
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .base import BaseModel
from .llada2_shared_batch import LLaDA2SharedBatch
from .llada2_joint_routing import BlockJointRoutingController, METHODS, validate_two_gpu_placement


class LLaDA2JointBatch(LLaDA2SharedBatch):
    def __init__(self, path, max_seq_len=4096, tokenizer_only=False, meta_template=None,
                 block_length=32, steps=32, confidence_threshold=.8,
                 epsilon=.1, route_method='batch', sampling='greedy', temperature=0.,
                 top_k=0, top_p=1., max_memory_per_gpu='34GiB', detailed_timing=False,
                 metrics_dir='results/block_joint_metrics', seed=0, **kwargs):
        # Reuse the proven prompt/decode/metrics path without installing the OLD
        # MASK-only controller. Do not change original shared-pair experiments.
        BaseModel.__init__(self, path=path, max_seq_len=max_seq_len, tokenizer_only=tokenizer_only,
                           meta_template=meta_template, **kwargs)
        if route_method not in METHODS or not 0 <= epsilon < 1:
            raise ValueError('Unsupported routing method or epsilon.')
        if not 1 <= steps <= block_length or not 0 <= confidence_threshold <= 1:
            raise ValueError('Invalid block/step/confidence settings.')
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.mask_id, self.eos_id = self.tokenizer.mask_token_id, self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_id
        if self.mask_id is None or self.pad_id is None:
            raise ValueError('Tokenizer must define MASK and PAD or EOS.')
        self.settings = dict(block_length=block_length, steps=steps, threshold=confidence_threshold,
                             sampling=sampling, temperature=temperature, top_k=top_k, top_p=top_p)
        self.epsilon = 0. if route_method == 'native' else epsilon
        self.route_method, self.seed, self.renormalize = route_method, seed, False
        self.metrics_dir = Path(metrics_dir)
        self.metrics_file = self.metrics_dir/f'metrics-{os.getpid()}-{uuid.uuid4().hex[:8]}.jsonl'
        self.calls = 0
        self.model = self.controller = None
        if not tokenizer_only:
            if torch.cuda.device_count() != 2:
                raise RuntimeError('Use exactly two visible GPUs and one process.')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Leave activation/KV workspace on BOTH 40GB cards. Accelerate's
            # no_split decoder layers make this layer model parallelism, NOT TP
            # or splitting the batch. No CPU/disk offload accepted below.
            # Reuse the current main-repository loading fix: checkpoint weights
            # and config, but repository HF code exposing the block KV API.
            code_dir = Path(__file__).resolve().parent / 'LLaDA2.0-mini'
            model_cls = get_class_from_dynamic_module(
                'modeling_llada2_moe.LLaDA2MoeModelLM', str(code_dir), local_files_only=True)
            model_config = model_cls.config_class.from_pretrained(path, local_files_only=True)
            self.model = model_cls.from_pretrained(
                path, config=model_config, torch_dtype=torch.bfloat16,
                device_map='balanced', max_memory={0:max_memory_per_gpu, 1:max_memory_per_gpu},
                low_cpu_mem_usage=True, attn_implementation='eager').eval()
            if 'store_kv' not in inspect.signature(self.model.model.forward).parameters:
                raise TypeError('Model must expose store_kv for prefix/block KV caching.')
            validate_two_gpu_placement(self.model)
            self.controller = BlockJointRoutingController(self.model, self.epsilon, route_method, detailed_timing)
