"""OpenCompass HF model: genuine batched Vanilla / Vanilla + shared routes."""
import inspect
import json
import os
import time
import uuid
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .base import BaseModel
from .llada2 import LLaDA2, _convert_chat_messages
from .llada2_batch_decode import generate_batch, synchronize_model
from .llada2_shared_routing import SharedRoutingController
from .sdar_utils import set_context


class LLaDA2SharedBatch(LLaDA2):
    def __init__(self, path, max_seq_len=4096, tokenizer_only=False, meta_template=None,
                 block_length=32, steps=32, confidence_threshold=.8,
                 epsilon=0., route_method='joint', renormalize=True,
                 sampling='native', temperature=0., top_k=0, top_p=1.,
                 max_memory_per_gpu='38GiB', metrics_dir='results/shared_route_metrics',
                 seed=0, **kwargs):
        BaseModel.__init__(self, path=path, max_seq_len=max_seq_len, tokenizer_only=tokenizer_only,
                           meta_template=meta_template, **kwargs)
        if not 0 <= epsilon < 1 or not 0 <= confidence_threshold <= 1:
            raise ValueError('Invalid epsilon or confidence threshold.')
        if not 1 <= steps <= block_length:
            raise ValueError('Require 1 <= steps <= block_length.')
        if route_method not in ('joint','independent','joint_no_fixed'):
            raise ValueError('Unsupported route method.')
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.mask_id = self.tokenizer.mask_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_id
        if self.mask_id is None or self.pad_id is None:
            raise ValueError('Tokenizer must define MASK and PAD or EOS.')
        self.settings = dict(block_length=block_length, steps=steps, threshold=confidence_threshold,
                             sampling=sampling, temperature=temperature, top_k=top_k, top_p=top_p)
        self.epsilon, self.route_method, self.seed = epsilon, route_method, seed
        self.renormalize = renormalize
        self.metrics_dir = Path(metrics_dir)
        self.metrics_file = self.metrics_dir/f'metrics-{os.getpid()}-{uuid.uuid4().hex[:8]}.jsonl'
        self.calls = 0
        self.model = None
        self.controller = None
        if not tokenizer_only:
            if torch.cuda.device_count() != 2:
                raise RuntimeError('This config requires exactly two visible GPUs; use num_gpus=2, one worker.')
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # The checkpoint's remote code may lack the repository's block-cache
            # API. Load code locally, while keeping checkpoint config/weights.
            code_dir = Path(__file__).resolve().parent / 'LLaDA2.0-mini'
            model_cls = get_class_from_dynamic_module(
                'modeling_llada2_moe.LLaDA2MoeModelLM', str(code_dir),
                local_files_only=True)
            model_config = model_cls.config_class.from_pretrained(
                path, local_files_only=True)
            self.model = model_cls.from_pretrained(
                path, config=model_config, torch_dtype=torch.bfloat16,
                device_map='balanced', max_memory={0:max_memory_per_gpu,1:max_memory_per_gpu},
                low_cpu_mem_usage=True, attn_implementation='eager').eval()
            if 'store_kv' not in inspect.signature(self.model.model.forward).parameters:
                raise TypeError('Loaded model lacks explicit store_kv support required by the repository block-cache loop.')
            placements = getattr(self.model, 'hf_device_map', {})
            if any(str(v) in ('cpu','disk') for v in placements.values()):
                raise RuntimeError('CPU/disk offload detected; refusing misleading GPU speed measurements.')
            self.controller = SharedRoutingController(self.model, epsilon, route_method, renormalize)

    def _prompt_ids(self, item):
        messages = _convert_chat_messages([item], include_system_prompt=True)[0]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # Mirrors the existing repository LLaDA2 wrapper's chat-template/tokenizer path.
        return self.tokenizer(text)['input_ids']

    def get_token_len(self, prompt):
        return len(self._prompt_ids(prompt))

    @torch.inference_mode()
    def generate(self, inputs, max_out_len):
        if self.model is None:
            raise RuntimeError('Cannot generate in tokenizer_only mode.')
        if not inputs:
            return []
        synchronize_model(self.model)
        wall_start = time.perf_counter()
        prompts = [self._prompt_ids(item) for item in inputs]
        length = self.settings['block_length']
        if any(((len(p)+max_out_len+length-1)//length)*length > self.max_seq_len for p in prompts):
            raise ValueError('Prompt + rounded generation length exceeds max_seq_len; increase it explicitly. No silent truncation.')
        set_context(is_decode=False, enable_token_eviction=False, strategy='none')
        try:
            generated, stats = generate_batch(
                self.model, prompts, mask_id=self.mask_id, pad_id=self.pad_id, eos_id=self.eos_id,
                gen_length=max_out_len, controller=self.controller, cache_factory=DynamicCache,
                **self.settings)
        finally:
            self.controller.compress = None
            set_context(is_decode=False, enable_token_eviction=False, strategy='none')
        texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
        synchronize_model(self.model)
        stats.update(record_type='batch_metrics', wall_seconds=time.perf_counter()-wall_start,
                     call_index=self.calls, first_call=self.calls == 0, epsilon=self.epsilon,
                     route_method=self.route_method, renormalize=self.renormalize, seed=self.seed,
                     settings=self.settings, model_path=self.path, max_out_len=max_out_len,
                     prompt_token_lengths=list(map(len,prompts)),
                     eos_finished=[self.eos_id is not None and self.eos_id in ids for ids in generated],
                     inference_token_count_includes_eos=True,
                     finished_rows_policy='static_batch_no_compaction',
                     tokenization_and_text_decode_in_wall=True,
                     model_loading_and_dataset_io_in_wall=False)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open('a', buffering=1) as stream:
            stream.write(json.dumps(stats)+'\n')
        self.calls += 1
        return texts
