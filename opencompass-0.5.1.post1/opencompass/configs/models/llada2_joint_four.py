"""All four variants: two GPUs per model, one serial worker, real request batch."""
import os

from opencompass.models.llada2_joint_batch import LLaDA2JointBatch

_methods = os.environ.get('JOINT_METHODS', 'native,independent,request,batch').split(',')
if not _methods or len(set(_methods)) != len(_methods) or any(
        m not in ('native', 'independent', 'request', 'batch') for m in _methods):
    raise ValueError('JOINT_METHODS: comma-separated unique native,independent,request,batch.')
_root = os.environ.get('ROUTE_METRICS_DIR', '../results/block_joint_eval/metrics')
_batch = int(os.environ.get('BATCH_SIZE', '8'))
models = []
for _method in _methods:
    _eps = 0. if _method == 'native' else float(os.environ.get('ROUTE_EPSILON', '0.1'))
    _abbr = f'llada2-block-{_method}-e{_eps:g}-b{_batch}'
    models.append(dict(
        type=LLaDA2JointBatch, abbr=_abbr,
        path=os.environ.get('MODEL_PATH', '/root/lkd/Models/LLaDA2.0-mini'),
        max_seq_len=int(os.environ.get('MAX_SEQ_LEN', '4096')),
        max_out_len=int(os.environ.get('MAX_OUT_LEN', '1024')),
        batch_size=_batch, block_length=int(os.environ.get('BLOCK_LENGTH', '32')),
        steps=int(os.environ.get('DENOISING_STEPS', '32')),
        confidence_threshold=float(os.environ.get('CONFIDENCE_THRESHOLD', '0.8')),
        epsilon=_eps, route_method=_method, sampling=os.environ.get('SAMPLING', 'greedy'),
        temperature=0., top_k=0, top_p=1.,
        max_memory_per_gpu=os.environ.get('MAX_MEMORY_PER_GPU', '34GiB'),
        detailed_timing=os.environ.get('DETAILED_TIMING', '0') == '1',
        metrics_dir=os.path.join(_root, _abbr), seed=int(os.environ.get('SEED', '0')),
        run_cfg=dict(num_gpus=2, num_procs=1),
    ))
