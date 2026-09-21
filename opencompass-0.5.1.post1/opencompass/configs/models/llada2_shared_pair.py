_base_ = []  # Use eager parsing for environment-derived configuration.
"""Matched HF Vanilla / shared-route pair. Both use two GPUs, one worker."""
import os

_mode = os.environ.get('SHARED_MODE', 'both')
if _mode not in ('both','vanilla','shared'):
    raise ValueError('SHARED_MODE must be both, vanilla or shared.')
_epsilon = float(os.environ.get('ROUTE_EPSILON','0.1'))
_batch = int(os.environ.get('BATCH_SIZE','8'))
_method = os.environ.get('ROUTE_METHOD','joint')
_root = os.environ.get('ROUTE_METRICS_DIR','../results/shared_route_eval/metrics')
models = []
for _name, _eps in [('vanilla',0.),('shared',_epsilon)]:
    if _mode != 'both' and _mode != _name:
        continue
    _abbr = f'llada2-{_name}-{_method}-e{_eps:g}-b{_batch}'
    models.append(dict(
        type='opencompass.models.llada2_shared_batch.LLaDA2SharedBatch', abbr=_abbr,
        path=os.environ.get('MODEL_PATH','/root/lkd/Models/LLaDA2.0-mini'),
        max_seq_len=int(os.environ.get('MAX_SEQ_LEN','4096')),
        max_out_len=int(os.environ.get('MAX_OUT_LEN','1024')),
        batch_size=_batch, block_length=int(os.environ.get('BLOCK_LENGTH','32')),
        steps=int(os.environ.get('DENOISING_STEPS','32')),
        confidence_threshold=float(os.environ.get('CONFIDENCE_THRESHOLD','0.8')),
        epsilon=_eps, route_method=_method,
        sampling=os.environ.get('SAMPLING','native'), temperature=0., top_k=0, top_p=1.,
        max_memory_per_gpu=os.environ.get('MAX_MEMORY_PER_GPU','38GiB'),
        metrics_dir=os.path.join(_root,_abbr), seed=int(os.environ.get('SEED','0')),
        run_cfg=dict(num_gpus=2, num_procs=1),
    ))
