"""First 32 examples; original GSM8K prompt and evaluator, smoke only."""
from mmengine.config import read_base

with read_base():
    from .gsm8k_0shot_v2_gen_17d799 import gsm8k_datasets

for _dataset in gsm8k_datasets:
    _dataset['abbr'] = 'gsm8k-shared-smoke32'
    _dataset['reader_cfg']['test_range'] = '[0:32]'
