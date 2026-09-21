"""Dependency-free regression checks for the shared evaluation entry point."""
import ast
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'opencompass-0.5.1.post1/opencompass/models'
CONFIG = MODELS.parent / 'configs/models/llada2_shared_pair.py'


class SharedEvalLoadingTest(unittest.TestCase):
    def test_config_is_eager_and_serializable(self):
        tree = ast.parse(CONFIG.read_text())
        self.assertIsInstance(tree.body[0], ast.Assign)
        self.assertEqual(tree.body[0].targets[0].id, '_base_')
        with patch.dict(os.environ, {'SHARED_MODE': 'both', 'BATCH_SIZE': '16'}):
            models = runpy.run_path(str(CONFIG))['models']
        restored = ast.literal_eval(repr(models))
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0]['epsilon'], 0.)
        for model in restored:
            self.assertEqual(model['batch_size'], 16)
            self.assertEqual(model['type'],
                             'opencompass.models.llada2_shared_batch.LLaDA2SharedBatch')

    def test_repository_model_has_block_cache_api(self):
        tree = ast.parse((MODELS / 'LLaDA2.0-mini/modeling_llada2_moe.py').read_text())
        for name in ('LLaDA2MoeModel', 'LLaDA2MoeModelLM'):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
            self.assertIn('store_kv', [a.arg for a in forward.args.args])


if __name__ == '__main__':
    unittest.main()
