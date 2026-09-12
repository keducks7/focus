import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    'replay_moe_layer', Path(__file__).parents[2] / 'benchmark/replay_moe_layer.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReplayInputsTest(unittest.TestCase):
    def test_exact_degrees_and_distinct_routes(self):
        rng = np.random.default_rng(17)
        for experts, topk in [(8, 8), (16, 4), (256, 8)]:
            for tokens in [1, 2, 17, 64]:
                original = np.array([rng.choice(experts, topk, replace=False) for _ in range(tokens)])
                loads = np.bincount(original.flatten(), minlength=experts)
                result = MODULE.synthesize_routes(loads, topk, seed=3)
                np.testing.assert_array_equal(np.bincount(result.flatten(), minlength=experts), loads)
                self.assertEqual(result.shape, (tokens, topk))
                self.assertTrue(all(len(set(row)) == topk for row in result))
                np.testing.assert_array_equal(result, MODULE.synthesize_routes(loads, topk, seed=3))

    def test_concentrated_load_and_zero_experts(self):
        result = MODULE.synthesize_routes([3, 3, 3, 0, 0], 3)
        self.assertTrue(all(set(row) == {0, 1, 2} for row in result))

    def test_invalid_histograms_rejected(self):
        for loads, k in [([0, 0], 1), ([3, 1], 2), ([-1, 3], 1), ([1, 2], 2)]:
            with self.assertRaises(ValueError):
                MODULE.synthesize_routes(loads, k)


if __name__ == '__main__':
    unittest.main()

