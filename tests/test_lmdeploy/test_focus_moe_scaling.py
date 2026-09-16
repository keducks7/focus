import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / 'benchmark' / 'profile_focus_moe_scaling.py'
SPEC = importlib.util.spec_from_file_location('focus_scaling', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScalingTests(unittest.TestCase):
    def test_summary_excludes_prefill_and_partial_batches(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for batch, experts in [(1, 8), (2, 12)]:
                run = root / f'focus_bs{batch}'
                run.mkdir()
                (run / 'complete.json').write_text('{}')
                row = dict(record_type='moe_forward', phase='decode', actual_batch=batch,
                           nonempty_requests=batch, group_id=0, layer_idx=2,
                           query_tokens=16 * batch, active_experts=experts, num_experts=256)
                records = [row, dict(row, phase='prefill', active_experts=255),
                           dict(row, nonempty_requests=0, active_experts=200)]
                (run / 'routes.jsonl').write_text('\n'.join(map(json.dumps, records)))
            MODULE.summarize(root, ['focus'], [1, 2])
            with (root / 'scaling.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]['mean_active_experts']), 8)
            self.assertEqual(float(rows[1]['expert_doubling_ratio']), 1.5)

    def test_incomplete_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                MODULE.summarize(Path(directory), ['focus'], [1])


if __name__ == '__main__':
    unittest.main()
