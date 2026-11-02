import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ct_transfer import pipeline


class RuntimeContractTests(unittest.TestCase):
    def test_private_context_is_supplied_only_by_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(pipeline.runtime_context(), {"session": None, "task": None})
        with mock.patch.dict(os.environ, {"FACE_CT_SESSION": "fixture-session", "FACE_CT_TASK": "fixture-task"}, clear=True):
            self.assertEqual(pipeline.runtime_context(), {"session": "fixture-session", "task": "fixture-task"})

    def test_common_batch_uses_full_throughput_and_all_scope_headroom(self):
        rows = [{"batch_size": batch, "scope": scope, "finite_logits": True, "minimum_free_bytes": 5 * 1024 ** 3, "images_per_second": speed} for batch, speed in ((4, 40.), (8, 80.), (16, 80.)) for scope in ("frozen", "lastblock", "full")]
        self.assertEqual(pipeline.select_common_batch(rows), 8)
        for row in rows:
            if row["batch_size"] == 8 and row["scope"] == "lastblock":
                row["minimum_free_bytes"] = 3 * 1024 ** 3
        self.assertEqual(pipeline.select_common_batch(rows), 16)
        for row in rows:
            row["minimum_free_bytes"] = 3 * 1024 ** 3
        with self.assertRaises(RuntimeError):
            pipeline.select_common_batch(rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
