import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ct_transfer import pipeline


class ResourceProgressTests(unittest.TestCase):
    def setUp(self):
        private = ROOT.parent / "work/private/tests"
        private.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=private)
        self.path = Path(self.temp.name)
        reference = self.path / "run02_snapshot"
        reference.mkdir()
        (reference / "gpu_benchmark.json").write_text(json.dumps({"measurements": [{"scope": "full", "peak_allocated_bytes": 1000}]}), encoding="utf-8")
        self.cohort = ({"records": [{"label": label} for label in (0, 1, 2, 0)]}, [], np.zeros((4, 3, 2, 2), dtype=np.float32))

    def tearDown(self):
        self.temp.cleanup()

    def test_all_skipped_candidates_preserve_their_guards_before_failure(self):
        memory = {"conservative_free_bytes": 0, "window_start_kst": "fixture", "window_end_kst": "fixture"}
        with mock.patch.object(pipeline, "PRIVATE", self.path), mock.patch.object(pipeline, "load_cohort", return_value=self.cohort), mock.patch.object(pipeline, "device", return_value=torch.device("cpu")), mock.patch.object(pipeline, "memory_snapshot", return_value=memory), mock.patch.object(pipeline.models, "build_model", side_effect=AssertionError("GPU 적재를 하면 안 됩니다")):
            with self.assertRaises(RuntimeError):
                pipeline.benchmark()
        progress_path = self.path / "resource_probe_progress.json"
        self.assertTrue(progress_path.exists(), "선택 실패 이전의 guard가 남아야 합니다")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress["status"], "no_eligible_batch")
        self.assertEqual(len(progress["skipped"]), 9)
        self.assertEqual(progress["measurements"], [])
        self.assertTrue(progress["source"])

    def test_completed_measurement_survives_a_later_memory_query_error(self):
        memory = {"conservative_free_bytes": 20 * 1024 ** 3, "window_start_kst": "fixture", "window_end_kst": "fixture"}
        fake_model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
        with mock.patch.object(pipeline, "PRIVATE", self.path), mock.patch.object(pipeline, "load_cohort", return_value=self.cohort), mock.patch.object(pipeline, "device", return_value=torch.device("cpu")), mock.patch.object(pipeline, "memory_snapshot", side_effect=[memory, memory, memory, OSError("fixture memory query")]), mock.patch.object(pipeline.models, "build_model", return_value=(fake_model, {"fixture": True})), mock.patch.object(pipeline, "_sync"), mock.patch.object(torch.cuda, "mem_get_info", return_value=(20 * 1024 ** 3, 24 * 1024 ** 3)), mock.patch.object(torch.cuda, "reset_peak_memory_stats"), mock.patch.object(torch.cuda, "max_memory_allocated", return_value=1000), mock.patch.object(torch.cuda, "max_memory_reserved", return_value=2000), mock.patch.object(torch.cuda, "empty_cache"):
            with self.assertRaises(OSError):
                pipeline.benchmark()
        progress_path = self.path / "resource_probe_progress.json"
        self.assertTrue(progress_path.exists(), "뒤의 조회 오류 전에 끝난 측정이 남아야 합니다")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(len(progress["measurements"]), 1)
        self.assertEqual(progress["measurements"][0]["scope"], "frozen")
        self.assertEqual(progress["measurements"][0]["peak_reserved_bytes"], 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
