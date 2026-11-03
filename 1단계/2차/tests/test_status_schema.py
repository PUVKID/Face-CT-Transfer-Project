"""화면에 전달하는 상태의 명시 타입과 미측정 값을 독립 fixture로 검사합니다."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen
from benchmark.acquisition import now
from benchmark.status import initial_status, read_public_status
from dashboard.server import create_server


def full_fixture():
    state = initial_status()
    state.update(status="running", updated_at=now(), stale=False)
    state["evaluation"].update(benchmark="LFW", images_completed=0, images_total=1, detection_coverage=0.0, failed_images=0)
    state["processes"] = [{"role": "검사용 역할", "pid": os.getpid(), "alive": False, "started_at": None}]
    state["model"]["measured_metrics"] = [{"benchmark": "검사용 자료", "metric": "검사용 지표", "value": 0.0, "unit": "%", "provenance": "제어 fixture; 실제 측정 아님"}]
    state["downloads"] = [{"id": "fixture", "name": "검사용 원본", "status": "pending", "received_bytes": 0, "total_bytes": None, "verified": False, "expected_digest": None, "digest_algorithm": None, "bytes_per_second": None, "eta_seconds": None, "error": None, "official_digest_verified": False, "verification_basis": "not_verified"}]
    state["resources"] = {"disk_free_bytes": 0, "gpu": {"name": "검사용 장치", "utilization_percent": 0.0, "memory_used_bytes": 0, "memory_total_bytes": 1, "measured_at": now()}}
    state["external_data"] = [{"name": "검사용 외부 자료", "status": None, "received_bytes": 0, "verified_bytes": None, "total_bytes": None, "updated_at": None}]
    for key in state["series"]:
        state["series"][key] = [{"time": now(), "value": 0.0}]
    return state


def replace_path(state, path, value):
    current = state
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


class StatusSchemaTests(unittest.TestCase):
    def read_fixture(self, state, root):
        (root / "status.json").write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")
        with patch("benchmark.status.process_alive", return_value=False):
            return read_public_status(root)

    def assert_rejected(self, state, root):
        result = self.read_fixture(state, root)
        self.assertEqual(result["status"], "failed")
        self.assertIs(result["stale"], True)
        self.assertIsNone(result["evaluation"]["metrics"])
        self.assertEqual(result["model"]["measured_metrics"], [])
        self.assertIn("검증 실패", result["current_task"]["title"])
        self.assertTrue(any("유효" in value for value in result["notices"]))

    def test_three_independent_counterexamples_are_rejected_unchanged(self):
        cases = []
        evaluation = initial_status()
        evaluation["evaluation"]["benchmark"] = {"bad": "object"}
        cases.append(evaluation)
        process = initial_status()
        process["processes"] = [{"role": {"bad": "object"}, "pid": None, "alive": False, "started_at": None}]
        cases.append(process)
        metric = initial_status()
        metric["model"]["measured_metrics"] = [{"benchmark": {"bad": "object"}, "metric": "accuracy", "value": 0.5, "unit": "%", "provenance": "fixture"}]
        cases.append(metric)
        with tempfile.TemporaryDirectory() as temporary:
            for index, state in enumerate(cases):
                with self.subTest(original_case=index):
                    state.update(status="running", updated_at=now(), stale=False)
                    self.assert_rejected(state, Path(temporary))

    def test_known_nullable_and_rendered_scalars_reject_objects_lists_and_wrong_bools(self):
        string_paths = [
            ("project", "revision"), ("current_task", "title"), ("current_task", "phase"), ("current_task", "detail"),
            ("model", "name"), ("model", "training_data"), ("model", "source_url"), ("model", "version"), ("model", "license_status"),
            ("evaluation", "benchmark"), ("evaluation", "precision"), ("legacy", "model"), ("legacy", "status"), ("legacy", "detail"),
            ("processes", 0, "role"), ("processes", 0, "started_at"),
            ("model", "measured_metrics", 0, "benchmark"), ("model", "measured_metrics", 0, "metric"), ("model", "measured_metrics", 0, "unit"), ("model", "measured_metrics", 0, "provenance"),
            ("model", "reported_metrics", 0, "benchmark"), ("downloads", 0, "id"), ("downloads", 0, "name"), ("downloads", 0, "error"), ("downloads", 0, "expected_digest"), ("downloads", 0, "digest_algorithm"), ("downloads", 0, "verification_basis"),
            ("steps", 0, "title"), ("steps", 0, "detail"), ("techniques", 0, "name"), ("techniques", 0, "detail"),
            ("external_data", 0, "name"), ("external_data", 0, "status"), ("external_data", 0, "updated_at"),
            ("log_sources", 0, "id"), ("log_sources", 0, "label"), ("resources", "gpu", "name"), ("resources", "gpu", "measured_at"),
            ("series", "throughput", 0, "time"), ("series", "gpu_utilization", 0, "time"), ("series", "metrics", 0, "time")
        ]
        numeric_paths = [
            ("progress", "completed"), ("progress", "total"), ("progress", "percent"), ("progress", "eta_seconds"),
            ("evaluation", "images_completed"), ("evaluation", "pairs_total"), ("evaluation", "fold_completed"), ("evaluation", "epoch"),
            ("resources", "disk_free_bytes"), ("resources", "gpu", "utilization_percent"), ("resources", "gpu", "memory_used_bytes"),
            ("model", "measured_metrics", 0, "value"), ("model", "reported_metrics", 0, "value"),
            ("downloads", 0, "received_bytes"), ("downloads", 0, "total_bytes"), ("downloads", 0, "bytes_per_second"), ("downloads", 0, "eta_seconds"),
            ("external_data", 0, "received_bytes"), ("external_data", 0, "verified_bytes"), ("external_data", 0, "total_bytes"),
            ("series", "throughput", 0, "value"), ("series", "gpu_utilization", 0, "value"), ("series", "metrics", 0, "value")
        ]
        bool_paths = [("model", "checkpoint_verified"), ("progress", "eta_is_estimate"), ("processes", 0, "alive"), ("downloads", 0, "verified"), ("downloads", 0, "official_digest_verified")]
        with tempfile.TemporaryDirectory() as temporary:
            for path in string_paths + numeric_paths + bool_paths:
                for bad in ([{}, []] if path in bool_paths else [{}, [], True]):
                    with self.subTest(path=path, bad_type=type(bad).__name__):
                        state = full_fixture()
                        replace_path(state, path, bad)
                        self.assert_rejected(state, Path(temporary))

    def test_empty_array_row_contracts_are_explicit(self):
        arrays = [("processes",), ("model", "measured_metrics"), ("model", "reported_metrics"), ("downloads",), ("steps",), ("techniques",), ("external_data",), ("log_sources",), ("series", "throughput"), ("series", "gpu_utilization"), ("series", "metrics")]
        with tempfile.TemporaryDirectory() as temporary:
            for path in arrays:
                for row in [{}, [], None, False]:
                    with self.subTest(path=path, row_type=type(row).__name__, empty=row == {}):
                        state = full_fixture()
                        replace_path(state, path, [row])
                        self.assert_rejected(state, Path(temporary))

    def test_counts_pid_finite_metrics_and_coverage_keep_numeric_contracts(self):
        cases = [
            (("processes", 0, "pid"), True), (("processes", 0, "pid"), 1.5), (("processes", 0, "pid"), None),
            (("evaluation", "failed_images"), True), (("evaluation", "failed_images"), -1), (("evaluation", "failed_images"), 1.5),
            (("evaluation", "detection_coverage"), True), (("evaluation", "detection_coverage"), -0.1), (("evaluation", "detection_coverage"), 1.1),
            (("model", "measured_metrics", 0, "value"), float("nan")), (("model", "measured_metrics", 0, "value"), float("inf")),
            (("series", "throughput", 0, "value"), float("inf")), (("external_data", 0, "verified_bytes"), True)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for path, bad in cases:
                with self.subTest(path=path, bad=bad):
                    state = full_fixture()
                    replace_path(state, path, bad)
                    self.assert_rejected(state, Path(temporary))

    def test_normal_rows_null_false_and_zero_are_preserved(self):
        state = full_fixture()
        state["processes"].append({"role": "생존 미확인", "pid": os.getpid(), "alive": None, "started_at": None})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "status.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            with patch("benchmark.status.process_alive", side_effect=[False, None]):
                actual = read_public_status(root)
        self.assertEqual(actual, state)
        self.assertIs(actual["processes"][0]["alive"], False)
        self.assertIsNone(actual["processes"][1]["alive"])
        self.assertEqual(type(actual["evaluation"]["failed_images"]), int)
        self.assertEqual(actual["evaluation"]["failed_images"], 0)
        self.assertIsNone(actual["evaluation"]["metrics"])
        self.assertIsNone(actual["downloads"][0]["expected_digest"])
        self.assertIs(actual["downloads"][0]["official_digest_verified"], False)

    def test_unobserved_timestamp_and_empty_arrays_are_valid_stale_state(self):
        state = initial_status()
        with tempfile.TemporaryDirectory() as temporary:
            actual = self.read_fixture(state, Path(temporary))
        self.assertEqual(actual["status"], "waiting")
        self.assertIs(actual["stale"], True)
        self.assertIsNone(actual["updated_at"])
        self.assertIsNone(actual["evaluation"]["benchmark"])
        self.assertIsNone(actual["evaluation"]["metrics"])
        self.assertEqual(actual["model"]["measured_metrics"], [])

    def test_typed_http_rejection_returns_valid_failed_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = full_fixture()
            state["evaluation"]["benchmark"] = {"bad": "object"}
            (root / "status.json").write_text(json.dumps(state), encoding="utf-8")
            server = create_server(root, root, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/status") as response:
                    result = json.load(response)
                    self.assertEqual(response.status, 200)
                self.assertEqual(result["status"], "failed")
                self.assertIs(result["stale"], True)
                self.assertIsNone(result["evaluation"]["metrics"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
