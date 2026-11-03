"""미측정 상태·읽기 경계·출력 상한을 검증합니다."""
import json
from pathlib import Path
import tempfile
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import unittest
from benchmark.status import initial_status, read_public_status, log_tail
from dashboard.server import create_server
from benchmark.acquisition import now


class StatusTests(unittest.TestCase):
    def test_log_wire_json_bytes_cap_handles_escaping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "server.log").write_bytes(b"\\" * 59999 + b"\n")
            server = create_server(root, root, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/logs?id=server&limit=200") as response:
                    raw = response.read()
                    self.assertEqual(response.status, 200)
                self.assertLessEqual(len(raw), 64 * 1024)
                value = json.loads(raw.decode("utf-8", errors="strict"))
                self.assertTrue(value["truncated"])
                self.assertEqual(value["lines"], [])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_log_tail_keeps_blank_lines_and_korean_at_wire_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            korean = "한글" * 3500
            (root / "evaluation.log").write_text("\\" * 32000 + "\n" + korean + "\n\n종료\n", encoding="utf-8")
            value = log_tail(root, "evaluation", 200)
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.assertLessEqual(len(raw), 64 * 1024)
            self.assertTrue(value["truncated"])
            self.assertEqual(value["lines"], [korean, "", "종료"])
            self.assertNotIn("\ufffd", raw.decode("utf-8", errors="strict"))

    def test_nonobject_and_wrong_nested_status_shapes_fail_safely(self):
        invalid = [[], None, True, "문자열", 1]
        for key, value in [("processes", [None]), ("processes", {}), ("model", []), ("resources", {"gpu": []}), ("evaluation", {"metrics": []}), ("series", {"throughput": {}}), ("notices", [None])]:
            candidate = initial_status()
            candidate.update(updated_at=now(), status="running")
            candidate[key] = value
            invalid.append(candidate)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in invalid:
                with self.subTest(value=value):
                    (root / "status.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                    result = read_public_status(root)
                    self.assertEqual(result["status"], "failed")
                    self.assertTrue(result["stale"])
                    self.assertIsNone(result["evaluation"]["metrics"])
                    self.assertTrue(any("유효" in notice for notice in result["notices"]))

    def test_http_wrong_status_shape_returns_failed_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "status.json").write_text("[]", encoding="utf-8")
            server = create_server(root, root, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/status") as response:
                    value = json.load(response)
                    self.assertEqual(response.status, 200)
                self.assertEqual(value["status"], "failed")
                self.assertTrue(value["stale"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_missing_state_is_stale_and_never_measured(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = read_public_status(Path(temporary))
            self.assertTrue(result["stale"])
            self.assertIsNone(result["evaluation"]["metrics"])
            self.assertIsNone(result["evaluation"]["detection_coverage"])
            self.assertIsNone(result["evaluation"]["failed_images"])
            self.assertEqual(result["model"]["measured_metrics"], [])
            self.assertEqual(result["series"]["metrics"], [])
            self.assertEqual(result["legacy"]["status"], "user_stopped")

    def test_malformed_state_does_not_claim_healthy_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "status.json").write_text("{broken", encoding="utf-8")
            result = read_public_status(root)
            self.assertTrue(result["stale"])
            self.assertEqual(result["status"], "failed")

    def test_log_source_is_allowlisted_and_secret_is_redacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acquisition.log").write_text("token=secret-value\n정상 처리\n", encoding="utf-8")
            result = log_tail(root, "acquisition", 100)
            self.assertNotIn("secret-value", "\n".join(result["lines"]))
            with self.assertRaises(ValueError):
                log_tail(root, "../../secret", 10)

    def test_server_rejects_mutations_and_path_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dist = root / "dist"
            dist.mkdir()
            (dist / "index.html").write_text("<p>현재 상태</p>", encoding="utf-8")
            server = create_server(root, dist, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/status") as response:
                    self.assertIsNone(json.load(response)["evaluation"]["metrics"])
                for path in ["/api/logs?id=../../secret", "/api/logs?id=server&path=x", "/%2e%2e/secret"]:
                    with self.assertRaises(HTTPError) as caught:
                        urlopen(base + path)
                    self.assertIn(caught.exception.code, [400, 403, 404])
                with self.assertRaises(HTTPError) as caught:
                    urlopen(Request(base + "/api/status", data=b"x", method="POST"))
                self.assertEqual(caught.exception.code, 405)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

