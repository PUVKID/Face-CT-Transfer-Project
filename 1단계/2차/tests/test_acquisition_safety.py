"""경로 이탈·미완료 전송·Range 무시를 실제 작은 HTTP 서버로 확인합니다."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from benchmark.acquisition import download_file, atomic_json
from benchmark.safety import safe_extract_tar, parse_lfw_pairs


class SafetyTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "실제 Windows 파일 공유 동작 검사입니다")
    def test_atomic_json_recovers_after_real_read_handle_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "상태.json"
            original = b'{"old": true}\n'
            target.write_bytes(original)
            handle = target.open("rb")
            blocked = threading.Event()
            attempts = []
            replace = Path.replace
            def observe(source, destination):
                attempts.append(time.monotonic())
                try:
                    return replace(source, destination)
                except PermissionError:
                    self.assertEqual(target.read_bytes(), original)
                    blocked.set()
                    raise
            def release():
                if blocked.wait(2):
                    time.sleep(0.08)
                    handle.close()
            thread = threading.Thread(target=release)
            thread.start()
            value = {"작업": "실제 Windows 점유 해제", "완료": True, "수": 0, "미측정": None}
            try:
                with patch.object(Path, "replace", new=observe):
                    atomic_json(target, value)
            finally:
                blocked.set()
                thread.join(timeout=2)
                handle.close()
            self.assertFalse(thread.is_alive())
            self.assertGreaterEqual(len(attempts), 2)
            self.assertLessEqual(len(attempts), 8)
            raw = target.read_bytes()
            self.assertEqual(json.loads(raw.decode("utf-8")), value)
            self.assertIn("실제 Windows 점유 해제".encode("utf-8"), raw)
            self.assertFalse(target.with_name(target.name + ".part").exists())

    @unittest.skipUnless(os.name == "nt", "실제 Windows 파일 공유 동작 검사입니다")
    def test_atomic_json_permanent_read_handle_is_bounded_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "상태.json"
            original = b'{"old": true}\n'
            target.write_bytes(original)
            value = {"새": "완전한 JSON", "미측정": None}
            attempts = []
            replace = Path.replace
            def observe(source, destination):
                attempts.append(time.monotonic())
                return replace(source, destination)
            started = time.monotonic()
            with target.open("rb"), patch.object(Path, "replace", new=observe):
                with self.assertRaises(PermissionError) as caught:
                    atomic_json(target, value)
            self.assertIn(caught.exception.winerror, {5, 32, 33})
            self.assertGreaterEqual(len(attempts), 2)
            self.assertLessEqual(len(attempts), 8)
            self.assertLess(time.monotonic() - started, 1.25)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(json.loads(target.with_name(target.name + ".part").read_text(encoding="utf-8")), value)

    @unittest.skipUnless(os.name == "nt", "Windows 일시 오류 코드 분기 검사입니다")
    def test_atomic_json_retries_only_known_windows_permission_errors(self):
        for code in [5, 32, 33]:
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "상태.json"
                original = b'{"old": true}\n'
                target.write_bytes(original)
                error = PermissionError(13, "제어된 Windows 일시 점유")
                error.winerror = code
                calls = []
                replace = Path.replace
                def intermittent(source, destination):
                    calls.append(source)
                    if len(calls) == 1:
                        self.assertEqual(target.read_bytes(), original)
                        raise error
                    return replace(source, destination)
                with patch.object(Path, "replace", new=intermittent):
                    atomic_json(target, {"새": code})
                self.assertEqual(len(calls), 2)
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"새": code})

    def test_atomic_json_other_errors_propagate_immediately_without_deleting_old_json(self):
        unknown = PermissionError(13, "제어된 다른 권한 오류")
        unknown.winerror = 87
        for error in [unknown, PermissionError(13, "Windows 코드 없는 권한 오류"), OSError(28, "제어된 공간 부족")]:
            with self.subTest(error=str(error)), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "상태.json"
                original = b'{"old": true}\n'
                target.write_bytes(original)
                with patch.object(Path, "replace", side_effect=error) as replace, patch("benchmark.acquisition.time.sleep") as sleep:
                    with self.assertRaises(type(error)) as caught:
                        atomic_json(target, {"새": "오류 전파"})
                self.assertIs(caught.exception, error)
                self.assertEqual(replace.call_count, 1)
                sleep.assert_not_called()
                self.assertEqual(target.read_bytes(), original)
                self.assertEqual(json.loads(target.with_name(target.name + ".part").read_text(encoding="utf-8")), {"새": "오류 전파"})

    def test_archive_suffix_collision_preserves_both_files_and_resume(self):
        entries = [("lfw/A/A_0001.jpg.extract.part", b"FIRST_PART"), ("lfw/A/A_0001.jpg", b"SECOND_IMAGE")]
        for ordered in [entries, entries[::-1]]:
            with self.subTest(order=[name for name, _ in ordered]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "collision.tar"
                with tarfile.open(source, "w") as archive:
                    for name, payload in ordered:
                        entry = tarfile.TarInfo(name)
                        entry.size = len(payload)
                        archive.addfile(entry, io.BytesIO(payload))
                for _ in range(2):
                    result = safe_extract_tar(source, root / "output", max_bytes=1024)
                    self.assertEqual(result["files"], 2)
                    for name, payload in entries:
                        self.assertEqual((root / "output" / name).read_bytes(), payload)

    def test_extract_preserves_existing_part_and_rejects_changed_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "safe.tar"
            output = root / "output"
            output.mkdir()
            part = output / "x.extract.part"
            part.write_bytes(b"BEFORE_EXTRACTION")
            with tarfile.open(source, "w") as archive:
                entry = tarfile.TarInfo("x")
                entry.size = 3
                archive.addfile(entry, io.BytesIO(b"abc"))
            for _ in range(2):
                safe_extract_tar(source, output, max_bytes=1024)
                self.assertEqual(part.read_bytes(), b"BEFORE_EXTRACTION")
                self.assertEqual((output / "x").read_bytes(), b"abc")
            (output / "x").write_bytes(b"FOREIGN")
            with self.assertRaises(ValueError):
                safe_extract_tar(source, output, max_bytes=1024)
            self.assertEqual(part.read_bytes(), b"BEFORE_EXTRACTION")
            self.assertEqual((output / "x").read_bytes(), b"FOREIGN")

    def test_archive_parent_path_and_symlink_never_write_outside(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "unsafe.tar"
                with tarfile.open(source, "w") as archive:
                    entry = tarfile.TarInfo(name)
                    entry.type = kind
                    entry.linkname = "../escape"
                    archive.addfile(entry)
                with self.assertRaises(ValueError):
                    safe_extract_tar(source, root / "output", max_bytes=1024)
                self.assertFalse((root / "escape").exists())

    def test_extract_is_safe_and_records_actual_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "safe.tar"
            with tarfile.open(source, "w") as archive:
                entry = tarfile.TarInfo("./lfw/person/name.jpg")
                entry.size = 3
                archive.addfile(entry, io.BytesIO(b"abc"))
            result = safe_extract_tar(source, root / "output", max_bytes=1024)
            self.assertEqual(result["files"], 1)
            self.assertEqual((root / "output/lfw/person/name.jpg").read_bytes(), b"abc")

    def test_lfw_paths_cannot_escape_dataset(self):
        lines = ["10\t300"] + ["../bad\t1\t2"] * 6000
        with self.assertRaises(ValueError):
            parse_lfw_pairs("\n".join(lines))

    def test_range_ignored_restarts_without_appending_corrupt_prefix(self):
        payload = b"0123456789" * 30
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "file.bin"
                destination.with_suffix(".bin.part").write_bytes(b"BAD")
                spec = {"id": "fixture", "name": "file.bin", "url": f"http://127.0.0.1:{server.server_port}/file", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                receipt = download_file(spec, destination, allowed_hosts={"127.0.0.1"}, minimum_free_bytes=0)
                self.assertEqual(destination.read_bytes(), payload)
                self.assertTrue(receipt["official_digest_verified"])
                self.assertFalse(destination.with_suffix(".bin.part").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

