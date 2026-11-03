"""합성 ZIP64·로컬 HTTP만 사용하며 실제 자료와 pickle을 읽지 않습니다."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import random
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
import zlib
from tools import official_lfw_range_member as subject
from tools import acquire_official_lfw as operator


def fixture_zip(name="faces_emore/lfw.bin", payload=b"synthetic-only" * 200, mode=0o100644):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=8) as archive:
        info = zipfile.ZipInfo(name)
        info.compress_type = 8
        info.create_system = 3
        info.external_attr = mode << 16
        with archive.open(info, "w", force_zip64=True) as output:
            output.write(payload)
    raw = stream.getvalue()
    end = struct.unpack("<4s4H2LH", raw[-22:])
    prefix = raw[:-22]
    # 작은 합성 자료에도 실제 ZIP64 EOCD/locator를 넣습니다.
    raw = prefix + struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, end[3], end[4], end[5], end[6]) + struct.pack("<4sLQL", b"PK\x06\x07", 0, len(prefix), 1) + struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 65535, 65535, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.infolist()[0]
    member = subject.Member(len(raw), name, info.compress_size, info.file_size, info.CRC, info.header_offset, 1)
    return raw, member, payload


@contextmanager
def server(raw, mode="good"):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Range"))
            first, last = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            part = raw[first:last + 1]
            if mode == "403":
                self.send_response(403); self.end_headers(); return
            if mode == "redirect":
                self.send_response(302); self.send_header("Location", "https://example.com/forbidden"); self.end_headers(); return
            self.send_response(200 if mode in {"200", "200_large"} else 206)
            self.send_header("Content-Type", "text/html" if mode == "html" else "application/octet-stream")
            self.send_header("Content-Length", str(1024 ** 3 if mode == "200_large" else len(part) + (1 if mode == "length" else 0)))
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(raw) + (1 if mode == 'range' else 0)}")
            self.end_headers()
            try:
                if mode == "slow" and first > 0:
                    for byte in part:
                        self.wfile.write(bytes([byte])); self.wfile.flush(); time.sleep(0.05)
                else:
                    self.wfile.write(part[:-1] if mode == "short" else part)
            except (BrokenPipeError, ConnectionResetError):
                # 거부된 합성 응답에서 client가 연결을 닫는 정상 검사 경로입니다.
                return
        def log_message(self, *args):
            return
    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True); thread.start()
    try:
        yield f"http://127.0.0.1:{instance.server_port}/fixture", requests
    finally:
        instance.shutdown(); instance.server_close(); thread.join(timeout=2)


class MemberTests(unittest.TestCase):
    def test_target_with_dos_directory_attribute_is_not_a_regular_file(self):
        raw, member, _ = fixture_zip(mode=0)
        raw = bytearray(raw)
        offset = raw.index(b"PK\x01\x02")
        raw[offset + 38:offset + 42] = struct.pack("<I", 0x10)
        with server(bytes(raw)) as (url, _):
            with self.assertRaises(subject.SafetyError):
                subject.inspect_member(subject.RangeClient(url, len(raw), fixture=True), member)

    def test_target_with_directory_mode_without_slash_is_not_a_regular_file(self):
        raw, member, _ = fixture_zip(mode=0o40755)
        with server(raw) as (url, _):
            with self.assertRaises(subject.SafetyError):
                subject.inspect_member(subject.RangeClient(url, len(raw), fixture=True), member)

    def test_pinned_official_metadata_loads_offline_without_any_network(self):
        research = Path(__file__).resolve().parents[2] / "work/stage1_revision2/aligned_lfw_source_research"
        with patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("실제 네트워크 금지")):
            url, member = subject.load_confirmed_source(research)
        self.assertEqual(member, subject.OFFICIAL)
        self.assertTrue(url.startswith("https://drive.usercontent.google.com/"))

    def test_zip64_single_member_roundtrip_receipt_and_no_pickle(self):
        raw, member, payload = fixture_zip()
        with server(raw) as (url, requests), tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "lfw.bin"
            client = subject.RangeClient(url, len(raw), fixture=True)
            receipt = subject.acquire(client, member, target)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(receipt["crc32"], f"{zlib.crc32(payload):08x}")
            self.assertFalse(receipt["official_sha256_published"])
            self.assertEqual(json.loads(target.with_name(target.name + ".receipt.json").read_text(encoding="utf-8")), receipt)
            self.assertLessEqual(len(requests), 64)

    def test_bad_http_contracts_stop_before_complete_or_large_body_read(self):
        raw, member, _ = fixture_zip()
        for mode in ["200", "200_large", "html", "length", "range", "short", "403", "redirect"]:
            with self.subTest(mode=mode), server(raw, mode) as (url, requests):
                client = subject.RangeClient(url, len(raw), fixture=True)
                with self.assertRaises(subject.SafetyError):
                    client.read_range(0, 10)
                self.assertEqual(len(requests), 1)
                self.assertLessEqual(client.received_bytes, 11)
                if mode != "short": self.assertEqual(client.received_bytes, 0)

    def test_production_url_rejects_plaintext_userinfo_foreign_hosts_and_fragments(self):
        for url in ["http://drive.usercontent.google.com/x", "https://example.com/x", "https://user@drive.usercontent.google.com/x", "https://drive.usercontent.google.com/x#fragment", "https://drive.usercontent.google.com:444/x"]:
            with self.subTest(url=url), self.assertRaises(subject.SafetyError):
                subject.RangeClient(url, 100)

    def test_request_byte_range_and_deadline_budgets_are_checked(self):
        raw, _, _ = fixture_zip()
        with server(raw) as (url, requests):
            client = subject.RangeClient(url, len(raw), fixture=True, limits=replace(subject.Limits(), max_requests=1))
            client.read_range(0, 1)
            with self.assertRaises(subject.SafetyError): client.read_range(1, 1)
            self.assertEqual(len(requests), 1)
            for start, size in [(-1, 1), (0, len(raw) + 1), (0, 1048577)]:
                with self.assertRaises(subject.SafetyError): client.read_range(start, size)
            client = subject.RangeClient(url, len(raw), fixture=True, limits=replace(subject.Limits(), max_bytes=1))
            with self.assertRaises(subject.SafetyError): client.read_range(0, 2)
            client = subject.RangeClient(url, len(raw), fixture=True, limits=replace(subject.Limits(), metadata_bytes=1))
            with self.assertRaises(subject.SafetyError): client.read_range(0, 2)
            client = subject.RangeClient(url, len(raw), fixture=True)
            with patch("tools.official_lfw_range_member.time.monotonic", return_value=client.started + 181):
                with self.assertRaises(subject.SafetyError): client.read_range(0, 1)

    def test_unsafe_and_symlink_member_metadata_is_rejected(self):
        for name, mode in [("../escape", 0o100644), ("/absolute", 0o100644), ("faces_emore\\lfw.bin", 0o100644), ("faces_emore/lfw.bin", 0o120777)]:
            raw, member, _ = fixture_zip(name=name, mode=mode)
            with self.subTest(name=name, mode=mode), server(raw) as (url, _):
                with self.assertRaises(subject.SafetyError): subject.inspect_member(subject.RangeClient(url, len(raw), fixture=True), member)

    def test_duplicate_names_encrypted_flags_and_local_mismatch_are_rejected(self):
        raw, member, _ = fixture_zip()
        bad = bytearray(raw)
        bad[8:10] = struct.pack("<H", 0)
        with server(bytes(bad)) as (url, _), self.assertRaises(subject.SafetyError):
            subject.inspect_member(subject.RangeClient(url, len(raw), fixture=True), member)
        for change in ["encrypted", "duplicate"]:
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w", compression=8) as archive:
                archive.writestr("faces_emore/lfw.bin", b"one")
                if change == "duplicate": archive.writestr("faces_emore/lfw.bin", b"two")
            raw = bytearray(stream.getvalue())
            if change == "encrypted":
                central = raw.index(b"PK\x01\x02"); raw[central + 8:central + 10] = struct.pack("<H", 1)
            with server(bytes(raw)) as (url, _), self.assertRaises(subject.SafetyError):
                subject.inspect_member(subject.RangeClient(url, len(raw), fixture=True), replace(member, archive_total=len(raw), entries=2 if change == "duplicate" else 1))

    def test_crc_eof_trailing_bytes_and_zipbomb_are_rejected_with_bounded_output(self):
        payload = b"x" * 100000
        compressor = zlib.compressobj(wbits=-15); compressed = compressor.compress(payload) + compressor.flush()
        for data, size, crc in [(compressed, len(payload), 0), (compressed[:-1], len(payload), zlib.crc32(payload)), (compressed + b"EXTRA", len(payload), zlib.crc32(payload)), (compressed, 10, zlib.crc32(payload))]:
            with self.subTest(size=size, compressed=len(data)), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "compressed.part"; output = Path(temporary) / "output.part"; source.write_bytes(data)
                with self.assertRaises(subject.SafetyError): subject.inflate(source, output, size, crc)
                self.assertLessEqual(output.stat().st_size, size)

    def test_existing_files_and_uncommitted_parts_are_never_overwritten(self):
        raw, member, _ = fixture_zip()
        for suffix in ["", ".part", ".deflate.part", ".receipt.json"]:
            with server(raw) as (url, requests), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "lfw.bin"; old = target.with_name(target.name + suffix); old.write_bytes(b"PRESERVE")
                with self.assertRaises(subject.SafetyError): subject.acquire(subject.RangeClient(url, len(raw), fixture=True), member, target)
                self.assertEqual(old.read_bytes(), b"PRESERVE")
                self.assertEqual(requests, [])

    def test_mid_transfer_failure_preserves_original_error_and_partial_without_receipt(self):
        raw, member, _ = fixture_zip(payload=random.Random(0).randbytes(140000))
        with server(raw) as (url, _), tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "lfw.bin"; client = subject.RangeClient(url, len(raw), fixture=True, limits=replace(subject.Limits(), max_single=65536))
            original = client.read_range
            member_calls = []
            def interrupted(start, size, *, kind="metadata"):
                if kind == "member":
                    member_calls.append(start)
                    if len(member_calls) == 2: raise RuntimeError("제어된 본문 전송 중단")
                return original(start, size, kind=kind)
            with patch.object(client, "read_range", side_effect=interrupted), self.assertRaisesRegex(RuntimeError, "제어된 본문 전송 중단"):
                subject.acquire(client, member, target)
            self.assertTrue(target.with_name(target.name + ".deflate.part").exists())
            self.assertEqual(target.with_name(target.name + ".deflate.part").stat().st_size, 65536)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name(target.name + ".receipt.json").exists())
            failure = json.loads(target.with_name(target.name + ".failure.json").read_text(encoding="utf-8"))
            self.assertIn("제어된 본문 전송 중단", failure["traceback"])


@unittest.skipUnless(os.name == "nt", "실제 Windows 단일 worker와 Job Object 경계 검사입니다")
class WatchdogTests(unittest.TestCase):
    def test_operator_plan_has_no_network_and_rejects_other_output(self):
        expected = Path(__file__).resolve().parents[2] / "data/face_benchmarks/raw/lfw_official_ms1mv2.bin"
        output = io.StringIO()
        with patch.object(operator, "supervise", side_effect=AssertionError("실제 실행 금지")), redirect_stdout(output):
            self.assertEqual(operator.main(["--source-evidence-dir", "unused", "--output", str(expected)]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "plan_only")
        self.assertEqual((value["soft_seconds"], value["hard_seconds"], value["network_requests"]), (180, 190, 0))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            operator.main(["--source-evidence-dir", "unused", "--output", str(expected.with_name("forbidden.bin"))])
        self.assertEqual(caught.exception.code, 2)

    def test_normal_base_worker_identity_and_process_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = "import sys,os,json; sys.stdin.readline(); print(json.dumps({'pid':os.getpid(),'executable':sys.executable}),flush=True)"
            result = operator.supervise([sys._base_executable, "-B", "-X", "utf8", "-c", code], Path(__file__).resolve().parents[1], Path(temporary) / "run", hard_seconds=3)
            self.assertEqual(result["status"], "worker_exited")
            self.assertEqual(result["worker_returncode"], 0)
            self.assertTrue(result["stop_verified"])
            self.assertEqual(result["job_active_processes_after"], 0)
            actual = json.loads(Path(result["log_path"]).read_text(encoding="utf-8").strip())
            self.assertEqual(actual["pid"], result["worker_pid"])
            self.assertEqual(Path(actual["executable"]).resolve(), Path(sys._base_executable).resolve())
            self.assertFalse(result["acquisition_completed"])

    def test_actual_slow_http_hard_limit_stops_only_owned_worker_and_keeps_partial(self):
        with tempfile.TemporaryDirectory() as temporary, server(b"KEEP" + b"x" * 200, "slow") as (url, requests):
            root = Path(temporary); partial = root / "partial.deflate.part"
            code = "import sys,os;sys.stdin.readline();from pathlib import Path;from tools.official_lfw_range_member import RangeClient;" + f"client=RangeClient({url!r},204,fixture=True);output=Path({str(partial)!r}).open('xb');output.write(client.read_range(0,4,kind='member'));output.flush();os.fsync(output.fileno());client.read_range(4,100,kind='member')"
            result = operator.supervise([sys._base_executable, "-B", "-X", "utf8", "-c", code], Path(__file__).resolve().parents[1], root / "run", hard_seconds=1)
            self.assertEqual(result["status"], "hard_timeout")
            self.assertTrue(result["stop_verified"])
            self.assertEqual(result["job_active_processes_after"], 0)
            self.assertEqual(partial.read_bytes(), b"KEEP")
            self.assertLess(result["elapsed_seconds"], 4)
            self.assertEqual(len(requests), 2)
            receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
            self.assertFalse(receipt["acquisition_completed"])
            self.assertEqual(receipt["status"], "hard_timeout")
            self.assertEqual(operator.SOFT_DEADLINE_SECONDS, 180)
            self.assertEqual(operator.HARD_DEADLINE_SECONDS, 190)

    def test_worker_error_preserves_raw_log_and_parent_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = "import sys;sys.stdin.readline();raise RuntimeError('제어된 worker 실패 원문')"
            result = operator.supervise([sys._base_executable, "-B", "-X", "utf8", "-c", code], Path(__file__).resolve().parents[1], Path(temporary) / "run", hard_seconds=3)
            self.assertEqual(result["status"], "worker_failed")
            self.assertNotEqual(result["worker_returncode"], 0)
            self.assertTrue(result["stop_verified"])
            self.assertIn("제어된 worker 실패 원문", Path(result["log_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
