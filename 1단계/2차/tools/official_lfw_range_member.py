"""공식 단일 ZIP member 안전 획득의 비공개 프로토타입입니다."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, parse_qsl
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile
import zlib


class SafetyError(ValueError):
    pass


@dataclass(frozen=True)
class Member:
    archive_total: int
    name: str
    compressed: int
    uncompressed: int
    crc: int
    offset: int
    entries: int


@dataclass(frozen=True)
class Limits:
    max_requests: int = 64
    max_bytes: int = 48100725 + 4 * 1024 * 1024
    max_single: int = 1024 * 1024
    metadata_bytes: int = 4 * 1024 * 1024
    socket_timeout: float = 5
    deadline_seconds: float = 180


OFFICIAL = Member(13247758075, "faces_emore/lfw.bin", 48100725, 64738112, 0x2D7D0C73, 13199656087, 11)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise SafetyError("관측된 직접 URL에서 리다이렉트가 발생해 본문을 읽지 않고 중단합니다")


class RangeClient:
    def __init__(self, url, total, *, fixture=False, limits=Limits()):
        parsed = urlsplit(url)
        if parsed.username or parsed.password or parsed.fragment:
            raise SafetyError("URL의 사용자정보·fragment는 허용하지 않습니다")
        allowed = parsed.scheme == "http" and parsed.hostname == "127.0.0.1" if fixture else parsed.scheme == "https" and parsed.hostname == "drive.usercontent.google.com" and parsed.port in {None, 443}
        if not allowed or type(total) is not int or total <= 0:
            raise SafetyError("허용한 URL 호스트·HTTPS·archive 전체 크기와 다릅니다")
        self.url, self.total, self.limits = url, total, limits
        self.started = time.monotonic()
        self.received_bytes = self.metadata_received = 0
        self.requests = []
        self.opener = build_opener(NoRedirect())

    def remaining(self):
        remaining = self.limits.deadline_seconds - (time.monotonic() - self.started)
        if remaining <= 0:
            raise SafetyError("전체 검사 시간 예산을 초과했습니다")
        return remaining

    def read_range(self, start, size, *, kind="metadata"):
        self.remaining()
        if type(start) is not int or type(size) is not int or start < 0 or size <= 0 or start + size > self.total or size > self.limits.max_single:
            raise SafetyError("요청 범위·단일 수신 상한을 벗어났습니다")
        if len(self.requests) >= self.limits.max_requests or self.received_bytes + size + 1 > self.limits.max_bytes:
            raise SafetyError("전체 요청 수·수신 바이트 예산을 초과했습니다")
        if kind not in {"metadata", "member"} or kind == "metadata" and self.metadata_received + size + 1 > self.limits.metadata_bytes:
            raise SafetyError("메타데이터 수신 예산을 초과했습니다")
        last = start + size - 1
        record = {"start": start, "end": last, "kind": kind, "status": None, "received": 0}
        self.requests.append(record)
        headers = {"Range": f"bytes={start}-{last}", "Accept-Encoding": "identity", "User-Agent": "OfficialLFWBoundedPrototype/1.0"}
        try:
            with self.opener.open(Request(self.url, headers=headers), timeout=min(self.limits.socket_timeout, self.remaining())) as response:
                record["status"] = response.status
                if response.status != 206 or response.headers.get_all("Content-Range") != [f"bytes {start}-{last}/{self.total}"] or response.headers.get_all("Content-Length") != [str(size)]:
                    raise SafetyError("정확한 206·Content-Range·Content-Length와 다르므로 본문을 읽지 않습니다")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity" or "html" in response.headers.get("Content-Type", "").lower():
                    raise SafetyError("HTML·내용 인코딩 응답을 거부합니다")
                chunks = []
                while record["received"] <= size:
                    self.remaining()
                    chunk = response.read1(min(65536, size + 1 - record["received"]))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    record["received"] += len(chunk)
                    self.received_bytes += len(chunk)
                    if kind == "metadata":
                        self.metadata_received += len(chunk)
                    self.remaining()
                raw = b"".join(chunks)
                if len(raw) != size:
                    raise SafetyError("실제 수신 본문 길이가 요청과 다릅니다")
                record["sha256"] = hashlib.sha256(raw).hexdigest()
                return raw
        except HTTPError as error:
            record["status"] = error.code
            raise SafetyError(f"HTTP {error.code} 응답입니다. 할당량·권한 오류를 우회하거나 재시도하지 않습니다") from error
        except (URLError, OSError) as error:
            raise SafetyError("제한된 범위 전송이 실패했습니다. 원문 예외를 비공개로 보존합니다") from error


class MetadataRange(io.RawIOBase):
    def __init__(self, client):
        super().__init__()
        self.client, self.position = client, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence not in {io.SEEK_SET, io.SEEK_CUR, io.SEEK_END}:
            raise SafetyError("지원하지 않는 ZIP seek입니다")
        target = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.client.total}[whence] + offset
        if type(target) is not int or not 0 <= target <= self.client.total:
            raise SafetyError("ZIP 메타데이터 위치가 archive 밖입니다")
        self.position = target
        return target

    def read(self, size=-1):
        if size < 0:
            size = self.client.total - self.position
        size = min(size, self.client.total - self.position)
        if size == 0:
            return b""
        raw = self.client.read_range(self.position, size)
        self.position += len(raw)
        return raw


def safe_name(name):
    parts = PurePosixPath(name).parts
    if not name or name.startswith("/") or "\\" in name or ":" in name or ".." in parts or any(ord(character) < 32 for character in name):
        raise SafetyError("ZIP 항목의 경로를 거부했습니다")


def inspect_member(client, expected):
    if client.total != expected.archive_total:
        raise SafetyError("고정 archive 전체 크기가 다릅니다")
    with zipfile.ZipFile(MetadataRange(client), allowZip64=True) as archive:
        entries = archive.infolist()
        if len(entries) != expected.entries or not 1 <= len(entries) <= 64:
            raise SafetyError("고정 ZIP 항목 개수가 다릅니다")
        names = set()
        for item in entries:
            safe_name(item.filename)
            key = item.filename.casefold()
            mode = (item.external_attr >> 16) & 0xFFFF
            if key in names or item.orig_filename != item.filename or item.flag_bits & 0x41 or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise SafetyError("중복·잘린 이름·암호화·링크·특수 항목을 거부했습니다")
            if not 0 <= item.header_offset < archive.start_dir or item.compress_size < 0 or item.file_size < 0:
                raise SafetyError("ZIP central 항목 범위가 다릅니다")
            names.add(key)
        selected = [item for item in entries if item.filename == expected.name]
        if len(selected) != 1:
            raise SafetyError("정확한 단일 대상이 없습니다")
        item = selected[0]
        if item.is_dir() or stat.S_IFMT((item.external_attr >> 16) & 0xFFFF) not in {0, stat.S_IFREG} or item.external_attr & 0x10 or item.compress_type != 8 or item.flag_bits != 0 or (item.compress_size, item.file_size, item.CRC, item.header_offset) != (expected.compressed, expected.uncompressed, expected.crc, expected.offset):
            raise SafetyError("고정 member의 크기·CRC·offset·일반파일·flags가 다릅니다")
        following = min([entry.header_offset for entry in entries if entry.header_offset > item.header_offset] + [archive.start_dir])
    header = struct.unpack("<4s5H3I2H", client.read_range(expected.offset, 30))
    signature, version, flags, method, _, _, crc, compressed, uncompressed, name_size, extra_size = header
    if signature != b"PK\x03\x04" or flags != 0 or method != 8 or crc != expected.crc or name_size == 0:
        raise SafetyError("local header와 central의 형식·CRC·flags가 다릅니다")
    variable = client.read_range(expected.offset + 30, name_size + extra_size)
    if variable[:name_size].decode("cp437") != expected.name:
        raise SafetyError("local header 이름이 다릅니다")
    extra = variable[name_size:]
    fields = {}
    while extra:
        if len(extra) < 4:
            raise SafetyError("잘린 ZIP extra field입니다")
        identifier, length = struct.unpack("<HH", extra[:4])
        if identifier in fields or len(extra) < 4 + length:
            raise SafetyError("중복·잘린 ZIP extra field입니다")
        fields[identifier] = extra[4:4 + length]
        extra = extra[4 + length:]
    values = fields.get(1, b"")
    if uncompressed == 0xFFFFFFFF:
        if len(values) < 8: raise SafetyError("ZIP64 출력 크기가 없습니다")
        uncompressed, values = struct.unpack("<Q", values[:8])[0], values[8:]
    if compressed == 0xFFFFFFFF:
        if len(values) < 8: raise SafetyError("ZIP64 압축 크기가 없습니다")
        compressed = struct.unpack("<Q", values[:8])[0]
    start = expected.offset + 30 + name_size + extra_size
    if (compressed, uncompressed) != (expected.compressed, expected.uncompressed) or start + compressed > following:
        raise SafetyError("local 크기 또는 압축 본문 범위가 central과 다릅니다")
    return start


def inflate(source, output, expected_size, expected_crc):
    decompressor = zlib.decompressobj(-15)
    total = crc = 0
    digest = hashlib.sha256()
    with Path(source).open("rb") as incoming, Path(output).open("xb") as outgoing:
        for chunk in iter(lambda: incoming.read(65536), b""):
            pending = chunk
            while pending:
                decoded = decompressor.decompress(pending, min(65536, expected_size - total + 1))
                if total + len(decoded) > expected_size:
                    raise SafetyError("해제 출력 상한을 초과했습니다")
                outgoing.write(decoded)
                total += len(decoded)
                crc = zlib.crc32(decoded, crc)
                digest.update(decoded)
                if decompressor.unused_data:
                    raise SafetyError("DEFLATE EOF 뒤 남는 압축 데이터가 있습니다")
                pending = decompressor.unconsumed_tail
        outgoing.flush()
        os.fsync(outgoing.fileno())
        if not decompressor.eof or decompressor.unconsumed_tail or total != expected_size or crc != expected_crc:
            raise SafetyError("DEFLATE EOF·출력 크기·CRC 검증에 실패했습니다")
    return digest.hexdigest()


def commit(source, destination):
    if os.name == "nt":
        os.rename(source, destination)
    else:
        os.link(source, destination)
        os.unlink(source)


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def acquire(client, member, destination):
    destination = Path(destination)
    paths = {suffix: destination.with_name(destination.name + suffix) for suffix in ["", ".part", ".deflate.part", ".receipt.json", ".receipt.json.part", ".failure.json"]}
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise SafetyError("기존 결과·부분·영수증·실패 원문을 덮어쓰지 않습니다")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        start = inspect_member(client, member)
        compressed_digest = hashlib.sha256()
        with paths[".deflate.part"].open("xb") as stream:
            for offset in range(0, member.compressed, client.limits.max_single):
                size = min(client.limits.max_single, member.compressed - offset)
                raw = client.read_range(start + offset, size, kind="member")
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
                compressed_digest.update(raw)
        if paths[".deflate.part"].stat().st_size != member.compressed:
            raise SafetyError("전체 압축 수신 크기가 다릅니다")
        client.remaining()
        digest = inflate(paths[".deflate.part"], paths[".part"], member.uncompressed, member.crc)
        client.remaining()
        receipt = {"status": "complete", "member": member.name, "archive_bytes": member.archive_total, "compressed_bytes": member.compressed, "bytes": member.uncompressed, "crc32": f"{member.crc:08x}", "sha256": digest, "compressed_sha256": compressed_digest.hexdigest(), "official_sha256_published": False, "hash_basis": "고정 ZIP CRC/크기 대조 후 최초 로컬 SHA 고정", "completed_at": datetime.now().astimezone().isoformat(), "requests": client.requests, "received_bytes": client.received_bytes, "unsafe_deserialization_runs": 0}
        write_json_new(paths[".receipt.json.part"], receipt)
        commit(paths[".part"], destination)
        commit(paths[".receipt.json.part"], paths[".receipt.json"])
        return receipt
    except Exception as error:
        failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "requests": client.requests, "received_bytes": client.received_bytes, "partial_files": {suffix: path.stat().st_size for suffix, path in paths.items() if path.is_file()}, "completed": False}
        try:
            write_json_new(paths[".failure.json"], failure)
        except Exception as record_error:
            raise ExceptionGroup("원본 실패와 비공개 실패 저장이 모두 실패했습니다", [error, record_error]) from error
        raise


def load_confirmed_source(research):
    research = Path(research)
    expected = {"official_zip_metadata_probe.private.json": "ff19439defdc31da11112ecd8308d469731d89491773337a26227ec022b879b8", "official_training_confirmation_form.private.json": "30d779ceb7c4b327f8e79435473da72d82b9e75d46721c4174011bee38798d2a", "official_training_confirmed_range.http.private.json": "2f7d6b7195813d82657345bb4c450e1831844c1de6c3438f3988d719ba0a2a91"}
    values = {}
    for name, digest in expected.items():
        raw = (research / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise SafetyError("고정 공식 출처 영수증의 SHA가 다릅니다")
        values[name] = json.loads(raw)
    response = values["official_training_confirmed_range.http.private.json"]
    form = values["official_training_confirmation_form.private.json"]["forms"][0]
    url = response["url"]
    parsed, action = urlsplit(url), urlsplit(form["attributes"]["action"])
    hidden = {item["name"]: item["value"] for item in form["inputs"] if item.get("type") == "hidden"}
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if response["status"] != 206 or response["final_url"] != url or form["attributes"]["method"].lower() != "get" or (parsed.scheme, parsed.netloc, parsed.path) != (action.scheme, action.netloc, action.path) or len(query) != len(hidden) or dict(query) != hidden:
        raise SafetyError("실제 관측된 정상 form/확인 URL과 다릅니다")
    RangeClient(url, OFFICIAL.archive_total)
    return url, OFFICIAL
