"""고정 공개 URL만 내려받고 Range·길이·해시를 검증합니다."""
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo
from .safety import digest_file


ALLOWED_HOSTS = frozenset({"drive.google.com", "drive.usercontent.google.com", "ndownloader.figshare.com", "figshare.com", "s3-eu-west-1.amazonaws.com", "raw.githubusercontent.com", "media.githubusercontent.com"})


def now():
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    # Windows 읽기 핸들의 일시 점유만 짧게 기다리며 기존 파일은 삭제하지 않습니다.
    deadline = time.monotonic() + 0.75
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            remaining = deadline - time.monotonic()
            if os.name != "nt" or getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 7 or remaining <= 0:
                raise
            time.sleep(min(0.05, remaining))
            if time.monotonic() >= deadline:
                raise


class RestrictedRedirect(HTTPRedirectHandler):
    def __init__(self, allowed_hosts):
        self.allowed_hosts = set(allowed_hosts)

    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlparse(newurl).hostname not in self.allowed_hosts or urlparse(newurl).scheme != "https":
            raise ValueError("허용하지 않은 다운로드 리디렉션입니다")
        return super().redirect_request(request, fp, code, message, headers, newurl)


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.method, self.fields = None, None, {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "form":
            self.action, self.method = values.get("action"), values.get("method", "get").lower()
        if tag == "input" and values.get("name"):
            self.fields[values["name"]] = values.get("value", "")


def _open_content(opener, url, offset, expected_id=None):
    headers = {"User-Agent": "FaceCTBenchmark-Research/1.0", "Accept-Encoding": "identity", "Range": f"bytes={offset}-"}
    response = opener.open(Request(url, headers=headers), timeout=30)
    if "html" not in response.headers.get("Content-Type", ""):
        return response
    with response:
        raw = response.read(131073)
    if len(raw) > 131072 or not expected_id:
        raise ValueError("예상하지 못한 HTML 응답이며 파일로 저장하지 않습니다")
    form = DownloadForm()
    form.feed(raw.decode("utf-8", errors="strict"))
    if form.method != "get" or urlparse(form.action or "").scheme != "https" or urlparse(form.action or "").hostname != "drive.usercontent.google.com" or form.fields.get("id") != expected_id or not set(form.fields).issubset({"id", "export", "confirm", "uuid"}):
        raise ValueError("인증 없는 공식 공개 다운로드 GET 양식이 아닙니다")
    return opener.open(Request(form.action + "?" + urlencode(form.fields), headers=headers), timeout=30)


def _check_git_blob(path, expected):
    digest = hashlib.sha1()
    digest.update(f"blob {Path(path).stat().st_size}\0".encode("ascii"))
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("원본 Git blob SHA-1과 소스 파일이 다릅니다")


def download_file(spec, destination, progress=None, *, allowed_hosts=ALLOWED_HOSTS, minimum_free_bytes=400 * 1024 ** 3, retries=3, failure_log=None):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(spec["url"])
    if parsed.hostname not in allowed_hosts or parsed.scheme != "https" and parsed.hostname != "127.0.0.1":
        raise ValueError("고정 다운로드 도메인이 아닙니다")
    size = spec["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("공식 파일 크기가 유효하지 않습니다")
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    if destination.exists():
        actual = digest_file(destination)
        if destination.stat().st_size != size or spec.get("sha256") and actual != spec["sha256"]:
            raise ValueError("기존 원본이 고정 크기·공식 해시와 다릅니다")
        if spec.get("git_blob_sha1"):
            _check_git_blob(destination, spec["git_blob_sha1"])
        if not spec.get("sha256") and not spec.get("git_blob_sha1"):
            if not receipt_path.exists() or json.loads(receipt_path.read_text(encoding="utf-8")).get("sha256") != actual:
                raise ValueError("공식 해시 없는 기존 파일에 최초 획득 영수증이 없습니다")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {"sha256": actual, "size": size, "official_digest_verified": True}
        if progress:
            progress({"status": "complete", "received_bytes": size, "total_bytes": size, "verified": True, "bytes_per_second": None, "eta_seconds": 0, "verification_basis": receipt.get("verification_basis", "official_sha256")})
        return receipt
    part = destination.with_name(destination.name + ".part")
    opener = build_opener(RestrictedRedirect(allowed_hosts))
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if offset > size:
            raise ValueError("부분 파일이 원본 전체보다 큽니다")
        if shutil.disk_usage(destination.parent).free < minimum_free_bytes + size - offset:
            raise OSError("원본 수신 후 필요한 디스크 여유가 부족합니다")
        try:
            if offset < size:
                with _open_content(opener, spec["url"], offset, spec.get("drive_id")) as response:
                    if "html" in response.headers.get("Content-Type", ""):
                        raise ValueError("파일 대신 HTML을 받았습니다")
                    if response.status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if not match or int(match[1]) != offset or int(match[2]) != size - 1 or int(match[3]) != size:
                            raise ValueError("Range 응답 시작점·전체 크기가 다릅니다")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ValueError("지원하지 않는 다운로드 HTTP 상태입니다")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) != size - offset:
                        raise ValueError("HTTP 길이가 고정 원본 길이와 다릅니다")
                    started, last = time.monotonic(), 0.0
                    start_offset = offset
                    with part.open("ab" if offset else "wb") as output:
                        while offset < size:
                            chunk = response.read(min(1024 * 1024, size - offset))
                            if not chunk:
                                raise OSError("고정 원본 길이 전에 전송이 끝났습니다")
                            output.write(chunk)
                            offset += len(chunk)
                            elapsed = time.monotonic() - started
                            if progress and elapsed - last >= 1:
                                speed = (offset - start_offset) / elapsed
                                progress({"status": "downloading", "received_bytes": offset, "total_bytes": size, "verified": False, "bytes_per_second": speed, "eta_seconds": (size - offset) / speed if speed else None})
                                last = elapsed
                        output.flush()
                        os.fsync(output.fileno())
            if progress:
                progress({"status": "verifying", "received_bytes": size, "total_bytes": size, "verified": False})
            actual = digest_file(part)
            if spec.get("sha256") and actual != spec["sha256"]:
                raise ValueError("다운로드 파일이 고정 공식 SHA-256과 다릅니다")
            if spec.get("git_blob_sha1"):
                _check_git_blob(part, spec["git_blob_sha1"])
            basis = "official_sha256" if spec.get("sha256") else "official_git_blob_sha1" if spec.get("git_blob_sha1") else "official_size_and_first_acquisition_sha256_no_published_digest"
            receipt = {"id": spec["id"], "source_url": spec["url"], "size": size, "sha256": actual, "verified_at": now(), "official_digest_verified": bool(spec.get("sha256") or spec.get("git_blob_sha1")), "verification_basis": basis}
            part.replace(destination)
            atomic_json(receipt_path, receipt)
            if progress:
                progress({"status": "complete", "received_bytes": size, "total_bytes": size, "verified": True, "bytes_per_second": None, "eta_seconds": 0, "verification_basis": receipt["verification_basis"]})
            return receipt
        except Exception as error:
            if failure_log:
                failure_log({"time_kst": now(), "id": spec["id"], "url": spec["url"], "attempt": attempt + 1, "error_type": type(error).__name__, "error_raw": str(error), "received_bytes": part.stat().st_size if part.exists() else 0})
            if attempt + 1 == retries or isinstance(error, ValueError):
                raise
            time.sleep(min(2 ** attempt * 2, 10))
    raise RuntimeError("유효한 다운로드 종료 상태가 없습니다")
