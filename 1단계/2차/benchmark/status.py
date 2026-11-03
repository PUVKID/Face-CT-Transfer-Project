"""실측·미측정·중단을 구분하는 읽기 전용 상태를 제공합니다."""
import ctypes
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo
from .acquisition import now


LOG_IDS = {"acquisition", "evaluation", "server"}
LOG_RESPONSE_MAX_BYTES = 64 * 1024

# 초기값에서 타입을 추론하지 않고 실제 API의 nullable 값과 배열 행을 선언합니다.
METRIC_ROW_SCHEMA = {"benchmark": "str", "metric": "str", "value": "number", "unit": "str", "provenance": "str"}
SERIES_ROW_SCHEMA = {"time": "str", "value": "number"}
STATUS_SCHEMA = {
    "schema_version": "int", "updated_at": "str?", "stale": "bool", "status": "str",
    "project": {"name": "str", "revision": "str", "stage": "str"},
    "current_task": {"title": "str", "phase": "str", "detail": "str"},
    "progress": {"completed": "count?", "total": "count?", "unit": "str", "percent": "number?", "eta_seconds": "number?", "eta_is_estimate": "bool"},
    "model": {"name": "str", "training_data": "str", "source_url": "str", "version": "str", "checkpoint_verified": "bool", "license_status": "str", "verification_detail": "str", "reported_metrics": [METRIC_ROW_SCHEMA], "measured_metrics": [METRIC_ROW_SCHEMA]},
    "downloads": [{"id": "str", "name": "str", "status": "str", "received_bytes": "count?", "total_bytes": "count?", "verified": "bool", "expected_digest": "str?", "digest_algorithm": "str?", "bytes_per_second": "number?", "eta_seconds": "number?", "error": "str?", "official_digest_verified": "bool", "verification_basis": "str"}],
    "steps": [{"id": "str", "title": "str", "status": "str", "detail": "str"}],
    "techniques": [{"id": "str", "name": "str", "status": "str", "detail": "str"}],
    "evaluation": {"status": "str", "benchmark": "str?", "precision": "str", "images_completed": "count?", "images_total": "count?", "pairs_completed": "count?", "pairs_total": "count?", "fold_completed": "count?", "fold_total": "count?", "epoch": "count?", "detection_coverage": "coverage?", "failed_images": "count?", "metrics": ("nullable", {"accuracy": "number", "std": "number", "author_input_equivalence": "bool"})},
    "resources": {"disk_free_bytes": "count?", "gpu": ("nullable", {"name": "str", "utilization_percent": "number", "memory_used_bytes": "count", "memory_total_bytes": "count", "measured_at": "str"})},
    "processes": [{"role": "str", "pid": "pid", "alive": "bool?", "started_at": "str?"}],
    "series": {"throughput": [SERIES_ROW_SCHEMA], "gpu_utilization": [SERIES_ROW_SCHEMA], "metrics": [SERIES_ROW_SCHEMA]},
    "legacy": {"model": "str", "status": "str", "detail": "str"},
    "external_data": [{"name": "str", "status": "str?", "received_bytes": "count?", "verified_bytes": "count?", "total_bytes": "count?", "updated_at": "str?"}],
    "log_sources": [{"id": "str", "label": "str"}], "notices": ["str"]
}
OPTIONAL_ROW_FIELDS = {"status.downloads[]": {"official_digest_verified", "verification_basis"}}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _finite_number(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_status_shape(value, schema=STATUS_SCHEMA, path="status"):
    if isinstance(schema, tuple) and schema[0] == "nullable":
        if value is None:
            return
        schema = schema[1]
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError("상태 객체 형식이 다릅니다: " + path)
        # 배열 행의 필수 필드는 비어 있는 초기 배열과 무관하게 검사합니다.
        if path.endswith("[]") and set(schema) - OPTIONAL_ROW_FIELDS.get(path, set()) - set(value):
            raise ValueError("상태 배열 행에 필수 필드가 없습니다: " + path)
        for key, child in value.items():
            if key in schema:
                _validate_status_shape(child, schema[key], path + "." + key)
    elif isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError("상태 배열 형식이 다릅니다: " + path)
        for item in value:
            _validate_status_shape(item, schema[0], path + "[]")
    else:
        nullable = schema.endswith("?")
        kind = schema[:-1] if nullable else schema
        if value is None and nullable:
            return
        valid = (
            kind == "str" and isinstance(value, str)
            or kind == "bool" and isinstance(value, bool)
            or kind == "number" and _finite_number(value)
            or kind == "int" and type(value) is int and _finite_number(value)
            or kind == "count" and type(value) is int and value >= 0 and _finite_number(value)
            or kind == "pid" and type(value) is int and 0 < value <= 0xFFFFFFFF
            or kind == "coverage" and _finite_number(value) and 0 <= value <= 1
        )
        if not valid:
            raise ValueError("상태 필드의 명시 타입이 다릅니다: " + path)


def initial_status():
    return {
        "schema_version": 1, "updated_at": None, "stale": True, "status": "waiting",
        "project": {"name": "얼굴 모델의 실제 성능 재현", "revision": "1단계 2차", "stage": "얼굴 벤치마크 재현"},
        "current_task": {"title": "실제 자료 수신 준비", "phase": "acquisition", "detail": "직접 얼굴 정확도는 아직 측정하지 않았습니다."},
        "progress": {"completed": None, "total": None, "unit": "bytes", "percent": None, "eta_seconds": None, "eta_is_estimate": True},
        "model": {"name": "TopoFR-R200 Glint360K", "training_data": "Glint360K", "source_url": "https://github.com/DanJun6737/TopoFR", "version": "5960a33220968e6f9f74adcffb3f1a4ca31f9a0a", "checkpoint_verified": False, "license_status": "저자 공개 연구용 로컬 사용; 체크포인트 별도 라이선스 미확인·재배포 금지", "reported_metrics": [{"benchmark": "IJB-C", "metric": "TAR@FAR=1e-4", "value": 97.84, "unit": "%", "provenance": "저자 보고"}, {"benchmark": "IJB-C", "metric": "TAR@FAR=1e-5", "value": 96.71, "unit": "%", "provenance": "저자 보고"}, {"benchmark": "LFW", "metric": "고정 10fold 정확도", "value": 99.87, "unit": "%", "provenance": "저자 보고"}], "measured_metrics": []},
        "downloads": [],
        "steps": [{"id": "selection", "title": "외부 성능·공개 버전 대조", "status": "complete", "detail": "주지표 IJB-C FAR1e-4; 공개 버전의 원보고값과 실제 측정을 구분합니다."}, {"id": "acquisition", "title": "원본 자료·모델 획득", "status": "waiting", "detail": "길이·해시·원본 영수증을 검증합니다."}, {"id": "alignment", "title": "얼굴 검출·5점 정렬", "status": "waiting", "detail": "원본의 단순 resize를 정렬로 표현하지 않습니다."}, {"id": "evaluation", "title": "실제 얼굴 성능 재현", "status": "waiting", "detail": "독립 검증 뒤 부모가 실행합니다."}, {"id": "ct", "title": "CT 전이 후속 단계", "status": "paused", "detail": "새 얼굴 성능 재현을 먼저 수행합니다."}],
        "techniques": [{"id": "fp32", "name": "FP32 고정 평가", "status": "planned", "detail": "원본 가중치의 평가 모드·경사 계산 없음"}, {"id": "alignment", "name": "5점 정렬·RGB 112px", "status": "planned", "detail": "얼굴 영상에만 적용, CT에 얼굴 랜드마크 생성 금지"}, {"id": "flip", "name": "좌우반전 특징 합", "status": "planned", "detail": "원저자 IJB 평가 조건"}, {"id": "template", "name": "media 평균·template 집계", "status": "planned", "detail": "특징 norm·faceness 보존"}, {"id": "lfw", "name": "LFW 고정 6000쌍·10fold", "status": "planned", "detail": "임계값은 나머지 9fold에서 선택"}, {"id": "ct_train", "name": "CT 미세조정", "status": "unused", "detail": "이번 얼굴 재현 단계에서는 학습하지 않음"}],
        "evaluation": {"status": "waiting", "benchmark": None, "precision": "float32", "images_completed": None, "images_total": None, "pairs_completed": None, "pairs_total": None, "fold_completed": None, "fold_total": None, "epoch": None, "metrics": None, "detection_coverage": None, "failed_images": None},
        "resources": {"gpu": None, "disk_free_bytes": None}, "processes": [],
        "series": {"throughput": [], "gpu_utilization": [], "metrics": []},
        "legacy": {"model": "AdaFace", "status": "user_stopped", "detail": "사용자 요청으로 기존 CT 실행을 중단하고 체크포인트를 보존했습니다. 재개하지 않습니다."},
        "external_data": [], "log_sources": [{"id": name, "label": label} for name, label in [("acquisition", "자료 획득"), ("evaluation", "얼굴 평가"), ("server", "대시보드 서버")]],
        "notices": ["세계 전체 최강을 보장하지 않습니다. 실제 얼굴 정확도 재현은 현재 미측정입니다."]}


def process_alive(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name != "nt":
        return Path(f"/proc/{pid}").exists()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        return True if result == 258 else False if result == 0 else None
    finally:
        kernel.CloseHandle(handle)


def gpu_snapshot():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, encoding="utf-8", errors="strict", creationflags=0x08000000 if os.name == "nt" else 0)
        if result.returncode != 0:
            return None
        name, utilization, used, total = [item.strip() for item in result.stdout.split("\n")[0].split(",")]
        return {"name": name, "utilization_percent": float(utilization), "memory_used_bytes": int(used) * 1024 ** 2, "memory_total_bytes": int(total) * 1024 ** 2, "measured_at": now()}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def read_public_status(state_dir):
    root = Path(state_dir)
    status = initial_status()
    path = root / "status.json"
    try:
        if not path.exists():
            return status
        if path.stat().st_size > 512 * 1024:
            raise ValueError("상태 출력 상한 초과")
        value = json.loads(path.read_text(encoding="utf-8"))
        _validate_status_shape(value)
        # 직렬화할 수 없는 비유한 수와 잘못된 Unicode도 응답 전에 거부합니다.
        json_bytes(value)
        if value.get("schema_version") != 1 or value.get("status") not in {"waiting", "downloading", "verifying", "ready", "running", "complete", "paused", "failed"}:
            raise ValueError("상태 형식이 유효하지 않음")
        for key in status:
            if key in value:
                if isinstance(status[key], dict):
                    status[key].update(value[key])
                else:
                    status[key] = value[key]
        if status["updated_at"] is None:
            status["stale"] = True
        else:
            recorded = datetime.fromisoformat(status["updated_at"]).timestamp()
            status["stale"] = time.time() - recorded > 30 or recorded - time.time() > 5
        for process in status["processes"]:
            process["alive"] = process_alive(process.get("pid"))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        status = initial_status()
        status.update(status="failed", stale=True)
        status["current_task"] = {"title": "원본 상태 검증 실패", "phase": "failed", "detail": "유효하지 않은 상태 형식을 확인해 진행·측정을 표시하지 않습니다."}
        status["notices"].append("원본 상태 파일이 없거나 유효하지 않아 현재 진행을 확인할 수 없습니다.")
    return status


def redact(text):
    text = re.sub(r"(?i)\b(token|password|authorization|api[_-]?key|secret)\s*[:=]\s*[^\s,;]+", r"\1=[비공개]", text)
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[비공개]", text)


def log_tail(state_dir, source, limit):
    if source not in LOG_IDS or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
        raise ValueError("허용하지 않은 로그 또는 줄 수입니다")
    path = Path(state_dir) / (source + ".log")
    lines, truncated = [], False
    if path.is_file() and not path.is_symlink():
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            start = max(0, size - 60000)
            stream.seek(start)
            raw = stream.read(60000)
        if start:
            raw = raw.partition(b"\n")[2]
        lines = raw.decode("utf-8", errors="replace").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        lines = [redact(line.rstrip("\r")) for line in lines]
        truncated = start > 0 or len(lines) > limit
        lines = lines[-limit:]
    result = {"source": source, "lines": lines, "truncated": truncated, "updated_at": now()}
    # 원문 크기와 달리 JSON의 역슬래시·따옴표 escape는 응답을 키웁니다.
    # UTF-8 문자열을 자르지 않고 오래된 완전한 줄부터 제외합니다.
    while len(json_bytes(result)) > LOG_RESPONSE_MAX_BYTES:
        result["lines"].pop(0)
        result["truncated"] = True
    return result
