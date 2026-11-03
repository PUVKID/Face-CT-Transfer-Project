"""고정한 공개 자료만 획득하며 장기 얼굴 GPU 평가는 실행하지 않습니다."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.acquisition import atomic_json, download_file, now
from benchmark.safety import digest_file, parse_lfw_pairs
from benchmark.status import initial_status, gpu_snapshot, process_alive

STAGE = ROOT.parent
WORK = STAGE / "work/stage1_revision2"
MANIFEST = ROOT / "benchmark/sources.json"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    lock_path = WORK / "acquisition.lock"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        if process_alive(previous.get("pid")) is not False:
            raise RuntimeError("이전 획득 프로세스의 종료를 확정할 수 없어 중복 실행하지 않습니다")
        archived = WORK / ("acquisition.lock.stale_" + str(time.time_ns()) + ".json")
        lock_path.replace(archived)
    with lock_path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_at": now()}, stream)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    specs = manifest["files"]
    state = initial_status()
    state["status"] = "downloading"
    state["processes"] = [{"role": "acquisition", "pid": os.getpid(), "alive": True, "started_at": now()}]
    state["steps"][1]["status"] = "active"
    rows = {spec["id"]: {"id": spec["id"], "name": spec["name"], "status": "pending", "received_bytes": 0, "total_bytes": spec["size"], "verified": False, "expected_digest": spec.get("sha256") or spec.get("git_blob_sha1"), "digest_algorithm": "sha256" if spec.get("sha256") else "git_blob_sha1" if spec.get("git_blob_sha1") else None, "bytes_per_second": None, "eta_seconds": None, "error": None, "verification_basis": "not_verified", "official_digest_verified": False} for spec in specs}
    mutex = threading.Lock()
    stop = threading.Event()

    def log(message):
        with (WORK / "acquisition.log").open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"{now()} {message}\n")

    def failure(event):
        event.update(session=os.environ.get("FACE_CT_SESSION"), task=os.environ.get("FACE_CT_TASK"), argv=sys.argv, cwd=os.getcwd(), changes_before=True, repair_used=os.environ.get("FACE_CT_REPAIR_USED"), input_fix_used=os.environ.get("FACE_CT_INPUT_FIX_USED"))
        with (WORK / "acquisition_failures.jsonl").open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save():
        state["updated_at"] = now()
        state["downloads"] = list(rows.values())
        received = sum(item["received_bytes"] for item in rows.values())
        total = sum(item["total_bytes"] for item in rows.values())
        speed = sum(item.get("bytes_per_second") or 0 for item in rows.values() if item["status"] == "downloading")
        state["progress"] = {"completed": received, "total": total, "unit": "bytes", "percent": received / total * 100, "eta_seconds": (total - received) / speed if speed > 0 else None, "eta_is_estimate": True}
        state["resources"]["disk_free_bytes"] = shutil.disk_usage(STAGE).free
        if speed:
            state["series"]["throughput"].append({"time": state["updated_at"], "value": speed})
            state["series"]["throughput"] = state["series"]["throughput"][-500:]
        luna = STAGE / "work/private/data_acquisition/progress.json"
        if luna.exists():
            try:
                value = json.loads(luna.read_text(encoding="utf-8"))
                state["external_data"] = [{"name": "LUNA25 기존 수집", "status": value.get("status"), "received_bytes": value.get("downloaded_bytes"), "verified_bytes": value.get("verified_bytes"), "total_bytes": value.get("total_bytes"), "updated_at": value.get("updated_kst")}]
            except (OSError, ValueError):
                state["external_data"] = [{"name": "LUNA25 기존 수집", "status": "unknown", "received_bytes": None, "verified_bytes": None, "total_bytes": None, "updated_at": None}]
        atomic_json(WORK / "status.json", state)

    def update(identity, event):
        with mutex:
            rows[identity].update(event)
            state["current_task"] = {"title": "공개 모델·얼굴 자료 수신 및 검증", "phase": "acquisition", "detail": "수신 바이트와 공식 해시 대조·최초 로컬 해시 고정을 구분합니다."}
            save()

    def heartbeat():
        while not stop.wait(5):
            measured = gpu_snapshot()
            with mutex:
                state["resources"]["gpu"] = measured
                if measured:
                    state["series"]["gpu_utilization"].append({"time": measured["measured_at"], "value": measured["utilization_percent"]})
                    state["series"]["gpu_utilization"] = state["series"]["gpu_utilization"][-500:]
                save()

    def acquire(spec):
        destination = (STAGE / spec["target"]).resolve()
        allowed = [STAGE / "data/face_benchmarks", STAGE / "models/topofr"]
        if not any(destination.is_relative_to(path.resolve()) for path in allowed):
            raise ValueError("고정 자료·모델 경계 밖의 획득 대상입니다")
        log("수신 시작: " + spec["name"])
        receipt = download_file(spec, destination, lambda event: update(spec["id"], event), failure_log=failure)
        update(spec["id"], {"official_digest_verified": receipt["official_digest_verified"], "verification_basis": receipt["verification_basis"]})
        log("길이·무결성 확인: " + spec["name"] + (" / 공식 digest 대조" if receipt["official_digest_verified"] else " / 최초 로컬 SHA 고정·공식 체크섬 미제공"))
        return receipt

    # 이전 조사에서 이미 받은 정확한 pair 원문은 재다운로드하지 않습니다.
    pairs = next(spec for spec in specs if spec["id"] == "lfw_pairs")
    target = STAGE / pairs["target"]
    existing_pair = STAGE / "work/private/face_benchmark_reassessment/figshare_lfw_pairs.json"
    if not target.exists() and existing_pair.exists():
        record = json.loads(existing_pair.read_text(encoding="utf-8"))
        raw = record["body"].encode("utf-8")
        import hashlib
        if hashlib.sha256(raw).hexdigest() != pairs["sha256"]:
            raise ValueError("재사용 pair 바이트가 공식 SHA와 다릅니다")
        parse_lfw_pairs(record["body"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        atomic_json(target.with_name(target.name + ".receipt.json"), {"id": pairs["id"], "size": len(raw), "sha256": pairs["sha256"], "official_digest_verified": True, "verification_basis": "official_sha256", "verified_at": now(), "source_url": pairs["url"], "reused_existing_metadata": True})
    with mutex:
        save()
    atomic_json(WORK / "acquisition_process.json", {"pid": os.getpid(), "started_at": state["processes"][0]["started_at"], "argv": sys.argv, "cwd": os.getcwd(), "source_manifest_sha256": digest_file(MANIFEST), "status": "running"})
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    receipts, failed = [], []
    try:
        # 코드·pair 작은 원본 검증 뒤 대용량 파일 두 개까지만 동시 수신합니다.
        for spec in specs:
            if spec["id"].startswith("vendor_") or spec["id"] == "lfw_pairs":
                receipts.append(acquire(spec))
        large = [spec for spec in specs if not spec["id"].startswith("vendor_") and spec["id"] != "lfw_pairs"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(acquire, spec): spec for spec in large}
            for future in as_completed(jobs):
                spec = jobs[future]
                try:
                    receipts.append(future.result())
                except Exception as error:
                    failed.append(spec["id"])
                    update(spec["id"], {"status": "failed", "error": type(error).__name__, "verified": False})
                    log("수신 실패: " + spec["name"] + " / 비공개 실패 원문 확인 필요")
    except Exception as error:
        failure({"time_kst": now(), "error_raw": str(error), "error_type": type(error).__name__, "phase": "acquisition"})
        failed.append("setup")
        log("획득 준비 실패: 비공개 실패 원문을 보존했습니다.")
    finally:
        stop.set()
        thread.join(timeout=6)
        with mutex:
            state["status"] = "failed" if failed else "ready"
            state["steps"][1]["status"] = "failed" if failed else "complete"
            state["current_task"] = {"title": "수신 오류 검토 필요" if failed else "원본 획득 완료·평가 준비 검토 대기", "phase": "acquisition", "detail": "실제 얼굴 정확도는 아직 미측정이며 strict 모델 적재·정렬·프로토콜 검토가 남았습니다."}
            save()
        atomic_json(WORK / "acquisition_receipts.json", {"receipts": receipts, "failed": failed, "finished_at": now()})
        atomic_json(WORK / "acquisition_process.json", {"pid": os.getpid(), "finished_at": now(), "argv": sys.argv, "cwd": os.getcwd(), "status": "failed" if failed else "complete", "exit_code": 1 if failed else 0})
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
