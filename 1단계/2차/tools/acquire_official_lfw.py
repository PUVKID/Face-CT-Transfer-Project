"""공식 단일 member를 별도 worker와 제한된 Windows Job에서 획득합니다."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import official_lfw_range_member as member_io

SOFT_DEADLINE_SECONDS = 180
HARD_DEADLINE_SECONDS = 190
START_TOKEN = b"START_OFFICIAL_LFW_WORKER\n"


class _BasicLimit(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong), ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t), ("maximum_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ["read_operations", "write_operations", "other_operations", "read_bytes", "write_bytes", "other_bytes"]]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("basic", _BasicLimit), ("io", _IoCounters), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in ["user_time", "kernel_time", "period_user_time", "period_kernel_time"]] + [(name, wintypes.DWORD) for name in ["page_faults", "total_processes", "active_processes", "terminated_processes"]]


class _OwnedWindowsJob:
    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("이 실행기의 강제 종료 경계는 검증한 Windows 환경에서만 지원합니다")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimit()
        limits.basic.flags = 0x2000  # 이 Job의 마지막 핸들이 닫힐 때 소속 프로세스만 종료합니다.
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        # CPython Windows Popen이 보유한 실제 생성 핸들을 사용하며 PID를 다시 검색하지 않습니다.
        if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def stop(self):
        if not self.api.TerminateJobObject(self.handle, 124):
            raise ctypes.WinError(ctypes.get_last_error())

    def accounting(self):
        value = _Accounting()
        if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.api.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def _atomic_new_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    member_io.write_json_new(temporary, value)
    member_io.commit(temporary, path)


def _stop_owned(process, job, assigned):
    if assigned:
        if process.poll() is None or job.accounting().active_processes:
            job.stop()
    elif process.poll() is None:
        # Job 배정 실패 시 START 이전에 대기 중인 단일 생성 프로세스 핸들만 종료합니다.
        process.kill()
    process.wait(timeout=5)


def supervise(argv, cwd, run_dir, *, hard_seconds=HARD_DEADLINE_SECONDS):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path, receipt_path = run_dir / "worker.log", run_dir / "supervisor_result.json"
    record = {"started_at": datetime.now().astimezone().isoformat(), "argv": list(argv), "cwd": str(cwd), "supervisor_pid": os.getpid(), "worker_pid": None, "hard_seconds": hard_seconds, "soft_deadline_seconds": SOFT_DEADLINE_SECONDS, "socket_timeout_seconds": 5, "status": "supervisor_failed", "worker_returncode": None, "stop_verified": False, "acquisition_completed": False, "job_active_processes_after": None, "log_path": str(log_path), "receipt_path": str(receipt_path), "routing_context": {key: os.environ.get(key) for key in ["FACE_SESSION_ID", "FACE_TASK_ID", "FACE_REPAIR_USED", "FACE_INPUT_FIX_USED"]}}
    _atomic_new_json(run_dir / "launch_request.json", record)
    started = time.monotonic()
    process = job = None
    assigned = False
    with log_path.open("xb") as output:
        try:
            job = _OwnedWindowsJob()
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            record["worker_pid"] = process.pid
            job.assign(process)
            assigned = True
            process.stdin.write(START_TOKEN)
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=max(0.001, hard_seconds - (time.monotonic() - started)))
            record["status"] = "worker_exited" if process.returncode == 0 else "worker_failed"
        except subprocess.TimeoutExpired:
            record.update(status="hard_timeout", termination_trigger_seconds=time.monotonic() - started, error="별도 worker의 운영 총시간 한도 초과")
        except KeyboardInterrupt:
            record.update(status="user_stopped", error="사용자가 실행기를 중단했습니다", traceback=traceback.format_exc())
        except Exception as error:
            record.update(status="supervisor_failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        finally:
            try:
                if process is not None:
                    _stop_owned(process, job, assigned)
                    record["worker_returncode"] = process.returncode
                if job is not None and assigned:
                    stop_deadline = time.monotonic() + 5
                    accounting = job.accounting()
                    while accounting.active_processes and time.monotonic() < stop_deadline:
                        time.sleep(0.02)
                        accounting = job.accounting()
                    record.update(job_active_processes_after=accounting.active_processes, job_total_processes=accounting.total_processes)
                    record["stop_verified"] = process.poll() is not None and accounting.active_processes == 0
                if not record["stop_verified"]:
                    record.update(status="supervisor_failed", stop_error="자기 worker의 완전 종료를 확인하지 못했습니다")
            except Exception as error:
                record.update(status="supervisor_failed", stop_error=str(error), stop_traceback=traceback.format_exc())
            finally:
                if process is not None and process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if job is not None:
                    try:
                        job.close()
                    except Exception as error:
                        record.update(status="supervisor_failed", job_close_error=str(error))
    record.update(ended_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic() - started)
    _atomic_new_json(receipt_path, record)
    return record


def _worker(source_evidence, destination, run_dir):
    if sys.stdin.buffer.readline() != START_TOKEN:
        raise RuntimeError("감독 실행기의 START 확인 없이 원격 획득을 시작하지 않습니다")
    _atomic_new_json(Path(run_dir) / "worker_started.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "executable": sys.executable, "python_version": sys.version, "argv": sys.orig_argv, "cwd": str(Path.cwd())})
    url, member = member_io.load_confirmed_source(source_evidence)
    receipt = member_io.acquire(member_io.RangeClient(url, member.archive_total), member, destination)
    print(json.dumps({"status": receipt["status"], "bytes": receipt["bytes"], "sha256": receipt["sha256"]}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="고정 공식 LFW 단일 파일 획득만 제한 실행합니다")
    parser.add_argument("--source-evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    stage = ROOT.parent
    expected_output = stage / "data/face_benchmarks/raw/lfw_official_ms1mv2.bin"
    if args.output.resolve() != expected_output.resolve():
        parser.error("출력은 현재 프로젝트의 지정한 lfw_official_ms1mv2.bin만 허용합니다")
    if args.worker:
        if args.run_dir is None:
            parser.error("내부 worker의 비공개 실행 기록 경로가 필요합니다")
        return _worker(args.source_evidence_dir, args.output, args.run_dir)
    if not args.execute:
        print(json.dumps({"status": "plan_only", "soft_seconds": SOFT_DEADLINE_SECONDS, "hard_seconds": HARD_DEADLINE_SECONDS, "output_name": expected_output.name, "network_requests": 0, "unsafe_deserialization_runs": 0}))
        return 0
    # URL은 명령 인자로 전달하지 않고 고정 SHA의 비공개 공식 영수증에서만 읽습니다.
    member_io.load_confirmed_source(args.source_evidence_dir)
    executable = Path(sys._base_executable).resolve()
    if not executable.is_file():
        raise RuntimeError("현재 venv가 사용하는 실제 base Python을 확인하지 못했습니다")
    run_dir = stage / "work/stage1_revision2/official_lfw_acquisition_runs" / str(time.time_ns())
    command = [str(executable), "-B", "-X", "utf8", str(Path(__file__).resolve()), "--worker", "--source-evidence-dir", str(args.source_evidence_dir.resolve()), "--output", str(args.output.resolve()), "--run-dir", str(run_dir)]
    result = supervise(command, ROOT, run_dir)
    completed = False
    try:
        if result["status"] == "worker_exited" and result["stop_verified"]:
            data_receipt = json.loads(args.output.with_name(args.output.name + ".receipt.json").read_text(encoding="utf-8"))
            with args.output.open("rb") as incoming:
                actual = hashlib.file_digest(incoming, "sha256").hexdigest()
            completed = data_receipt.get("status") == "complete" and data_receipt.get("member") == member_io.OFFICIAL.name and data_receipt.get("bytes") == args.output.stat().st_size == member_io.OFFICIAL.uncompressed and data_receipt.get("sha256") == actual
            if not completed:
                raise RuntimeError("worker 종료0 이후 실제 결과·완료 영수증 대조에 실패했습니다")
    except Exception as error:
        result.update(validation_error=str(error), validation_traceback=traceback.format_exc())
    result.update(acquisition_completed=completed, status="complete" if completed else result["status"] if result["status"] in {"hard_timeout", "user_stopped"} else "failed")
    _atomic_new_json(run_dir / "acquisition_result.json", result)
    print(json.dumps({"status": result["status"], "acquisition_completed": completed, "worker_pid": result["worker_pid"], "stop_verified": result["stop_verified"]}), flush=True)
    return 0 if completed else 124 if result["status"] == "hard_timeout" else 130 if result["status"] == "user_stopped" else 1


if __name__ == "__main__":
    raise SystemExit(main())
