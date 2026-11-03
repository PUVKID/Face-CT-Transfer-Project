"""부모가 별도 검증 후 시작하는 LFW 준비 또는 평가 단일 작업입니다."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.acquisition import atomic_json, now
from benchmark.evaluation import prepare_lfw, evaluate_lfw
from benchmark.status import initial_status, gpu_snapshot, process_alive


class StatusWriter:
    def __init__(self, work, phase):
        self.work, self.phase = work, phase
        self.lock = threading.RLock()
        path = work / "status.json"
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else initial_status()
        # 새 작업의 미측정 집계를 초기화하기 전에 이전 관측 원문을 비공개로 보존합니다.
        with (work / "progress_history.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"time": now(), "pid": os.getpid(), "phase": phase, "previous_progress": self.state["progress"], "previous_evaluation": self.state["evaluation"]}, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.techniques = {item["id"]: item for item in self.state["techniques"]}
        self.alignment_observed = False
        self.inference_observed = False
        previous_alignment = self.techniques["alignment"]["status"] == "complete" or any(item["id"] == "alignment" and item["status"] == "complete" for item in self.state["steps"])
        for identifier in ["fp32", "flip", "lfw"]:
            self.techniques[identifier].update(status="planned", detail="이번 작업에서 미적용 · 정렬 조건과 평가 입력 검증 뒤 적용합니다.")
        self.techniques["alignment"].update(status="complete" if phase == "evaluate" and previous_alignment else "planned", detail="이전 정렬 완료 기록 · 이번 평가의 캐시 검증 대기입니다." if phase == "evaluate" and previous_alignment else "미적용 · 원본 추출과 정렬 입력 검증 뒤 시작합니다.")
        self.techniques["template"].update(status="unused", detail="이번 LFW에서 미사용 · IJB-C media/template 집계는 후속 계획입니다.")
        self.techniques["ct_train"].update(status="unused", detail="이번 얼굴 재현 단계에서는 CT 학습을 적용하지 않습니다.")
        for step in self.state["steps"]:
            if step["id"] == ("alignment" if phase == "prepare" else "evaluation"):
                step.update(status="waiting", detail="입력 검증 뒤 실제 처리 단계에서 시작합니다.")
        self.throughput_unit = self.state["progress"]["unit"]
        self._reset_throughput("", "new_job")
        self.state["progress"] = {"completed": None, "total": None, "unit": "", "percent": None, "eta_seconds": None, "eta_is_estimate": True}
        self.state["status"] = "running"
        self.state["processes"] = [item for item in self.state["processes"] if item["role"] != "LFW 평가"]
        self.state["processes"].append({"role": "LFW 평가", "pid": os.getpid(), "alive": True, "started_at": now()})
        self.state["evaluation"].update(status="running", benchmark="LFW", precision="float32", epoch=None, metrics=None, images_completed=None, images_total=None, pairs_completed=None, pairs_total=None, fold_completed=None, fold_total=None)
        if phase == "prepare":
            self.state["evaluation"].update(detection_coverage=None, failed_images=None)
        self.state["current_task"] = {"title": "LFW 독립 정렬 준비" if phase == "prepare" else "LFW 실제 얼굴 평가", "phase": phase, "detail": "저자 aligned/bin과 입력 동일성이 확인되지 않은 독립 재측정입니다. IJB-C는 외부 할당량 대기입니다."}
        self.last_resource = 0
        self.flush()

    def _reset_throughput(self, next_unit, reason):
        previous = self.state["series"]["throughput"]
        if previous:
            # 이전 관측은 공개 현재 그래프에서 분리하되 비공개 운영 원문에 보존합니다.
            record = {"time": now(), "pid": os.getpid(), "phase": self.phase, "reason": reason, "previous_unit": self.throughput_unit, "next_unit": next_unit, "points": previous}
            with (self.work / "throughput_history.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self.state["series"]["throughput"] = []
        self.throughput_unit = next_unit

    def flush(self):
        with self.lock:
            if time.monotonic() - self.last_resource >= 5:
                self.state["resources"] = {"gpu": gpu_snapshot(), "disk_free_bytes": shutil.disk_usage(self.work).free}
                self.last_resource = time.monotonic()
                gpu = self.state["resources"]["gpu"]
                if gpu is not None:
                    self.state["series"]["gpu_utilization"].append({"time": gpu["measured_at"], "value": gpu["utilization_percent"]})
                    self.state["series"]["gpu_utilization"] = self.state["series"]["gpu_utilization"][-500:]
            self.state["updated_at"] = now()
            atomic_json(self.work / "status.json", self.state)

    def progress(self, phase, values):
        with self.lock:
            count = values.get("completed", values.get("bytes"))
            total = values.get("total", values.get("total_bytes"))
            elapsed = values.get("elapsed_seconds")
            resumed = values.get("resumed_at", 0)
            speed = (count - resumed) / elapsed if elapsed and count is not None and count > resumed else None
            eta = (total - count) / speed if speed and total is not None else None
            unit = "bytes" if phase == "extract" else "images"
            if unit != self.throughput_unit:
                self._reset_throughput(unit, "unit_change")
            self.state["progress"] = {"completed": count, "total": total, "unit": unit, "percent": 100 * count / total if total else None, "eta_seconds": eta, "eta_is_estimate": True}
            self.state["current_task"]["phase"] = phase
            if phase != "extract":
                self.state["evaluation"].update(images_completed=count, images_total=total)
            if phase == "extract":
                detail = "원본 archive 추출 중 · 얼굴 정렬과 특징 추론은 아직 적용하지 않았습니다."
            elif phase == "alignment":
                self.alignment_observed = True
                detail = "검출 실패 " + str(values.get("failed", 0)) + "개 · 실패 pair를 제외한 정확도를 만들지 않습니다."
                self.techniques["alignment"].update(status="active", detail="실제 얼굴 검출·5점 정렬 처리 중 · " + detail)
            else:
                self.inference_observed = True
                self.state["evaluation"].update(pairs_total=6000, fold_total=10)
                detail = "FP32·원본/반전 특징·실제 처리 영상 수이며 정확도는 전체 고정 pair 완료 뒤 계산합니다."
                self.techniques["alignment"].update(status="complete", detail="평가 진입 전 정렬 캐시의 결합·해시·전체 coverage 검증을 통과했습니다.")
                self.techniques["fp32"].update(status="active", detail=f"실제 FP32 평가 모드 추론 중 · 재개 위치 {resumed}개, 현재 {count}/{total}개입니다.")
                self.techniques["flip"].update(status="active", detail="원본과 좌우반전 특징을 실제 추론 중 · 합산 후 L2 정규화 조건을 유지합니다.")
                if count == total:
                    self.techniques["lfw"].update(status="active", detail="특징 처리 완료 뒤 고정 6000쌍·10fold 점수와 임계값 계산 단계로 진입합니다.")
            self.state["current_task"]["detail"] = detail
            for step in self.state["steps"]:
                if phase != "extract" and step["id"] == ("alignment" if phase == "alignment" else "evaluation"):
                    step.update(status="active", detail=self.state["current_task"]["detail"])
            if speed is not None:
                self.state["series"]["throughput"].append({"time": now(), "value": speed})
                self.state["series"]["throughput"] = self.state["series"]["throughput"][-500:]
            self.log(phase, {key: values[key] for key in ["completed", "total", "failed", "files", "bytes", "total_bytes"] if key in values})
            self.flush()

    def log(self, phase, values):
        with (self.work / "evaluation.log").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"time": now(), "phase": phase, **values}, ensure_ascii=False) + "\n")

    def finish(self, result):
        with self.lock:
            unmet = result.get("status") == "protocol_unmet"
            self.state["status"] = "paused"
            self.state["evaluation"]["status"] = "protocol_unmet" if unmet else "ready" if self.phase == "prepare" else "complete"
            self.state["current_task"] = {"title": "LFW 정렬 실패 coverage 검토 대기" if unmet else "LFW 정렬 완료 · 별도 평가 시작 대기" if self.phase == "prepare" else "LFW 직접 측정 완료 · IJB-C 외부 대기", "phase": "protocol_unmet" if unmet else "ready" if self.phase == "prepare" else "complete", "detail": "IJB-C는 Google Drive 할당량 해제와 원본 라이선스·metadata 검토가 남았습니다. LFW만으로 IJB-C 재현을 완료 처리하지 않습니다."}
            if unmet:
                self.state["current_task"]["detail"] = f"성공 {result['successful_images']} / 전체 {result['images']} · 실패 {len(result['failed_images'])}개. 정확도 미측정 · 전체 pair 평가 조건 미충족 검토 대기입니다. 실패 영상을 임의 대체하거나 pair에서 제외하지 않습니다."
            if self.phase == "prepare":
                self.state["evaluation"].update(images_completed=result["images"], images_total=result["images"], detection_coverage=result["coverage"], failed_images=len(result["failed_images"]))
                detail = f"성공 {result['successful_images']} / 전체 {result['images']} · 실패 {len(result['failed_images'])}개 · coverage {100 * result['coverage']:.2f}%. "
                detail += "전체 pair 평가 조건 미충족 · 정확도 미측정, 검토 대기입니다." if unmet else "실제 정렬 처리 완료입니다." if self.alignment_observed else "검증된 정렬 캐시 재사용입니다."
                self.techniques["alignment"].update(status="paused" if unmet else "complete", detail=detail)
                if unmet:
                    for identifier in ["fp32", "flip", "lfw"]:
                        self.techniques[identifier].update(status="planned", detail="미적용 · 정렬 coverage 조건 미충족으로 평가를 시작하지 않았습니다.")
            else:
                self.techniques["alignment"].update(status="complete", detail="평가에 사용한 정렬 캐시의 결합·해시·전체 coverage 검증을 통과했습니다.")
                self.techniques["fp32"].update(status="complete", detail="실제 FP32 추론과 특징 검증을 완료했습니다." if self.inference_observed else "검증된 FP32 특징 캐시 재사용 · 이번 작업의 신규 추론 없음.")
                self.techniques["flip"].update(status="complete", detail="실제 원본·좌우반전 특징 합산 후 L2 정규화를 완료했습니다." if self.inference_observed else "검증된 원본·반전 특징 캐시 재사용 후 합산·L2 정규화 완료 · 신규 추론 없음.")
                self.techniques["lfw"].update(status="complete", detail="고정 6000쌍·10fold 계산 완료 · 나머지 9fold에서 임계값 선택, 저자 입력 동일성은 미확인입니다.")
                self.state["evaluation"].update(pairs_completed=6000, pairs_total=6000, fold_completed=10, fold_total=10, metrics={"accuracy": result["accuracy"], "std": result["std"], "author_input_equivalence": False})
                measured = {"benchmark": "LFW", "metric": "고정 6000쌍 10fold 정확도 · YuNet 독립 정렬", "value": 100 * result["accuracy"], "unit": "%", "provenance": "직접 측정 · 저자 입력 동일성 미확인"}
                self.state["model"]["measured_metrics"] = [item for item in self.state["model"]["measured_metrics"] if item["benchmark"] != "LFW"] + [measured]
            for step in self.state["steps"]:
                if step["id"] == ("alignment" if self.phase == "prepare" else "evaluation"):
                    step.update(status="paused" if unmet else "complete", detail=self.state["current_task"]["detail"])
            for process in self.state["processes"]:
                if process.get("pid") == os.getpid():
                    process["alive"] = False
            self.log("protocol_unmet" if unmet else "complete", {"measured_face_accuracy": self.phase == "evaluate"})
            self.flush()

    def fail(self):
        with self.lock:
            self.state["status"] = "failed"
            self.state["evaluation"]["status"] = "failed"
            self.state["current_task"] = {"title": "LFW 작업 실패 · 원문 보존", "phase": "failed", "detail": "비공개 실패 영수증을 보존했습니다. 확인되지 않은 정확도는 표시하지 않습니다."}
            # 실패 전에 실제 진입한 기법만 실패 처리하고 미적용 계획은 유지합니다.
            for item in self.state["techniques"]:
                if item["status"] == "active":
                    item.update(status="failed", detail="실제 적용 중 작업이 실패했습니다. 완료와 정확도는 확인되지 않았으며 원문을 보존했습니다.")
            for step in self.state["steps"]:
                if step["id"] in {"alignment", "evaluation"} and step["status"] == "active":
                    step.update(status="failed", detail=self.state["current_task"]["detail"])
            self.flush()


def main():
    parser = argparse.ArgumentParser(description="고정 자료의 LFW 단일 단계만 실행합니다")
    parser.add_argument("phase", choices=["prepare", "evaluate"])
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4, 8], default=8)
    args = parser.parse_args()
    work = ROOT.parent / "work/stage1_revision2"
    work.mkdir(parents=True, exist_ok=True)
    lock = work / "lfw_job.lock.json"
    if lock.exists():
        old = json.loads(lock.read_text(encoding="utf-8"))
        if process_alive(old.get("pid")) is not False:
            raise RuntimeError("LFW 작업 lock의 프로세스가 실행 중이거나 생존 확인이 불가능합니다")
        archive = lock.with_name("lfw_job.previous_" + str(time.time_ns()) + ".json")
        lock.replace(archive)
    process = {"pid": os.getpid(), "argv": [sys.executable, *sys.argv], "orig_argv": list(sys.orig_argv), "cwd": str(Path.cwd()), "started_at": now(), "phase": args.phase, "status": "running", "exit_code": None}
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(process, stream, ensure_ascii=False)
    atomic_json(work / "evaluation_process.json", process)
    writer = StatusWriter(work, args.phase)
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(5):
            writer.flush()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    code = 1
    try:
        result = prepare_lfw(ROOT.parent, writer.progress) if args.phase == "prepare" else evaluate_lfw(ROOT.parent, args.device, args.batch_size, writer.progress)
        writer.finish(result)
        code = 2 if result.get("status") == "protocol_unmet" else 0
        if code:
            raise RuntimeError("정렬 실패로 고정 전체 pair 평가 조건이 충족되지 않았습니다. coverage 증거를 보존했습니다")
    except Exception as error:
        failure = {"time_kst": now(), "session": os.environ.get("FACE_SESSION_ID"), "task": os.environ.get("FACE_TASK_ID"), "argv": process["argv"], "orig_argv": process["orig_argv"], "cwd": process["cwd"], "exit_code": code, "error": str(error), "traceback": traceback.format_exc(), "changed_files": False, "repair_used": int(os.environ.get("FACE_REPAIR_USED", "0")), "input_fix_used": int(os.environ.get("FACE_INPUT_FIX_USED", "0"))}
        with (work / "evaluation_failures.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
        if code != 2:
            writer.fail()
        writer.log("failed", {"detail": "원문은 비공개 실패 영수증에 보존했습니다", "exit_code": code})
    finally:
        stop.set()
        thread.join(timeout=10)
        process.update(status="complete" if code == 0 else "protocol_unmet" if code == 2 else "failed", exit_code=code, ended_at=now())
        atomic_json(work / "evaluation_process.json", process)
        atomic_json(lock, process)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
