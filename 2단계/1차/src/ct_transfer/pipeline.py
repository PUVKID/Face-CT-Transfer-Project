"""자료 준비·실측 예산·승인 후 순차 실행을 연결합니다."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from . import data, evaluation, models, splits, training
from .io import append_event, atomic_json, digest_file, signature, timestamp
from .budget import DeadlineExceeded, RunDeadline


SOURCE = Path(__file__).resolve().parents[2]
STAGE = SOURCE.parent
PROJECT = STAGE.parent
PRIVATE = STAGE / "work/private"
CACHE = STAGE / "data"
RESULT_MODELS = STAGE / "models"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime_context():
    return {"session": os.environ.get("FACE_CT_SESSION"), "task": os.environ.get("FACE_CT_TASK")}


def source_identity():
    return {str(path.relative_to(SOURCE)).replace("\\", "/"): digest_file(path)
            for directory in (SOURCE / "src", SOURCE / "tools", SOURCE / "configs")
            for path in sorted(directory.rglob("*")) if path.is_file() and path.suffix in (".py", ".json")}


def prepare():
    manifests = {}
    for name, prepare_function, original in (
        ("luna", data.prepare_luna, PROJECT / "1단계/data/luna25"),
        ("iq", data.prepare_iq, Path("D:/2026/폐암 논문/원본 데이터 셋 분류")),
    ):
        manifests[name] = prepare_function(original, CACHE)
        records = manifests[name]["records"]
        folds = splits.make_folds(records, n_splits=7, seed=42)
        atomic_json(CACHE / f"{name}_folds.json", {"index_signature": signature(manifests[name]), "seed": 42, "folds": folds})
    result = {name: {"images": len(index["records"]), "groups": len({row["group"] for row in index["records"]}), "patient_mapping_known": index["patient_mapping_known"]} for name, index in manifests.items()}
    atomic_json(PRIVATE / "preparation_summary.json", result)
    return result


def load_cohort(name):
    index = read_json(CACHE / f"{name}_index.json")
    split = read_json(CACHE / f"{name}_folds.json")
    if signature(index) != split["index_signature"] or digest_file(index["cache"]) != index["cache_sha256"]:
        raise ValueError("자료 또는 분할 출처 해시가 다릅니다")
    for fold in split["folds"]:
        splits.assert_fold(index["records"], fold)
    return index, split["folds"], np.load(index["cache"], mmap_mode="r", allow_pickle=False)


def device():
    if not torch.cuda.is_available():
        raise RuntimeError("실제 장기 실행과 GPU 벤치마크에는 CUDA가 필요합니다")
    return torch.device("cuda")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_snapshot():
    started = timestamp()
    query = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8", timeout=10, check=True)
    rows = [line.strip() for line in query.stdout.split("\n") if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("현재 자원 측정 명세는 한 개 GPU의 일치하는 지표가 필요합니다")
    physical_free, physical_total, utilization = (int(item.strip()) for item in rows[0].split(","))
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    physical_free *= 1024 ** 2
    physical_total *= 1024 ** 2
    return {"window_start_kst": started, "window_end_kst": timestamp(), "nvidia_free_bytes": physical_free, "nvidia_total_bytes": physical_total, "nvidia_utilization_percent": utilization, "cuda_free_bytes": cuda_free, "cuda_total_bytes": cuda_total, "conservative_free_bytes": min(physical_free, cuda_free)}


def select_common_batch(results, *, headroom_bytes=4 * 1024 ** 3):
    eligible = []
    for batch in sorted({row["batch_size"] for row in results}):
        rows = [row for row in results if row["batch_size"] == batch]
        if {row["scope"] for row in rows} != {"frozen", "lastblock", "full"} or any(not row["finite_logits"] or row["minimum_free_bytes"] < headroom_bytes for row in rows):
            continue
        full = next(row for row in rows if row["scope"] == "full")
        eligible.append((batch, full["images_per_second"]))
    if not eligible:
        raise RuntimeError("모든 학습 범위에서 최소 4GiB 여유를 보존한 공통 배치가 없습니다")
    return sorted(eligible, key=lambda pair: (-pair[1], pair[0]))[0][0]


def benchmark():
    index, iq_folds, values = load_cohort("iq")
    target_device = device()
    torch.set_num_threads(4)
    results = []
    skipped = []
    def persist(status="partial", current=None):
        atomic_json(PRIVATE / "resource_probe_progress.json", {"time_kst": timestamp(), "source": source_identity(), "status": status, "state_represents": "마지막으로 관측한 경계이며 프로세스 생존 여부는 별도 process 기록을 따름", "current": current, "measurements": results, "skipped": skipped})
    persist(current={"phase": "starting"})
    reference_path = PRIVATE / "run02_snapshot/gpu_benchmark.json"
    reference = read_json(reference_path)
    observed_full_peak = max(row["peak_allocated_bytes"] for row in reference["measurements"] if row["scope"] == "full")
    headroom = 4 * 1024 ** 3
    selected = [next(number for number, row in enumerate(index["records"]) if row["label"] == label) for label in range(3)]
    selected.append(next(number for number in range(len(values)) if number not in selected))
    base_x = np.array(values[selected])
    base_y = [index["records"][number]["label"] for number in selected]
    for batch_size, scope in ((batch, scope) for batch in (4, 8, 16) for scope in ("frozen", "lastblock", "full")):
        # 큰 후보는 이미 측정한 full peak의 보수적 비례 상한으로 먼저 거릅니다.
        persist(current={"phase": "before_guard", "scope": scope, "batch_size": batch_size})
        guard = memory_snapshot()
        predicted_peak = math.ceil(observed_full_peak * batch_size / 4)
        persist(current={"phase": "guard_observed", "scope": scope, "batch_size": batch_size, "predicted_peak_bytes": predicted_peak, "memory": guard})
        if guard["conservative_free_bytes"] - predicted_peak < headroom:
            skipped.append({"scope": scope, "batch_size": batch_size, "reason": "예상 full peak 이후 4GiB 여유 부족", "predicted_peak_bytes": predicted_peak, "memory": guard})
            persist(current={"phase": "guard_skipped", "scope": scope, "batch_size": batch_size})
            continue
        training.seed_everything(11)
        model, evidence = models.build_model(3, scope)
        state_hash = hashlib.sha256()
        for key, value in model.state_dict().items():
            state_hash.update(key.encode("utf-8"))
            state_hash.update(value.numpy().tobytes())
        before = memory_snapshot()
        persist(current={"phase": "before_gpu_load", "scope": scope, "batch_size": batch_size, "memory": before})
        if before["conservative_free_bytes"] - predicted_peak < headroom:
            skipped.append({"scope": scope, "batch_size": batch_size, "reason": "GPU 적재 직전 재확인 여유 부족", "predicted_peak_bytes": predicted_peak, "memory": before})
            persist(current={"phase": "load_skipped", "scope": scope, "batch_size": batch_size})
            del model
            continue
        model.to(target_device).train()
        x = torch.tensor(np.tile(base_x, (batch_size // 4, 1, 1, 1)), device=target_device)
        y = torch.tensor(base_y * (batch_size // 4), device=target_device)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4)
        amp, amp_dtype = training.amp_policy(target_device, {"amp": scope != "frozen", "amp_dtype": "bfloat16"})
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        torch.cuda.reset_peak_memory_stats()
        measurements = []
        free_samples = []
        for step in range(4):
            _sync()
            begun = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=amp_dtype):
                loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("GPU 자원 측정 손실이 유한하지 않습니다")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            _sync()
            if step:
                measurements.append(time.perf_counter() - begun)
            free_samples.append(torch.cuda.mem_get_info()[0])
        model.eval()
        _sync()
        begun = time.perf_counter()
        with torch.inference_mode():
            result = model(x)
        _sync()
        evaluation_seconds = time.perf_counter() - begun
        after = memory_snapshot()
        median_step = float(np.median(measurements))
        peak_reserved = torch.cuda.max_memory_reserved()
        results.append({"scope": scope, "batch_size": batch_size, "sample_repeat_factor": batch_size // 4, "initial_state_sha256": state_hash.hexdigest(), "seed": 11, "training_precision": "bfloat16_amp" if amp else "float32", "evaluation_precision": "float32", "warmup_steps": 1, "measured_steps": 3, "step_seconds": measurements, "step_seconds_median": median_step, "images_per_second": batch_size / median_step, "evaluation_batch_seconds": evaluation_seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": peak_reserved, "memory_before": before, "memory_after": after, "minimum_free_bytes": min(before["conservative_free_bytes"], after["conservative_free_bytes"], *free_samples), "finite_logits": bool(torch.isfinite(result).all()), "model_evidence": evidence})
        if batch_size == 4 and scope == "full":
            observed_full_peak = max(observed_full_peak, peak_reserved)
        persist(current={"phase": "measured", "scope": scope, "batch_size": batch_size, "observed_full_peak_reference_bytes": observed_full_peak})
        del model, optimizer, scaler, x, y, loss, result
        torch.cuda.empty_cache()
    try:
        chosen = select_common_batch(results, headroom_bytes=headroom)
    except RuntimeError:
        persist(status="no_eligible_batch", current={"phase": "selection", "headroom_bytes": headroom})
        raise
    output = {"time_kst": timestamp(), "kind": "real_iq_gpu_smoke_not_clinical_result", "device": torch.cuda.get_device_name(), "torch": str(torch.__version__), "source": source_identity(), "measurements": results, "skipped": skipped, "selected_batch_size": chosen, "selection_rule": "전 범위 최소 4GiB 여유를 충족하는 full 처리량 최대; 동률 작은 배치", "guard_reference_sha256": digest_file(reference_path), "memory_limitations": "두 도구는 기록된 짧은 시간 창에서 순차 조회합니다. 시점·드라이버 메모리 관리 차이 원인은 확정하지 않으며 더 작은 여유를 사용합니다. 다른 GPU 작업을 종료하지 않았습니다."}
    atomic_json(PRIVATE / "gpu_benchmark.json", output)
    # 선택한 공통 배치와 고정 epoch의 실제 분할별 단계 수로 비용을 추정합니다.
    config = {"max_epochs": 20, "batch_size": chosen, "patience": 5, "lr": 1e-3, "backbone_lr": 1e-5, "weight_decay": .01, "warmup_epochs": 2, "amp": True, "amp_dtype": "bfloat16", "wall_seconds": 24 * 3600, "gradient_clip": 5.}
    selected_results = [row for row in results if row["batch_size"] == chosen]
    train_steps = sum(math.ceil(len(fold["train"]) / chosen) for fold in iq_folds)
    val_steps = sum(math.ceil(len(fold["validation"]) / chosen) for fold in iq_folds)
    test_steps = sum(math.ceil(len(fold["test"]) / chosen) for fold in iq_folds)
    estimate = sum(10 * (config["max_epochs"] * train_steps * row["step_seconds_median"] + (config["max_epochs"] * val_steps + test_steps) * row["evaluation_batch_seconds"]) for row in selected_results if row["scope"] != "frozen")
    aux_estimate = 2 * math.ceil(6163 / chosen) * next(row["step_seconds_median"] for row in selected_results if row["scope"] == "full")
    budget = {"schema": 1, "created_kst": timestamp(), "source": source_identity(), "benchmark_sha256": digest_file(PRIVATE / "gpu_benchmark.json"), "config": config, "auxiliary": {"epochs": 2, "seed": 73, "scope": "full"}, "luna_head_epochs": 40, "estimated_iq_train_seconds_with_1_5_factor": float(estimate * 1.5), "estimated_auxiliary_seconds_with_1_5_factor": float(aux_estimate * 1.5), "estimate_limitations": [f"선택한 배치 {chosen}의 실제 IQ 측정과 고정 최대 epoch로 추정하며 디스크·체크포인트·경합 지연은 변동합니다.", "frozen 특징 일회 추출과 head 학습은 별도이며 최초 실행에 고정한 전체 UTC 7일 상한을 재개에도 유지합니다."], "total_wall_seconds": 7 * 24 * 3600}
    atomic_json(PRIVATE / "run_budget.json", budget)
    persist(status="complete", current={"phase": "budget_written", "selected_batch_size": chosen, "benchmark_sha256": budget["benchmark_sha256"]})
    return output


def _features(name, initialization, values, adapted, binding, batch_size, deadline):
    deadline.check()
    target = CACHE / f"{name}_features_{initialization}.npy"
    receipt = target.with_suffix(".json")
    expected = signature({"binding": binding, "initialization": initialization, "adapted": digest_file(adapted) if initialization == "luna-adapted" else None})
    if target.exists() and receipt.exists():
        record = read_json(receipt)
        if record["binding"] == expected and record["sha256"] == digest_file(target):
            return np.load(target, mmap_mode="r", allow_pickle=False)
        raise ValueError("저장된 특징의 출처가 다릅니다")
    backbone, _ = models.face_backbone()
    if initialization == "luna-adapted":
        checkpoint = torch.load(adapted, map_location="cpu", weights_only=True)
        if checkpoint.get("purpose") != "luna_auxiliary_for_iq_only":
            raise ValueError("IQ 보조 학습 목적의 체크포인트가 아닙니다")
        backbone.load_state_dict(checkpoint["backbone"], strict=True)
    backbone.to(device()).eval()
    temporary = target.with_suffix(".part.npy")
    progress_path = target.with_suffix(".progress.json")
    if temporary.exists() or progress_path.exists():
        if not temporary.exists() or not progress_path.exists():
            raise ValueError("미완료 특징의 데이터와 진행 기록 중 하나가 없습니다")
        progress = read_json(progress_path)
        if progress.get("binding") != expected or type(progress.get("rows_completed")) is not int or not 0 <= progress["rows_completed"] <= len(values):
            raise ValueError("미완료 특징의 출처 또는 진행 위치가 다릅니다")
        output = np.load(temporary, mmap_mode="r+", allow_pickle=False)
        if output.shape != (len(values), 512) or output.dtype != np.float32:
            raise ValueError("미완료 특징 배열의 형상 또는 dtype이 다릅니다")
        start_row = progress["rows_completed"]
    else:
        output = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(values), 512))
        start_row = 0
        atomic_json(progress_path, {"binding": expected, "rows_completed": 0})
    try:
        with torch.inference_mode():
            for start in range(start_row, len(values), batch_size):
                deadline.check()
                result = backbone(training._batch(values, slice(start, start + batch_size), device()))
                if isinstance(result, tuple):
                    result = result[0]
                result = result.float().cpu().numpy()
                if result.shape != (min(batch_size, len(values) - start), 512) or not np.isfinite(result).all():
                    raise ValueError("원본 얼굴 특징의 형상 또는 값이 다릅니다")
                output[start:start + batch_size] = result
                output.flush()
                atomic_json(progress_path, {"binding": expected, "rows_completed": min(start + batch_size, len(values))})
    finally:
        output.flush()
        del output
        del backbone
        torch.cuda.empty_cache()
    os.replace(temporary, target)
    atomic_json(receipt, {"binding": expected, "sha256": digest_file(target)})
    progress_path.unlink()
    return np.load(target, mmap_mode="r", allow_pickle=False)


def _auxiliary(index, values, budget, binding, deadline):
    deadline.check()
    destination = RESULT_MODELS / "luna_auxiliary_for_iq.pt"
    out = RESULT_MODELS / "auxiliary"
    out.mkdir(parents=True, exist_ok=True)
    aux = budget["auxiliary"]
    expected = signature({"binding": binding, "auxiliary": aux})
    if destination.exists():
        if torch.load(destination, map_location="cpu", weights_only=True).get("binding") != expected:
            raise ValueError("보조 학습 체크포인트 출처가 다릅니다")
        return destination
    training.seed_everything(aux["seed"])
    model, _ = models.build_model(2, aux["scope"])
    model.to(device())
    labels = torch.tensor([row["label"] for row in index["records"]])
    weights = training.class_weights(labels, 2).to(device())
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=budget["config"]["backbone_lr"], weight_decay=budget["config"]["weight_decay"])
    amp, amp_dtype = training.amp_policy(device(), budget["config"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    latest = out / "latest.pt"
    epoch_start, batch_start = 0, 0
    if latest.exists():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        if checkpoint["binding"] != expected:
            raise ValueError("보조 학습 재개 출처가 다릅니다")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch_start, batch_start = checkpoint["epoch"], checkpoint["next_batch"]
        torch.set_rng_state(checkpoint["rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    batch_size = budget["config"]["batch_size"]
    def store(epoch, next_batch):
        training.atomic_torch(latest, {"binding": expected, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "next_batch": next_batch, "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
    for epoch in range(epoch_start, aux["epochs"]):
        model.train()
        indices = torch.randperm(len(labels), generator=torch.Generator().manual_seed(aux["seed"] + epoch))
        for batch, start in enumerate(range(0, len(indices), batch_size)):
            if epoch == epoch_start and batch < batch_start:
                continue
            try:
                deadline.check()
            except DeadlineExceeded:
                store(epoch, batch)
                append_event(out / "progress.jsonl", {"status": "paused", "reason": "global_deadline", "epoch": epoch, "next_batch": batch})
                raise
            selected = indices[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=amp_dtype):
                logits = model(training._batch(values, selected, device()))
                loss = nn.functional.cross_entropy(logits, labels[selected].to(device()), weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("보조 학습 손실에 비유한 값이 있습니다")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if (batch + 1) % 100 == 0 or start + batch_size >= len(indices):
                next_epoch = epoch + 1 if start + batch_size >= len(indices) else epoch
                next_batch = 0 if next_epoch != epoch else batch + 1
                store(next_epoch, next_batch)
                append_event(out / "progress.jsonl", {"epoch": epoch, "batch": batch, "loss": float(loss.detach()), "purpose": "전체 보조 학습이며 독립 평가가 아님"})
    training.atomic_torch(destination, {"purpose": "luna_auxiliary_for_iq_only", "binding": expected, "backbone": model.backbone.state_dict(), "epochs": aux["epochs"], "source": "LUNA25 official binary annotations"})
    del model, optimizer
    torch.cuda.empty_cache()
    return destination


def _fold_fit(name, index, fold, values, case, config, adapted, common, deadline):
    deadline.check()
    splits.validate_analysis(index["cohort"], case["initialization"])
    out = RESULT_MODELS / name / case["id"] / f"fold_{fold['fold']}"
    provenance = {"source": common["source"], "index": signature(index), "fold": fold, "case": case, "adapted_sha256": digest_file(adapted) if case["initialization"] == "luna-adapted" else None}
    binding = signature({"config": config, "provenance": provenance})
    result_path = out / "result.json"
    if result_path.exists():
        record = read_json(result_path)
        if record.get("training_binding") != binding or record.get("status") != "complete":
            raise ValueError("완료 fold 재사용의 출처가 다릅니다")
        if digest_file(record["checkpoint"]) != record["checkpoint_sha256"]:
            raise ValueError("완료 fold 모델의 해시가 다릅니다")
        return record
    training.seed_everything(case["seed"])
    if case["scope"] == "frozen":
        model = nn.Linear(512, common["classes"])
    else:
        model, _ = models.build_model(common["classes"], case["scope"], case["initialization"], adapted)
    # backbone 생성 중 소비한 RNG와 무관하게 같은 시드의 head 초기값을 맞춥니다.
    torch.manual_seed(case["seed"])
    (model if case["scope"] == "frozen" else model.head).reset_parameters()
    model.to(device())
    y = np.asarray([row["label"] for row in index["records"]], dtype=np.int64)
    train, val, test = (np.asarray(fold[key]) for key in ("train", "validation", "test"))
    result = training.fit_model(model, values[train], y[train], values[val], y[val], config, out, provenance, deadline=deadline)
    if result["status"] != "complete":
        return {"status": "paused", "fold": fold["fold"]}
    validation_logits = training.predict_logits(model, values[val], config["batch_size"], deadline=deadline)
    temperature = training.fit_temperature(validation_logits, y[val], deadline=deadline)
    test_logits = training.predict_logits(model, values[test], config["batch_size"], deadline=deadline)
    probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    raw_probabilities = torch.softmax(test_logits, dim=1).numpy()
    payload = {"status": "complete", "fold": fold["fold"], "training_binding": binding, "best_epoch": result["best_epoch"], "best_score": result["best_score"], "temperature": temperature, "metrics": evaluation.metrics(y[test], probabilities, common["classes"]), "uncalibrated_metrics": evaluation.metrics(y[test], raw_probabilities, common["classes"]), "indices": test.tolist(), "probabilities": probabilities.tolist(), "logits": test_logits.tolist(), "model_kind": "embedding_head" if case["scope"] == "frozen" else "image_classifier", "checkpoint": str(out / "best.pt"), "checkpoint_sha256": digest_file(out / "best.pt"), "case": case, "preprocessing": data.PREPROCESSING, "adapted_checkpoint": str(adapted) if case["initialization"] == "luna-adapted" else None, "adapted_sha256": provenance["adapted_sha256"]}
    atomic_json(result_path, payload)
    del model
    torch.cuda.empty_cache()
    return payload


def run(budget_path):
    budget = read_json(budget_path)
    current_source = source_identity()
    if budget["source"] != current_source or digest_file(PRIVATE / "gpu_benchmark.json") != budget["benchmark_sha256"]:
        raise ValueError("실측 예산과 현재 동결 소스가 다릅니다")
    identity = signature({"source": current_source, "budget": budget, "indices": {name: digest_file(CACHE / f"{name}_index.json") for name in ("luna", "iq")}})
    deadline = RunDeadline(PRIVATE / "run_deadline.json", budget["total_wall_seconds"], identity)
    try:
        deadline.check()
        result = _run(budget, current_source, deadline)
        if result.get("status") == "paused":
            atomic_json(PRIVATE / "run_state.json", {**result, "deadline_utc": deadline.deadline_utc, "binding": identity})
        return result
    except DeadlineExceeded:
        atomic_json(PRIVATE / "run_state.json", {"status": "paused", "reason": "global_deadline", "deadline_utc": deadline.deadline_utc, "binding": identity})
        return {"status": "paused", "reason": "global_deadline", "deadline_utc": deadline.deadline_utc}


def _run(budget, current_source, deadline):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    begun = time.monotonic()
    common = {"source": current_source}
    luna, luna_folds, luna_values = load_cohort("luna")
    iq, iq_folds, iq_values = load_cohort("iq")
    deadline.check()
    binding = {"source": current_source, "luna": signature(luna), "iq": signature(iq), "budget": signature(budget)}
    luna_features = _features("luna", "face", luna_values, None, binding, budget["config"]["batch_size"], deadline)
    luna_case = {"id": "original_face_frozen", "scope": "frozen", "initialization": "face", "seed": 11}
    luna_config = {**budget["config"], "seed": 11, "max_epochs": budget["luna_head_epochs"], "batch_size": 128, "warmup_epochs": 2, "amp": False}
    a_results = [_fold_fit("luna_binary", luna, fold, luna_features, luna_case, luna_config, None, {**common, "classes": 2}, deadline) for fold in luna_folds]
    if any(result["status"] != "complete" for result in a_results):
        return {"status": "paused", "phase": "luna_binary"}
    a_oof = np.zeros((len(luna["records"]), 2))
    for result in a_results:
        a_oof[result["indices"]] = result["probabilities"]
    a_y = np.asarray([row["label"] for row in luna["records"]])
    atomic_json(PRIVATE / "luna_binary_oof.json", {"folds": a_results, "aggregate": evaluation.metrics(a_y, a_oof, 2), "patient_confidence": evaluation.patient_bootstrap(a_y, a_oof, [row["patient_id"] for row in luna["records"]], known_patients=True, deadline=deadline), "analysis": "원래 얼굴 고정 특징의 환자별 7fold 이진 평가"})
    adapted = _auxiliary(luna, luna_values, budget, binding, deadline)
    completed_cases = []
    for case in models.experiment_cases():
        deadline.check()
        values = _features("iq", case["initialization"], iq_values, adapted, binding, budget["config"]["batch_size"], deadline) if case["scope"] == "frozen" else iq_values
        config = {**budget["config"], "seed": case["seed"], "amp": case["scope"] != "frozen"}
        results = [_fold_fit("iq", iq, fold, values, case, config, adapted, {**common, "classes": 3}, deadline) for fold in iq_folds]
        if any(result["status"] != "complete" for result in results):
            return {"status": "paused", "phase": "iq", "case": case}
        oof = np.zeros((len(iq["records"]), 3))
        for result in results:
            oof[result["indices"]] = result["probabilities"]
        y = [row["label"] for row in iq["records"]]
        aggregate = evaluation.metrics(y, oof, 3)
        deadline.check()
        atomic_json(PRIVATE / f"iq_case_{case['id']}_oof.json", {"case": case, "folds": results, "aggregate": aggregate, "patient_confidence_interval": None, "patient_mapping": "unavailable"})
        evaluation.write_case_report(SOURCE / f"reports/설정_{case['id']}.md", case, results, aggregate)
        completed_cases.append(case["id"])
        atomic_json(PRIVATE / "run_state.json", {"status": "running", "completed_cases": completed_cases, "case": case})
    selected = []
    for fold in range(7):
        deadline.check()
        candidates = [read_json(RESULT_MODELS / f"iq/{case['id']}/fold_{fold}/result.json") for case in models.experiment_cases()]
        chosen = sorted(candidates, key=lambda result: (-result["best_score"], result["case"]["id"]))[0]
        selected.append(chosen)
    atomic_json(PRIVATE / "stage3_handoff.json", {"selection": "각 outer fold의 내부 validation macro-F1 최대; 동률은 사전 설정 ID 오름차순", "folds": selected, "index": str(CACHE / "iq_index.json"), "split": str(CACHE / "iq_folds.json"), "patient_mapping_known": False, "source": current_source})
    atomic_json(PRIVATE / "run_state.json", {"status": "complete", "completed_cases": completed_cases, "seconds": time.monotonic() - begun})
    return {"status": "complete", "cases": len(completed_cases)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="2단계 실제 CT 자료 준비·측정·순차 학습")
    parser.add_argument("command", choices=("prepare", "benchmark", "run"))
    parser.add_argument("--budget", type=Path, default=PRIVATE / "run_budget.json")
    args = parser.parse_args(argv)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    try:
        result = prepare() if args.command == "prepare" else benchmark() if args.command == "benchmark" else run(args.budget)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as error:
        append_event(PRIVATE / "runtime_failures.jsonl", {**runtime_context(), "argv": [__import__("sys").executable, *__import__("sys").argv], "cwd": str(Path.cwd()), "exit_code": 1, "error_type": type(error).__name__, "error": str(error), "traceback": __import__("traceback").format_exc(), "changed": "호출에 따라 캐시 또는 체크포인트가 생성되었을 수 있음", "counters": "부모 중앙 기록과 대조 필요; 이 런타임은 중앙 누적을 변경하지 않음"})
        raise


if __name__ == "__main__":
    main()
