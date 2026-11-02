"""분할에 결합한 원자 체크포인트와 검증 전용 조기 종료입니다."""

import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

from .io import append_event, atomic_json, signature
from .budget import DeadlineExceeded


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def amp_policy(device, config):
    enabled = bool(config["amp"] and device.type == "cuda")
    if enabled:
        if config.get("amp_dtype") != "bfloat16":
            raise ValueError("CUDA AMP는 사전 고정한 bfloat16을 명시해야 합니다")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("현재 CUDA 장치는 native bfloat16을 지원하지 않습니다")
    return enabled, torch.bfloat16


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def class_weights(labels, classes):
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=classes).float()
    if len(counts) != classes or torch.any(counts == 0):
        raise ValueError("학습 분할에 모든 클래스가 필요합니다")
    return counts.sum() / (classes * counts)


def _validate_tensor_state(model, state):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("체크포인트 tensor 키가 모델과 다릅니다")
    for key, reference in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
            raise ValueError("체크포인트 tensor 형상·dtype·유한값이 다릅니다: " + key)


def restore_best(model, path, binding, best_epoch):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("binding") != binding:
        raise ValueError("최적 체크포인트의 분할·설정 결합이 다릅니다")
    if type(checkpoint.get("epoch")) is not int or checkpoint["epoch"] != best_epoch:
        raise ValueError("최적 체크포인트 epoch가 기록한 best_epoch와 다릅니다")
    _validate_tensor_state(model, checkpoint.get("model"))
    # 모든 상태를 검사한 뒤 한 번에 복원해 부분적인 모델 변경을 막습니다.
    model.load_state_dict(checkpoint["model"], strict=True)


def _batch(values, indices, device):
    if isinstance(values, np.ndarray):
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise ValueError("NumPy 배치에는 1차원 정수 인덱스가 필요합니다")
        # Torch 단일 원소의 scalar __index__ 대신 NumPy 고급 인덱싱을 명시합니다.
        selected = np.asarray(values[indices])
        if not selected.flags.writeable:
            selected = selected.copy()
        result = torch.from_numpy(selected)
    else:
        result = values[indices]
    result = result.to(device, dtype=torch.float32)
    if result.ndim == 4 and result.shape[1] == 1:
        result = result.repeat(1, 3, 1, 1)
    return result.contiguous()


def predict_logits(model, inputs, batch_size=32, device=None, *, deadline=None):
    device = device or next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            if deadline is not None:
                deadline.check()
            result = model(_batch(inputs, slice(start, start + batch_size), device))
            if isinstance(result, tuple):
                result = result[0]
            outputs.append(result.float().cpu())
    if not outputs:
        raise ValueError("빈 추론 자료입니다")
    return torch.cat(outputs)


def fit_model(model, train_x, train_y, val_x, val_y, config, output, provenance, *, max_epochs_this_call=None, deadline=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if len(train_x) != len(train_y) or len(val_x) != len(val_y) or not len(train_y) or not len(val_y):
        raise ValueError("학습·검증 길이가 유효하지 않습니다")
    seed_everything(int(config["seed"]))
    device = next(model.parameters()).device
    train_y = torch.as_tensor(train_y, dtype=torch.long)
    val_y = torch.as_tensor(val_y, dtype=torch.long)
    classes = int(max(train_y.max(), val_y.max()).item()) + 1
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_y, classes).to(device))
    if hasattr(model, "backbone"):
        groups = [{"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config["backbone_lr"]},
                  {"params": list(model.head.parameters()), "lr": config["lr"]}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config["lr"]}]
    groups = [group for group in groups if group["params"]]
    base_lrs = [group["lr"] for group in groups]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    # 제너레이터를 소모하지 않도록 위 그룹의 params는 실제 목록으로 고정합니다.
    amp, amp_dtype = amp_policy(device, config)
    # BF16은 FP32와 같은 지수 범위를 사용하므로 경사 스케일링을 적용하지 않습니다.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    binding = signature({"config": config, "provenance": provenance})
    latest = output / "latest.pt"
    best = output / "best.pt"
    start_epoch, best_score, best_epoch, stale, history = 0, -1., -1, 0, []
    resume_sample, resume_losses = 0, []
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state.get("binding") != binding:
            raise ValueError("재개 분할·소스·설정 결합이 다릅니다")
        _validate_tensor_state(model, state.get("model"))
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1 if state.get("epoch_complete", True) else state["epoch"]
        resume_sample = 0 if state.get("epoch_complete", True) else state["next_sample"]
        resume_losses = [] if state.get("epoch_complete", True) else state["epoch_losses"]
        best_score, best_epoch, stale, history = state["best_score"], state["best_epoch"], state["stale"], state["history"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if state["status"] == "complete":
            restore_best(model, best, binding, best_epoch)
            return {"status": "complete", "best_epoch": best_epoch, "best_score": best_score, "history": history, "binding": binding}
    begun = time.monotonic()
    status = "paused"
    def store(epoch, epoch_complete, status, next_sample, losses):
        state = {"status": status, "binding": binding, "epoch": epoch, "epoch_complete": epoch_complete, "next_sample": next_sample, "epoch_losses": losses, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "best_score": best_score, "best_epoch": best_epoch, "stale": stale, "history": history, "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        atomic_torch(latest, state)
        atomic_json(output / "state.json", {"status": status, "binding": binding, "epoch": epoch, "epoch_complete": epoch_complete, "next_sample": next_sample, "best_epoch": best_epoch, "best_score": best_score})
    def paused(epoch, next_sample, losses):
        store(epoch, False, "paused", next_sample, losses)
        append_event(output / "epochs.jsonl", {"status": "paused", "reason": "global_deadline", "epoch": epoch, "next_sample": next_sample})
        return {"status": "paused", "best_epoch": best_epoch, "best_score": best_score, "history": history, "binding": binding}
    for epoch in range(start_epoch, int(config["max_epochs"])):
        if epoch > start_epoch and time.monotonic() - begun > config["wall_seconds"]:
            break
        model.train()
        warmup = int(config["warmup_epochs"])
        multiplier = (epoch + 1) / warmup if epoch < warmup else .5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, config["max_epochs"] - warmup)))
        for group, initial_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = initial_lr * multiplier
        permutation = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(config["seed"] + epoch))
        losses = list(resume_losses) if epoch == start_epoch else []
        for start in range(0, len(permutation), config["batch_size"]):
            if epoch == start_epoch and start < resume_sample:
                continue
            if deadline is not None:
                try:
                    deadline.check()
                except DeadlineExceeded:
                    return paused(epoch, start, losses)
            indices = permutation[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp, dtype=amp_dtype):
                logits = model(_batch(train_x, indices, device))
                loss = loss_fn(logits, train_y[indices].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("학습 손실에 비유한 값이 있습니다")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 5.)), error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        try:
            validation = predict_logits(model, val_x, config["batch_size"], device, deadline=deadline)
        except DeadlineExceeded:
            return paused(epoch, len(permutation), losses)
        score = float(f1_score(val_y.numpy(), validation.argmax(1).numpy(), average="macro", labels=list(range(classes)), zero_division=0))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            atomic_torch(best, {"model": model.state_dict(), "binding": binding, "epoch": epoch})
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": score})
        status = "complete" if epoch + 1 == config["max_epochs"] or stale >= config["patience"] else "paused"
        store(epoch, True, status, 0, [])
        append_event(output / "epochs.jsonl", history[-1])
        if status == "complete" or (max_epochs_this_call is not None and epoch - start_epoch + 1 >= max_epochs_this_call):
            break
    if status == "complete":
        restore_best(model, best, binding, best_epoch)
    return {"status": status, "best_epoch": best_epoch, "best_score": best_score, "history": history, "binding": binding}


def fit_temperature(validation_logits, validation_labels, *, deadline=None):
    logits = torch.as_tensor(validation_logits).detach().float().cpu()
    labels = torch.as_tensor(validation_labels, dtype=torch.long).cpu()
    if len(logits) != len(labels) or not torch.isfinite(logits).all():
        raise ValueError("온도 보정 자료가 유효하지 않습니다")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=50, line_search_fn="strong_wolfe")
    def closure():
        if deadline is not None:
            deadline.check()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_temperature.exp().clamp(.05, 20.), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(.05, 20.))
