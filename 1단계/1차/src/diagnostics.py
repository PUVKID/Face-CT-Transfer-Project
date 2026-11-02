"""서로 다른 합성 엔지니어링 검사를 실행하며 임상 수치를 만들지 않습니다."""

import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .stage1 import load_checkpoint, preprocess, seed_everything

CASE_IDS = (
    "rgb_endpoints", "grayscale_replication", "resize_non_square", "bgr_explicit",
    "float_normalization", "reject_nan", "reject_infinity", "reject_range",
    "reject_channels", "reject_shape", "inference_single", "inference_batch",
    "batch_independence", "deterministic_repeat", "input_sensitivity", "frozen_gradient",
    "frozen_bn", "head_gradient", "head_update", "unfrozen_gradient", "refreeze",
    "checkpoint_roundtrip", "missing_checkpoint", "mismatch_checkpoint", "seed_reproduction",
    "cuda_fp32", "cuda_fp16", "cuda_bfloat16", "channels_last", "cpu_cuda_parity",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if len(cases) != 30 or {c.get("id") for c in cases} != set(CASE_IDS):
        raise ValueError("정의된 30개 진단은 중복·누락 없이 있어야 합니다")
    if any(not isinstance(c.get("hypothesis"), str) or not c["hypothesis"].strip() for c in cases):
        raise ValueError("각 진단에 가설이 필요합니다")
    return cases


def run_case(case_id, model, *, device, private_temp):
    """부작용을 격리해 실제 백본으로 각 가설을 검사합니다."""
    buffers = {k: v.detach().clone() for k, v in model.backbone.named_buffers()}
    head = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    frozen, training = model.frozen, model.training
    try:
        return _run_case(case_id, model, device=device, private_temp=private_temp)
    finally:
        # 실패한 검사도 다음 검사의 사전학습 상태를 바꾸지 않습니다.
        with torch.no_grad():
            current = dict(model.backbone.named_buffers())
            for key, value in buffers.items():
                current[key].copy_(value)
        model.head.load_state_dict(head, strict=True)
        model.zero_grad(set_to_none=True)
        model.set_frozen(frozen)
        model.train(training)


def _run_case(case_id, model, *, device, private_temp):
    seed_everything(20261103)
    model.zero_grad(set_to_none=True)
    model.set_frozen(True)
    model.eval()
    model.to(device=device, memory_format=torch.contiguous_format)
    x = torch.rand(2, 3, 112, 112, device=device).mul(2).sub(1)
    if case_id == "rgb_endpoints":
        y = preprocess(torch.stack([torch.zeros(112, 112), torch.full((112, 112), 255), torch.zeros(112, 112)]).byte())
        require(y[0].eq(-1).all() and y[1].eq(1).all(), "정규화 끝값 불일치")
        return {"min": y.min().item(), "max": y.max().item()}
    if case_id == "grayscale_replication":
        y = preprocess(torch.linspace(0, 1, 32 * 40).reshape(1, 32, 40))
        require(torch.equal(y[0], y[1]) and torch.equal(y[1], y[2]), "채널 복제 불일치")
        return {"shape": list(y.shape)}
    if case_id == "resize_non_square":
        y = preprocess(torch.rand(3, 60, 180))
        require(y.shape == (3, 112, 112) and torch.isfinite(y).all(), "크기 변환 실패")
        return {"original_shape": [3, 60, 180], "output_shape": list(y.shape)}
    if case_id == "bgr_explicit":
        raw = torch.rand(3, 112, 112)
        require(torch.equal(preprocess(raw).flip(0), preprocess(raw, color_order="BGR")), "BGR 변환 불일치")
        return {"explicit_channel_reversal": True}
    if case_id == "float_normalization":
        y = preprocess(torch.full((3, 112, 112), 0.5))
        require(y.eq(0).all(), "중간값 정규화 불일치")
        return {"normalized_midpoint": 0.0}
    invalid = {"reject_nan": torch.full((3, 8, 8), float("nan")),
               "reject_infinity": torch.full((3, 8, 8), float("inf")),
               "reject_range": torch.full((3, 8, 8), 1.01),
               "reject_channels": torch.zeros(4, 8, 8), "reject_shape": torch.zeros(8, 8)}
    if case_id in invalid:
        try:
            preprocess(invalid[case_id])
        except ValueError as error:
            return {"expected_rejection": type(error).__name__, "message": str(error)}
        raise AssertionError("잘못된 입력이 거부되지 않았습니다")
    if case_id in ("inference_single", "inference_batch"):
        batch = 1 if case_id == "inference_single" else 2
        with torch.inference_mode():
            features = model.features(x[:batch])
            logits = model.head(features)
        require(features.shape == (batch, 512) and logits.shape == (batch, 3) and torch.isfinite(logits).all(), "추론 형상/유한성 실패")
        return {"features_shape": list(features.shape), "logits_shape": list(logits.shape), "feature_l2": features.norm(dim=1).tolist()}
    if case_id in ("batch_independence", "deterministic_repeat", "input_sensitivity"):
        with torch.inference_mode():
            a = model.features(x[:1])
            b = model.features(x)[0:1] if case_id == "batch_independence" else model.features(x[:1] if case_id == "deterministic_repeat" else -x[:1])
        difference = (a - b).abs().max().item()
        if case_id == "input_sensitivity":
            require(difference > 1e-6, "입력 변화에 특징이 반응하지 않습니다")
        else:
            require(torch.allclose(a, b, atol=1e-4, rtol=1e-4), "추론 일관성 실패")
        return {"max_abs_difference": difference, "atol": 1e-4, "rtol": 1e-4}
    if case_id in ("frozen_gradient", "frozen_bn", "head_gradient", "head_update", "unfrozen_gradient", "refreeze"):
        if case_id in ("unfrozen_gradient", "refreeze"):
            model.set_frozen(False)
        if case_id == "refreeze":
            model.set_frozen(True)
        model.train()
        bn_before = {k: v.clone() for k, v in model.backbone.named_buffers() if "running_" in k or "num_batches_tracked" in k}
        head_before = model.head.weight.detach().clone()
        loss = F.cross_entropy(model(x), torch.tensor([0, 2], device=device))
        require(torch.isfinite(loss), "비유한 손실")
        loss.backward()
        gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
        head_norm = model.head.weight.grad.norm().item()
        require(head_norm > 0 and torch.isfinite(model.head.weight.grad).all(), "헤드 경사 실패")
        if case_id == "unfrozen_gradient":
            require(gradients and all(torch.isfinite(g).all() for g in gradients), "백본 경사 누락/비유한")
            require(sum(g.abs().sum().item() for g in gradients) > 0, "백본 경사가 모두 0입니다")
        else:
            require(not gradients, "동결 백본에 경사가 생겼습니다")
            current = dict(model.backbone.named_buffers())
            require(all(torch.equal(v, current[k]) for k, v in bn_before.items()), "동결 BN 통계가 변경되었습니다")
        change = 0.0
        if case_id == "head_update":
            optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01)
            optimizer.step()
            change = (head_before - model.head.weight).abs().max().item()
            require(change > 0, "헤드 파라미터가 갱신되지 않았습니다")
            optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        model.set_frozen(True)
        model.eval()
        return {"loss": loss.item(), "head_gradient_norm": head_norm,
                "backbone_gradient_tensors": len(gradients), "head_max_update": change,
                "bn_buffer_count": len(bn_before)}
    if case_id in ("checkpoint_roundtrip", "missing_checkpoint", "mismatch_checkpoint"):
        with tempfile.TemporaryDirectory(dir=private_temp) as tmp:
            path = Path(tmp) / "diagnostic.pt"
            if case_id == "checkpoint_roundtrip":
                before = {k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()}
                torch.save(before, path)
                metadata = load_checkpoint(model.backbone, path)
                require(all(torch.equal(v, model.backbone.state_dict()[k].cpu()) for k, v in before.items()), "왕복 상태 불일치")
                return metadata
            if case_id == "mismatch_checkpoint":
                torch.save({"wrong_key": torch.zeros(1)}, path)
            try:
                load_checkpoint(model.backbone, path)
            except (FileNotFoundError, RuntimeError) as error:
                return {"expected_rejection": type(error).__name__}
            raise AssertionError("잘못된 체크포인트를 거부하지 않았습니다")
    if case_id == "seed_reproduction":
        seed_everything(7)
        a = nn.Linear(512, 3).weight.detach().clone()
        seed_everything(7)
        b = nn.Linear(512, 3).weight.detach().clone()
        require(torch.equal(a, b), "헤드 초기값 재현 실패")
        return {"equal_initial_weights": True}
    if case_id in ("cuda_fp32", "cuda_fp16", "cuda_bfloat16", "cpu_cuda_parity") and not torch.cuda.is_available():
        return {"unsupported": "CUDA 장치 없음"}
    if case_id in ("cuda_fp32", "cuda_fp16", "cuda_bfloat16"):
        if case_id == "cuda_bfloat16" and not torch.cuda.is_bf16_supported():
            return {"unsupported": "장치의 bfloat16 미지원"}
        model.cuda()
        cuda_x = x.cuda()
        dtype = torch.float16 if case_id == "cuda_fp16" else torch.bfloat16
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=case_id != "cuda_fp32"):
            y = model(cuda_x)
        require(torch.isfinite(y).all(), "CUDA 추론의 비유한 출력")
        return {"output_dtype": str(y.dtype), "shape": list(y.shape), "device": str(y.device)}
    if case_id == "channels_last":
        with torch.inference_mode():
            baseline = model(x)
            try:
                y = model(x.contiguous(memory_format=torch.channels_last))
            except RuntimeError as error:
                if "view size is not compatible" in str(error):
                    return {"unsupported": "공식 백본의 view 기반 Flatten이 channels_last를 지원하지 않음", "error": str(error)}
                raise
        require(torch.allclose(baseline, y, atol=1e-4, rtol=1e-4), "메모리 배치 출력 불일치")
        return {"max_abs_difference": (baseline-y).abs().max().item(), "atol": 1e-4, "rtol": 1e-4}
    if case_id == "cpu_cuda_parity":
        raw = x[:1].cpu()
        model.cpu()
        with torch.inference_mode():
            a = model(raw)
        # 장치 이동은 일반 Tensor 버퍼를 유지하도록 추론 구간 밖에서 수행합니다.
        model.cuda()
        with torch.inference_mode():
            b = model(raw.cuda()).cpu()
        require(torch.allclose(a, b, atol=1e-3, rtol=1e-3), "CPU/CUDA 출력 불일치")
        model.to(device)
        return {"max_abs_difference": (a-b).abs().max().item(), "atol": 1e-3, "rtol": 1e-3}
    raise ValueError(f"정의되지 않은 진단: {case_id}")
