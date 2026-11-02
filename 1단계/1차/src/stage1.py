"""합성 진단과 정상·양성·악성 전이에 공통으로 사용하는 엄격한 경계."""

import hashlib
import os
import random
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("시드는 0 이상 2**32 미만 정수여야 합니다")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def preprocess(image: torch.Tensor, *, color_order: str = "RGB") -> torch.Tensor:
    """CHW uint8[0,255] 또는 실수[0,1]을 [-1,1] 3×112×112로 변환합니다.

    의료 원본의 HU 변환·윈도우는 이 함수가 수행하지 않습니다. 호출자는
    사전 정의한 윈도우로 [0,1]을 만든 뒤 전달해야 합니다.
    """
    if color_order not in ("RGB", "BGR"):
        raise ValueError("색상 순서는 RGB 또는 BGR이어야 합니다")
    if not isinstance(image, torch.Tensor):
        raise TypeError("입력은 torch.Tensor여야 합니다")
    if image.ndim != 3 or image.shape[0] not in (1, 3) or min(image.shape[1:]) < 1:
        raise ValueError("입력은 비어 있지 않은 1 또는 3 채널 CHW여야 합니다")
    if image.dtype == torch.uint8:
        x = image.to(torch.float32) / 255.0
    elif image.is_floating_point():
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("실수 입력은 유한한 [0,1] 범위여야 합니다")
        x = image.to(torch.float32)
    else:
        raise TypeError("uint8 또는 실수 입력만 지원합니다")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if color_order == "BGR":
        x = x.flip(0)
    if x.shape[1:] != (112, 112):
        x = F.interpolate(x.unsqueeze(0), size=(112, 112), mode="bilinear",
                          align_corners=False, antialias=True).squeeze(0)
    return x.sub(0.5).div(0.5).contiguous()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(module: nn.Module, path: str | Path, *, state_key: str | None = None,
                    prefix: str = "", expected_sha256: str | None = None) -> dict:
    """검토한 로컬 파일만 weights_only=True 및 strict=True로 적재합니다."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("체크포인트 SHA-256 불일치")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_key is not None:
        if not isinstance(state, Mapping) or state_key not in state:
            raise ValueError("명시한 state_dict 키가 없습니다")
        state = state[state_key]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("비어 있지 않은 state_dict 매핑이 필요합니다")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("state_dict는 문자열 키와 Tensor 값만 허용합니다")
    if prefix:
        if not all(k.startswith(prefix) for k in state):
            raise ValueError("모든 키에 명시한 접두사가 있어야 합니다")
        state = {k[len(prefix):]: v for k, v in state.items()}
    expected = module.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(f"state_dict 키 불일치: 누락={sorted(set(expected)-set(state))}, 추가={sorted(set(state)-set(expected))}")
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise RuntimeError(f"state_dict 형상 불일치: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"비유한 체크포인트 값: {key}")
    module.load_state_dict(state, strict=True)
    return {"sha256": digest, "state_key": state_key, "removed_prefix": prefix,
            "tensor_count": len(state), "strict": True, "weights_only": True}


class TransferClassifier(nn.Module):
    """512차원 백본 위의 정상·양성·악성 세 로짓 헤드입니다."""

    def __init__(self, backbone: nn.Module, *, frozen: bool = True):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(512, 3)
        self.set_frozen(frozen)

    def set_frozen(self, frozen: bool) -> None:
        if not isinstance(frozen, bool):
            raise TypeError("동결 설정은 bool이어야 합니다")
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        # 이전 훈련의 경사를 재동결 상태에 남기지 않습니다.
        if frozen:
            for parameter in self.backbone.parameters():
                parameter.grad = None
        self.backbone.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[0] < 1 or x.shape[1:] != (3, 112, 112):
            raise ValueError("백본 입력은 N×3×112×112여야 합니다")
        if not x.is_floating_point() or not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1:
            raise ValueError("백본 입력은 유한한 실수 [-1,1]이어야 합니다")
        if self.frozen:
            with torch.no_grad():
                result = self.backbone(x)
        else:
            result = self.backbone(x)
        if not isinstance(result, torch.Tensor) or result.shape != (x.shape[0], 512):
            raise ValueError("백본은 N×512 Tensor를 반환해야 합니다")
        if not torch.isfinite(result).all():
            raise ValueError("백본 특징이 비유한 값입니다")
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(x))
        if not torch.isfinite(logits).all():
            raise ValueError("분류 로짓이 비유한 값입니다")
        return logits
