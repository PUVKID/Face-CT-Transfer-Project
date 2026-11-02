"""동결된 원본 얼굴 모델과 명시한 학습 범위를 구성합니다."""

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn

from .io import digest_file


def project_root():
    return Path(__file__).resolve().parents[4]


def experiment_cases():
    return [{"id": f"{index:02d}", "initialization": initialization, "scope": scope, "seed": seed}
            for index, (initialization, scope, seed) in enumerate(
                ((initialization, scope, seed) for initialization in ("face", "luna-adapted")
                 for scope in ("frozen", "lastblock", "full") for seed in (11, 22, 33, 44, 55)), 1)]


def face_backbone(root=None):
    root = Path(root or project_root())
    source = root / "1단계/1차/src"
    name = "_stage1_frozen_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / "__init__.py", submodule_search_locations=[str(source)])
        if spec is None or spec.loader is None:
            raise ImportError("동결된 1단계 적재 모듈을 열 수 없습니다")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    from importlib import import_module
    loader = import_module(name + ".adaface")
    return loader.load_adaface(root / "1단계/work/vendor/cvlface/models/iresnet/model.py", root / "1단계/models/adaface_ir101_webface12m.pt")


class Classifier(nn.Module):
    def __init__(self, backbone, classes, scope):
        super().__init__()
        if scope not in ("frozen", "lastblock", "full"):
            raise ValueError("지원하지 않는 학습 범위입니다")
        self.backbone = backbone
        self.head = nn.Linear(512, classes)
        self.scope = scope
        for parameter in backbone.parameters():
            parameter.requires_grad_(scope == "full")
        if scope == "lastblock":
            for parameter in backbone.body[-1].parameters():
                parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.scope == "frozen":
            self.backbone.eval()
        # 작은 배치와 범위별 통계 차이를 막기 위해 BN 통계를 공통 고정합니다.
        for layer in self.backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def forward(self, images):
        features = self.backbone(images)
        if isinstance(features, tuple):
            features = features[0]
        return self.head(features)


def build_model(classes, scope, initialization="face", adapted_path=None, root=None):
    backbone, evidence = face_backbone(root)
    if initialization == "luna-adapted":
        if adapted_path is None:
            raise ValueError("의료 적응 체크포인트가 필요합니다")
        checkpoint = torch.load(adapted_path, map_location="cpu", weights_only=True)
        if checkpoint.get("purpose") != "luna_auxiliary_for_iq_only":
            raise ValueError("적응 체크포인트의 사용 목적이 다릅니다")
        backbone.load_state_dict(checkpoint["backbone"], strict=True)
        evidence["adapted_sha256"] = digest_file(adapted_path)
    elif initialization != "face":
        raise ValueError("지원하지 않는 초기화입니다")
    return Classifier(backbone, classes, scope), evidence


def restore_for_stage3(fold_record, root=None):
    if digest_file(fold_record["checkpoint"]) != fold_record["checkpoint_sha256"]:
        raise ValueError("3단계에 전달한 fold 체크포인트 해시가 다릅니다")
    case = fold_record["case"]
    if case["initialization"] == "luna-adapted" and digest_file(fold_record["adapted_checkpoint"]) != fold_record["adapted_sha256"]:
        raise ValueError("3단계 보조 적응 backbone의 해시가 다릅니다")
    model, _ = build_model(3, case["scope"], case["initialization"], fold_record["adapted_checkpoint"], root)
    checkpoint = torch.load(fold_record["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["binding"] != fold_record["training_binding"]:
        raise ValueError("3단계 모델과 분할·학습 설정 결합이 다릅니다")
    if fold_record["model_kind"] == "embedding_head":
        model.head.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), float(fold_record["temperature"])
