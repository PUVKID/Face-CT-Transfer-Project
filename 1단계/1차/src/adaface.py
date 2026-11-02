"""검토한 공식 AdaFace IR101 버전의 로컬 적재만 허용합니다."""

import importlib.util
from pathlib import Path

from .stage1 import load_checkpoint, sha256

REVISION = "54f602a0737bd1ee4a4e7e9fd089a485f397fefd"
SOURCE_SHA256 = "e31be55c60b538c15151887e911b7535e2f0e1114a13427ba06483e1cd2a63f9"
WEIGHTS_SHA256 = "e312d79222d28027f146ed495e182e48fe0dddf404cdbbdabcccdbdd07cc3758"


def load_adaface(source: str | Path, checkpoint: str | Path):
    source = Path(source)
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("공식 모델 소스 SHA-256 불일치: 실행을 거부합니다")
    spec = importlib.util.spec_from_file_location("pinned_cvlface_ir101", source)
    if spec is None or spec.loader is None:
        raise ImportError("공식 모델 모듈을 열 수 없습니다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backbone = module.IR_101(input_size=(112, 112), output_dim=512)
    evidence = load_checkpoint(backbone, checkpoint, prefix="net.", expected_sha256=WEIGHTS_SHA256)
    evidence.update({"model": "AdaFace IR101 WebFace12M", "revision": REVISION,
                     "source_sha256": SOURCE_SHA256, "color_order": "RGB",
                     "input_shape": [3, 112, 112], "normalization": "(x - 0.5) / 0.5"})
    return backbone, evidence
