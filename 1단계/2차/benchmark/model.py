"""고정 저자 R200 구조와 원본 가중치를 안전하게 적재합니다."""
import importlib.util
import json
from pathlib import Path
import sys
from .acquisition import _check_git_blob, now
from .safety import digest_file


def safe_state_dict(path):
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        raise ValueError("안전한 weights_only 역직렬화에 실패했습니다") from error
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("원래의 비어 있지 않은 tensor state_dict가 아닙니다")
    return state


def validate_state_dict(state, expected):
    import torch
    if set(state) != set(expected):
        raise ValueError("저자 R200 구조와 tensor 키가 다릅니다")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype or not torch.isfinite(tensor).all().item():
            raise ValueError("tensor 형상·dtype·유한값이 저자 구조와 다릅니다: " + key)


def construct_topofr(package):
    # 고정 저자 eval_ijbc_glint360k.py와 glint360k_r50.py의 생성 인자입니다.
    return package.get_model("r200", dropout=0, fp16=False, num_features=512, num_classes=360232)


def load_topofr(stage, *, threads=4):
    import torch
    stage = Path(stage).resolve()
    torch.set_num_threads(threads)
    source_root = stage / "2차"
    manifest = json.loads((source_root / "benchmark/sources.json").read_text(encoding="utf-8"))
    revision = manifest["topofr_revision"]
    vendor = stage / "models/topofr/vendor" / revision / "backbones"
    source_hashes = {}
    for item in manifest["files"]:
        if item["id"].startswith("vendor_"):
            path = stage / item["target"]
            _check_git_blob(path, item["git_blob_sha1"])
            source_hashes[item["name"]] = digest_file(path)
    weight_spec = next(item for item in manifest["files"] if item["id"] == "topofr_r200")
    checkpoint = stage / weight_spec["target"]
    receipt = json.loads(checkpoint.with_name(checkpoint.name + ".receipt.json").read_text(encoding="utf-8"))
    actual = digest_file(checkpoint)
    if checkpoint.stat().st_size != weight_spec["size"] or receipt.get("sha256") != actual or receipt.get("source_url") != weight_spec["url"]:
        raise ValueError("최초 획득한 원본 모델의 길이·로컬 SHA·출처가 다릅니다")
    package_name = "topofr_fixed_" + revision
    spec = importlib.util.spec_from_file_location(package_name, vendor / "__init__.py", submodule_search_locations=[str(vendor)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
    model = construct_topofr(package)
    state = safe_state_dict(checkpoint)
    validate_state_dict(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    evidence = {"verified_at": now(), "architecture": "r200", "constructor": {"dropout": 0, "fp16": False, "num_features": 512, "num_classes": 360232}, "name": "TopoFR-R200 Glint360K", "source_revision": revision, "source_sha256": source_hashes, "checkpoint_bytes": checkpoint.stat().st_size, "checkpoint_sha256": actual, "checkpoint_official_digest_verified": False, "hash_note": "최초 로컬 SHA 고정과 대조이며 저자 공식 체크섬 대조가 아닙니다", "weights_only": True, "strict": True, "tensor_finite": True, "precision": "float32", "device": "cpu", "parameter_count": sum(value.numel() for value in model.parameters()), "face_accuracy_measured": False}
    return model, evidence
