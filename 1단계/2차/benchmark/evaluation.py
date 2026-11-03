"""원본 LFW의 독립 정렬과 원저자 지표 규칙을 분리해 실행합니다."""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from .acquisition import atomic_json, now
from .metrics import lfw_ten_fold
from .model import load_topofr
from .protocol import similarity_matrix
from .safety import digest_file, parse_lfw_pairs, safe_extract_tar
from . import lfw_primitive


ALIGNMENT_PROTOCOL = {
    "detector": "OpenCV FaceDetectorYN YuNet 2023mar",
    "opencv_distribution": "opencv-python-headless", "opencv_version": "4.13.0.92",
    "source_revision": "48b10fa505aeb51fbd127fc60a5c660040e823b7",
    "weight_sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "score_threshold": 0.9, "nms_threshold": 0.3, "top_k": 5000,
    "backend": "OpenCV DNN CPU", "selection": "영상 중심까지 bbox 중심 제곱거리 최소; 동률은 큰 면적, 높은 점수, 좌표 사전순",
    "landmarks": "YuNet의 오른눈·왼눈·코·오른입·왼입 5점",
    "warp": "5점 최소제곱 similarity; OpenCV INTER_LINEAR; 상수 검정 border; 112x112 RGB uint8",
    "failure": "실패 영상과 coverage를 보존; 실패 pair 삭제·임의 resize·정확도 산출 금지",
    "author_input_equivalence": False,
    "difference": "저자 aligned112/bin의 영상 바이트·정렬과 동일성 미확인. 원본 LFW에 새 검출기를 적용한 독립 재측정입니다."
}

OFFICIAL_ALIGNMENT_PROTOCOL = {
    "input_kind": "official-ms1mv2", "source": "공식 MS1MV2 faces_emore/lfw.bin",
    "images": 12000, "pairs": 6000, "shape": [112, 112, 3], "dtype": "uint8",
    "pillow_version": "13.0.0", "opencv_version": "4.13.0.92",
    "decode": "Pillow header만 검사: JPEG/PNG RGB 단일112x112; OpenCV IMREAD_UNCHANGED BGR uint8 → RGB",
    "resize": False, "realignment": False, "detector_used": False,
    "pair_order": "연속 두 영상이 한 pair; 공식 pairs.txt의 6000 issame와 10x600 순서 전부 대조",
    "image_identity_independently_verified": False, "author_input_equivalence": False,
    "difference": "bin에는 영상 ID가 없어 인물 ID 독립 대조 불가. TopoFR 당시 파일 SHA 및 MXNet과 OpenCV 픽셀 동일성 미확인."
}


def decode_official_image(raw):
    import cv2
    from PIL import Image
    if type(raw) is not bytes or not 0 < len(raw) <= lfw_primitive.MAX_IMAGE_BYTES:
        raise ValueError("공식 입력 영상의 encoded bytes 상한과 다릅니다")
    with Image.open(io.BytesIO(raw), formats=["JPEG", "PNG"]) as header:
        if header.size != (112, 112) or header.mode != "RGB" or getattr(header, "n_frames", 1) != 1:
            raise ValueError("제공된 정렬 영상 header가 단일 RGB112x112가 아닙니다")
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.shape != (112, 112, 3) or bgr.dtype != np.uint8:
        raise ValueError("공식 정렬 영상 decode의 shape·dtype가 다릅니다")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _official_binding(stage, pair_hash):
    return {"input_kind": "official-ms1mv2", "input_sha256": lfw_primitive.FILE_SHA256,
            "receipt_sha256": lfw_primitive.RECEIPT_SHA256, "pairs_sha256": pair_hash,
            "protocol": OFFICIAL_ALIGNMENT_PROTOCOL, "source_identity": source_identity(Path(stage) / "2차")}


def prepare_official_lfw(stage, progress=None):
    stage = Path(stage)
    if (importlib.metadata.version("opencv-python-headless") != OFFICIAL_ALIGNMENT_PROTOCOL["opencv_version"]
            or importlib.metadata.version("Pillow") != OFFICIAL_ALIGNMENT_PROTOCOL["pillow_version"]):
        raise ValueError("고정 공식 입력 decoder 배포 버전과 다릅니다")
    pair_path, pair_hash = _official_file(stage, "lfw_pairs")
    pairs = parse_lfw_pairs(pair_path.read_text(encoding="utf-8"))
    images, labels = lfw_primitive.load_verified(stage)
    lfw_primitive.validate_pair_order(labels, pairs)
    if progress:
        progress("official_input", {})
    binding = _official_binding(stage, pair_hash)
    cache = stage / "data/face_benchmarks/derived/lfw_official_ms1mv2"
    cache.mkdir(parents=True, exist_ok=True)
    cache_path, metadata_path, partial = cache / "aligned.npy", cache / "aligned.json", cache / "aligned.part.npy"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (existing.get("binding") == binding and existing.get("status") == "complete"
                and existing.get("images") == 12000 and existing.get("pairs") == 6000
                and type(existing.get("labels")) is list and all(type(value) is bool for value in existing["labels"])
                and existing.get("labels") == labels and existing.get("failed_images") == [] and existing.get("coverage") is None
                and cache_path.exists() and digest_file(cache_path) == existing.get("cache_sha256")):
            return existing
        raise ValueError("다른 조건 또는 미완료 공식 입력 캐시를 덮어쓰지 않습니다")
    if partial.exists() or cache_path.exists():
        raise ValueError("이전 공식 입력 부분·완료 배열을 덮어쓰지 않습니다")
    aligned = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(12000, 112, 112, 3))
    started = time.monotonic()
    try:
        for index, encoded in enumerate(images):
            aligned[index] = decode_official_image(encoded)
            if (index + 1) % 25 == 0 or index + 1 == len(images):
                aligned.flush()
                if progress:
                    progress("provided_alignment", {"completed": index + 1, "total": len(images), "elapsed_seconds": time.monotonic() - started})
    finally:
        aligned.flush()
        aligned._mmap.close()
    # Windows rename은 존재하는 완료 배열을 덮어쓰지 않습니다.
    partial.rename(cache_path)
    evidence = {"binding": binding, "status": "complete", "completed_at": now(), "images": 12000,
                "successful_images": 12000, "failed_images": [], "coverage": None, "pairs": 6000,
                "labels": labels, "pair_labels_and_fold_order_verified": True,
                "image_identity_independently_verified": False, "cache_sha256": digest_file(cache_path),
                "actual_face_accuracy_measured": False}
    atomic_json(metadata_path, evidence)
    return evidence


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def source_identity(root):
    root = Path(root)
    files = sorted((root / "benchmark").glob("*.py")) + [root / "benchmark/sources.json", root / "tools/run_lfw.py"]
    return _json_digest({str(path.relative_to(root)): digest_file(path) for path in files})


def _spec(stage, identifier):
    manifest = json.loads((Path(stage) / "2차/benchmark/sources.json").read_text(encoding="utf-8"))
    return next(item for item in manifest["files"] if item["id"] == identifier)


def _official_file(stage, identifier):
    spec = _spec(stage, identifier)
    path = Path(stage) / spec["target"]
    if not spec.get("sha256") or path.stat().st_size != spec["size"] or digest_file(path) != spec["sha256"]:
        raise ValueError("공식 길이·SHA-256과 평가 입력이 다릅니다: " + identifier)
    return path, spec["sha256"]


def select_face(faces, width, height):
    if faces is None:
        raise ValueError("얼굴 검출 결과가 없습니다")
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 15 or not len(faces) or not np.isfinite(faces).all() or np.any(faces[:, 2:4] <= 0):
        raise ValueError("얼굴 검출 좌표가 유효하지 않습니다")
    def key(face):
        x, y, w, h = map(float, face[:4])
        return ((x + w / 2 - width / 2) ** 2 + (y + h / 2 - height / 2) ** 2, -w * h, -float(face[14]), x, y, w, h)
    return min(faces, key=key)


def checked_embedding_sum(original, flipped):
    original, flipped = np.asarray(original, dtype=np.float64), np.asarray(flipped, dtype=np.float64)
    if original.ndim != 2 or original.shape != flipped.shape or not np.isfinite(original).all() or not np.isfinite(flipped).all():
        raise ValueError("원본·반전 특징 형상·유한값이 유효하지 않습니다")
    summed = original + flipped
    norms = np.linalg.norm(summed, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("원본·반전 특징 합의 norm이 0 또는 비유한값입니다")
    return summed / norms


def infer_feature_batch(model, rgb, device):
    import torch
    rgb = np.asarray(rgb)
    if rgb.ndim != 4 or rgb.shape[1:] != (112, 112, 3) or rgb.dtype != np.uint8 or not len(rgb):
        raise ValueError("추론 batch는 RGB uint8 112x112 영상이어야 합니다")
    original = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    original.div_(255).sub_(0.5).div_(0.5)
    inputs = torch.cat([original, torch.flip(original, dims=[3])], dim=0)
    outputs = model(inputs, phase="infer")
    if outputs.shape != (2 * len(rgb), 512) or not torch.isfinite(outputs).all().item():
        raise ValueError("모델의 원본·반전 출력 형상·유한값이 다릅니다")
    output = outputs.detach().cpu().numpy()
    return output[:len(rgb)], output[len(rgb):]


def validate_resume(state, binding, feature_path, count):
    if state.get("binding") != binding:
        raise ValueError("다른 소스·자료·가중치의 특징 재개 상태입니다")
    done = state.get("next_index")
    if isinstance(done, bool) or not isinstance(done, int) or not 0 <= done <= count:
        raise ValueError("특징 재개 위치가 유효하지 않습니다")
    try:
        data = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("재개 특징 배열이 없거나 안전한 NumPy 형식이 아닙니다") from error
    try:
        if data.shape != (count, 2, 512) or data.dtype != np.float32:
            raise ValueError("특징 재개 배열 형식이 다릅니다")
        # 원자 재개 영수증에 확정된 prefix만 검사하며 미기록 suffix는 다시 계산합니다.
        for start in range(0, done, 256):
            if not np.isfinite(data[start:min(done, start + 256)]).all():
                raise ValueError("재개 특징의 확정 구간이 비유한값입니다")
    finally:
        data._mmap.close()
    return done


def create_yunet_detector(path, expected_sha256):
    import cv2
    # 검증한 원본 바이트를 공식 버퍼 API로 전달해 한글 파일명 처리를 피합니다.
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("YuNet 원본 바이트의 SHA-256이 공식 고정값과 다릅니다")
    return cv2.FaceDetectorYN.create("onnx", np.frombuffer(raw, dtype=np.uint8), np.empty(0, dtype=np.uint8), (250, 250), 0.9, 0.3, 5000, cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)


def prepare_lfw(stage, progress=None, input_kind="yunet"):
    if input_kind == "official-ms1mv2":
        return prepare_official_lfw(stage, progress)
    if input_kind != "yunet":
        raise ValueError("허용하지 않은 LFW 입력 종류입니다")
    import cv2
    stage = Path(stage)
    if importlib.metadata.version("opencv-python-headless") != ALIGNMENT_PROTOCOL["opencv_version"]:
        raise ValueError("고정 OpenCV 배포 버전과 다릅니다")
    archive, archive_hash = _official_file(stage, "lfw_archive")
    pair_path, pair_hash = _official_file(stage, "lfw_pairs")
    detector_path, detector_hash = _official_file(stage, "yunet")
    pairs = parse_lfw_pairs(pair_path.read_text(encoding="utf-8"))
    names = sorted({name for pair in pairs for name in pair[:2]})
    destination = stage / "data/face_benchmarks/extracted"
    safe_extract_tar(archive, destination, progress=lambda value: progress("extract", value) if progress else None)
    cache = stage / "data/face_benchmarks/derived/lfw_yunet"
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / "aligned.npy"
    metadata_path = cache / "aligned.json"
    binding = {"archive_sha256": archive_hash, "pairs_sha256": pair_hash, "protocol": ALIGNMENT_PROTOCOL, "source_identity": source_identity(stage / "2차")}
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("binding") != binding:
            raise ValueError("다른 정렬 조건의 캐시를 덮어쓰지 않습니다")
        if existing.get("status") == "complete" and cache_path.exists() and digest_file(cache_path) == existing.get("cache_sha256"):
            return existing
        raise ValueError("이전 미완료 정렬 증거가 있습니다. 원문 보존과 부모 재판단이 필요합니다")
    detector = create_yunet_detector(detector_path, detector_hash)
    partial = cache / "aligned.part.npy"
    if partial.exists():
        raise ValueError("이전 부분 정렬 배열을 덮어쓰지 않습니다")
    aligned = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(len(names), 112, 112, 3))
    failures, multi_face = [], 0
    started = time.monotonic()
    try:
        for index, name in enumerate(names):
            try:
                path = destination / "lfw" / name
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3:
                    raise ValueError("원본 영상 디코딩 실패")
                height, width = image.shape[:2]
                detector.setInputSize((width, height))
                _, faces = detector.detect(image)
                face = select_face(faces, width, height)
                multi_face += int(len(faces) > 1)
                matrix = similarity_matrix(face[4:14].reshape(5, 2))
                warped = cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                aligned[index] = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            except (ValueError, OSError, cv2.error) as error:
                aligned[index] = 0
                failures.append({"image": name, "error": str(error)})
            if (index + 1) % 25 == 0 or index + 1 == len(names):
                aligned.flush()
                if progress:
                    progress("alignment", {"completed": index + 1, "total": len(names), "failed": len(failures), "elapsed_seconds": time.monotonic() - started})
    finally:
        aligned.flush()
        aligned._mmap.close()
    evidence = {"binding": binding, "status": "protocol_unmet" if failures else "complete", "completed_at": now(), "images": len(names), "successful_images": len(names) - len(failures), "failed_images": failures, "coverage": (len(names) - len(failures)) / len(names), "multi_face_images": multi_face, "names": names, "pairs": len(pairs), "cache_sha256": None, "actual_face_accuracy_measured": False}
    if not failures:
        partial.replace(cache_path)
        evidence["cache_sha256"] = digest_file(cache_path)
    atomic_json(metadata_path, evidence)
    return evidence


def evaluate_lfw(stage, device="cuda:0", batch_size=8, progress=None, input_kind="yunet"):
    import torch
    if device not in {"cpu", "cuda:0"} or isinstance(batch_size, bool) or batch_size not in {1, 2, 4, 8}:
        raise ValueError("허용하지 않은 평가 장치·batch입니다")
    if input_kind not in {"yunet", "official-ms1mv2"}:
        raise ValueError("허용하지 않은 LFW 입력 종류입니다")
    official = input_kind == "official-ms1mv2"
    protocol = OFFICIAL_ALIGNMENT_PROTOCOL if official else ALIGNMENT_PROTOCOL
    stage = Path(stage)
    cache = stage / ("data/face_benchmarks/derived/lfw_official_ms1mv2" if official else "data/face_benchmarks/derived/lfw_yunet")
    metadata = json.loads((cache / "aligned.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or metadata.get("failed_images") or (metadata.get("coverage") is not None if official else metadata.get("coverage") != 1.0):
        raise ValueError("LFW 정렬 protocol이 충족되지 않아 pair 삭제 없이 평가를 차단합니다")
    cache_path = cache / "aligned.npy"
    if metadata["binding"].get("source_identity") != source_identity(stage / "2차") or metadata["binding"].get("protocol") != protocol or digest_file(cache_path) != metadata.get("cache_sha256"):
        raise ValueError("정렬 캐시의 소스·조건·해시가 다릅니다")
    pair_path, pair_hash = _official_file(stage, "lfw_pairs")
    if pair_hash != metadata["binding"]["pairs_sha256"]:
        raise ValueError("정렬과 평가 pair 메타데이터가 다릅니다")
    pairs = parse_lfw_pairs(pair_path.read_text(encoding="utf-8"))
    if official:
        if metadata["binding"] != _official_binding(stage, pair_hash) or metadata.get("images") != 12000 or metadata.get("pairs") != 6000:
            raise ValueError("공식 입력 캐시의 결합·영상·pair 수가 다릅니다")
        _, official_labels = lfw_primitive.load_verified(stage)
        lfw_primitive.validate_pair_order(official_labels, pairs)
        lfw_primitive.validate_pair_order(metadata.get("labels", []), pairs)
        if metadata.get("labels") != official_labels:
            raise ValueError("공식 입력 캐시의 label 순서가 원본과 다릅니다")
        if progress:
            progress("official_input", {})
        names = list(range(12000))
    else:
        names = metadata["names"]
        if names != sorted({name for pair in pairs for name in pair[:2]}):
            raise ValueError("캐시 영상 ID와 고정 pair 영상이 다릅니다")
    model, model_evidence = load_topofr(stage)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA 평가 장치를 사용할 수 없습니다")
    # 학습·AMP·TF32 없이 원래 FP32 가중치와 eval 경로를 사용합니다.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = model.to(device)
    features_path = cache / "features.npy"
    resume_path = cache / "features_resume.json"
    binding = {"source_identity": source_identity(stage / "2차"), "checkpoint_sha256": model_evidence["checkpoint_sha256"], "cache_sha256": metadata["cache_sha256"], "pair_sha256": pair_hash, "torch_version": str(torch.__version__), "device": device, "batch_size": batch_size, "precision": "float32"}
    if official:
        binding.update(input_kind=input_kind, input_sha256=lfw_primitive.FILE_SHA256, receipt_sha256=lfw_primitive.RECEIPT_SHA256, preprocessing=OFFICIAL_ALIGNMENT_PROTOCOL)
    if resume_path.exists():
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        start = validate_resume(resume, binding, features_path, len(names))
        features = np.lib.format.open_memmap(features_path, mode="r+")
    else:
        if features_path.exists():
            raise ValueError("재개 영수증 없는 특징 배열을 덮어쓰지 않습니다")
        features = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float32, shape=(len(names), 2, 512))
        features.flush()
        start = 0
        atomic_json(resume_path, {"binding": binding, "next_index": 0, "updated_at": now()})
    aligned = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    started = time.monotonic()
    try:
        if aligned.shape != (len(names), 112, 112, 3) or aligned.dtype != np.uint8:
            raise ValueError("정렬 캐시 형식이 다릅니다")
        with torch.inference_mode():
            for begin in range(start, len(names), batch_size):
                end = min(begin + batch_size, len(names))
                original, flipped = infer_feature_batch(model, aligned[begin:end], device)
                features[begin:end, 0] = original
                features[begin:end, 1] = flipped
                features.flush()
                atomic_json(resume_path, {"binding": binding, "next_index": end, "updated_at": now()})
                if progress:
                    progress("inference", {"completed": end, "total": len(names), "resumed_at": start, "elapsed_seconds": time.monotonic() - started})
        normalized = checked_embedding_sum(features[:, 0], features[:, 1])
        index = {name: number for number, name in enumerate(names)}
        left = np.arange(0, 12000, 2) if official else np.asarray([index[pair[0]] for pair in pairs])
        right = np.arange(1, 12000, 2) if official else np.asarray([index[pair[1]] for pair in pairs])
        distances = np.sum((normalized[left] - normalized[right]) ** 2, axis=1)
        labels = np.asarray([pair[2] for pair in pairs])
        result = lfw_ten_fold(distances, labels, input_kind="squared_l2")
    finally:
        aligned._mmap.close()
        features.flush()
        features._mmap.close()
        del model
    interpretation = "공식 MS1MV2 제공 정렬 LFW의 제한 로더·RGB 재측정. 저자 당시 입력 SHA·MXNet 디코딩 픽셀 동일성은 미확인이고 IJB-C 재현이 아닙니다." if official else "원본 LFW에 고정 YuNet 정렬을 적용한 독립 재측정. 저자 aligned/bin과 동일 입력 재현 또는 IJB-C 재현이 아닙니다."
    result.update({"benchmark": "LFW", "completed_at": now(), "binding": binding, "alignment_protocol": protocol, "author_reported_accuracy": 0.9987, "difference_percentage_points": 100 * (result["accuracy"] - 0.9987), "author_input_equivalence": False, "interpretation": interpretation, "features_sha256": digest_file(features_path), "model_evidence": model_evidence})
    atomic_json(cache / "result.json", result)
    return result
