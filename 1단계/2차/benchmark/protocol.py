"""정렬 규격과 IJB-C 영상·template 대응을 검사합니다."""
import hashlib
from pathlib import Path, PurePosixPath
import numpy as np
from .safety import digest_file


CANONICAL_LANDMARKS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def similarity_matrix(source, destination=CANONICAL_LANDMARKS):
    source, destination = np.asarray(source, dtype=np.float64), np.asarray(destination, dtype=np.float64)
    if source.shape != (5, 2) or destination.shape != (5, 2) or not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("유한한 5점 좌표가 필요합니다")
    design = np.zeros((10, 4), dtype=np.float64)
    design[0::2] = np.column_stack([source[:, 0], -source[:, 1], np.ones(5), np.zeros(5)])
    design[1::2] = np.column_stack([source[:, 1], source[:, 0], np.zeros(5), np.ones(5)])
    result, _, rank, _ = np.linalg.lstsq(design, destination.reshape(-1), rcond=None)
    if rank != 4:
        raise ValueError("퇴화한 얼굴 랜드마크입니다")
    a, b, x, y = result
    return np.array([[a, -b, x], [b, a, y]])


def _name(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("IJB 영상 이름에 경로 이탈 문자가 있습니다")
    return path.as_posix()


def validate_ijb_metadata(root):
    root = Path(root).resolve()
    meta = root / "meta"
    tid = meta / "ijbc_face_tid_mid.txt"
    landmarks = meta / "ijbc_name_5pts_score.txt"
    pairs = meta / "ijbc_template_pair_label.txt"
    records, names, templates = [], set(), set()
    with tid.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 3 or not all(value.isdecimal() for value in fields[1:]):
                raise ValueError("IJB face/template/media 형식이 다릅니다")
            name = _name(fields[0])
            if name in names:
                raise ValueError("IJB 영상 이름이 중복됩니다")
            names.add(name)
            templates.add(int(fields[1]))
            records.append((name, int(fields[1]), int(fields[2])))
    if not records:
        raise ValueError("IJB 영상 목록이 비어 있습니다")
    counted = 0
    with landmarks.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            fields = line.split()
            if index >= len(records) or len(fields) != 12 or _name(fields[0]) != records[index][0]:
                raise ValueError("IJB 영상과 5점/점수 목록의 ID·순서가 다릅니다")
            values = np.asarray([float(value) for value in fields[1:]], dtype=np.float64)
            if not np.isfinite(values).all() or not 0 <= values[-1] <= 1:
                raise ValueError("IJB 좌표·faceness가 유효하지 않습니다")
            image = (root / "loose_crop" / records[index][0]).resolve()
            if not image.is_relative_to(root / "loose_crop") or not image.is_file():
                raise ValueError("IJB 원본 영상이 없거나 root 밖입니다")
            counted += 1
    if counted != len(records):
        raise ValueError("IJB 좌표와 영상 수가 다릅니다")
    pair_count, positive = 0, 0
    with pairs.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 3 or not all(value.isdecimal() for value in fields) or int(fields[2]) not in [0, 1] or any(int(value) not in templates for value in fields[:2]):
                raise ValueError("IJB pair의 template 참조·라벨이 유효하지 않습니다")
            pair_count += 1
            positive += int(fields[2])
    if not 0 < positive < pair_count:
        raise ValueError("IJB 동일인·타인 pair가 모두 필요합니다")
    licenses = list(root.rglob("*license*.csv")) + list(root.rglob("*LICENSE*"))
    return {"images": counted, "templates": len(templates), "pairs": pair_count, "genuine_pairs": positive, "impostor_pairs": pair_count - positive, "image_name_order_verified": True, "metadata_sha256": {path.name: digest_file(path) for path in [tid, landmarks, pairs]}, "license_files": [{"name": str(path.relative_to(root)), "sha256": digest_file(path)} for path in licenses if path.is_file()], "license_review_complete": False}
