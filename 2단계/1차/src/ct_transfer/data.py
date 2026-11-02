"""원자료를 읽기만 하며 실행 가능한 pickle을 수용하지 않습니다."""

import csv
import io
import math
from pathlib import Path, PurePosixPath
import pickle
import pickletools
import struct
import zipfile

import numpy as np
from numpy.lib import format as npformat
from PIL import Image
from scipy.ndimage import affine_transform

from .io import atomic_json, digest_file, signature


BLOCKS = (
    ("luna25_nodule_blocks.zip.001", 4697620480, "0e825281aae77d18dcc02d4909570a7b"),
    ("luna25_nodule_blocks.zip.002", 2099185052, "8df04bfbc9824ccf112a8c90059ba650"),
)
ANNOTATION_MD5 = "1349fcd9fdd120a990f43a2301f1609b"
PREPROCESSING = {"shape": [3, 112, 112], "color": "RGB", "normalization": "[-1,1]", "luna_hu": [-1000, 400], "luna_fov_mm": 50, "luna_plane": "central_axial", "luna_channels": "replicated_grayscale", "iq_patient_mapping": "unavailable"}


class ConcatReader(io.RawIOBase):
    """원본 조각을 복제하지 않고 단일 seek 가능 바이트 스트림으로 읽습니다."""

    def __init__(self, paths):
        super().__init__()
        self.handles = [Path(path).open("rb") for path in paths]
        self.lengths = [handle.seek(0, 2) for handle in self.handles]
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else sum(self.lengths) + offset if whence == 2 else -1
        if value < 0:
            raise ValueError("유효하지 않은 archive seek입니다")
        self.position = value
        return value

    def read(self, size=-1):
        remaining = max(0, sum(self.lengths) - self.position)
        size = remaining if size < 0 else min(size, remaining)
        chunks = []
        offset = self.position
        for handle, length in zip(self.handles, self.lengths):
            if offset >= length:
                offset -= length
                continue
            handle.seek(offset)
            chunk = handle.read(min(size, length - offset))
            chunks.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
            offset = 0
            if not size:
                break
        return b"".join(chunks)

    def close(self):
        for handle in self.handles:
            handle.close()
        super().close()


class _DType:
    def __init__(self, descriptor, *_args):
        if descriptor not in {"O8", "O4", "f8", "f4", "i8", "i4", "u8", "u4"}:
            raise ValueError("메타데이터 dtype을 허용하지 않습니다")
        self.descriptor = descriptor
        self.endian = "="

    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) < 2 or state[1] not in ("<", ">", "|", "="):
            raise ValueError("메타데이터 dtype 상태가 유효하지 않습니다")
        if any(item is not None for item in state[2:5]):
            raise ValueError("구조화된 메타데이터 dtype은 금지합니다")
        self.endian = state[1]


class _Array:
    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 5:
            raise ValueError("메타데이터 배열 상태가 유효하지 않습니다")
        _version, shape, dtype, fortran, payload = state
        if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 or n > 16 for n in shape) or math.prod(shape) > 16:
            raise ValueError("메타데이터 배열 할당 상한을 초과합니다")
        if not isinstance(dtype, _DType):
            raise ValueError("허용되지 않은 메타데이터 dtype입니다")
        if dtype.descriptor.startswith("O"):
            if shape != () or not isinstance(payload, list) or len(payload) != 1:
                raise ValueError("루트 객체 스칼라만 허용합니다")
            self.value = payload[0]
        else:
            if not isinstance(payload, bytes) or len(payload) > 128:
                raise ValueError("메타데이터 숫자 바이트 상한을 초과합니다")
            array = np.frombuffer(payload, dtype=np.dtype(dtype.endian + dtype.descriptor))
            if array.size != math.prod(shape):
                raise ValueError("메타데이터 배열 형상과 바이트 수가 다릅니다")
            self.value = array.reshape(shape, order="F" if fortran else "C").copy()


def _reconstruct(subtype, shape, dtype):
    if subtype is not _Array or shape != (0,) or dtype != b"b":
        raise ValueError("허용되지 않은 배열 생성 요청입니다")
    return _Array()


class _MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy", "ndarray"): _Array,
            ("numpy", "dtype"): _DType,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("메타데이터의 임의 전역 함수 실행을 거부합니다")
        return allowed[(module, name)]


def decode_metadata(raw):
    if len(raw) > 16384:
        raise ValueError("메타데이터 파일 크기 상한을 초과합니다")
    stream = io.BytesIO(raw)
    version = npformat.read_magic(stream)
    if version == (1, 0):
        shape, _fortran, dtype = npformat.read_array_header_1_0(stream, max_header_size=4096)
    elif version == (2, 0):
        shape, _fortran, dtype = npformat.read_array_header_2_0(stream, max_header_size=4096)
    else:
        raise ValueError("검토하지 않은 NPY 메타데이터 버전입니다")
    if shape != () or not dtype.hasobject:
        raise ValueError("루트 객체 스칼라 NPY만 허용합니다")
    payload = stream.read()
    for operation, _argument, _position in pickletools.genops(payload):
        if operation.name in {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID", "NEWOBJ", "NEWOBJ_EX", "INST", "OBJ"}:
            raise pickle.UnpicklingError("검토하지 않은 pickle 실행 연산입니다")
    root = _MetadataReader(io.BytesIO(payload)).load()
    if not isinstance(root, _Array) or not isinstance(root.value, dict) or set(root.value) != {"origin", "spacing", "transform"}:
        raise ValueError("메타데이터 키가 공식 형식과 다릅니다")
    result = {}
    for key, expected_shape in (("origin", (3,)), ("spacing", (3,)), ("transform", (3, 3))):
        value = root.value[key]
        if not isinstance(value, _Array) or not isinstance(value.value, np.ndarray) or value.value.shape != expected_shape:
            raise ValueError("메타데이터 벡터 또는 행렬 형상이 다릅니다")
        array = value.value.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError("메타데이터에 비유한 값이 있습니다")
        result[key] = array
    if np.any(result["spacing"] <= 0) or abs(np.linalg.det(result["transform"])) < 1e-8:
        raise ValueError("유효하지 않은 CT 공간 변환입니다")
    return result


def validate_archive_members(names):
    if len(names) != len(set(names)):
        raise ValueError("중복 archive 경로를 허용하지 않습니다")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError("archive 경로 이탈을 허용하지 않습니다")


def decode_volume(raw):
    if len(raw) > 40_000_000:
        raise ValueError("CT 배열 바이트 상한을 초과합니다")
    stream = io.BytesIO(raw)
    version = npformat.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = npformat.read_array_header_1_0(stream, max_header_size=4096)
    elif version == (2, 0):
        shape, fortran, dtype = npformat.read_array_header_2_0(stream, max_header_size=4096)
    else:
        raise ValueError("검토하지 않은 CT NPY 버전입니다")
    if len(shape) != 3 or any(type(size) is not int or size <= 0 or size > 4096 for size in shape) or math.prod(shape) > 16_000_000 or dtype.kind not in "iuf" or dtype.itemsize > 8:
        raise ValueError("CT NPY의 형상 또는 숫자 dtype이 유효하지 않습니다")
    payload = stream.read()
    if len(payload) != math.prod(shape) * dtype.itemsize:
        raise ValueError("CT 배열 형상과 실제 바이트 수가 다릅니다")
    return np.frombuffer(payload, dtype=dtype).reshape(shape, order="F" if fortran else "C")


def luna_plane(volume, metadata, *, size_px=112, size_mm=50.):
    if volume.ndim != 3 or volume.size > 16_000_000 or volume.dtype.kind not in "iuf" or not np.isfinite(volume).all():
        raise ValueError("유효한 숫자 CT 블록만 허용합니다")
    spacing = metadata["spacing"]
    transform = metadata["transform"]
    # 제작자 좌표 순서인 z,y,x와 고정된 물리적 시야를 보존합니다.
    orientation = (np.linalg.inv(transform).dot(np.eye(3)) * spacing).T
    orientation = (orientation.T / np.sqrt(np.sum(orientation * orientation, axis=1))).T
    orientation = np.linalg.inv(orientation.T)
    forward = np.diag([size_px / size_mm] * 3).dot(orientation).dot(np.diag(spacing))
    inverse = np.linalg.inv(forward)
    center = np.asarray(volume.shape, dtype=np.float64) // 2
    output_shape = np.array([1, size_px, size_px])
    offset = center - inverse.dot((output_shape - 1) / 2.)
    plane = affine_transform(volume.astype(np.float32), inverse, offset=offset, output_shape=tuple(output_shape), order=1, mode="constant", cval=-1000., prefilter=False)
    return ((np.clip(plane, -1000., 400.) + 1000.) / 1400. * 2. - 1.).astype(np.float32)


def iq_records(root):
    mapping = {"Normal cases": 0, "Bengin cases": 1, "Malignant cases": 2}
    records = []
    group_labels = {}
    import hashlib
    for directory, label in mapping.items():
        location = Path(root) / directory
        if not location.is_dir():
            raise FileNotFoundError(location)
        for path in sorted(location.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                pixel_hash = hashlib.sha256(struct.pack("<II", *rgb.size) + rgb.tobytes()).hexdigest()
            group = "pixels:" + pixel_hash
            if group in group_labels and group_labels[group] != label:
                raise ValueError("동일 화소의 IQ 영상에 상충하는 라벨이 있습니다")
            group_labels[group] = label
            records.append({"id": "iq:" + digest_file(path), "path": str(path), "label": label, "group": group, "patient_id": None, "source": "IQ-OTH/NCCD", "label_basis": "publisher_class_folder"})
    if not records:
        raise ValueError("IQ 영상이 없습니다")
    return records


def prepare_luna(source_root, cache_root):
    source_root, cache_root = Path(source_root), Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, size, md5 in BLOCKS:
        path = source_root / "nodule_blocks" / name
        if not path.is_file() or path.stat().st_size != size or digest_file(path, "md5") != md5:
            raise ValueError("LUNA25 공식 원본 조각의 크기 또는 MD5가 다릅니다: " + name)
        paths.append(path)
    csv_path = source_root / "annotations" / "LUNA25_Public_Training_Development_Data.csv"
    if digest_file(csv_path, "md5") != ANNOTATION_MD5:
        raise ValueError("LUNA25 공식 주석 MD5가 다릅니다")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["AnnotationID"] for row in rows]
    if len(ids) != 6163 or len(set(ids)) != len(ids):
        raise ValueError("LUNA25 주석 수 또는 고유 키가 다릅니다")
    target = cache_root / "luna.npy"
    index_path = cache_root / "luna_index.json"
    preprocessing_source = digest_file(__file__)
    if target.exists() and index_path.exists():
        import json
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("preprocessing_source_sha256") == preprocessing_source and existing.get("cache_sha256") == digest_file(target):
            return existing
        raise ValueError("기존 LUNA 캐시와 현재 전처리·파일 해시가 다릅니다")
    temporary = target.with_suffix(".part.npy")
    records = []
    with ConcatReader(paths) as reader, zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        validate_archive_members(names)
        expected = {f"luna25_nodule_blocks/{kind}/{identifier}.npy" for kind in ("image", "metadata") for identifier in ids}
        if {name for name in names if not name.endswith("/")} != expected:
            raise ValueError("archive 파일과 CSV 주석이 일치하지 않습니다")
        output = npformat.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(rows), 1, 112, 112))
        for index, row in enumerate(rows):
            identifier = row["AnnotationID"]
            image_name = f"luna25_nodule_blocks/image/{identifier}.npy"
            metadata_name = f"luna25_nodule_blocks/metadata/{identifier}.npy"
            if archive.getinfo(image_name).file_size > 40_000_000 or archive.getinfo(metadata_name).file_size > 16384:
                raise ValueError("archive 항목 크기 상한을 초과합니다")
            volume = decode_volume(archive.read(image_name))
            metadata = decode_metadata(archive.read(metadata_name))
            output[index] = luna_plane(volume, metadata)
            label = int(row["label"])
            if label not in (0, 1) or identifier != f"{row['PatientID']}_{row['LesionID']}_{row['StudyDate']}":
                raise ValueError("LUNA25 라벨 또는 주석 키가 유효하지 않습니다")
            records.append({"id": identifier, "label": label, "group": row["PatientID"], "patient_id": row["PatientID"], "nodule_id": row["NoduleID"], "study_date": row["StudyDate"], "source": "LUNA25/NLST", "label_basis": "official_binary_annotation"})
            if (index + 1) % 250 == 0 or index + 1 == len(rows):
                output.flush()
                atomic_json(cache_root / "preparation_progress.json", {"cohort": "luna", "status": "preparing", "rows_completed": index + 1, "rows_total": len(rows), "official_md5_verified": True})
        output.flush()
        del output
    import os
    os.replace(temporary, target)
    manifest = {"cohort": "luna_binary_cv", "records": records, "cache": str(target), "cache_sha256": digest_file(target), "preprocessing": PREPROCESSING, "preprocessing_source_sha256": preprocessing_source, "patient_mapping_known": True, "source_md5": [item[2] for item in BLOCKS] + [ANNOTATION_MD5], "analysis_unit": "annotation_timepoint", "patients": len({row["patient_id"] for row in records}), "distinct_nodules": len({row["nodule_id"] for row in records})}
    atomic_json(cache_root / "luna_index.json", manifest)
    atomic_json(cache_root / "preparation_progress.json", {"cohort": "luna", "status": "complete", "rows_completed": len(records), "rows_total": len(rows), "official_md5_verified": True, "cache_sha256": manifest["cache_sha256"]})
    return manifest


def prepare_iq(source_root, cache_root):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    records = iq_records(source_root)
    target = cache_root / "iq.npy"
    index_path = cache_root / "iq_index.json"
    preprocessing_source = digest_file(__file__)
    if target.exists() and index_path.exists():
        import json
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing.get("preprocessing_source_sha256") == preprocessing_source and existing.get("records") == records and existing.get("cache_sha256") == digest_file(target):
            return existing
        raise ValueError("기존 IQ 캐시와 현재 원본·전처리·파일 해시가 다릅니다")
    temporary = target.with_suffix(".part.npy")
    output = npformat.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(records), 3, 112, 112))
    for index, row in enumerate(records):
        with Image.open(row["path"]) as image:
            array = np.asarray(image.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR), dtype=np.float32)
        output[index] = np.transpose(array / 127.5 - 1., (2, 0, 1))
    output.flush()
    del output
    import os
    os.replace(temporary, target)
    manifest = {"cohort": "iq_exploratory", "records": records, "cache": str(target), "cache_sha256": digest_file(target), "preprocessing": PREPROCESSING, "preprocessing_source_sha256": preprocessing_source, "patient_mapping_known": False, "limitations": ["정확 중복 그룹이며 환자 독립성이 확인되지 않았습니다.", "IQ 3분류와 LUNA25 결절 이진 자료를 라벨 테이블로 혼합하지 않습니다."]}
    atomic_json(cache_root / "iq_index.json", manifest)
    return manifest
