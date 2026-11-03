"""신뢰하지 않는 archive는 링크·경로 이탈·기존 파일 덮어쓰기 없이 읽습니다."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile


def digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_tar(source, destination, max_bytes=80 * 1024 ** 3, progress=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count, total = 0, 0
    with tarfile.open(source, "r:*") as archive:
        members = archive.getmembers()
        planned = []
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or ":" in member.name or not (member.isdir() or member.isfile()):
                raise ValueError("archive의 경로·링크·파일 형식을 거부했습니다")
            target = (destination / Path(*name.parts)).resolve()
            if not target.is_relative_to(destination) or target == destination and not member.isdir():
                raise ValueError("archive 경로가 대상 밖입니다")
            if member.size < 0 or member.size > 2 * 1024 ** 3:
                raise ValueError("archive 개별 파일 크기가 상한을 넘습니다")
            total += member.size
            if total > max_bytes or len(planned) >= 1000000:
                raise ValueError("archive 전체 크기·파일 수가 상한을 넘습니다")
            planned.append((member, target))
        reserved = set()
        for _, target in planned:
            reserved.add(target)
            reserved.update(parent for parent in target.parents if parent.is_relative_to(destination))
        # 독점 생성한 staging root가 원본 파일·디렉터리의 경로와 겹치면 쓰기 전에 거부합니다.
        with tempfile.TemporaryDirectory(prefix=".face-extract-", dir=destination) as temporary:
            staging = Path(temporary).resolve()
            if not staging.is_relative_to(destination) or staging in reserved:
                raise ValueError("추출 임시 공간과 archive 경로가 충돌합니다")
            done_bytes = 0
            for member, target in planned:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    # 재개 시 기존 원본은 비교만 하며 임시 접미사의 기존 파일도 건드리지 않습니다.
                    with archive.extractfile(member) as incoming, target.open("rb") as existing:
                        while True:
                            left, right = incoming.read(1024 * 1024), existing.read(1024 * 1024)
                            if left != right:
                                raise ValueError("기존 추출 파일이 원본과 달라 덮어쓰지 않습니다")
                            if not left:
                                break
                else:
                    descriptor, temporary_name = tempfile.mkstemp(dir=staging)
                    part = Path(temporary_name)
                    with os.fdopen(descriptor, "wb") as output, archive.extractfile(member) as incoming:
                        written = 0
                        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                            output.write(chunk)
                            written += len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if written != member.size:
                        raise ValueError("추출 바이트 수가 header와 다릅니다")
                    if os.name == "nt":
                        # Windows rename은 최종 경로가 생겼으면 실패하며 replace처럼 덮어쓰지 않습니다.
                        os.rename(part, target)
                    else:
                        # 다른 OS에서도 최종 파일은 독점 생성하여 기존 파일을 덮어쓰지 않습니다.
                        with part.open("rb") as incoming, target.open("xb") as output:
                            shutil.copyfileobj(incoming, output, 1024 * 1024)
                            output.flush()
                            os.fsync(output.fileno())
                count += 1
                done_bytes += member.size
                if progress and (count % 200 == 0):
                    progress({"files": count, "bytes": done_bytes, "total_bytes": total})
    return {"files": count, "bytes": total, "source_sha256": digest_file(source)}


def parse_lfw_pairs(text):
    lines = text.split("\n")
    lines = [line.rstrip("\r") for line in lines if line.strip()]
    if lines[0].split() != ["10", "300"] or len(lines) != 6001:
        raise ValueError("LFW pair header·6000쌍 수가 다릅니다")
    result = []
    for index, line in enumerate(lines[1:]):
        fields = line.split()
        same = index % 600 < 300
        if len(fields) != (3 if same else 4):
            raise ValueError("LFW 동일인·타인 pair 형식이 다릅니다")
        names = [fields[0]] if same else [fields[0], fields[2]]
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
            raise ValueError("LFW 이름에 경로 문자가 있습니다")
        numbers = [fields[1], fields[2]] if same else [fields[1], fields[3]]
        if any(not number.isdecimal() or not 1 <= int(number) <= 9999 for number in numbers):
            raise ValueError("LFW 영상 번호가 유효하지 않습니다")
        first, second = (names[0], names[0]) if same else names
        result.append((f"{first}/{first}_{int(numbers[0]):04d}.jpg", f"{second}/{second}_{int(numbers[1]):04d}.jpg", int(same)))
    return result
