"""관측한 11 opcode로만 고정 LFW의 bytes·bool primitive를 구성합니다."""
import hashlib
import io
import json
from pathlib import Path
import pickletools
import zlib


FILE_BYTES = 64_738_112
FILE_SHA256 = "9b711dca71a47fc9f295a28514abe8cc4e457632a8d634264691398c99569593"
RECEIPT_SHA256 = "758f691ca614e53d847aaca4c6e0b2dd876ee9ae334dd619746a7083ff9ec9b1"
CRC32 = "2d7d0c73"
IMAGE_COUNT, PAIR_COUNT = 12_000, 6_000
MAX_IMAGE_BYTES, MAX_OPS, MAX_STACK, MAX_MARKS = 16_384, 40_000, 12_010, 4
ALLOWED = {"PROTO", "EMPTY_LIST", "MARK", "APPENDS", "BINSTRING", "BINPUT", "LONG_BINPUT", "NEWTRUE", "NEWFALSE", "TUPLE2", "STOP"}


class PrimitiveError(ValueError):
    """제한 형식 또는 자원 경계와 다릅니다."""


class _Reader(io.BytesIO):
    def read(self, size=-1):
        # 길이 선언을 신뢰한 큰 할당과 허용하지 않은 textual operand를 막습니다.
        if size < 0 or size > MAX_IMAGE_BYTES or size > len(self.getbuffer()) - self.tell():
            if size == 1 and self.tell() == len(self.getbuffer()):
                return b""
            raise PrimitiveError("operand 길이가 고정 경계를 넘었습니다")
        return super().read(size)

    def readline(self, size=-1):
        raise PrimitiveError("관측하지 않은 textual opcode는 허용하지 않습니다")


def parse_primitive(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= FILE_BYTES:
        raise PrimitiveError("primitive 파일 길이 경계가 다릅니다")
    reader = _Reader(raw)
    stack, marks, lists = [], [], []
    memo_count = operations = string_bytes = string_count = tuple_count = 0
    stopped = False
    for opcode, arg, offset in pickletools.genops(reader):
        name = opcode.name
        operations += 1
        if operations > MAX_OPS or name not in ALLOWED:
            raise PrimitiveError("허용 opcode 또는 개수 경계와 다릅니다")
        if operations == 1 and (name != "PROTO" or arg != 2 or offset != 0):
            raise PrimitiveError("최초 PROTO2가 필요합니다")
        if name == "PROTO":
            if operations != 1 or arg != 2:
                raise PrimitiveError("PROTO 위치·버전이 다릅니다")
        elif name == "EMPTY_LIST":
            if len(lists) >= 2:
                raise PrimitiveError("두 primitive list만 허용합니다")
            value = []
            lists.append(value)
            stack.append(value)
        elif name == "MARK":
            if len(marks) >= MAX_MARKS:
                raise PrimitiveError("MARK 깊이 경계를 넘었습니다")
            marks.append(len(stack))
        elif name == "BINSTRING":
            if type(arg) is not str:
                raise PrimitiveError("BINSTRING operand가 문자열이 아닙니다")
            # pickletools의 Python2 문자열은 명시적 latin1 역변환만 허용합니다.
            value = arg.encode("latin-1", errors="strict")
            string_bytes += len(value)
            string_count += 1
            if not 0 < len(value) <= MAX_IMAGE_BYTES or string_bytes > FILE_BYTES or string_count > IMAGE_COUNT:
                raise PrimitiveError("영상 bytes의 개별·누적·개수 경계를 넘었습니다")
            stack.append(value)
        elif name in {"NEWTRUE", "NEWFALSE"}:
            stack.append(name == "NEWTRUE")
        elif name in {"BINPUT", "LONG_BINPUT"}:
            if not stack or type(arg) is not int or arg != memo_count or not 0 <= arg <= IMAGE_COUNT + 2:
                raise PrimitiveError("memo 중복·범위·쓰기 순서 또는 stack이 다릅니다")
            if (name == "BINPUT") != (arg < 256):
                raise PrimitiveError("memo index와 operand 폭이 다릅니다")
            # GET을 허용하지 않으므로 객체 조회나 동적 함수 호출은 없습니다.
            memo_count += 1
        elif name == "APPENDS":
            if not marks:
                raise PrimitiveError("APPENDS에 MARK가 없습니다")
            mark = marks.pop()
            if mark < 1 or type(stack[mark - 1]) is not list or mark == len(stack):
                raise PrimitiveError("APPENDS 대상 또는 항목이 다릅니다")
            target, items = stack[mark - 1], stack[mark:]
            if lists and target is lists[0]:
                if any(type(item) is not bytes for item in items) or len(target) + len(items) > IMAGE_COUNT:
                    raise PrimitiveError("영상 list 항목·개수와 다릅니다")
            elif len(lists) == 2 and target is lists[1]:
                if any(type(item) is not bool for item in items) or len(target) + len(items) > PAIR_COUNT:
                    raise PrimitiveError("라벨 list 항목·개수와 다릅니다")
            else:
                raise PrimitiveError("APPENDS 대상 list가 다릅니다")
            target.extend(items)
            del stack[mark:]
        elif name == "TUPLE2":
            if marks or tuple_count or len(stack) != 2 or len(lists) != 2 or stack[0] is not lists[0] or stack[1] is not lists[1]:
                raise PrimitiveError("최종 tuple2 구조와 다릅니다")
            stack[:] = [(stack[0], stack[1])]
            tuple_count += 1
        elif name == "STOP":
            stopped = True
            if reader.tell() != len(raw):
                raise PrimitiveError("STOP 이후 trailing bytes가 있습니다")
        if len(stack) > MAX_STACK:
            raise PrimitiveError("stack 개수 경계를 넘었습니다")
    if (not stopped or marks or len(stack) != 1 or type(stack[0]) is not tuple
            or tuple_count != 1 or memo_count != IMAGE_COUNT + 3):
        raise PrimitiveError("종료 primitive 또는 memo 개수가 다릅니다")
    images, labels = stack[0]
    if (type(images) is not list or len(images) != IMAGE_COUNT or any(type(value) is not bytes for value in images)
            or type(labels) is not list or len(labels) != PAIR_COUNT or any(type(value) is not bool for value in labels)):
        raise PrimitiveError("최종 12000 bytes·6000 bool schema와 다릅니다")
    return images, labels


def validate_pair_order(labels, pairs):
    if len(labels) != PAIR_COUNT or len(pairs) != PAIR_COUNT:
        raise PrimitiveError("공식 pair 개수가 다릅니다")
    for index, (label, pair) in enumerate(zip(labels, pairs)):
        if type(label) is not bool or len(pair) != 3 or pair[2] not in {0, 1} or label != bool(pair[2]) or label != (index % 600 < 300):
            raise PrimitiveError("공식 issame·연속 10fold label 순서와 다릅니다")


def load_verified(stage):
    path = Path(stage) / "data/face_benchmarks/raw/lfw_official_ms1mv2.bin"
    receipt_path = path.with_name(path.name + ".receipt.json")
    if path.stat().st_size != FILE_BYTES or receipt_path.stat().st_size > 1_048_576:
        raise PrimitiveError("고정 원본·획득 영수증 크기와 다릅니다")
    receipt_raw = receipt_path.read_bytes()
    if hashlib.sha256(receipt_raw).hexdigest() != RECEIPT_SHA256:
        raise PrimitiveError("획득 완료 영수증 SHA와 다릅니다")
    receipt = json.loads(receipt_raw)
    if (receipt.get("status") != "complete" or receipt.get("member") != "faces_emore/lfw.bin"
            or type(receipt.get("bytes")) is not int or receipt["bytes"] != FILE_BYTES
            or receipt.get("sha256") != FILE_SHA256 or receipt.get("crc32") != CRC32):
        raise PrimitiveError("획득 영수증의 완료·길이·SHA·CRC와 다릅니다")
    with path.open("rb") as stream:
        raw = stream.read(FILE_BYTES + 1)
    if len(raw) != FILE_BYTES or hashlib.sha256(raw).hexdigest() != FILE_SHA256 or f"{zlib.crc32(raw):08x}" != CRC32:
        raise PrimitiveError("실제 원본 바이트의 길이·SHA·CRC와 다릅니다")
    images, labels = parse_primitive(raw)
    return images, labels
