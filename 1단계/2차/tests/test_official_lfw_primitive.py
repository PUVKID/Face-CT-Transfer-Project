"""공식 정렬 bin의 제한 VM과 합성 영상 경계만 검사합니다."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from benchmark import lfw_primitive as primitive
from benchmark import evaluation
from benchmark.status import _validate_status_shape
from tools.run_lfw import StatusWriter


def fixture(payload=b"image", labels=None):
    # 실제 파일과 같은 primitive 구조를 손으로 구성하며 pickle 실행을 사용하지 않습니다.
    labels = [index % 600 < 300 for index in range(6000)] if labels is None else labels
    result = bytearray(b"\x80\x02]q\x00(")
    for index in range(12000):
        result += b"T" + len(payload).to_bytes(4, "little", signed=True) + payload
        number = index + 1
        result += b"q" + bytes([number]) if number < 256 else b"r" + number.to_bytes(4, "little")
    result += b"e]r" + (12001).to_bytes(4, "little") + b"("
    result += bytes(0x88 if value else 0x89 for value in labels)
    result += b"e\x86r" + (12002).to_bytes(4, "little") + b"."
    return bytes(result)


def pairs():
    return [("a", "b", int(index % 600 < 300)) for index in range(6000)]


class OfficialPrimitiveTests(unittest.TestCase):
    def test_exact_schema_latin1_and_no_execution(self):
        raw = fixture(b"\xff\x00\x80")
        with patch("pickle.load", side_effect=AssertionError("실행 금지")), patch("pickle.loads", side_effect=AssertionError("실행 금지")), patch("os.system", side_effect=AssertionError("실행 금지")):
            images, labels = primitive.parse_primitive(raw)
        self.assertEqual(len(images), 12000)
        self.assertEqual(images[0], b"\xff\x00\x80")
        self.assertEqual(len(labels), 6000)
        self.assertTrue(all(type(value) is bool for value in labels))
        primitive.validate_pair_order(labels, pairs())

    def test_dangerous_and_unobserved_opcodes_are_rejected(self):
        for opcode in [b"cos\nsystem\n", b"R", b"h\x00", b"g0\n", b"2", b"0", b"N", b"X\x01\x00\x00\x00a"]:
            with self.subTest(opcode=opcode[:1]), self.assertRaises(ValueError):
                primitive.parse_primitive(b"\x80\x02" + opcode + b".")

    def test_protocol_stop_trailing_and_truncation_rejected(self):
        valid = fixture()
        for raw in [valid + b"N.", valid[:-1], valid[:10], b"\x80\x03" + valid[2:], valid[2:]]:
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                primitive.parse_primitive(raw)

    def test_memo_duplicate_range_and_nonsequential_rejected(self):
        for raw in [b"\x80\x02]q\x00q\x00.", b"\x80\x02]r\xff\xff\xff\xff.", b"\x80\x02]q\x01.", b"\x80\x02q\x00."]:
            with self.assertRaises(ValueError):
                primitive.parse_primitive(raw)

    def test_stack_mark_list_and_tuple_shape_limits(self):
        for raw in [b"\x80\x02]e.", b"\x80\x02((((((.", b"\x80\x02]]].", b"\x80\x02\x86.", b"\x80\x02](\x88e."]:
            with self.assertRaises(ValueError):
                primitive.parse_primitive(raw)
        with self.assertRaises(ValueError):
            primitive.parse_primitive(fixture(labels=[True] * 5999))

    def test_image_length_and_operation_bounds(self):
        with self.assertRaises(ValueError):
            primitive.parse_primitive(b"\x80\x02T" + (primitive.MAX_IMAGE_BYTES + 1).to_bytes(4, "little") + b"x" * (primitive.MAX_IMAGE_BYTES + 1) + b".")
        with self.assertRaises(ValueError):
            primitive.parse_primitive(b"\x80\x02T\xff\xff\xff\x7f.")
        with self.assertRaises(ValueError):
            primitive.parse_primitive(b"\x80\x02" + b"(" * (primitive.MAX_OPS + 1) + b".")
        # 합성 limit만 축소해 실제 연산 상한 분기를 짧게 확인하며 운영 상한은 바꾸지 않습니다.
        with patch.object(primitive, "MAX_OPS", 10), self.assertRaisesRegex(ValueError, "개수 경계"):
            primitive.parse_primitive(fixture())
        with patch.object(primitive, "MAX_STACK", 10), self.assertRaisesRegex(ValueError, "stack 개수"):
            primitive.parse_primitive(fixture())

    def test_pair_labels_and_fold_order_cannot_be_reordered(self):
        labels = [index % 600 < 300 for index in range(6000)]
        primitive.validate_pair_order(labels, pairs())
        labels[0], labels[300] = labels[300], labels[0]
        with self.assertRaises(ValueError):
            primitive.validate_pair_order(labels, pairs())
        with self.assertRaises(ValueError):
            primitive.validate_pair_order([1] * 6000, pairs())

    def test_image_header_rejection_happens_before_opencv_decode(self):
        for mode, shape in [("RGB", (113, 112)), ("L", (112, 112)), ("RGBA", (112, 112))]:
            stream = io.BytesIO()
            Image.new(mode, shape).save(stream, format="PNG")
            with self.subTest(mode=mode, shape=shape), patch("cv2.imdecode", side_effect=AssertionError("header 거부 전 decode 금지")):
                with self.assertRaises(ValueError):
                    evaluation.decode_official_image(stream.getvalue())

    def test_synthetic_rgb_pixel_order_dtype_and_shape(self):
        rgb = np.zeros((112, 112, 3), dtype=np.uint8)
        rgb[:] = [250, 17, 63]
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        actual = evaluation.decode_official_image(stream.getvalue())
        self.assertEqual(actual.dtype, np.uint8)
        self.assertEqual(actual.shape, (112, 112, 3))
        np.testing.assert_array_equal(actual, rgb)

    def test_native_decode_shape_and_dtype_are_rechecked(self):
        stream = io.BytesIO()
        Image.new("RGB", (112, 112)).save(stream, format="PNG")
        for decoded in [None, np.zeros((112, 112, 4), np.uint8), np.zeros((112, 112, 3), np.uint16)]:
            with patch("cv2.imdecode", return_value=decoded), self.assertRaises(ValueError):
                evaluation.decode_official_image(stream.getvalue())

    def test_official_status_never_claims_our_detection_or_old_coverage(self):
        with tempfile.TemporaryDirectory() as directory, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            work = Path(directory)
            writer = StatusWriter(work, "prepare", input_kind="official-ms1mv2")
            self.assertIsNone(writer.state["evaluation"]["detection_coverage"])
            self.assertIn("공식", writer.state["current_task"]["title"])
            writer.progress("provided_alignment", {"completed": 1, "total": 12000, "elapsed_seconds": 1})
            row = next(item for item in writer.state["techniques"] if item["id"] == "alignment")
            self.assertEqual(row["status"], "active")
            self.assertIn("제공된", row["detail"])
            self.assertNotIn("검출·5점", row["detail"])
            writer.finish({"status": "complete", "images": 12000, "successful_images": 12000, "failed_images": [], "coverage": None})
            self.assertIsNone(writer.state["evaluation"]["detection_coverage"])
            self.assertIsNone(writer.state["evaluation"]["failed_images"])
            self.assertIsNone(writer.state["evaluation"]["metrics"])
            _validate_status_shape(writer.state)

    def test_official_cache_validation_rejects_foreign_source_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            cache = stage / "data/face_benchmarks/derived/lfw_official_ms1mv2"
            cache.mkdir(parents=True)
            (cache / "aligned.json").write_text(json.dumps({"status": "complete", "images": 12000, "pairs": 6000, "binding": {"source_identity": "foreign"}, "failed_images": [], "coverage": None}), encoding="utf-8")
            with patch("benchmark.evaluation.source_identity", return_value="current"), patch("benchmark.evaluation.load_topofr", side_effect=AssertionError("binding 검증 전 모델 금지")):
                with self.assertRaises(ValueError):
                    evaluation.evaluate_lfw(stage, "cpu", 4, input_kind="official-ms1mv2")

    def test_official_resume_rejects_other_receipt_or_input_kind(self):
        binding = {"source_identity": "fixed", "input_kind": "official-ms1mv2", "receipt_sha256": "fixed-receipt", "input_sha256": "fixed-input"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npy"
            np.save(path, np.ones((1, 2, 512), dtype=np.float32), allow_pickle=False)
            state = {"binding": binding, "next_index": 1}
            self.assertEqual(evaluation.validate_resume(state, binding, path, 1), 1)
            for other in [{**binding, "receipt_sha256": "other"}, {**binding, "input_kind": "yunet"}, {**binding, "input_sha256": "other"}]:
                with self.assertRaises(ValueError):
                    evaluation.validate_resume(state, other, path, 1)

    def test_official_job_resets_old_coverage_and_preserves_observation_history(self):
        for phase in ["prepare", "evaluate"]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory, patch("tools.run_lfw.gpu_snapshot", return_value=None):
                work = Path(directory)
                old = StatusWriter(work, "prepare")
                old.finish({"status": "protocol_unmet", "images": 5, "successful_images": 4, "failed_images": [{}], "coverage": 0.8})
                writer = StatusWriter(work, phase, input_kind="official-ms1mv2")
                self.assertIsNone(writer.state["evaluation"]["detection_coverage"])
                self.assertIsNone(writer.state["evaluation"]["failed_images"])
                self.assertIsNone(writer.state["evaluation"]["images_completed"])
                self.assertIsNone(writer.state["progress"]["percent"])
                self.assertEqual(next(row for row in writer.state["steps"] if row["id"] == "alignment")["status"], "waiting")
                records = [json.loads(line) for line in (work / "progress_history.jsonl").read_text(encoding="utf-8").split("\n") if line]
                self.assertEqual(records[-1]["previous_evaluation"]["detection_coverage"], 0.8)
                writer.progress("official_input", {})
                row = writer.state["downloads"][-1]
                self.assertTrue(row["verified"])
                self.assertFalse(row["official_digest_verified"])
                self.assertEqual(row["received_bytes"], 64738112)
                self.assertIsNone(writer.state["progress"]["percent"])
                _validate_status_shape(writer.state)

    def test_official_inference_and_complete_keep_input_step_consistent(self):
        with tempfile.TemporaryDirectory() as directory, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(directory), "evaluate", input_kind="official-ms1mv2")
            writer.progress("inference", {"completed": 4, "total": 12000, "elapsed_seconds": 1})
            step = next(row for row in writer.state["steps"] if row["id"] == "alignment")
            self.assertEqual(step["status"], "complete")
            self.assertIn("제공", step["detail"])
            writer.finish({"accuracy": 0.5, "std": 0.1})
            self.assertEqual(writer.state["evaluation"]["status"], "complete")
            self.assertIn("공식 MS1MV2", writer.state["model"]["measured_metrics"][-1]["metric"])
            self.assertEqual(step["status"], "complete")
            self.assertIsNone(writer.state["evaluation"]["detection_coverage"])
            _validate_status_shape(writer.state)

    def test_official_prepare_failure_does_not_leave_active_or_claim_fp32(self):
        with tempfile.TemporaryDirectory() as directory, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(directory), "prepare", input_kind="official-ms1mv2")
            writer.progress("provided_alignment", {"completed": 25, "total": 12000, "elapsed_seconds": 1})
            writer.fail()
            self.assertFalse(any(row["status"] == "active" for row in writer.state["steps"] + writer.state["techniques"]))
            self.assertEqual(next(row for row in writer.state["techniques"] if row["id"] == "alignment")["status"], "failed")
            self.assertEqual(next(row for row in writer.state["techniques"] if row["id"] == "fp32")["status"], "planned")
            self.assertIsNone(writer.state["evaluation"]["metrics"])


if __name__ == "__main__":
    unittest.main()
