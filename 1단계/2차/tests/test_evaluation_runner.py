"""실제 얼굴 평가의 정렬 실패·원본 결합·재개 경계를 검사합니다."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from benchmark.evaluation import select_face, checked_embedding_sum, validate_resume, infer_feature_batch
from tools.run_lfw import StatusWriter
from tools import run_lfw
from benchmark import evaluation
from benchmark.status import initial_status


class EvaluationRunnerTests(unittest.TestCase):
    def test_yunet_buffer_loads_verified_original_at_korean_path_without_detect(self):
        stage = Path(__file__).resolve().parents[2]
        manifest = json.loads((stage / "2차/benchmark/sources.json").read_text(encoding="utf-8"))
        spec = next(item for item in manifest["files"] if item["id"] == "yunet")
        path = stage / spec["target"]
        self.assertTrue(any("가" <= character <= "힣" for character in str(path)))
        # 모델 생성만 확인하며 detect·얼굴 영상·GPU를 실행하지 않습니다.
        detector = evaluation.create_yunet_detector(path, spec["sha256"])
        self.assertEqual(detector.getInputSize(), (250, 250))
        self.assertAlmostEqual(detector.getScoreThreshold(), 0.9, places=6)
        self.assertAlmostEqual(detector.getNMSThreshold(), 0.3, places=6)
        self.assertEqual(detector.getTopK(), 5000)

    def test_yunet_buffer_rejects_bytes_outside_verified_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "검증되지않은모델.onnx"
            path.write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "YuNet.*SHA-256"):
                evaluation.create_yunet_detector(path, "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")

    def test_new_job_resets_observation_counts_and_preserves_input_and_history(self):
        for phase in ["prepare", "evaluate"]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
                root = Path(temporary)
                previous = initial_status()
                previous["progress"].update(completed=100, total=100, unit="bytes", percent=100.0, eta_seconds=0.0)
                previous["evaluation"].update(images_completed=4, images_total=4, pairs_completed=6000, pairs_total=6000, fold_completed=10, fold_total=10, detection_coverage=1.0, failed_images=0)
                previous["model"]["measured_metrics"] = [{"benchmark": "보존", "metric": "이전 측정", "value": 0.5, "unit": "%", "provenance": "제어 fixture"}]
                previous["series"]["metrics"] = [{"time": "이전 시각", "value": 0.5}]
                (root / "status.json").write_text(json.dumps(previous, ensure_ascii=False), encoding="utf-8")
                writer = StatusWriter(root, phase)
                for key in ["completed", "total", "percent", "eta_seconds"]:
                    self.assertIsNone(writer.state["progress"][key], key)
                self.assertEqual(writer.state["progress"]["unit"], "")
                for key in ["images_completed", "images_total", "pairs_completed", "pairs_total", "fold_completed", "fold_total"]:
                    self.assertIsNone(writer.state["evaluation"][key], key)
                self.assertEqual(writer.state["evaluation"]["detection_coverage"], 1.0 if phase == "evaluate" else None)
                self.assertEqual(writer.state["evaluation"]["failed_images"], 0 if phase == "evaluate" else None)
                self.assertEqual(writer.state["model"]["measured_metrics"], previous["model"]["measured_metrics"])
                self.assertEqual(writer.state["series"]["metrics"], previous["series"]["metrics"])
                history = [json.loads(line) for line in (root / "progress_history.jsonl").read_text(encoding="utf-8").split("\n") if line]
                self.assertEqual(history[-1]["previous_progress"], previous["progress"])
                self.assertEqual(history[-1]["previous_evaluation"], previous["evaluation"])

    def test_process_keeps_original_interpreter_flags_separate_from_legacy_argv(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            revision = Path(temporary) / "2차"
            original = [run_lfw.sys.executable, "-B", "-X", "utf8", "tools/run_lfw.py", "evaluate"]
            with patch.object(run_lfw, "ROOT", revision), patch.object(run_lfw.sys, "argv", ["tools/run_lfw.py", "evaluate"]), patch.object(run_lfw.sys, "orig_argv", original), patch.object(run_lfw, "evaluate_lfw", side_effect=ValueError("제어된 입력 차단")):
                self.assertEqual(run_lfw.main(), 1)
            work = revision.parent / "work/stage1_revision2"
            process = json.loads((work / "evaluation_process.json").read_text(encoding="utf-8"))
            self.assertEqual(process["argv"], [run_lfw.sys.executable, "tools/run_lfw.py", "evaluate"])
            self.assertEqual(process["orig_argv"], original)
            failure = json.loads((work / "evaluation_failures.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(failure["orig_argv"], original)

    def test_techniques_prepare_extract_alignment_and_success_follow_actual_phase(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(temporary), "prepare")
            rows = {row["id"]: row for row in writer.state["techniques"]}
            self.assertEqual(rows["template"]["status"], "unused")
            self.assertEqual(rows["ct_train"]["status"], "unused")
            writer.progress("extract", {"bytes": 100, "total_bytes": 200, "files": 1})
            self.assertEqual(rows["alignment"]["status"], "planned")
            self.assertEqual(next(row for row in writer.state["steps"] if row["id"] == "alignment")["status"], "waiting")
            writer.progress("alignment", {"completed": 2, "total": 4, "failed": 0, "elapsed_seconds": 1})
            self.assertEqual(rows["alignment"]["status"], "active")
            for identifier in ["fp32", "flip", "lfw"]:
                self.assertEqual(rows[identifier]["status"], "planned")
                self.assertIn("미적용", rows[identifier]["detail"])
            writer.finish({"status": "complete", "images": 4, "successful_images": 4, "failed_images": [], "coverage": 1.0})
            self.assertEqual(rows["alignment"]["status"], "complete")
            self.assertIsNone(writer.state["evaluation"]["metrics"])

    def test_techniques_unmet_pauses_alignment_without_applying_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(temporary), "prepare")
            writer.progress("alignment", {"completed": 2, "total": 4, "failed": 1, "elapsed_seconds": 1})
            writer.finish({"status": "protocol_unmet", "images": 4, "successful_images": 3, "failed_images": [{}], "coverage": 0.75})
            rows = {row["id"]: row for row in writer.state["techniques"]}
            self.assertEqual(rows["alignment"]["status"], "paused")
            self.assertIn("75.00%", rows["alignment"]["detail"])
            for identifier in ["fp32", "flip", "lfw"]:
                self.assertEqual(rows[identifier]["status"], "planned")
                self.assertIn("미적용", rows[identifier]["detail"])
            self.assertFalse(any(row["status"] == "active" for row in writer.state["techniques"]))
            self.assertEqual(writer.state["evaluation"]["status"], "protocol_unmet")
            self.assertIsNone(writer.state["evaluation"]["metrics"])

    def test_techniques_evaluate_inference_and_metric_completion_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(temporary), "evaluate")
            rows = {row["id"]: row for row in writer.state["techniques"]}
            for identifier in ["fp32", "flip", "lfw"]:
                self.assertEqual(rows[identifier]["status"], "planned")
            writer.progress("inference", {"completed": 2, "total": 4, "resumed_at": 0, "elapsed_seconds": 1})
            self.assertEqual(rows["alignment"]["status"], "complete")
            self.assertEqual(rows["fp32"]["status"], "active")
            self.assertEqual(rows["flip"]["status"], "active")
            self.assertEqual(rows["lfw"]["status"], "planned")
            writer.progress("inference", {"completed": 4, "total": 4, "resumed_at": 0, "elapsed_seconds": 2})
            self.assertEqual(rows["lfw"]["status"], "active")
            # 반환값은 완료 상태 전환만 검사하는 제어 fixture이며 실제 얼굴 정확도가 아닙니다.
            writer.finish({"accuracy": 0.5, "std": 0.1})
            for identifier in ["fp32", "flip", "lfw"]:
                self.assertEqual(rows[identifier]["status"], "complete")
            self.assertEqual(rows["template"]["status"], "unused")
            self.assertEqual(rows["ct_train"]["status"], "unused")

    def test_techniques_validated_cache_reuse_and_previous_alignment_stay_consistent(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            root = Path(temporary)
            prepared = StatusWriter(root, "prepare")
            prepared.finish({"status": "complete", "images": 4, "successful_images": 4, "failed_images": [], "coverage": 1.0})
            self.assertIn("캐시 재사용", next(row for row in prepared.state["techniques"] if row["id"] == "alignment")["detail"])
            writer = StatusWriter(root, "evaluate")
            rows = {row["id"]: row for row in writer.state["techniques"]}
            self.assertEqual(rows["alignment"]["status"], "complete")
            self.assertEqual(rows["fp32"]["status"], "planned")
            writer.finish({"accuracy": 0.5, "std": 0.1})
            for identifier in ["fp32", "flip"]:
                self.assertEqual(rows[identifier]["status"], "complete")
                self.assertIn("재사용", rows[identifier]["detail"])
                self.assertIn("신규 추론 없음", rows[identifier]["detail"])
            self.assertEqual(rows["lfw"]["status"], "complete")

    def test_techniques_failure_main_path_clears_only_applied_active_methods(self):
        for phase in ["prepare", "evaluate"]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
                revision = Path(temporary) / "2차"
                work = revision.parent / "work/stage1_revision2"
                def fail_prepare(stage, progress):
                    progress("alignment", {"completed": 2, "total": 4, "failed": 0, "elapsed_seconds": 1})
                    raise RuntimeError("제어된 준비 실패")
                def fail_evaluate(stage, device, batch, progress):
                    progress("inference", {"completed": 2, "total": 4, "resumed_at": 0, "elapsed_seconds": 1})
                    raise RuntimeError("제어된 추론 실패")
                with patch.object(run_lfw, "ROOT", revision), patch.object(run_lfw.sys, "argv", ["run_lfw.py", phase]), patch.object(run_lfw, "prepare_lfw", side_effect=fail_prepare), patch.object(run_lfw, "evaluate_lfw", side_effect=fail_evaluate):
                    self.assertEqual(run_lfw.main(), 1)
                state = json.loads((work / "status.json").read_text(encoding="utf-8"))
                rows = {row["id"]: row for row in state["techniques"]}
                for identifier in (["alignment"] if phase == "prepare" else ["fp32", "flip"]):
                    self.assertEqual(rows[identifier]["status"], "failed")
                self.assertEqual(rows["lfw"]["status"], "planned")
                self.assertFalse(any(row["status"] == "active" for row in state["techniques"] + state["steps"]))
                self.assertEqual(state["status"], "failed")

    def test_techniques_blocked_evaluation_is_unapplied(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            revision = Path(temporary) / "2차"
            work = revision.parent / "work/stage1_revision2"
            with patch.object(run_lfw, "ROOT", revision), patch.object(run_lfw.sys, "argv", ["run_lfw.py", "evaluate"]), patch.object(run_lfw, "evaluate_lfw", side_effect=ValueError("제어된 캐시 검증 차단")):
                self.assertEqual(run_lfw.main(), 1)
            state = json.loads((work / "status.json").read_text(encoding="utf-8"))
            for row in state["techniques"]:
                self.assertNotEqual(row["status"], "active")
                if row["id"] in {"fp32", "flip", "lfw"}:
                    self.assertEqual(row["status"], "planned")
                    self.assertIn("미적용", row["detail"])
            self.assertIsNone(state["evaluation"]["metrics"])

    def test_techniques_partial_resume_uses_observed_remaining_inference(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(temporary), "evaluate")
            writer.progress("inference", {"completed": 3, "total": 4, "resumed_at": 2, "elapsed_seconds": 1})
            rows = {row["id"]: row for row in writer.state["techniques"]}
            self.assertEqual(rows["fp32"]["status"], "active")
            self.assertIn("2", rows["fp32"]["detail"])
            self.assertEqual(rows["lfw"]["status"], "planned")
            writer.progress("inference", {"completed": 4, "total": 4, "resumed_at": 2, "elapsed_seconds": 2})
            writer.finish({"accuracy": 0.5, "std": 0.1})
            self.assertEqual(rows["fp32"]["status"], "complete")
            self.assertEqual(rows["lfw"]["status"], "complete")

    def test_new_job_resets_only_throughput_and_preserves_old_points_privately(self):
        for phase in ["prepare", "evaluate"]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
                root = Path(temporary)
                old = initial_status()
                old["series"]["throughput"] = [{"time": "2026-11-03T00:00:00+09:00", "value": 100000.0}]
                old["series"]["metrics"] = [{"time": "2026-11-03T00:00:00+09:00", "value": 0.5}]
                old["model"]["measured_metrics"] = [{"benchmark": "제어 fixture", "metric": "검사용", "value": 0.5, "unit": "%", "provenance": "실측 아님"}]
                (root / "status.json").write_text(json.dumps(old), encoding="utf-8")
                writer = StatusWriter(root, phase)
                self.assertEqual(writer.state["series"]["throughput"], [])
                self.assertEqual(writer.state["series"]["metrics"], old["series"]["metrics"])
                self.assertEqual(writer.state["model"]["measured_metrics"], old["model"]["measured_metrics"])
                records = [json.loads(line) for line in (root / "throughput_history.jsonl").read_text(encoding="utf-8").split("\n") if line]
                self.assertEqual(records[-1]["points"], old["series"]["throughput"])
                self.assertEqual(records[-1]["previous_unit"], "bytes")
                self.assertEqual(records[-1]["reason"], "new_job")

    def test_throughput_bytes_to_images_never_mixes_units(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            root = Path(temporary)
            writer = StatusWriter(root, "prepare")
            writer.progress("extract", {"completed": 100000, "total": 200000, "elapsed_seconds": 1})
            writer.progress("alignment", {"completed": 10, "total": 20, "elapsed_seconds": 1})
            self.assertEqual(writer.state["progress"]["unit"], "images")
            self.assertEqual([point["value"] for point in writer.state["series"]["throughput"]], [10.0])
            records = [json.loads(line) for line in (root / "throughput_history.jsonl").read_text(encoding="utf-8").split("\n") if line]
            self.assertEqual(records[-1]["previous_unit"], "bytes")
            self.assertEqual(records[-1]["next_unit"], "images")
            self.assertEqual(records[-1]["reason"], "unit_change")
            self.assertEqual([point["value"] for point in records[-1]["points"]], [100000.0])

    def test_throughput_same_unit_observations_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(temporary), "prepare")
            writer.progress("alignment", {"completed": 10, "total": 100, "elapsed_seconds": 1})
            writer.progress("alignment", {"completed": 30, "total": 100, "elapsed_seconds": 2})
            self.assertEqual([point["value"] for point in writer.state["series"]["throughput"]], [10.0, 15.0])

    def test_prepare_unmet_reports_real_coverage_and_no_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(temporary), "prepare")
            writer.finish({"status": "protocol_unmet", "images": 5, "successful_images": 3, "failed_images": [{}, {}], "coverage": 0.6})
            state = writer.state
            self.assertEqual(state["evaluation"]["status"], "protocol_unmet")
            self.assertEqual(state["evaluation"]["detection_coverage"], 0.6)
            self.assertEqual(state["evaluation"]["failed_images"], 2)
            self.assertIn("성공 3 / 전체 5", state["current_task"]["detail"])
            self.assertIn("실패 2", state["current_task"]["detail"])
            self.assertIn("정확도 미측정", state["current_task"]["detail"])
            self.assertIn("검토 대기", state["current_task"]["detail"])
            self.assertNotIn("할당량", state["current_task"]["detail"])
            self.assertIsNone(state["evaluation"]["metrics"])
            self.assertEqual(state["model"]["measured_metrics"], [])

    def test_prepare_success_preserves_unmeasured_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary, patch("tools.run_lfw.gpu_snapshot", return_value=None):
            writer = StatusWriter(Path(temporary), "prepare")
            writer.finish({"status": "complete", "images": 5, "successful_images": 5, "failed_images": [], "coverage": 1.0})
            self.assertEqual(writer.state["evaluation"]["status"], "ready")
            self.assertEqual(writer.state["evaluation"]["detection_coverage"], 1.0)
            self.assertEqual(writer.state["evaluation"]["failed_images"], 0)
            self.assertIsNone(writer.state["evaluation"]["metrics"])
            self.assertEqual(writer.state["model"]["measured_metrics"], [])

    def test_singleton_batch_keeps_axis_rgb_normalization_and_flip(self):
        import torch
        class Probe:
            def __call__(self, inputs, phase):
                self.inputs, self.phase = inputs, phase
                return torch.ones((len(inputs), 512), dtype=torch.float32)
        model = Probe()
        rgb = np.zeros((1, 112, 112, 3), dtype=np.uint8)
        rgb[0, :, 0, 0] = 255
        first, second = infer_feature_batch(model, rgb, "cpu")
        self.assertEqual(first.shape, (1, 512))
        self.assertEqual(second.shape, (1, 512))
        self.assertEqual(model.phase, "infer")
        self.assertEqual(model.inputs.shape, (2, 3, 112, 112))
        self.assertEqual(model.inputs[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(model.inputs[0, 1, 0, 0].item(), -1.0)
        self.assertEqual(model.inputs[1, 0, 0, -1].item(), 1.0)

    def test_face_selection_uses_center_then_area_then_score(self):
        faces = np.zeros((3, 15), dtype=np.float32)
        faces[:, :4] = [[0, 0, 20, 20], [40, 40, 20, 20], [30, 30, 40, 40]]
        faces[:, 14] = [0.99, 0.99, 0.91]
        np.testing.assert_array_equal(select_face(faces, 100, 100), faces[2])
        with self.assertRaises(ValueError):
            select_face(None, 100, 100)

    def test_flip_sum_is_normalized_after_addition(self):
        original = np.array([[3., 0.], [0., 2.]], dtype=np.float32)
        flipped = np.array([[0., 4.], [2., 0.]], dtype=np.float32)
        actual = checked_embedding_sum(original, flipped)
        np.testing.assert_allclose(actual[0], [0.6, 0.8], atol=1e-12)
        self.assertEqual(actual.dtype, np.float64)
        with self.assertRaises(ValueError):
            checked_embedding_sum(original, -original)

    def test_resume_rejects_foreign_binding_and_uncommitted_data(self):
        binding = {"checkpoint": "fixed", "cache": "fixed", "source": "fixed"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npy"
            data = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(3, 2, 512))
            data[:] = 1
            data.flush()
            del data
            state = {"binding": binding, "next_index": 1}
            self.assertEqual(validate_resume(state, binding, path, 3), 1)
            with self.assertRaises(ValueError):
                validate_resume(state, {**binding, "cache": "other"}, path, 3)
            with self.assertRaises(ValueError):
                validate_resume({"binding": binding, "next_index": 4}, binding, path, 3)
            with self.assertRaises(ValueError):
                validate_resume(state, binding, path.with_name("missing.npy"), 3)


if __name__ == "__main__":
    unittest.main()
