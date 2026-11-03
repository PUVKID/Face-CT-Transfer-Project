"""안전한 역직렬화·형상·정렬·메타데이터 대응을 검증합니다."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from benchmark.model import safe_state_dict, validate_state_dict, construct_topofr
from benchmark.protocol import similarity_matrix, validate_ijb_metadata
from benchmark.metrics import lfw_ten_fold


class UnsafeFixture:
    def __reduce__(self):
        return (Path.write_text, (Path("never_execute_fixture.txt"), "실행 금지"))


class ModelProtocolTests(unittest.TestCase):
    def test_author_r200_constructor_keeps_class_centers(self):
        class AuthorFactory:
            def get_model(self, name, **kwargs):
                self.name, self.kwargs = name, kwargs
                if kwargs.get("num_classes") is None:
                    raise ValueError("저자 R200 클래스 중심 생성 인자가 없습니다")
                return "strict로 전체 적재할 원본 구조"
        package = AuthorFactory()
        self.assertEqual(construct_topofr(package), "strict로 전체 적재할 원본 구조")
        self.assertEqual(package.name, "r200")
        self.assertEqual(package.kwargs, {"dropout": 0, "fp16": False, "num_features": 512, "num_classes": 360232})

    def test_weights_only_rejects_custom_callable_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "unsafe.pt"
            torch.save(UnsafeFixture(), file)
            with self.assertRaises(ValueError):
                safe_state_dict(file)
            self.assertFalse(Path("never_execute_fixture.txt").exists())

    def test_strict_state_rejects_shape_dtype_and_nonfinite(self):
        model = torch.nn.Linear(2, 2)
        for bad in [dict(weight=torch.zeros(1, 2), bias=torch.zeros(2)), dict(weight=torch.zeros(2, 2, dtype=torch.float64), bias=torch.zeros(2)), dict(weight=torch.full((2, 2), float("nan")), bias=torch.zeros(2))]:
            with self.assertRaises(ValueError):
                validate_state_dict(bad, model.state_dict())
        validate_state_dict(model.state_dict(), model.state_dict())

    def test_similarity_transform_preserves_scale_translation(self):
        points = np.array([[0., 0.], [2., 0.], [1., 1.], [0., 2.], [2., 2.]])
        actual = similarity_matrix(points, points * 2 + [3, 4])
        np.testing.assert_allclose(actual, [[2., 0., 3.], [0., 2., 4.]], atol=1e-12)

    def test_squared_l2_grid_is_used_without_cosine_threshold_conversion(self):
        labels = np.tile(np.r_[np.ones(300), np.zeros(300)], 10)
        distances = np.where(labels == 1, 0.011, 0.3)
        result = lfw_ten_fold(distances, labels, input_kind="squared_l2")
        self.assertEqual(result["folds"][0]["threshold"], 0.02)
        self.assertEqual(result["accuracy"], 1.0)

    def test_ijb_reordered_names_and_unknown_pair_templates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "loose_crop").mkdir()
            for name in ["a.jpg", "b.jpg", "c.jpg"]:
                (root / "loose_crop" / name).write_bytes(b"fixture")
            (root / "meta/ijbc_face_tid_mid.txt").write_text("a.jpg 10 1\nb.jpg 11 2\nc.jpg 12 3\n", encoding="utf-8")
            landmark = "1 1 2 1 1.5 2 1 3 2 3 0.9"
            score_file = root / "meta/ijbc_name_5pts_score.txt"
            score_file.write_text("\n".join(name + " " + landmark for name in ["a.jpg", "b.jpg", "c.jpg"]), encoding="utf-8")
            pairs = root / "meta/ijbc_template_pair_label.txt"
            pairs.write_text("10 11 1\n10 12 0\n", encoding="utf-8")
            self.assertEqual(validate_ijb_metadata(root)["images"], 3)
            score_file.write_text("\n".join(name + " " + landmark for name in ["b.jpg", "a.jpg", "c.jpg"]), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_ijb_metadata(root)
            score_file.write_text("\n".join(name + " " + landmark for name in ["a.jpg", "b.jpg", "c.jpg"]), encoding="utf-8")
            pairs.write_text("10 99 1\n10 12 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_ijb_metadata(root)
