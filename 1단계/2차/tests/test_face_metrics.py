"""동률 점수·임계값 누출·media 집계 오류를 잡는 작은 실제 계산 검사입니다."""
import unittest
import numpy as np
from benchmark.metrics import verification_roc, lfw_ten_fold, aggregate_templates


class MetricsTests(unittest.TestCase):
    def test_ties_do_not_create_impossible_low_far_acceptance(self):
        result = verification_roc([0.9, 0.8, 0.8, 0.1], [1, 1, 0, 0], [0.0, 0.5])
        self.assertEqual(result[0]["tar"], 0.5)
        self.assertEqual(result[0]["actual_far"], 0.0)
        self.assertEqual(result[1]["tar"], 1.0)
        self.assertEqual(result[1]["actual_far"], 0.5)

    def test_single_class_and_nonfinite_scores_are_rejected(self):
        for scores, labels in [([0.1, 0.2], [1, 1]), ([0.1, float("nan")], [0, 1])]:
            with self.assertRaises(ValueError):
                verification_roc(scores, labels)

    def test_lfw_fold_threshold_uses_other_nine_folds(self):
        labels = np.tile(np.r_[np.ones(300), np.zeros(300)], 10)
        scores = np.where(labels == 1, 0.9, 0.1)
        first = lfw_ten_fold(scores, labels)
        altered = labels.copy()
        altered[:600] = 1 - altered[:600]
        second = lfw_ten_fold(scores, altered)
        self.assertEqual(first["accuracy"], 1.0)
        self.assertEqual(first["folds"][0]["threshold"], second["folds"][0]["threshold"])
        self.assertEqual(second["folds"][0]["accuracy"], 0.0)

    def test_lfw_requires_canonical_six_thousand_pairs(self):
        with self.assertRaises(ValueError):
            lfw_ten_fold(np.zeros(5999), np.zeros(5999))

    def test_video_frames_are_averaged_before_template_sum(self):
        ids, result = aggregate_templates(np.array([[2., 0.], [0., 2.], [2., 0.]]), [7, 7, 7], [3, 3, 4], [1., 1., 1.])
        self.assertEqual(ids.tolist(), [7])
        np.testing.assert_allclose(result[0], [3 / np.sqrt(10), 1 / np.sqrt(10)])

