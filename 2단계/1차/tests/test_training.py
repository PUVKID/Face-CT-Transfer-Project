import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from ct_transfer import evaluation, models, training
except ModuleNotFoundError as error:
    if not error.name.startswith("ct_transfer"):
        raise
    evaluation = models = training = None


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.assertIsNotNone(training, "학습 경계 구현 전 제어 실패")
        private = ROOT.parent / "work" / "private" / "tests"
        private.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=private)
        self.path = Path(self.temp.name)
        self.x = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]] * 4)
        self.y = torch.tensor([0, 0, 1, 1] * 4)
        self.config = {"max_epochs": 2, "batch_size": 4, "patience": 5, "lr": .01, "backbone_lr": .001, "weight_decay": 0., "warmup_epochs": 0, "amp": False, "seed": 17, "wall_seconds": 60}

    def tearDown(self):
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def test_small_real_head_resume_matches_uninterrupted_training(self):
        torch.manual_seed(9)
        initial = nn.Linear(2, 2)
        uninterrupted, resumed = copy.deepcopy(initial), copy.deepcopy(initial)
        full = training.fit_model(uninterrupted, self.x, self.y, self.x, self.y, self.config, self.path / "full", {"split": "same"})
        paused = training.fit_model(resumed, self.x, self.y, self.x, self.y, self.config, self.path / "resume", {"split": "same"}, max_epochs_this_call=1)
        self.assertEqual(paused["status"], "paused")
        result = training.fit_model(resumed, self.x, self.y, self.x, self.y, self.config, self.path / "resume", {"split": "same"})
        self.assertEqual(result["status"], "complete")
        for key in uninterrupted.state_dict():
            torch.testing.assert_close(uninterrupted.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(full["best_epoch"], result["best_epoch"])

    def test_resume_rejects_a_different_split_instead_of_reusing_weights(self):
        model = nn.Linear(2, 2)
        training.fit_model(model, self.x, self.y, self.x, self.y, self.config, self.path / "fit", {"split": "A"}, max_epochs_this_call=1)
        with self.assertRaises(ValueError):
            training.fit_model(model, self.x, self.y, self.x, self.y, self.config, self.path / "fit", {"split": "B"})

    def test_completed_resume_rejects_foreign_best_checkpoint_binding(self):
        model = nn.Linear(2, 2)
        destination = self.path / "foreign_best"
        training.fit_model(model, self.x, self.y, self.x, self.y, self.config, destination, {"split": "A"})
        checkpoint = torch.load(destination / "best.pt", weights_only=True)
        checkpoint["binding"] = "foreign-split-binding"
        checkpoint["model"] = {key: torch.zeros_like(value) for key, value in checkpoint["model"].items()}
        training.atomic_torch(destination / "best.pt", checkpoint)
        with self.assertRaises(ValueError):
            training.fit_model(model, self.x, self.y, self.x, self.y, self.config, destination, {"split": "A"})

    def test_best_checkpoint_rejects_wrong_epoch_and_invalid_tensor(self):
        model = nn.Linear(2, 2)
        for mutation in ("epoch", "nan", "shape"):
            with self.subTest(mutation=mutation):
                checkpoint = {"binding": "same", "epoch": 3, "model": {key: value.clone() for key, value in model.state_dict().items()}}
                if mutation == "epoch":
                    checkpoint["epoch"] = 4
                elif mutation == "nan":
                    checkpoint["model"]["weight"][0, 0] = float("nan")
                else:
                    checkpoint["model"]["weight"] = torch.zeros(1)
                training.atomic_torch(self.path / "bad_best.pt", checkpoint)
                with self.assertRaises(ValueError):
                    training.restore_best(model, self.path / "bad_best.pt", "same", 3)

    def test_new_completion_also_rejects_foreign_best_checkpoint(self):
        model = nn.Linear(2, 2)
        original = training.atomic_torch
        def inject(path, value):
            if Path(path).name == "best.pt":
                value = {**value, "binding": "foreign"}
            original(path, value)
        with mock.patch.object(training, "atomic_torch", side_effect=inject):
            with self.assertRaises(ValueError):
                training.fit_model(model, self.x, self.y, self.x, self.y, self.config, self.path / "new_completion", {"split": "A"})

    def test_class_weights_use_only_training_labels(self):
        weights = training.class_weights(torch.tensor([0, 0, 0, 1]), 2)
        torch.testing.assert_close(weights, torch.tensor([2 / 3, 2.]))

    def test_missing_class_auc_is_unavailable_instead_of_zero(self):
        result = evaluation.metrics(np.array([0, 0]), np.array([[.8, .1, .1], [.7, .2, .1]]), 3)
        self.assertIsNone(result["roc_auc_ovr"])

    def test_patient_confidence_interval_refuses_unknown_iq_patient_mapping(self):
        with self.assertRaises(ValueError):
            evaluation.patient_bootstrap(np.array([0, 1]), np.array([[.8, .2], [.2, .8]]), np.array(["a", "b"]), known_patients=False, iterations=5)

    def test_incomplete_folds_cannot_generate_a_result_report(self):
        with self.assertRaises(ValueError):
            evaluation.write_case_report(self.path / "report.md", {"id": "01"}, [{"status": "complete"}] * 6, {})
        self.assertFalse((self.path / "report.md").exists())

    def test_thirty_settings_are_six_methods_with_five_explicit_seeds(self):
        cases = models.experiment_cases()
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({case["id"] for case in cases}), 30)
        self.assertEqual({case["scope"] for case in cases}, {"frozen", "lastblock", "full"})
        self.assertEqual({case["initialization"] for case in cases}, {"face", "luna-adapted"})

    def test_trainable_image_head_and_batchnorm_statistics_contract(self):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512))
                self.bn = nn.BatchNorm1d(512)
            def forward(self, x):
                return self.bn(self.body(x))
        model = models.Classifier(Backbone(), 3, "lastblock").train()
        self.assertFalse(model.backbone.bn.training)
        self.assertFalse(model.backbone.body[0].weight.requires_grad)
        self.assertTrue(model.backbone.body[-1].weight.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)
        before = model.head.weight.detach().clone()
        x = torch.randn(12, 512)
        y = torch.tensor([0, 1, 2] * 4)
        training.fit_model(model, x, y, x, y, self.config, self.path / "head", {"unit": "head"})
        self.assertFalse(torch.equal(before, model.head.weight))

    def test_temperature_fit_has_positive_finite_output(self):
        result = training.fit_temperature(torch.tensor([[6., 0.], [5., 0.], [0., 6.], [0., 5.]]), torch.tensor([0, 1, 1, 0]))
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
