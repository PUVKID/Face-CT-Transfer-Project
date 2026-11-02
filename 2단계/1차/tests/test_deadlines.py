from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from ct_transfer import budget, training
except ImportError:
    budget = None


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(budget, "전체 실행 상한 구현 전 제어된 RED")
        private = ROOT.parent / "work/private/tests"
        private.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=private)
        self.path = Path(self.temp.name)

    def tearDown(self):
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def test_resume_keeps_the_first_deadline_and_rejects_changed_binding(self):
        now = [100.]
        first = budget.RunDeadline(self.path / "deadline.json", 10, "same", clock=lambda: now[0])
        self.assertEqual(first.deadline_utc, 110.)
        now[0] = 108.
        resumed = budget.RunDeadline(self.path / "deadline.json", 10, "same", clock=lambda: now[0])
        self.assertEqual(resumed.deadline_utc, 110.)
        resumed.check()
        now[0] = 110.
        with self.assertRaises(budget.DeadlineExceeded):
            resumed.check()
        with self.assertRaises(ValueError):
            budget.RunDeadline(self.path / "deadline.json", 10, "different", clock=lambda: now[0])

    def test_expired_fold_training_saves_resumable_partial_batch_state(self):
        torch.set_num_threads(1)
        x = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]] * 4)
        y = torch.tensor([0, 0, 1, 1] * 4)
        config = {"max_epochs": 2, "batch_size": 4, "patience": 5, "lr": .01, "backbone_lr": .001, "weight_decay": 0., "warmup_epochs": 0, "amp": False, "seed": 17, "wall_seconds": 60}
        clock = [0.]
        def ticking_clock():
            value = clock[0]
            clock[0] += 1.
            return value
        deadline = budget.RunDeadline(self.path / "run.json", 3, "same", clock=ticking_clock)
        torch.manual_seed(9)
        first = nn.Linear(2, 2)
        initial = {key: value.clone() for key, value in first.state_dict().items()}
        paused = training.fit_model(first, x, y, x, y, config, self.path / "paused", {"split": "A"}, deadline=deadline)
        self.assertEqual(paused["status"], "paused")
        checkpoint = torch.load(self.path / "paused/latest.pt", weights_only=True)
        self.assertFalse(checkpoint["epoch_complete"])
        self.assertGreater(checkpoint["next_sample"], 0)
        self.assertLess(checkpoint["next_sample"], len(x))
        training.fit_model(first, x, y, x, y, config, self.path / "paused", {"split": "A"})
        second = nn.Linear(2, 2)
        second.load_state_dict(initial)
        training.fit_model(second, x, y, x, y, config, self.path / "whole", {"split": "A"})
        for key in first.state_dict():
            torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
