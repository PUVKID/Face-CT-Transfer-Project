import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.stage1 import TransferClassifier, load_checkpoint, preprocess, seed_everything


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(3, 512)

    def forward(self, x):
        return self.linear(self.pool(self.bn(x)).flatten(1))


class Stage1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(41)

    def test_rgb_endpoint_normalization(self):
        x = torch.zeros(3, 112, 112, dtype=torch.uint8)
        x[2] = 255
        y = preprocess(x)
        self.assertEqual(tuple(y.shape), (3, 112, 112))
        self.assertTrue(torch.equal(y[0], -torch.ones(112, 112)))
        self.assertTrue(torch.equal(y[2], torch.ones(112, 112)))

    def test_explicit_bgr_order(self):
        x = torch.zeros(3, 112, 112, dtype=torch.uint8)
        x[0] = 255
        self.assertTrue(torch.equal(preprocess(x, color_order="BGR")[2], torch.ones(112, 112)))

    def test_medical_grayscale_replicates_channels(self):
        y = preprocess(torch.full((1, 32, 40), 0.25))
        self.assertTrue(torch.equal(y[0], y[2]))
        self.assertTrue(torch.allclose(y, torch.full_like(y, -0.5)))

    def test_invalid_inputs_rejected(self):
        bad = [torch.zeros(112, 112), torch.zeros(4, 8, 8),
               torch.full((3, 8, 8), float("nan")),
               torch.full((3, 8, 8), float("inf")),
               torch.full((3, 8, 8), -0.01), torch.full((3, 8, 8), 1.01),
               torch.zeros(3, 8, 8, dtype=torch.int64), torch.zeros(3, 0, 8)]
        for x in bad:
            with self.subTest(shape=tuple(x.shape), dtype=x.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    preprocess(x)
        with self.assertRaises(ValueError):
            preprocess(torch.zeros(3, 8, 8), color_order="UNKNOWN")

    def test_frozen_backbone_and_bn_statistics(self):
        model = TransferClassifier(TinyBackbone(), frozen=True).train()
        before = model.backbone.bn.running_mean.clone()
        model(torch.rand(2, 3, 112, 112)).sum().backward()
        self.assertTrue(torch.equal(before, model.backbone.bn.running_mean))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertIsNotNone(model.head.weight.grad)
        self.assertFalse(model.backbone.training)

    def test_unfreeze_restores_gradient_and_three_logits(self):
        model = TransferClassifier(TinyBackbone(), frozen=True)
        model.set_frozen(False)
        model.train()
        y = model(torch.rand(2, 3, 112, 112))
        self.assertEqual(tuple(y.shape), (2, 3))
        y.square().sum().backward()
        self.assertGreater(model.backbone.linear.weight.grad.abs().sum().item(), 0)

    def test_invalid_backbone_features_rejected(self):
        for backbone in [nn.Identity(), nn.Sequential(nn.Flatten(), nn.Linear(3 * 112 * 112, 4))]:
            with self.assertRaises(ValueError):
                TransferClassifier(backbone)(torch.zeros(2, 3, 112, 112))

    def test_invalid_model_input_rejected(self):
        model = TransferClassifier(TinyBackbone())
        for x in [torch.zeros(2, 1, 112, 112), torch.zeros(2, 3, 8, 8),
                  torch.full((2, 3, 112, 112), float("nan")),
                  torch.full((2, 3, 112, 112), 2.0)]:
            with self.assertRaises(ValueError):
                model(x)

    def test_checkpoint_strict_roundtrip_and_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            original = TinyBackbone()
            for wrapper in [None, "state_dict", "model"]:
                state = original.state_dict()
                torch.save(state if wrapper is None else {wrapper: state}, path)
                target = TinyBackbone()
                load_checkpoint(target, path, state_key=wrapper)
                self.assertTrue(all(torch.equal(state[k], target.state_dict()[k]) for k in state))

    def test_missing_malformed_and_mismatch_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(TinyBackbone(), path)
            for state in [{"unknown": torch.zeros(1)}, {"linear.weight": torch.zeros(1)},
                          {"not_tensor": "bad"}]:
                torch.save(state, path)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_checkpoint(TinyBackbone(), path)

    def test_prefix_is_explicit_not_silently_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            state = TinyBackbone().state_dict()
            torch.save({"module." + k: v for k, v in state.items()}, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(TinyBackbone(), path)
            load_checkpoint(TinyBackbone(), path, prefix="module.")

    def test_seed_repeatability(self):
        seed_everything(7)
        a = torch.rand(4)
        seed_everything(7)
        self.assertTrue(torch.equal(a, torch.rand(4)))

    def test_pinned_loader_rejects_source_tampering_before_execution(self):
        from src.adaface import load_adaface
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.py"
            source.write_text("raise AssertionError('실행되면 안 됨')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_adaface(source, Path(tmp) / "absent.pt")

    def test_case_manifest_is_complete_and_diagnostic(self):
        from src.diagnostics import read_cases
        cases = read_cases(Path(__file__).resolve().parents[1] / "configs" / "stage1_cases.json")
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({case["id"] for case in cases}), 30)
        self.assertTrue(all(case["hypothesis"] for case in cases))

    def test_duplicate_case_manifest_rejected(self):
        import json
        from src.diagnostics import read_cases
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps({"cases": [{"id": "rgb_endpoints", "hypothesis": "중복"}] * 30}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_cases(path)

    def test_diagnostic_training_restores_parameters_and_bn(self):
        from src.diagnostics import run_case
        with tempfile.TemporaryDirectory() as tmp:
            model = TransferClassifier(TinyBackbone()).eval()
            for case_id in ("head_update", "unfrozen_gradient"):
                before = {k: v.clone() for k, v in model.state_dict().items()}
                run_case(case_id, model, device="cpu", private_temp=tmp)
                self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()), case_id)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA 장치가 없는 환경")
    def test_cpu_cuda_comparison_keeps_buffers_mutable(self):
        from src.diagnostics import run_case
        with tempfile.TemporaryDirectory() as tmp:
            model = TransferClassifier(TinyBackbone()).cuda().eval()
            run_case("cpu_cuda_parity", model, device="cuda", private_temp=tmp)
            with torch.no_grad():
                model.backbone.bn.running_mean.zero_()


if __name__ == "__main__":
    unittest.main()
