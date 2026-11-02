from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ct_transfer import training


class PrecisionTests(unittest.TestCase):
    def test_amp_requires_explicit_native_bf16_without_fallback(self):
        with self.assertRaises(ValueError):
            training.amp_policy(torch.device("cuda"), {"amp": True, "amp_dtype": "float16"})
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            with self.assertRaises(RuntimeError):
                training.amp_policy(torch.device("cuda"), {"amp": True, "amp_dtype": "bfloat16"})
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            enabled, dtype = training.amp_policy(torch.device("cuda"), {"amp": True, "amp_dtype": "bfloat16"})
        self.assertTrue(enabled)
        self.assertEqual(dtype, torch.bfloat16)

    def test_cpu_and_frozen_no_amp_path_remains_float32(self):
        enabled, _dtype = training.amp_policy(torch.device("cpu"), {"amp": False})
        self.assertFalse(enabled)


if __name__ == "__main__":
    unittest.main(verbosity=2)
