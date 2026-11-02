import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ct_transfer import training


class NumpyBatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        private = ROOT.parent / "work/private/tests"
        private.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=private)
        self.path = Path(self.temp.name)
        self.mappings = []
        self.config = {"max_epochs": 2, "batch_size": 4, "patience": 5, "lr": .01, "backbone_lr": .001, "weight_decay": 0., "warmup_epochs": 0, "amp": False, "seed": 17, "wall_seconds": 60}
        self.x = np.random.default_rng(101).normal(size=(9, 512)).astype(np.float32)
        self.y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)

    def tearDown(self):
        for mapped in self.mappings:
            mapped._mmap.close()
        self.temp.cleanup()

    def test_singleton_features_gray_rgb_keep_the_numpy_batch_axis(self):
        for sample_shape in ((512,), (1, 4, 4), (3, 4, 4)):
            values = np.arange(3 * int(np.prod(sample_shape)), dtype=np.float32).reshape((3, *sample_shape))
            path = self.path / ("array_" + "_".join(map(str, sample_shape)) + ".npy")
            np.save(path, values, allow_pickle=False)
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            self.mappings.append(mapped)
            for kind, inputs in (("numpy", values), ("memmap", mapped), ("tensor", torch.from_numpy(values))):
                for index_kind, indices in (("torch_one", torch.tensor([1])), ("torch_many", torch.tensor([2, 0])), ("numpy_one", np.array([1])), ("list_one", [1]), ("slice_one", slice(1, 2))):
                    with self.subTest(sample_shape=sample_shape, inputs=kind, indices=index_kind):
                        expected = torch.tensor(values[np.array([2, 0]) if index_kind == "torch_many" else np.array([1])])
                        if len(sample_shape) == 3 and sample_shape[0] == 1:
                            expected = expected.repeat(1, 3, 1, 1)
                        actual = training._batch(inputs, indices, torch.device("cpu"))
                        self.assertEqual(actual.shape, expected.shape)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_last_numpy_batch_of_one_trains_like_tensor_inputs(self):
        torch.manual_seed(9)
        numpy_model = nn.Linear(512, 2)
        tensor_model = copy.deepcopy(numpy_model)
        result = training.fit_model(numpy_model, self.x, self.y, self.x, self.y, self.config, self.path / "numpy", {"fixture": "numpy_last_one"})
        self.assertEqual(result["status"], "complete")
        training.fit_model(tensor_model, torch.from_numpy(self.x), torch.from_numpy(self.y), torch.from_numpy(self.x), torch.from_numpy(self.y), self.config, self.path / "tensor", {"fixture": "numpy_last_one"})
        for key in numpy_model.state_dict():
            torch.testing.assert_close(numpy_model.state_dict()[key], tensor_model.state_dict()[key], rtol=0, atol=0)

    def test_readonly_memmap_last_batch_of_one_resumes_exactly(self):
        path = self.path / "features.npy"
        np.save(path, self.x, allow_pickle=False)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        self.mappings.append(values)
        torch.manual_seed(9)
        whole = nn.Linear(512, 2)
        resumed = copy.deepcopy(whole)
        training.fit_model(whole, values, self.y, values, self.y, self.config, self.path / "whole", {"fixture": "memmap_last_one"})
        paused = training.fit_model(resumed, values, self.y, values, self.y, self.config, self.path / "resumed", {"fixture": "memmap_last_one"}, max_epochs_this_call=1)
        self.assertEqual(paused["status"], "paused")
        result = training.fit_model(resumed, values, self.y, values, self.y, self.config, self.path / "resumed", {"fixture": "memmap_last_one"})
        self.assertEqual(result["status"], "complete")
        for key in whole.state_dict():
            torch.testing.assert_close(whole.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
