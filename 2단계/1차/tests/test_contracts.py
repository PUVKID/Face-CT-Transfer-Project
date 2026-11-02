import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from ct_transfer import data, splits
except ModuleNotFoundError as error:
    if not error.name.startswith("ct_transfer"):
        raise
    data = splits = None


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "데이터 경계 구현 전 제어 실패")
        private = ROOT.parent / "work" / "private" / "tests"
        private.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=private)
        self.path = Path(self.temp.name)

    def tearDown(self):
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def test_numeric_metadata_is_decoded_without_unrestricted_numpy_pickle(self):
        original = {"origin": np.array([1., 2., 3.]), "spacing": np.array([2., .7, .7]), "transform": np.eye(3)}
        stream = io.BytesIO()
        np.save(stream, original)
        result = data.decode_metadata(stream.getvalue())
        np.testing.assert_array_equal(result["origin"], [1., 2., 3.])
        np.testing.assert_array_equal(result["spacing"], [2., .7, .7])
        np.testing.assert_array_equal(result["transform"], np.eye(3))

    def test_malicious_metadata_cannot_execute_a_global_function(self):
        marker = self.path / "must-not-exist"
        expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
        class Attack:
            def __reduce__(self):
                return eval, (expression,)
        stream = io.BytesIO()
        np.save(stream, Attack())
        with self.assertRaises((ValueError, pickle.UnpicklingError)):
            data.decode_metadata(stream.getvalue())
        self.assertFalse(marker.exists())

    def test_metadata_size_limit_rejects_before_parsing(self):
        with self.assertRaises(ValueError):
            data.decode_metadata(b"x" * 20000)

    def test_numeric_volume_header_is_checked_before_large_allocation(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False, "shape": (1000000, 1000000, 1000000)})
        with self.assertRaises(ValueError):
            data.decode_volume(stream.getvalue())
        valid = io.BytesIO()
        np.save(valid, np.ones((2, 3, 4), dtype=np.int16), allow_pickle=False)
        np.testing.assert_array_equal(data.decode_volume(valid.getvalue()), np.ones((2, 3, 4), dtype=np.int16))

    def test_path_escape_and_duplicate_archive_members_are_rejected(self):
        with self.assertRaises(ValueError):
            data.validate_archive_members(["../escape.npy"])
        with self.assertRaises(ValueError):
            data.validate_archive_members(["root/a.npy", "root/a.npy"])

    def test_split_zip_can_be_read_without_copying_original_parts(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_LZMA) as archive:
            archive.writestr("sample.txt", "실제 ZIP 읽기")
        raw = stream.getvalue()
        first, second = self.path / "part1", self.path / "part2"
        first.write_bytes(raw[:31])
        second.write_bytes(raw[31:])
        with data.ConcatReader([first, second]) as reader:
            with zipfile.ZipFile(reader) as archive:
                self.assertEqual(archive.read("sample.txt").decode(), "실제 ZIP 읽기")
        self.assertEqual(first.read_bytes(), raw[:31])

    def test_luna_preprocessing_preserves_hu_window_and_rgb_contract(self):
        volume = np.full((8, 16, 16), -1000, dtype=np.int16)
        metadata = {"origin": np.zeros(3), "spacing": np.ones(3), "transform": np.eye(3)}
        image = data.luna_plane(volume, metadata, size_px=112, size_mm=10.)
        self.assertEqual(image.shape, (1, 112, 112))
        self.assertTrue(np.isfinite(image).all())
        np.testing.assert_array_equal(image, np.full((1, 112, 112), -1., dtype=np.float32))

    def test_luna_physical_fov_uses_the_voxel_spacing(self):
        volume = np.broadcast_to(np.arange(64, dtype=np.float32)[None, None, :] * 10 - 1000, (8, 64, 64)).copy()
        metadata = {"origin": np.zeros(3), "spacing": np.array([2., 2., 2.]), "transform": np.eye(3)}
        result = data.luna_plane(volume, metadata, size_px=5, size_mm=10.)
        # 2mm 화소에서 10mm/5 간격은 원본 한 화소씩 이동해야 합니다.
        expected = (np.array([30., 31., 32., 33., 34.]) * 10) / 1400 * 2 - 1
        np.testing.assert_allclose(result[0, 2], expected, atol=1e-6)

    def test_iq_filename_never_becomes_a_patient_identifier(self):
        for name, color in (("Normal cases", 50), ("Bengin cases", 100), ("Malignant cases", 150)):
            directory = self.path / name
            directory.mkdir()
            Image.new("L", (16, 16), color).save(directory / "case (1).png")
        records = data.iq_records(self.path)
        self.assertEqual(len(records), 3)
        self.assertTrue(all(record["patient_id"] is None for record in records))
        self.assertTrue(all(record["group"].startswith("pixels:") for record in records))

    def test_same_pixels_with_conflicting_labels_block_the_cohort(self):
        for name in ("Normal cases", "Bengin cases", "Malignant cases"):
            directory = self.path / name
            directory.mkdir()
            Image.new("L", (16, 16), 100).save(directory / "one.png")
        with self.assertRaises(ValueError):
            data.iq_records(self.path)

    def test_outer_inner_test_patient_groups_are_disjoint_with_class_coverage(self):
        records = [{"id": f"g{label}-{group}-{visit}", "group": f"g{label}-{group}", "label": label} for label in range(3) for group in range(14) for visit in range(2)]
        folds = splits.make_folds(records, n_splits=7, seed=42)
        self.assertEqual(len(folds), 7)
        seen = []
        for fold in folds:
            groups = [{records[index]["group"] for index in fold[key]} for key in ("train", "validation", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            for key in ("train", "validation", "test"):
                self.assertEqual({records[index]["label"] for index in fold[key]}, {0, 1, 2})
            seen.extend(fold["test"])
        self.assertEqual(sorted(seen), list(range(len(records))))

    def test_insufficient_class_groups_do_not_create_fake_seven_folds(self):
        records = [{"id": str(index), "group": str(index), "label": int(index == 0)} for index in range(20)]
        with self.assertRaises(ValueError):
            splits.make_folds(records, n_splits=7, seed=42)

    def test_global_luna_adaptation_cannot_enter_luna_test_folds(self):
        with self.assertRaises(ValueError):
            splits.validate_analysis("luna_binary_cv", "luna-adapted")
        splits.validate_analysis("iq_exploratory", "luna-adapted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
