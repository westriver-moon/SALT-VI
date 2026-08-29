import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

import pipeline as p


class RepeatX4(torch.nn.Module):
    def forward(self, x):
        return x.repeat_interleave(4, -2).repeat_interleave(4, -1)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = p.load_config(p.HERE / "config.json")

    def test_legacy_channel_exact_parity(self):
        old = Path("/home/lab929/ybj/SALT-VI/src/salt_vi/utils/super_resolution/build_sysu_swinir_x2.py")
        if not old.exists():
            self.skipTest("Legacy source is only on the research server")
        tree = ast.parse(old.read_text())
        functions = {"normalize_ir", "prepare_images", "assert_finite_output", "finalize_images"}
        selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions], type_ignores=[])
        scope = {"np": np, "torch": torch}
        exec(compile(selected, str(old), "exec"), scope)
        source = np.random.RandomState(9).randint(0, 256, (2, 17, 23, 3), dtype=np.uint8)
        output = torch.from_numpy(np.random.RandomState(3).uniform(-0.3, 1.3, (2, 3, 68, 92)).astype(np.float32))
        for modality in ("rgb", "ir"):
            self.assertTrue(torch.equal(p.prepare_images(source, modality), scope["prepare_images"](source, modality)))
            np.testing.assert_array_equal(p.finalize_images(output.clone(), modality, (68, 92)),
                                          scope["finalize_images"](output.clone(), modality, (68, 92)))

    def test_native_dimensions_and_no_resampling(self):
        # Odd, tiny and large inputs cover padding, direct inference and tiling.
        for h, w in ((1, 1), (7, 3), (17, 23), (128, 64), (419, 217)):
            pixels = np.random.RandomState(h).randint(0, 256, (1, h, w, 3), dtype=np.uint8)
            expected = pixels[0].repeat(4, 0).repeat(4, 1)
            for tile in (0, 192):
                result = p.infer(RepeatX4(), pixels, "rgb", "cpu", dict(self.config, tile=tile))
                np.testing.assert_array_equal(result, expected)

    def test_default_is_whole_image(self):
        self.assertEqual(self.config["tile"], 0)
        self.assertEqual(p.work_pixels(485, 877, self.config), 488*880)

    def test_ir_channels_equal(self):
        pixels = np.random.RandomState(3).randint(0, 256, (1, 19, 31, 3), dtype=np.uint8)
        result = p.infer(RepeatX4(), pixels, "ir", "cpu", self.config)
        np.testing.assert_array_equal(result, p.normalize_ir(pixels)[0].repeat(4, 0).repeat(4, 1))
        self.assertTrue(np.array_equal(result[..., 0], result[..., 2]))

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf")):
            output = torch.zeros((1, 3, 4, 4))
            output[0, 0, 0, 0] = value
            with self.assertRaises(FloatingPointError):
                p.finalize_images(output, "ir", (4, 4))

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            p.finalize_images(torch.zeros((1, 3, 8, 8)), "rgb", (4, 4))

    def test_resume_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"image.png"
            row = {"width": 4, "height": 5, "modality": "ir"}
            self.assertFalse(p.output_valid(path, row))
            pixels = np.full((20, 16, 3), 124, dtype=np.uint8)
            p.save_output(path, pixels, self.config)
            self.assertTrue(p.output_valid(path, row))
            self.assertFalse(path.with_name("image.png.tmp").exists())
            pixels[0, 0, 1] = 125
            p.save_output(path, pixels, self.config)
            self.assertFalse(p.output_valid(path, row))
            path.write_bytes(b"broken PNG")
            self.assertFalse(p.output_valid(path, row))
            p.save_output(path, np.zeros((3, 3, 3), dtype=np.uint8), self.config)
            self.assertFalse(p.output_valid(path, row))

    def test_alias_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"dataset": "llcm", "output": "vis/0001/a.png", "aliases": ["test_vis/cam1/0001/a.jpg"]}
            path = root/"llcm"/row["output"]
            p.save_output(path, np.zeros((8, 8, 3), dtype=np.uint8), self.config)
            p.ensure_aliases(root, row)
            p.ensure_aliases(root, row)
            alias = root/"llcm/test_vis/cam1/0001/a.png"
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), path.resolve())

    def test_tile_coverage_and_alignment(self):
        for length in range(8, 1601, 8):
            counts = np.zeros(length)
            for start in p.tile_starts(length, 192, 32):
                self.assertEqual(start % 8, 0)
                counts[start:start+192] += 1
            self.assertTrue((counts > 0).all())

    def test_source_change_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.png"
            path.write_bytes(b"original")
            row = {"dataset": "sysu", "source": path.name, "source_bytes": path.stat().st_size,
                   "source_mtime_ns": path.stat().st_mtime_ns}
            config = {"roots": {"sysu": directory}}
            self.assertEqual(p.source_path(row, config), path)
            path.write_bytes(b"modified source")
            with self.assertRaises(RuntimeError):
                p.source_path(row, config)

    def test_inventory_scope_and_dedup(self):
        manifest = Path(self.config["manifest_dir"])/"images.jsonl"
        if not manifest.exists():
            self.skipTest("Run inventory first")
        rows = p.load_rows(self.config)
        keys = [(r["dataset"], r["source"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        regdb = [r for r in rows if r["dataset"] == "regdb"]
        self.assertEqual(len(regdb), 8240)
        self.assertTrue(all(len(r["references"]) == 10 for r in regdb))
        self.assertFalse(any("modify/" in r["source"] or r["source"].endswith(".npy") for r in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
