import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import pipeline as p
import geometry as g
import geometry_stage as stage


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.settings = stage.load_settings(p.HERE/"geometry_config.json")
        self.config = p.load_config(p.HERE/"config.json")

    def sample(self, size=(80, 200)):
        return Image.fromarray(np.random.RandomState(13).randint(0, 256, (size[1], size[0], 3), dtype=np.uint8))

    def test_default_off_native_contract_unchanged(self):
        self.assertFalse(self.settings["enabled"])
        self.assertNotIn("geometry", self.config)
        self.assertEqual(self.config["tile"], 0)
        self.assertFalse(self.config["post_resize"])

    def test_yolo_disabled_matches_legacy_pixels(self):
        path = Path("/home/lab929/ybj/SALT-VI/pasd_plugin/geometry.py")
        if not path.exists():
            self.skipTest("Legacy source is on research server")
        module = types.ModuleType("legacy_geometry_for_test")
        sys.modules[module.__name__] = module
        try:
            exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
            for size in ((80, 200), (200, 80), (256, 512), (1, 7)):
                source = self.sample(size)
                old, _ = module.prepare_control_image(source, module.PersonDetection((0, 0, *size), 0., "full_frame"))
                new, _ = g.fit_with_background(source, self.settings)
                np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
        finally:
            sys.modules.pop(module.__name__, None)

    def test_detection_moves_canvas_without_cropping(self):
        settings = dict(self.settings, foreground_feather_radius=0.)
        source = self.sample()
        detections = [{"bbox_xyxy": [50, 20, 78, 190], "confidence": .95}]
        output, meta = g.fit_with_background(source, settings, detections, "ok")
        self.assertTrue(meta["offset_changed_by_yolo"])
        self.assertEqual(meta["person_guided_offset"], [0, 0])
        self.assertEqual(meta["padding"], [0, 0, 51, 0])
        self.assertIsNone(meta["crop_box"])
        foreground = source.resize(tuple(meta["resized_size"]), g.RESAMPLING.LANCZOS)
        np.testing.assert_array_equal(np.asarray(output.crop(tuple(meta["foreground_box"]))), np.asarray(foreground))

    def test_vertical_person_alignment(self):
        _, meta = g.fit_with_background(self.sample((200, 80)), self.settings,
                                        [{"bbox_xyxy": [30, 0, 170, 25], "confidence": .9}], "ok")
        self.assertEqual(meta["person_guided_offset"][0], 0)
        self.assertNotEqual(meta["person_guided_offset"][1], meta["centered_offset"][1])
        self.assertTrue(all(v >= 0 for v in meta["padding"]))

    def test_same_aspect_cannot_translate_or_change_pixels(self):
        source = self.sample((256, 512))
        result, meta = g.fit_with_background(source, self.settings,
                                            [{"bbox_xyxy": [100, 30, 240, 500], "confidence": .9}], "ok")
        self.assertFalse(meta["offset_changed_by_yolo"])
        self.assertEqual(meta["padding"], [0, 0, 0, 0])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(source))

    def test_no_detection_and_ambiguous_fallback(self):
        _, empty = g.fit_with_background(self.sample(), self.settings, [], "ok")
        self.assertEqual(empty["placement_reason"], "no_reliable_person")
        detections = [{"bbox_xyxy": [5, 10, 35, 190], "confidence": .9},
                      {"bbox_xyxy": [45, 10, 75, 190], "confidence": .9}]
        _, meta = g.fit_with_background(self.sample(), self.settings, detections, "ok")
        self.assertEqual(meta["placement_reason"], "ambiguous_people")
        self.assertFalse(meta["offset_changed_by_yolo"])
        self.assertEqual(len(meta["protected_person_boxes"]), 2)

    def test_invalid_and_tiny_boxes_ignored(self):
        selected, candidates, _ = g.select_person([
            {"bbox_xyxy": [0, 0, 1, 1], "confidence": .99},
            {"bbox_xyxy": [float("nan"), 0, 50, 200], "confidence": .99},
            {"bbox_xyxy": [50, 0, 1, 200], "confidence": .99},
            {"bbox_xyxy": [0, 0, 80, 200], "confidence": .1}], (80, 200), self.settings["detector"])
        self.assertIsNone(selected)
        self.assertEqual(candidates, [])

    def test_protected_body_not_feathered(self):
        source = self.sample()
        result, meta = g.fit_with_background(source, self.settings,
                                            [{"bbox_xyxy": [1, 0, 40, 190], "confidence": .95}], "ok")
        foreground = source.resize(tuple(meta["resized_size"]), g.RESAMPLING.LANCZOS)
        box = meta["protected_person_boxes"][0]
        left, top = meta["person_guided_offset"]
        local_box = [box[0]-left, box[1]-top, box[2]-left, box[3]-top]
        np.testing.assert_array_equal(np.asarray(result.crop(tuple(box))), np.asarray(foreground.crop(tuple(local_box))))

    def test_ir_channels_and_rounding(self):
        for size in ((79, 199), (239, 41), (1, 1)):
            source = self.sample(size).convert("L").convert("RGB")
            output, meta = g.fit_with_background(source, self.settings)
            output = g.ensure_ir_channels(output, "ir")
            pixels = np.asarray(output)
            np.testing.assert_array_equal(pixels[..., 0], pixels[..., 1])
            np.testing.assert_array_equal(pixels[..., 1], pixels[..., 2])
            self.assertEqual(output.size, (256, 512))
            for original, resized in zip(size, meta["resized_size"]):
                self.assertLessEqual(abs(original*meta["scale"]-resized), 1.)

    def test_off_cli_does_not_load_yolo_or_touch_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"must_not_exist"
            result = subprocess.run([sys.executable, str(p.HERE/"pipeline.py"), "postprocess",
                                     "--geometry", "off", "--output-root", str(root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"geometry": "off"', result.stdout)
            self.assertFalse(root.exists())

    def test_detector_disabled_does_not_import_ultralytics(self):
        settings = dict(self.settings["detector"], enabled=False, model_path="missing.pt")
        with mock.patch.dict(sys.modules, {"ultralytics": None}):
            detector = g.CpuPersonDetector(settings)
            self.assertEqual(detector.detect(self.sample()), ([], "disabled"))

    def test_geometry_command_is_cpu_and_run_limit_does_not_hide_new_sr(self):
        args = types.SimpleNamespace(geometry_config=str(p.HERE/"geometry_config.json"),
                                     config=str(p.HERE/"config.json"), source_root=None, output_root=None,
                                     datasets=["regdb"], limit=10, available_only=False, command="run")
        with mock.patch.object(p.subprocess, "run") as run:
            p.launch_geometry(args, self.config)
        command = run.call_args[0][0]
        self.assertEqual(command[0], self.settings["python"])
        self.assertIn(".venvs/qri-v1/bin/python", command[0])
        self.assertIn("--available-only", command)
        self.assertNotIn("--limit", command)
        self.assertEqual(run.call_args[1]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("/extras/CUPTI/lib64", run.call_args[1]["env"]["LD_LIBRARY_PATH"])

    def test_resume_rebuild_and_aliases(self):
        class Detector:
            calls = 0
            def detect(self, image):
                self.calls += 1
                return [{"bbox_xyxy": [50, 20, 78, 390], "confidence": .95}], "ok"
        with tempfile.TemporaryDirectory() as directory:
            native, output = Path(directory)/"native", Path(directory)/"geometry"
            row = {"dataset": "llcm", "source": "nir/1/a.jpg", "output": "nir/1/a.png",
                   "width": 20, "height": 100, "modality": "ir", "aliases": ["test_nir/cam1/1/a.jpg"]}
            source = self.sample((80, 400)).convert("L").convert("RGB")
            native_path = native/"llcm"/row["output"]
            p.save_output(native_path, np.asarray(source), self.config)
            original_bytes = native_path.read_bytes()
            detector = Detector()
            first = stage.process_one(row, native, output, self.config, self.settings, detector)
            self.assertEqual(first["status"], "generated")
            second = stage.process_one(row, native, output, self.config, self.settings, detector)
            self.assertEqual(second["status"], "skipped")
            self.assertEqual(detector.calls, 1)
            image_path, marker = stage.paths(output, row)
            self.assertTrue((output/"llcm/test_nir/cam1/1/a.png").is_symlink())
            image_path.write_bytes(b"corrupt")
            stage.process_one(row, native, output, self.config, self.settings, detector)
            self.assertEqual(detector.calls, 2)
            marker.unlink()
            stage.process_one(row, native, output, self.config, self.settings, detector)
            self.assertEqual(detector.calls, 3)
            self.assertEqual(native_path.read_bytes(), original_bytes)
            new_source = np.full((400, 80, 3), 127, dtype=np.uint8)
            p.save_output(native_path, new_source, self.config)
            stage.process_one(row, native, output, self.config, self.settings, detector)
            self.assertEqual(detector.calls, 4)
            self.assertTrue(stage.completed(output, native_path, row, self.settings))

    def test_contract_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"geometry"
            stage.prepare_contract(root, {"algorithm": "a", "target": [256, 512]})
            stage.prepare_contract(root, {"algorithm": "a", "target": [256, 512]})
            with self.assertRaises(ValueError):
                stage.prepare_contract(root, {"algorithm": "a", "target": [128, 256]})

    def test_run_on_orchestrates_sr_then_cpu_geometry(self):
        argv = [str(p.HERE/"pipeline.py"), "run", "--gpu", "0", "--geometry", "on", "--limit", "1"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(p.subprocess, "run") as run, \
                mock.patch.object(p, "load_model") as model, mock.patch.object(p, "assert_idle_gpu") as idle:
            p.main()
        self.assertEqual(run.call_count, 2)
        sr_command = run.call_args_list[0][0][0]
        geometry_command = run.call_args_list[1][0][0]
        self.assertEqual(sr_command[-2:], ["--geometry", "off"])
        self.assertIn(str(p.HERE/"geometry_stage.py"), geometry_command)
        self.assertEqual(run.call_args_list[1][1]["env"]["CUDA_VISIBLE_DEVICES"], "")
        model.assert_not_called()
        idle.assert_not_called()

    def test_bad_geometry_rejected_before_sr(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory)/"geometry.json"
            settings = dict(self.settings, target_width=0)
            config_path.write_text(json.dumps(settings))
            argv = [str(p.HERE/"pipeline.py"), "run", "--gpu", "0", "--geometry", "on", "--geometry-config", str(config_path)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(p.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    p.main()
            run.assert_not_called()

    def test_native_root_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                stage.run_stage(self.config, self.settings, [], directory, directory)


if __name__ == "__main__":
    unittest.main(verbosity=2)
