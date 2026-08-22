from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers", reason="PASD inference dependencies are optional")

import pasd_plugin.runtime as runtime_module
from pasd_plugin.config import PluginConfig
from pasd_plugin.runtime import PASDGenerator


class FakePipeline:
    def __init__(self):
        self.prompts = None
        self.options = None

    def __call__(self, args, prompts, working, **options):
        self.prompts = list(prompts)
        self.options = dict(options)
        return SimpleNamespace(images=[working.copy() for _ in prompts])


def test_direct_rewrite_accepts_qri_prompt_and_scale_overrides(monkeypatch, tmp_path):
    source = tmp_path / "control.png"
    Image.new("RGB", (256, 512), "gray").save(source)
    config = PluginConfig(
        dataset="sysu",
        dataset_root=tmp_path,
        captions={"rgb": tmp_path / "rgb.json", "ir": tmp_path / "ir.json"},
        output_root=tmp_path / "output",
        pretrained_model_path=tmp_path / "sd",
        pasd_model_path=tmp_path / "pasd",
        geometry_mode="direct_rewrite",
    )
    pipeline = FakePipeline()
    generator = PASDGenerator.__new__(PASDGenerator)
    generator.config = config
    generator.device = torch.device("cpu")
    generator.detector = None
    generator.pipeline = pipeline
    monkeypatch.setattr(runtime_module, "wavelet_color_fix", lambda image, _: image)

    images, geometry = generator.generate_views(
        source,
        ["person"],
        [7],
        modality="rgb",
        added_prompt="localized detail",
        negative_prompts=["changed identity"],
        guidance_scale=9.0,
        conditioning_scale=0.5,
    )

    assert [image.size for image in images] == [(256, 512)]
    assert geometry["mode"] == "direct_rewrite"
    assert pipeline.prompts == ["person, localized detail"]
    assert pipeline.options["negative_prompt"] == ["changed identity"]
    assert pipeline.options["guidance_scale"] == 9.0
    assert pipeline.options["conditioning_scale"] == 0.5
