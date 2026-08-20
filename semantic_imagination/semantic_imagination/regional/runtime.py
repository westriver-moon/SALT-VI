from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .config import RegionalConfig


def _autocast_context(torch, device):
    enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


@dataclass
class PASDGeneration:
    images: list[Image.Image]
    geometry: dict


@dataclass(frozen=True)
class PASDGenerationOptions:
    guidance_scale: float
    conditioning_scale: float
    added_prompt: str
    negative_prompt: str

    def manifest(self) -> dict[str, object]:
        return {
            "guidance_scale": float(self.guidance_scale),
            "conditioning_scale": float(self.conditioning_scale),
            "added_prompt": self.added_prompt,
            "negative_prompt": self.negative_prompt,
        }


class PASDBackend(Protocol):
    def generate(
        self,
        control_path: Path,
        captions: list[str],
        seeds: list[int],
        modality: str,
        options: PASDGenerationOptions | None = None,
    ) -> PASDGeneration: ...


def validate_qri_pasd_generation(
    generation: PASDGeneration,
    expected_size: tuple[int, int],
) -> PASDGeneration:
    expected = [int(expected_size[0]), int(expected_size[1])]
    geometry = generation.geometry
    if geometry.get("mode") != "direct_rewrite":
        raise ValueError("QRI requires PASD direct_rewrite geometry")
    if (
        geometry.get("source_size") != expected
        or geometry.get("target_size") != expected
    ):
        raise ValueError("QRI PASD source and target canvases must match SwinIR")
    if geometry.get("resized_size") != expected or geometry.get("padding") != [
        0,
        0,
        0,
        0,
    ]:
        raise ValueError(
            "QRI PASD direct rewrite cannot resize or pad the control canvas"
        )
    if geometry.get("transform") != {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
    }:
        raise ValueError("QRI PASD direct rewrite requires identity coordinates")
    if geometry.get("background_restoration") is not False:
        raise ValueError("QRI PASD direct rewrite cannot restore a background canvas")
    for image in generation.images:
        if image.size != tuple(expected):
            raise ValueError(
                f"QRI PASD image size {image.size} does not match canonical {tuple(expected)}"
            )
    return generation


class IdentityBackend(Protocol):
    def feature(self, image: Image.Image, modality: str) -> np.ndarray: ...


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_assets(config: RegionalConfig) -> dict[str, dict[str, object]]:
    results = {}
    for name, asset in sorted(config.assets.items()):
        if not asset.path.is_file():
            raise FileNotFoundError(f"missing QRI asset {name}: {asset.path}")
        digest = sha256_file(asset.path)
        if digest != asset.sha256:
            raise ValueError(
                f"QRI asset checksum mismatch for {name}: expected {asset.sha256}, got {digest}"
            )
        results[name] = {
            "path": str(asset.path),
            "sha256": digest,
            "bytes": asset.path.stat().st_size,
        }
    return results


class OfficialSwinIRBackend:
    def __init__(self, swinir_root: str | Path, model_path: str | Path, device: str):
        from salt_vi.utils.super_resolution.build_sysu_swinir_x2 import load_swinir

        self.device = device
        self.model, self.implementation = load_swinir(swinir_root, model_path, device)

    def restore(self, image: Image.Image, modality: str) -> Image.Image:
        from salt_vi.utils.super_resolution.build_sysu_swinir_x2 import infer

        array = np.asarray(image.convert("RGB"), dtype=np.uint8)[None, ...]
        output = infer(self.model, array, modality, self.device)[0]
        return Image.fromarray(output, mode="RGB")


class ExistingPASDBackend:
    def __init__(self, config_path: str | Path, device: str | None = None):
        from pasd_plugin.config import PluginConfig
        from pasd_plugin.runtime import PASDGenerator

        config = PluginConfig.from_yaml(config_path)
        if device is not None:
            config.device = str(device)
        if config.geometry_mode != "direct_rewrite":
            raise ValueError(
                "QRI PASD adapters must use geometry_mode=direct_rewrite; person-fit and "
                "background restoration do not share the SwinIR/ROI coordinate system"
            )
        self.config = config
        self.generator = PASDGenerator(config)

    def generate(
        self,
        control_path: Path,
        captions: list[str],
        seeds: list[int],
        modality: str,
        options: PASDGenerationOptions | None = None,
    ) -> PASDGeneration:
        from pasd_plugin.validation import validate_geometry

        images, geometry = self.generator.generate_views(
            control_path,
            captions,
            seeds,
            modality=modality,
            batch_size=1,
            added_prompt=(options.added_prompt if options is not None else None),
            negative_prompts=(
                [options.negative_prompt] * len(captions)
                if options is not None
                else None
            ),
            guidance_scale=(options.guidance_scale if options is not None else None),
            conditioning_scale=(
                options.conditioning_scale if options is not None else None
            ),
        )
        validate_geometry(geometry, self.config)
        return PASDGeneration(
            images=[image.convert("RGB") for image in images],
            geometry=geometry,
        )


class SALTIdentityBackend:
    def __init__(
        self,
        config_path: str | Path,
        checkpoint: str | Path,
        device: str,
    ):
        import torch

        from salt_vi.engine import build_model
        from salt_vi.entrypoints.train import _load_compatible_state_dict
        from salt_vi.utils.utils import load_train_configs

        self.torch = torch
        self.device = torch.device(device)
        self.config = load_train_configs(str(config_path))
        self.config.pid_num = 395
        self.model = build_model(self.config).to(self.device)
        _load_compatible_state_dict(
            self.model, str(Path(checkpoint).expanduser().resolve()), self.device
        )
        self.model.set_eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(
            1, 3, 1, 1
        )

    def feature(self, image: Image.Image, modality: str) -> np.ndarray:
        torch = self.torch
        image = image.convert("RGB").resize(
            (int(self.config.img_w), int(self.config.img_h)),
            getattr(Image, "Resampling", Image).BICUBIC,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        mode = "IR" if modality.lower() == "ir" else "RGB"
        with torch.no_grad(), _autocast_context(torch, self.device):
            visual = self.model.encode_image_featmap(tensor, modality.lower())
            feature = self.model.classifier(
                self.model.extract_global_feat(visual), mode
            )
            feature = torch.nn.functional.normalize(feature.float(), dim=1)
        return feature[0].detach().cpu().numpy()
