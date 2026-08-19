from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .config import RegionalConfig


class PASDBackend(Protocol):
    def generate(
        self,
        control_path: Path,
        captions: list[str],
        seeds: list[int],
        modality: str,
    ) -> list[Image.Image]: ...


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
        self.generator = PASDGenerator(config)

    def generate(
        self,
        control_path: Path,
        captions: list[str],
        seeds: list[int],
        modality: str,
    ) -> list[Image.Image]:
        images, _ = self.generator.generate_views(
            control_path,
            captions,
            seeds,
            modality=modality,
            batch_size=1,
        )
        return [image.convert("RGB") for image in images]


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
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

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
        with torch.no_grad():
            visual = self.model.encode_image_featmap(tensor, modality.lower())
            feature = self.model.classifier(
                self.model.extract_global_feat(visual), mode
            )
            feature = torch.nn.functional.normalize(feature.float(), dim=1)
        return feature[0].detach().cpu().numpy()
