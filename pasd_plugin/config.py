"""Configuration for the unified PASD dataset plugin."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


SUPPORTED_DATASETS = ("sysu", "regdb", "llcm")


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class PluginConfig:
    dataset: str
    dataset_root: Path
    captions: dict[str, Path]
    output_root: Path
    pretrained_model_path: Path
    pasd_model_path: Path
    seed: int = 20_260_817
    device: str = "cuda:0"
    mixed_precision: str = "fp16"
    process_size: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 4.0
    conditioning_scale: float = 1.0
    added_prompt: str = (
        "faithful pedestrian restoration, preserve the same identity, body shape, "
        "pose, clothing structure, accessories, natural surveillance texture, sharp details"
    )
    negative_prompt: str = (
        "different person, changed identity, changed pose, changed body proportions, "
        "different clothing, new accessories, altered face, painting, cartoon, artificial "
        "texture, blurry, noise, raster lines, over-smoothed"
    )
    decoder_tiled_size: int = 2048
    encoder_tiled_size: int = 2048
    latent_tiled_size: int = 320
    latent_tiled_overlap: int = 8
    init_latent_with_noise: bool = False
    added_noise_level: int = 0
    offset_noise_scale: float = 0.0
    enable_xformers: bool = True
    target_height: int = 512
    target_width: int = 256
    views_per_source: int = 1
    asset_sha256: dict[str, str] = field(default_factory=dict)
    png_compress_level: int = 4
    person_detector_model: Path | None = None
    person_detector_confidence: float = 0.25
    person_margin: float = 0.05
    background_blur_radius: float = 24.0
    foreground_feather_radius: float = 2.0
    gpu_allowlist: tuple[int, ...] = (1, 2, 3)
    min_free_memory_gib: float = 22.0
    max_gpu_utilization: int = 5
    worker_chunk_size: int = 100
    build_sha256: str = field(default="", init=False)

    @property
    def records_path(self) -> Path:
        return self.output_root / "records.jsonl"

    def output_contract(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_root": str(self.dataset_root),
            "captions": {name: str(path) for name, path in sorted(self.captions.items())},
            "mixed_precision": self.mixed_precision,
            "process_size": self.process_size,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "conditioning_scale": self.conditioning_scale,
            "added_prompt": self.added_prompt,
            "negative_prompt": self.negative_prompt,
            "decoder_tiled_size": self.decoder_tiled_size,
            "encoder_tiled_size": self.encoder_tiled_size,
            "latent_tiled_size": self.latent_tiled_size,
            "latent_tiled_overlap": self.latent_tiled_overlap,
            "init_latent_with_noise": self.init_latent_with_noise,
            "added_noise_level": self.added_noise_level,
            "offset_noise_scale": self.offset_noise_scale,
            "enable_xformers": self.enable_xformers,
            "target_height": self.target_height,
            "target_width": self.target_width,
            "views_per_source": self.views_per_source,
            "seed": self.seed,
            "asset_sha256": dict(sorted(self.asset_sha256.items())),
            "png_compress_level": self.png_compress_level,
            "person_detector_model": str(self.person_detector_model) if self.person_detector_model else None,
            "person_detector_confidence": self.person_detector_confidence,
            "person_margin": self.person_margin,
            "background_blur_radius": self.background_blur_radius,
            "foreground_feather_radius": self.foreground_feather_radius,
        }

    def validate_assets(self) -> None:
        files = {
            "sd_text_encoder": self.pretrained_model_path / "text_encoder/model.safetensors",
            "sd_vae": self.pretrained_model_path / "vae/diffusion_pytorch_model.safetensors",
            "pasd_unet": self.pasd_model_path / "unet/diffusion_pytorch_model.safetensors",
            "pasd_controlnet": self.pasd_model_path / "controlnet/diffusion_pytorch_model.safetensors",
            "person_detector": self.person_detector_model,
        }
        unknown = set(self.asset_sha256).difference(files)
        if unknown:
            raise ValueError(f"unknown asset hashes: {sorted(unknown)}")
        for name, expected in self.asset_sha256.items():
            candidate = files[name]
            if candidate is None or not candidate.is_file():
                raise FileNotFoundError(f"missing configured asset: {name}={candidate}")
            hasher = hashlib.sha256()
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            if digest != expected:
                raise ValueError(f"asset SHA-256 mismatch: {name}={candidate}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PluginConfig":
        path = Path(path).expanduser().resolve()
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        required = ("dataset", "dataset_root", "captions", "output_root", "pretrained_model_path", "pasd_model_path")
        missing = [key for key in required if not values.get(key)]
        if missing:
            raise ValueError(f"missing required plugin settings: {missing}")
        values["dataset"] = str(values["dataset"]).lower()
        if values["dataset"] not in SUPPORTED_DATASETS:
            raise ValueError(f"dataset must be one of {SUPPORTED_DATASETS}")
        base = path.parent
        for key in ("dataset_root", "output_root", "pretrained_model_path", "pasd_model_path"):
            values[key] = _resolve(values[key], base)
        if not values["dataset_root"].is_dir():
            raise FileNotFoundError(values["dataset_root"])
        for key in ("pretrained_model_path", "pasd_model_path"):
            if not values[key].exists():
                raise FileNotFoundError(values[key])
        captions = values["captions"]
        if not isinstance(captions, Mapping) or set(captions) != {"rgb", "ir"}:
            raise ValueError("captions must contain exactly rgb and ir JSON paths")
        values["captions"] = {name: _resolve(value, base) for name, value in captions.items()}
        for caption in values["captions"].values():
            if not caption.is_file():
                raise FileNotFoundError(caption)
        detector = values.get("person_detector_model")
        values["person_detector_model"] = _resolve(detector, base) if detector else None
        if values["person_detector_model"] and not values["person_detector_model"].is_file():
            raise FileNotFoundError(values["person_detector_model"])
        values["gpu_allowlist"] = tuple(int(value) for value in values.get("gpu_allowlist", (1, 2, 3)))
        if not values["gpu_allowlist"] or 0 in values["gpu_allowlist"] or any(value not in (1, 2, 3) for value in values["gpu_allowlist"]):
            raise ValueError("gpu_allowlist must be a non-empty subset of physical GPUs 1, 2, 3")
        if int(values.get("views_per_source", 1)) != 1:
            raise ValueError("the unified PASD protocol requires exactly one view per source")
        values["views_per_source"] = 1
        if (int(values.get("target_width", 256)), int(values.get("target_height", 512))) != (256, 512):
            raise ValueError("the unified PASD protocol requires target_width=256 and target_height=512")
        if not 0 <= int(values.get("png_compress_level", 4)) <= 9:
            raise ValueError("png_compress_level must be in [0, 9]")
        if float(values.get("background_blur_radius", 24.0)) <= 0:
            raise ValueError("background_blur_radius must be positive")
        return cls(**values)
