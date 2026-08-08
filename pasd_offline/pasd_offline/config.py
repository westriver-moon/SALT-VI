from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class GenerationConfig:
    pretrained_model_path: Path
    pasd_model_path: Path
    output_root: Path
    device: str = "cuda:0"
    mixed_precision: str = "fp16"
    process_size: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 9.0
    conditioning_scale: float = 1.0
    added_prompt: str = "clean, high-resolution, detailed, sharp"
    negative_prompt: str = "blurry, noise, unclear, lowres, over-smoothed"
    seed: int = 42
    decoder_tiled_size: int = 224
    encoder_tiled_size: int = 1024
    latent_tiled_size: int = 320
    latent_tiled_overlap: int = 8
    init_latent_with_noise: bool = False
    added_noise_level: int = 900
    offset_noise_scale: float = 0.0
    enable_xformers: bool = True
    target_height: int = 512
    target_width: int = 256
    png_compress_level: int = 4
    person_detector_model: Path | None = None
    person_detector_confidence: float = 0.25
    person_margin: float = 0.05
    gpu_allowlist: tuple[int, ...] = (1, 2, 3)
    min_free_memory_gib: float = 22.0
    max_gpu_utilization: int = 5
    worker_chunk_size: int = 100

    def output_contract(self) -> dict:
        return {
            "pretrained_model_path": str(self.pretrained_model_path),
            "pasd_model_path": str(self.pasd_model_path),
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
            "png_compress_level": self.png_compress_level,
            "person_detector_model": (
                str(self.person_detector_model) if self.person_detector_model else None
            ),
            "person_detector_confidence": self.person_detector_confidence,
            "person_margin": self.person_margin,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GenerationConfig":
        path = Path(path).resolve()
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("pretrained_model_path", "pasd_model_path", "person_detector_model"):
            if values.get(key) in (None, ""):
                values[key] = None
                continue
            value = Path(values[key]).expanduser()
            values[key] = value if value.is_absolute() else (path.parent / value).resolve()
        values["output_root"] = Path(values["output_root"]).expanduser().resolve()
        if "gpu_allowlist" in values:
            values["gpu_allowlist"] = tuple(int(value) for value in values["gpu_allowlist"])
        target_height = int(values.get("target_height", 512))
        target_width = int(values.get("target_width", 256))
        if target_height <= 0 or target_width <= 0:
            raise ValueError("target_height and target_width must be positive")
        if target_height % 8 or target_width % 8:
            raise ValueError("PASD target dimensions must be divisible by 8")
        if not 0 <= int(values.get("png_compress_level", 4)) <= 9:
            raise ValueError("png_compress_level must be in [0, 9]")
        allowlist = tuple(values.get("gpu_allowlist", (1, 2, 3)))
        if not allowlist or 0 in allowlist or any(value not in (1, 2, 3) for value in allowlist):
            raise ValueError("gpu_allowlist must be a non-empty subset of physical GPUs 1, 2, 3")
        return cls(**values)
