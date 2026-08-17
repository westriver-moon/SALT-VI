from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from .config import PluginConfig
from .geometry import PersonDetector, prepare_control_image, restore_blurred_background


MODULE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = MODULE_ROOT / "vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from pasd.models.pasd.controlnet import ControlNetModel  # noqa: E402
from pasd.models.pasd.unet_2d_condition import UNet2DConditionModel  # noqa: E402
from pasd.myutils.wavelet_color_fix import wavelet_color_fix  # noqa: E402
from pasd.pipelines.pipeline_pasd import StableDiffusionControlNetPipeline  # noqa: E402


class PASDGenerator:
    def __init__(self, config: PluginConfig):
        self.config = config
        config.validate_assets()
        self.device = torch.device(config.device)
        self.dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[config.mixed_precision]
        if config.num_inference_steps != 20:
            raise ValueError("the unified PASD protocol requires exactly 20 inference steps")
        self.detector = PersonDetector(
            config.person_detector_model, config.person_detector_confidence
        )
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> StableDiffusionControlNetPipeline:
        base = str(self.config.pretrained_model_path)
        model = str(self.config.pasd_model_path)
        scheduler = UniPCMultistepScheduler.from_pretrained(base, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
        feature_extractor = CLIPImageProcessor.from_pretrained(f"{base}/feature_extractor")
        unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")
        controlnet = ControlNetModel.from_pretrained(model, subfolder="controlnet")

        for component in (text_encoder, vae, unet, controlnet):
            component.requires_grad_(False)
            component.to(self.device, dtype=self.dtype)

        if self.config.enable_xformers:
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()

        pipeline = StableDiffusionControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline._init_tiled_vae(
            encoder_tile_size=self.config.encoder_tiled_size,
            decoder_tile_size=self.config.decoder_tiled_size,
        )
        return pipeline

    def _working_image(self, image: Image.Image) -> Image.Image:
        width = image.width
        height = image.height
        scale = max(1.0, self.config.process_size / min(width, height))
        width = int(width * scale) // 8 * 8
        height = int(height * scale) // 8 * 8
        return image.resize((width, height), Image.Resampling.BILINEAR)

    def prepare(self, image_path: str | Path) -> tuple[Image.Image, Image.Image, dict]:
        with Image.open(image_path) as image:
            source = image.convert("RGB")
        detection = self.detector.detect(source)
        control, geometry = prepare_control_image(
            source,
            detection,
            target_size=(self.config.target_width, self.config.target_height),
            margin=self.config.person_margin,
            background_blur_radius=self.config.background_blur_radius,
            foreground_feather_radius=self.config.foreground_feather_radius,
        )
        return self._working_image(control), control, geometry

    def generate_views(
        self,
        image_path: str | Path,
        captions: list[str],
        seeds: list[int],
        modality: str,
        batch_size: int = 1,
    ) -> tuple[list[Image.Image], dict]:
        if len(captions) != len(seeds) or not captions:
            raise ValueError("captions and seeds must be non-empty lists of equal length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        working, source_background, geometry = self.prepare(image_path)
        args = SimpleNamespace(
            init_latent_with_noise=self.config.init_latent_with_noise,
            offset_noise_scale=self.config.offset_noise_scale,
            num_inference_steps=self.config.num_inference_steps,
            added_noise_level=self.config.added_noise_level,
            latent_tiled_size=self.config.latent_tiled_size,
            latent_tiled_overlap=self.config.latent_tiled_overlap,
        )
        results: list[Image.Image] = []
        for start in range(0, len(captions), batch_size):
            caption_batch = captions[start : start + batch_size]
            seed_batch = seeds[start : start + batch_size]
            prompts = [
                ", ".join(
                    part
                    for part in (caption.strip(), self.config.added_prompt.strip())
                    if part
                )
                for caption in caption_batch
            ]
            generators = [
                torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seed_batch
            ]
            generated = self.pipeline(
                args,
                prompts,
                working,
                num_inference_steps=self.config.num_inference_steps,
                generator=generators,
                guidance_scale=self.config.guidance_scale,
                negative_prompt=[self.config.negative_prompt] * len(prompts),
                conditioning_scale=self.config.conditioning_scale,
            ).images
            for image in generated:
                image = wavelet_color_fix(image, working).resize(
                    (self.config.target_width, self.config.target_height),
                    Image.Resampling.LANCZOS,
                )
                image = restore_blurred_background(image, geometry, source_background)
                if modality.lower() == "ir":
                    image = image.convert("L").convert("RGB")
                results.append(image)
        return results, geometry

    def generate(
        self,
        image_path: str | Path,
        caption: str,
        seed: int,
        modality: str,
    ) -> Image.Image:
        images, _ = self.generate_views(
            image_path, [caption], [seed], modality=modality, batch_size=1
        )
        return images[0]

    def benchmark_batches(
        self,
        image_path: str | Path,
        caption: str,
        seed: int,
        candidates: tuple[int, ...] = (5, 2, 1),
        memory_limit_gib: float = 22.0,
    ) -> dict:
        attempts = []
        for batch_size in candidates:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            try:
                self.generate_views(
                    image_path,
                    [caption] * batch_size,
                    [seed + index for index in range(batch_size)],
                    modality="rgb",
                    batch_size=batch_size,
                )
                peak = torch.cuda.max_memory_allocated(self.device) / 2**30
                elapsed = time.perf_counter() - started
                accepted = peak <= memory_limit_gib
                attempts.append(
                    {
                        "batch_size": batch_size,
                        "status": "ok" if accepted else "over_memory_limit",
                        "peak_memory_gib": peak,
                        "elapsed_seconds": elapsed,
                    }
                )
                if accepted:
                    return {"selected_batch_size": batch_size, "attempts": attempts}
            except torch.cuda.OutOfMemoryError as error:
                attempts.append(
                    {"batch_size": batch_size, "status": "oom", "error": str(error)}
                )
                torch.cuda.empty_cache()
        raise RuntimeError(f"no PASD batch size satisfied the GPU contract: {attempts}")
