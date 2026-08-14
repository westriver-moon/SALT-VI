#!/usr/bin/env python3
"""Isolated InternVL2.5-8B smoke test for SALT Semantic Imagination."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from semantic_imagination import (
    CATEGORY_STATES,
    DEFAULT_SAMPLING_STRATA,
    build_hypothesis_manifest,
    to_pasd_record,
    validate_atomic_response,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAMPLING_STRATA = DEFAULT_SAMPLING_STRATA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--embed-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--max-image-tiles", type=int, default=2)
    parser.add_argument("images", nargs="+", type=Path)
    return parser.parse_args()


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def closest_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff or (
            diff == best_diff
            and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_diff = diff
            best = ratio
    return best


def dynamic_preprocess(
    image: Image.Image, image_size: int = 448, max_num: int = 2
) -> list[Image.Image]:
    width, height = image.size
    ratios = sorted(
        {
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if 1 <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    ratio = closest_ratio(width / height, ratios, width, height, image_size)
    target_width = image_size * ratio[0]
    target_height = image_size * ratio[1]
    resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    tiles = []
    for index in range(ratio[0] * ratio[1]):
        left = (index % ratio[0]) * image_size
        top = (index // ratio[0]) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    return tiles


class InternVL25Backend:
    model_id = "OpenGVLab/InternVL2_5-8B"

    def __init__(
        self,
        model_path: Path,
        embed_model_path: Path,
        perturb_dir: Path,
        max_image_tiles: int,
    ) -> None:
        self.model_path = model_path.resolve()
        self.embed_model_path = embed_model_path.resolve()
        self.perturb_dir = perturb_dir
        self.perturb_dir.mkdir(parents=True, exist_ok=True)
        self.max_image_tiles = max_image_tiles
        self.image_transform = build_transform()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False, local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().cuda()
        self.embed_tokenizer = None
        self.embed_model = None

    def _pixels(self, image: Image.Image) -> torch.Tensor:
        tiles = dynamic_preprocess(
            image.convert("RGB"), max_num=self.max_image_tiles
        )
        return torch.stack([self.image_transform(tile) for tile in tiles]).to(
            device="cuda", dtype=torch.bfloat16
        )

    def _chat(
        self, image: Image.Image, question: str, seed: int, do_sample: bool
    ) -> str:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        pixels = self._pixels(image)
        config = {
            "max_new_tokens": 96,
            "do_sample": do_sample,
            "temperature": 0.75 if do_sample else 1.0,
            "top_p": 0.9 if do_sample else 1.0,
        }
        try:
            with torch.inference_mode():
                response = self.model.chat(
                    self.tokenizer,
                    pixels,
                    "<image>\n" + question,
                    config,
                )
            return str(response).strip()
        finally:
            del pixels

    def observe(self, image: Path) -> str:
        prompt = (
            "Describe this pedestrian in one concise English sentence. State only "
            "identity-relevant facts that are clearly visible: apparent presentation, "
            "upper/lower clothing type and color, footwear, carried objects, and visible "
            "accessories. Do not infer obscured details and do not describe the background."
        )
        return self._chat(Image.open(image).convert("RGB"), prompt, 0, False)

    def perturb(self, image: Path, seed: int) -> Image.Image:
        rng = random.Random(seed)
        value = Image.open(image).convert("RGB")
        width, height = value.size
        scale = rng.uniform(0.60, 0.86)
        small = value.resize(
            (max(8, round(width * scale)), max(16, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
        value = small.resize((width, height), Image.Resampling.BICUBIC)
        value = value.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.65)))
        value = ImageEnhance.Brightness(value).enhance(rng.uniform(0.94, 1.06))
        value = ImageEnhance.Contrast(value).enhance(rng.uniform(0.94, 1.06))
        value.save(self.perturb_dir / f"perturb-{seed}.jpg", quality=88)
        return value

    def imagine(
        self, image: Image.Image, observed: str, instruction: str, seed: int
    ) -> str:
        target_match = re.search(r"Target category:\s*([a-z0-9_]+)", instruction, re.I)
        target_category = target_match.group(1).casefold() if target_match else None
        if target_category not in CATEGORY_STATES:
            raise ValueError(f"missing or invalid target category: {target_category!r}")
        allowed_states = ", ".join(sorted(CATEGORY_STATES[target_category]))
        base_prompt = f"""Visible facts already established:
{observed}

{instruction}
Use exactly the Target category stated above. Valid categories are:
eyewear, wrist_accessory, headwear, body_marking,
clothing_detail, carried_object, pocket_item, footwear_detail, other.
For Target category {target_category}, choose exactly one canonical state from:
{allowed_states}
Return exactly one line in this format:
ATOM | <category> | <canonical state> | <one short value> | <body location or none>
The value and location must not contain 'and', 'or', commas, semicolons, or a second detail.
For a positive state, the value must explicitly name evidence for that state, for example
'red heart graphic', 'thin frame', 'black footwear strap', or 'small shoulder bag'.
Do not repeat a value already stated in Visible facts. Use value 'absent' only when the
selected category is not visible. Use value 'no_additional_detail' when the category is
already visible but no compatible extra detail can be inferred; never use 'unknown'. Never
contradict the visible facts, invent an identity, or describe the background."""
        return self._chat(image, base_prompt, seed, True)

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self.embed_tokenizer is None:
            self.embed_tokenizer = AutoTokenizer.from_pretrained(
                self.embed_model_path, local_files_only=True
            )
            self.embed_model = AutoModel.from_pretrained(
                self.embed_model_path, local_files_only=True
            ).eval().cpu()
        embedding_texts = []
        for text in texts:
            fields = dict(
                part.split("=", 1) for part in text.split("; ") if "=" in part
            )
            value = fields.get("value", text).replace("_", " ")
            location = fields.get("location", "none").replace("_", " ")
            embedding_texts.append(f"{value} at {location}")
        batch = self.embed_tokenizer(
            embedding_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = self.embed_model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().float().tolist()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.model, args.embed_model, *args.images):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    backend = InternVL25Backend(
        args.model,
        args.embed_model,
        output_dir / "perturbations",
        args.max_image_tiles,
    )
    records = []
    manifests = []
    summary_items = []
    instruction = (
        "Infer one atomic detail that the low-resolution image leaves genuinely ambiguous. "
        "Keep it compatible with the clearly observed pedestrian."
    )

    def compose_atomic(observed: str, hypothesis: str) -> str:
        fields = dict(
            part.split("=", 1) for part in hypothesis.split("; ") if "=" in part
        )
        value = fields.get("value", "uncertain detail")
        location = fields.get("location", "none")
        category = fields.get("category", "detail").replace("_", " ")
        if value == "no_additional_detail":
            possible = f"The pedestrian may have no additional {category} detail"
        elif value == "absent":
            possible = f"The pedestrian may have no {category}"
        else:
            possible = f"The pedestrian may have {value}"
        if location not in {"none", "absent"}:
            possible += f" at {location}"
        return f"{observed} {possible}."
    for index, image in enumerate(args.images):
        image = image.resolve()
        item_dir = output_dir / f"sample-{index:02d}"
        item_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, item_dir / "source.jpg")
        manifest = build_hypothesis_manifest(
            image=image,
            source_key=str(image),
            backend=backend,
            instruction=instruction,
            sample_count=args.sample_count,
            seed=args.seed + index,
            similarity_threshold=args.similarity_threshold,
            compose=compose_atomic,
            contract={
                "model_path": str(args.model.resolve()),
                "embed_model": str(args.embed_model.resolve()),
                "perturbation": "downsample-upsample+gaussian-blur+brightness+contrast-v1",
                "max_image_tiles": args.max_image_tiles,
                "hypothesis_schema": "atomic-category-state-value-location-v2",
                "canonical_state_taxonomy": "closed-category-specific-v1",
                "atomic_validation": "schema+state-value+category+contradiction+absence+abstention+not-observed-v6",
                "atomic_max_attempts": 4,
                "stratification": "round-robin-eight-identity-attribute-categories-v1",
                "embedding_projection": "atomic-value-location-v1",
            },
            cluster_linkage="complete",
            sampling_strata=SAMPLING_STRATA,
            validator=validate_atomic_response,
            max_attempts=4,
            validation_failure_policy="exclude",
        )
        record = to_pasd_record(
            manifest,
            output_dir=item_dir / "pasd_views",
            pasd_seed=args.seed,
            modality="rgb",
        )
        (item_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (item_dir / "pasd_record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        records.append(record)
        manifests.append(manifest)
        summary_items.append(
            {
                "source": str(image),
                "observed": manifest["observed"],
                "sample_count": len(manifest["samples"]),
                "cluster_count": len(manifest["hypotheses"]),
                "weights": [item["weight"] for item in manifest["hypotheses"]],
                "weight_intervals_95": [
                    item["weight_interval_95"] for item in manifest["hypotheses"]
                ],
                "conditional_weights": [
                    item["conditional_weight"] for item in manifest["hypotheses"]
                ],
                "categories": [
                    item["category"] for item in manifest["hypotheses"]
                ],
                "representatives": [
                    item["representative"] for item in manifest["hypotheses"]
                ],
                "semantic_weight_mass": sum(
                    item["weight"] for item in manifest["hypotheses"]
                ),
                "sampling_diagnostics": manifest["sampling_diagnostics"],
            }
        )

    elapsed = time.time() - started
    peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
    summary = {
        "schema_version": 2,
        "model": backend.model_id,
        "model_path": str(args.model.resolve()),
        "embed_model": str(args.embed_model.resolve()),
        "images": len(args.images),
        "samples_per_image": args.sample_count,
        "similarity_threshold": args.similarity_threshold,
        "cluster_linkage": "complete",
        "sampling_strata": list(SAMPLING_STRATA),
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_gib": peak_gib,
        "atomic_validation_attempts": sum(
            len(sample.get("attempts", []))
            for manifest in manifests
            for sample in manifest["samples"]
        ),
        "atomic_validation_retries": sum(
            max(0, len(sample.get("attempts", [])) - 1)
            for manifest in manifests
            for sample in manifest["samples"]
        ),
        "atomic_validation_failures": sum(
            manifest["sampling_diagnostics"]["validation_failed"]
            for manifest in manifests
        ),
        "atomic_active_abstentions": sum(
            manifest["sampling_diagnostics"]["active_abstentions"]
            for manifest in manifests
        ),
        "atomic_valid_rate": sum(
            manifest["sampling_diagnostics"]["valid"] for manifest in manifests
        )
        / max(1, args.sample_count * len(args.images)),
        "all_weights_valid": all(
            item["weights"]
            and all(weight > 0 and math.isfinite(weight) for weight in item["weights"])
            and 0 < sum(item["weights"]) <= 1.0 + 1e-9
            for item in summary_items
        ),
        "items": summary_items,
    }
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
