from __future__ import annotations

import time
import zlib
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ..regional.roi import HumanROIGenerator
from ..regional.schema import Region, SourceItem
from ..regional.tta import (
    SwinBackend,
    blur_information,
    restore_tta_set,
    robust_category_normalize,
    swin_instability,
)
from .config import TextAnnotationConfig
from .reasoner import normalized_entropy, sample_joint_text_worlds


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


class AnnotationReasoner(Protocol):
    model_id: str

    def annotate(
        self,
        lr: Image.Image,
        swin: Image.Image,
        regions: list[Region],
        *,
        modality: str,
        seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


class ReferenceStore(Protocol):
    def image(self, source: SourceItem) -> Image.Image: ...


class TextAnnotationPipeline:
    def __init__(
        self,
        config: TextAnnotationConfig,
        *,
        swin: SwinBackend,
        roi: HumanROIGenerator,
        reasoner: AnnotationReasoner,
        category_stats: dict[str, dict[str, float]] | None = None,
        reference_store: ReferenceStore | None = None,
    ):
        self.config = config.validate()
        self.swin = swin
        self.roi = roi
        self.reasoner = reasoner
        self.category_stats = category_stats or {}
        self.reference_store = reference_store

    def _load_lr(self, source: SourceItem) -> Image.Image:
        with Image.open(source.image) as image:
            image = image.convert("RGB")
        target_h, target_w = self.config.source_size_hw
        image = image.resize((target_w, target_h), BICUBIC)
        if source.modality.lower() == "ir":
            image = image.convert("L").convert("RGB")
        return image

    def _score_regions(
        self,
        reference: Image.Image,
        variants: list[Image.Image],
        regions: list[Region],
    ) -> None:
        raw = []
        for region in regions:
            region.u_swin = swin_instability(reference, variants, region.mask)
            region.u_blur = blur_information(reference, region.mask)
            raw.append(region.u_swin)
        global_median = float(np.median(raw))
        q1, q3 = np.percentile(raw, [25, 75])
        global_iqr = float(q3 - q1)
        for region in regions:
            stats = self.category_stats.get(region.category, {})
            region.u_swin_normalized = robust_category_normalize(
                region.u_swin,
                float(stats.get("median", global_median)),
                float(stats.get("iqr", global_iqr)),
            )

    def _score_blur_only(
        self, reference: Image.Image, regions: list[Region]
    ) -> None:
        raw = []
        for region in regions:
            region.u_swin = 0.0
            region.u_blur = blur_information(reference, region.mask)
            raw.append(region.u_blur)
        median = float(np.median(raw))
        q1, q3 = np.percentile(raw, [25, 75])
        iqr = float(q3 - q1)
        for region in regions:
            region.u_swin_normalized = robust_category_normalize(
                region.u_blur, median, iqr
            )

    def _select_regions(self, regions: list[Region]) -> tuple[list[Region], str]:
        ranked = sorted(
            regions,
            key=lambda region: (
                -region.u_swin_normalized,
                -region.u_blur,
                region.region_id,
            ),
        )
        eyes = next((region for region in regions if region.region_id == "eyes"), None)
        qualified = [
            region
            for region in ranked
            if float(region.u_swin_normalized)
            >= float(self.config.roi_selection_threshold)
        ]
        selected = [eyes] if eyes is not None else []
        selected.extend(region for region in qualified if region is not eyes)
        for region in ranked:
            if len(selected) >= int(self.config.selected_region_count):
                break
            if region not in selected:
                selected.append(region)
        selected = selected[: int(self.config.max_selected_region_count)]
        score_name = (
            "normalized-U_swin"
            if self.config.exact_selection_mode == "full_tta"
            else "normalized-regional-blur"
        )
        return selected, (
            f"eyes-guard-plus-{score_name}-threshold-"
            f"{float(self.config.roi_selection_threshold):.3f}-"
            f"min-{int(self.config.selected_region_count)}-"
            f"max-{int(self.config.max_selected_region_count)}"
        )

    def process(self, source: SourceItem) -> dict[str, Any]:
        process_started = time.perf_counter()
        lr = self._load_lr(source)
        lr_ready = time.perf_counter()
        reference = (
            self.reference_store.image(source)
            if self.reference_store is not None
            else self.swin.restore(lr, source.modality).convert("RGB")
        )
        reference_ready = time.perf_counter()
        expected = (self.config.output_size_hw[1], self.config.output_size_hw[0])
        if reference.size != expected:
            raise ValueError(f"SwinIR reference size {reference.size} != {expected}")
        if self.config.exact_selection_mode == "full_tta":
            reference, variants = restore_tta_set(
                self.swin, lr, source.modality, reference=reference
            )
        else:
            variants = []
        tta_ready = time.perf_counter()
        regions = self.roi.regions(reference, source.modality)
        if self.config.exact_selection_mode == "full_tta":
            self._score_regions(reference, variants, regions)
        else:
            self._score_blur_only(reference, regions)
        roi_ready = time.perf_counter()
        selected, selection_rule = self._select_regions(regions)
        if not (
            int(self.config.selected_region_count)
            <= len(selected)
            <= int(self.config.max_selected_region_count)
        ):
            raise ValueError(
                f"ROI stack produced {len(selected)} selected regions, expected "
                f"between {self.config.selected_region_count} and "
                f"{self.config.max_selected_region_count}"
            )
        source_seed = (
            int(self.config.seed) + zlib.crc32(source.source_key.encode("utf-8"))
        ) % (2**31 - 1)
        annotation, telemetry = self.reasoner.annotate(
            lr,
            reference,
            selected,
            modality=source.modality,
            seed=source_seed,
        )
        qwen_ready = time.perf_counter()
        sampled = None
        if self.config.probability_mode == "vlm_reported":
            sampled = sample_joint_text_worlds(
                annotation["regions"],
                sample_count=int(self.config.world_sample_count),
                max_worlds=int(self.config.max_worlds),
                seed=source_seed,
            )
            for regional in annotation["regions"]:
                regional["normalized_entropy"] = normalized_entropy(
                    regional["hypotheses"]
                )
        finished = time.perf_counter()
        telemetry["pipeline"] = {
            "lr_load_seconds": lr_ready - process_started,
            "reference_load_or_restore_seconds": reference_ready - lr_ready,
            "tta_restore_seconds": tta_ready - reference_ready,
            "roi_and_score_seconds": roi_ready - tta_ready,
            "qwen_seconds": qwen_ready - roi_ready,
            "postprocess_seconds": finished - qwen_ready,
            "total_seconds": finished - process_started,
            "reference_source": "precomputed" if self.reference_store else "live_swinir",
        }
        selected_ids = {region.region_id for region in selected}
        record = {
            "schema_version": int(self.config.schema_version),
            "annotation_version": self.config.annotation_version,
            "run_signature": self.config.run_signature(),
            "source_key": source.source_key,
            "image": str(source.image),
            "identity": source.identity,
            "camera": source.camera,
            "modality": source.modality,
            "split": source.split,
            "status": "complete",
            "selection_rule": selection_rule,
            "selected_region_ids": [region.region_id for region in selected],
            "annotation_provenance": {
                "anchor_source_key": source.source_key,
                "direct_vlm": True,
                "semantic_scope": "source_image",
                "roi_geometry_scope": "source_image",
                "vlm_visual_input": "swinir_only",
            },
            "roi_candidates": [
                {
                    **region.manifest(),
                    "selected": region.region_id in selected_ids,
                    "mask_area_fraction": float(np.asarray(region.mask).mean()),
                }
                for region in regions
            ],
            "annotation": annotation,
            "probability_design": {
                "mode": self.config.probability_mode,
                "specification": self.config.probability_spec,
                "vlm_self_reported_probability": (
                    self.config.probability_mode == "vlm_reported"
                ),
                "status": (
                    "available"
                    if self.config.probability_mode == "vlm_reported"
                    else "deferred"
                ),
            },
            "telemetry": telemetry,
        }
        if sampled is not None:
            record["sampled_text_worlds"] = sampled
        return record
