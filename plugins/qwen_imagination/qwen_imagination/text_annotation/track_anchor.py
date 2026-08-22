from __future__ import annotations

import copy
import json
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageFilter

from ..regional.roi import HumanROIGenerator
from ..regional.schema import Region, SourceItem
from ..regional.sysu_sources import load_train_source_records
from ..regional.tta import blur_information, robust_category_normalize
from .config import TextAnnotationConfig
from .manifest import atomic_json
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


class PrecomputedSwinIRStore:
    """Read the canonical SYSU SwinIR build without loading the SR network."""

    def __init__(self, dataset_root: str | Path, derived_root: str | Path):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.derived_root = Path(derived_root).expanduser().resolve()
        self._indices = {
            modality: {
                record.source_key: int(record.index)
                for record in load_train_source_records(self.dataset_root, modality)
            }
            for modality in ("rgb", "ir")
        }
        self._arrays: dict[str, np.ndarray] = {}

    def _array(self, modality: str) -> np.ndarray:
        if modality not in self._arrays:
            path = self.derived_root / f"train_{modality}_swinir_x2_img.npy"
            self._arrays[modality] = np.load(path, mmap_mode="r")
        return self._arrays[modality]

    def image(self, source: SourceItem) -> Image.Image:
        modality = source.modality.lower()
        if source.split == "train":
            try:
                index = self._indices[modality][source.source_key]
            except KeyError as error:
                raise KeyError(
                    f"precomputed SwinIR index omits {source.source_key}"
                ) from error
            array = np.array(self._array(modality)[index], dtype=np.uint8, copy=True)
            image = Image.fromarray(array, mode="RGB")
        else:
            path = self.derived_root / "eval" / source.source_key
            with Image.open(path) as stored:
                image = stored.convert("RGB")
        if image.size != (256, 512):
            raise ValueError(
                f"precomputed SwinIR size {image.size} != (256, 512) for "
                f"{source.source_key}"
            )
        return image


@dataclass
class PreparedSource:
    source: SourceItem
    lr: Image.Image
    reference: Image.Image
    regions: list[Region]
    preparation_elapsed_seconds: float
    global_quality: float


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _high_pass(image: Image.Image) -> np.ndarray:
    gray = _gray(image)
    blurred = (
        np.asarray(
            image.convert("L").filter(ImageFilter.GaussianBlur(1.0)),
            dtype=np.float32,
        )
        / 255.0
    )
    return gray - blurred


def _source_seed(config: TextAnnotationConfig, source_key: str) -> int:
    return (
        int(config.seed) + zlib.crc32(source_key.encode("utf-8"))
    ) % (2**31 - 1)


class TrackAnchorTextAnnotationPipeline:
    """Per-image ROI pre-scan with one Qwen 3.8 call per camera/identity track."""

    def __init__(
        self,
        config: TextAnnotationConfig,
        *,
        roi: HumanROIGenerator,
        reasoner: AnnotationReasoner,
        store: PrecomputedSwinIRStore,
        category_stats: dict[str, dict[str, float]] | None = None,
    ):
        self.config = config.validate()
        if self.config.strategy != "track_anchor":
            raise ValueError("track anchor pipeline requires strategy=track_anchor")
        self.roi = roi
        self.reasoner = reasoner
        self.store = store
        self.category_stats = category_stats or {}

    def _load_lr(self, source: SourceItem) -> Image.Image:
        with Image.open(source.image) as image:
            image = image.convert("RGB")
        target_h, target_w = self.config.source_size_hw
        image = image.resize((target_w, target_h), BICUBIC)
        if source.modality.lower() == "ir":
            image = image.convert("L").convert("RGB")
        return image

    def _score_regions(
        self, lr: Image.Image, reference: Image.Image, regions: list[Region]
    ) -> None:
        bicubic = lr.resize(reference.size, BICUBIC)
        residual = np.abs(_high_pass(reference) - _high_pass(bicubic))
        raw_residual = []
        raw_blur = []
        for region in regions:
            selected = np.asarray(region.mask, dtype=bool)
            values = residual[selected]
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            region.u_swin = center + 1.4826 * mad
            region.u_blur = blur_information(reference, selected)
            raw_residual.append(region.u_swin)
            raw_blur.append(region.u_blur)
        residual_median = float(np.median(raw_residual))
        residual_iqr = float(np.subtract(*np.percentile(raw_residual, [75, 25])))
        blur_median = float(np.median(raw_blur))
        blur_iqr = float(np.subtract(*np.percentile(raw_blur, [75, 25])))
        for region in regions:
            stats = self.category_stats.get(region.category, {})
            residual_score = robust_category_normalize(
                region.u_swin,
                float(stats.get("fast_residual_median", residual_median)),
                float(stats.get("fast_residual_iqr", residual_iqr)),
            )
            blur_score = robust_category_normalize(
                region.u_blur,
                float(stats.get("blur_median", blur_median)),
                float(stats.get("blur_iqr", blur_iqr)),
            )
            region.u_swin_normalized = 0.65 * residual_score + 0.35 * blur_score

    def _prepare(self, source: SourceItem) -> PreparedSource:
        started = time.perf_counter()
        lr = self._load_lr(source)
        reference = self.store.image(source)
        regions = self.roi.regions(reference, source.modality)
        self._score_regions(lr, reference, regions)
        global_quality = float(
            np.mean(np.abs(_high_pass(reference))) + 0.25 * np.std(_gray(reference))
        )
        return PreparedSource(
            source=source,
            lr=lr,
            reference=reference,
            regions=regions,
            preparation_elapsed_seconds=time.perf_counter() - started,
            global_quality=global_quality,
        )

    def _prescan_path(self, source_key: str) -> Path:
        source = Path(source_key)
        return (
            self.config.output_root
            / "prescan"
            / source.parent
            / f"{source.stem}.json"
        )

    def _prescan_signature(self) -> dict[str, Any]:
        return {
            "version": "track-prescan-v1",
            "source_size_hw": list(self.config.source_size_hw),
            "output_size_hw": list(self.config.output_size_hw),
            "precomputed_swinir_root": str(self.config.precomputed_swinir_root),
            "roi_assets": {
                name: str(self.config.assets[name])
                for name in ("yolo_pose", "schp_lip", "sam_vit_b")
            },
        }

    def _prescan_summary(self, item: PreparedSource) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "prescan_signature": self._prescan_signature(),
            "source_key": item.source.source_key,
            "global_quality": item.global_quality,
            "preparation_elapsed_seconds": item.preparation_elapsed_seconds,
            "roi_candidates": [
                {
                    **region.manifest(),
                    "mask_area_fraction": float(
                        np.asarray(region.mask, dtype=bool).mean()
                    ),
                }
                for region in item.regions
            ],
        }

    def _load_prescan(self, source: SourceItem) -> dict[str, Any] | None:
        path = self._prescan_path(source.source_key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("source_key") != source.source_key
            or payload.get("prescan_signature") != self._prescan_signature()
            or not payload.get("roi_candidates")
        ):
            return None
        return payload

    def _select_region_ids(self, summaries: list[dict[str, Any]]) -> list[str]:
        shared_ids = {
            region["region_id"] for region in summaries[0]["roi_candidates"]
        }
        for item in summaries[1:]:
            shared_ids.intersection_update(
                region["region_id"] for region in item["roi_candidates"]
            )
        if len(shared_ids) < self.config.selected_region_count:
            raise ValueError(
                f"track has only {len(shared_ids)} ROI ids shared by every frame"
            )
        scores: dict[str, list[float]] = defaultdict(list)
        raw_scores: dict[str, list[float]] = defaultdict(list)
        for item in summaries:
            for region in item["roi_candidates"]:
                region_id = region["region_id"]
                if region_id in shared_ids:
                    scores[region_id].append(float(region["u_swin_normalized"]))
                    raw_scores[region_id].append(float(region["u_swin"]))
        ranked = sorted(
            shared_ids,
            key=lambda region_id: (
                -float(np.median(scores[region_id])),
                -float(np.median(raw_scores[region_id])),
                region_id,
            ),
        )
        return ranked[: self.config.selected_region_count]

    @staticmethod
    def _select_anchor(
        summaries: list[dict[str, Any]], selected_ids: list[str]
    ) -> dict[str, Any]:
        wanted = set(selected_ids)

        def score(item: dict[str, Any]) -> tuple[float, str]:
            regions = {
                region["region_id"]: region for region in item["roi_candidates"]
            }
            roi_clarity = sum(1.0 - float(regions[name]["u_blur"]) for name in wanted)
            return (
                float(item["global_quality"]) + 0.1 * roi_clarity,
                str(item["source_key"]),
            )

        return max(summaries, key=score)

    def process_track(self, sources: list[SourceItem]) -> list[dict[str, Any]]:
        if not sources:
            return []
        track_keys = {(source.split, source.camera, source.identity) for source in sources}
        if len(track_keys) != 1:
            raise ValueError(f"process_track received mixed tracks: {sorted(track_keys)}")
        summaries = []
        prepared_by_key = {}
        for source in sources:
            summary = self._load_prescan(source)
            if summary is None:
                prepared = self._prepare(source)
                prepared_by_key[source.source_key] = prepared
                summary = self._prescan_summary(prepared)
                atomic_json(self._prescan_path(source.source_key), summary)
            summaries.append(summary)
        selected_ids = self._select_region_ids(summaries)
        anchor_summary = self._select_anchor(summaries, selected_ids)
        anchor_source = next(
            source
            for source in sources
            if source.source_key == anchor_summary["source_key"]
        )
        anchor = prepared_by_key.get(anchor_source.source_key)
        if anchor is None:
            anchor = self._prepare(anchor_source)
        anchor_by_id = {region.region_id: region for region in anchor.regions}
        selected_anchor_regions = [anchor_by_id[name] for name in selected_ids]
        anchor_seed = _source_seed(self.config, anchor.source.source_key)
        annotation, anchor_telemetry = self.reasoner.annotate(
            anchor.lr,
            anchor.reference,
            selected_anchor_regions,
            modality=anchor.source.modality,
            seed=anchor_seed,
        )
        for regional in annotation["regions"]:
            regional["normalized_entropy"] = normalized_entropy(
                regional["hypotheses"]
            )
        track_key = "/".join(track_keys.pop())
        records = []
        summary_by_key = {item["source_key"]: item for item in summaries}
        for source in sources:
            item = summary_by_key[source.source_key]
            source_seed = _source_seed(self.config, source.source_key)
            source_annotation = copy.deepcopy(annotation)
            regions_by_id = {
                region["region_id"]: region for region in item["roi_candidates"]
            }
            missing = [name for name in selected_ids if name not in regions_by_id]
            if missing:
                raise ValueError(
                    f"source {source.source_key} omits track-selected ROIs {missing}"
                )
            selected_set = set(selected_ids)
            direct_vlm = source.source_key == anchor.source.source_key
            records.append(
                {
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
                    "annotation_provenance": {
                        "strategy": "track_anchor",
                        "track_key": track_key,
                        "anchor_source_key": anchor.source.source_key,
                        "direct_vlm": direct_vlm,
                        "semantic_scope": "camera_identity_track",
                        "roi_geometry_scope": "source_image",
                    },
                    "selection_rule": (
                        "top-3-by-track-median-fast-swin-residual-plus-blur"
                    ),
                    "selected_region_ids": list(selected_ids),
                    "roi_candidates": [
                        {
                            **region,
                            "selected": region["region_id"] in selected_set,
                        }
                        for region in item["roi_candidates"]
                    ],
                    "annotation": source_annotation,
                    "sampled_text_worlds": sample_joint_text_worlds(
                        source_annotation["regions"],
                        sample_count=int(self.config.world_sample_count),
                        max_worlds=int(self.config.max_worlds),
                        seed=source_seed,
                    ),
                    "telemetry": {
                        "preparation_elapsed_seconds": (
                            item["preparation_elapsed_seconds"]
                        ),
                        "direct_vlm": direct_vlm,
                        "anchor_vlm": anchor_telemetry,
                    },
                }
            )
        return records
