from __future__ import annotations

import hashlib
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .calibration import (
    CalibrationWeights,
    calibrate_world_weights,
    cosine_identity_energy,
)
from .composite import (
    atomic_mask,
    atomic_png,
    edit_energy,
    lr_cycle_energy,
    masked_composite,
    paste_roi_realization,
    roi_control_image,
    roi_crop_box,
    soft_mask,
    union_mask,
)
from .config import RegionalConfig
from .manifest import canonical_sha256, save_source_record, sha256_file
from .qwen import RegionalReasoner, sample_joint_worlds
from .roi import HumanROIGenerator
from .runtime import (
    IdentityBackend,
    PASDBackend,
    PASDGenerationOptions,
    validate_qri_pasd_generation,
)
from .schema import Region, SourceItem, fallback_world
from .tta import (
    SwinBackend,
    blur_information,
    restore_tta_set,
    robust_category_normalize,
    swin_instability,
)
from .worlds import build_worlds, edited_region_ids


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


class RegionalImaginationPipeline:
    def __init__(
        self,
        config: RegionalConfig,
        *,
        swin: SwinBackend,
        roi: HumanROIGenerator,
        reasoner: RegionalReasoner,
        pasd: PASDBackend,
        identity: IdentityBackend,
        category_stats: dict[str, dict[str, float]] | None = None,
    ):
        self.config = config.validate()
        self.swin = swin
        self.roi = roi
        self.reasoner = reasoner
        self.pasd = pasd
        self.identity = identity
        self.category_stats = category_stats or {}
        self.build_payload = {
            "schema_version": config.schema_version,
            "plugin": config.plugin_id,
            "config": {
                key: value
                for key, value in asdict(config).items()
                if key not in {"dataset_root", "output_root", "assets"}
            },
            "dataset_root": str(config.dataset_root),
            "assets": {
                name: {"path": str(asset.path), "sha256": asset.sha256}
                for name, asset in sorted(config.assets.items())
            },
            "qwen_backend": reasoner.model_id,
            "category_stats_sha256": (
                canonical_sha256(self.category_stats) if self.category_stats else None
            ),
            "selection": "top-3-by-normalized-U_swin-no-hard-gate",
            "failure_policy": "one-SwinIR-world-weight-1",
        }
        self.build_sha256 = canonical_sha256(self.build_payload)

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.config.output_root.resolve()).as_posix()

    def _artifact_dir(self, source: SourceItem) -> Path:
        key = Path(source.source_key)
        return self.config.output_root / "sources" / key.parent / key.stem

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
            median = float(stats.get("median", global_median))
            iqr = float(stats.get("iqr", global_iqr))
            region.u_swin_normalized = robust_category_normalize(
                region.u_swin, median, iqr
            )

    def _save_regions(self, artifact_dir: Path, regions: list[Region]) -> None:
        for region in regions:
            path = artifact_dir / "regions" / f"{region.region_id}.png"
            atomic_mask(
                path,
                Image.fromarray(
                    np.asarray(region.mask, dtype=np.uint8) * 255, mode="L"
                ),
            )
            region.mask_path = self._relative(path)
            region.mask_sha256 = sha256_file(path)

    def _materialize_worlds(
        self,
        source: SourceItem,
        artifact_dir: Path,
        lr: Image.Image,
        reference: Image.Image,
        regions: list[Region],
        worlds,
    ) -> dict | None:
        if (
            self.config.plugin_version == "qri-v2"
            and self.config.pasd.get("realization")
            == "roi-direct-rewrite-then-soft-mask-composite"
        ):
            return self._materialize_localized_worlds(
                source, artifact_dir, lr, reference, regions, worlds
            )
        control = artifact_dir / "swin_reference.png"
        atomic_png(control, reference)
        reference_feature = self.identity.feature(reference, source.modality)
        by_id = {region.region_id: region for region in regions}
        editable = [world for world in worlds if edited_region_ids(world)]
        generation = (
            self.pasd.generate(
                control,
                [world.caption for world in editable],
                [world.seed for world in editable],
                source.modality,
            )
            if editable
            else None
        )
        generated = generation.images if generation is not None else []
        if generation is not None:
            validate_qri_pasd_generation(generation, reference.size)
        if len(generated) != len(editable):
            raise RuntimeError("PASD did not return one image for every editable world")
        generated_by_id = {
            world.world_id: image for world, image in zip(editable, generated)
        }

        for world in worlds:
            edited = edited_region_ids(world)
            masks = [by_id[region_id].mask for region_id in sorted(edited)]
            binary = union_mask(masks, self.config.output_size_hw)
            feathered = soft_mask(
                binary,
                dilation_px=self.config.mask_dilation_px,
                feather_px=self.config.mask_feather_px,
            )
            mask_path = artifact_dir / "world_masks" / f"{world.world_id}.png"
            atomic_mask(mask_path, feathered)
            world.mask_path = self._relative(mask_path)
            world.mask_sha256 = sha256_file(mask_path)

            if edited:
                pasd_image = generated_by_id[world.world_id].convert("RGB")
                pasd_path = artifact_dir / "pasd_full" / f"{world.world_id}.png"
                atomic_png(pasd_path, pasd_image)
                world.pasd_output = self._relative(pasd_path)
                world.pasd_output_sha256 = sha256_file(pasd_path)
                composite = masked_composite(reference, pasd_image, feathered)
                world.e_edit = edit_energy(pasd_image, reference, feathered)
            else:
                composite = reference.copy()
                world.e_edit = 0.0
            output_path = artifact_dir / "views" / f"{world.world_id}.png"
            atomic_png(output_path, composite)
            world.output = self._relative(output_path)
            world.output_sha256 = sha256_file(output_path)
            world.output_bytes = output_path.stat().st_size
            world.e_lr = lr_cycle_energy(composite, lr, source.modality)
            candidate_feature = self.identity.feature(composite, source.modality)
            world.e_id = cosine_identity_energy(reference_feature, candidate_feature)

        calibrate_world_weights(
            worlds,
            CalibrationWeights(
                alpha=self.config.calibration_alpha,
                beta=self.config.calibration_beta,
                gamma=self.config.calibration_gamma,
                delta=self.config.calibration_delta,
            ),
        )
        return generation.geometry if generation is not None else None

    def _materialize_localized_worlds(
        self,
        source: SourceItem,
        artifact_dir: Path,
        lr: Image.Image,
        reference: Image.Image,
        regions: list[Region],
        worlds,
    ) -> dict:
        """Realize every V2 assignment in an enlarged ROI before full-canvas fusion."""
        control = artifact_dir / "swin_reference.png"
        atomic_png(control, reference)
        reference_feature = self.identity.feature(reference, source.modality)
        by_id = {region.region_id: region for region in regions}
        context_scale = float(self.config.pasd["roi_context_scale"])
        pasd_target = (256, 512)
        options = PASDGenerationOptions(
            guidance_scale=float(self.config.pasd["guidance_scale"]),
            conditioning_scale=float(self.config.pasd["conditioning_scale"]),
            added_prompt=str(self.config.pasd["localized_added_prompt"]),
            negative_prompt=str(self.config.pasd["localized_negative_prompt"]),
        )
        geometry_worlds: dict[str, list[dict]] = {}

        for world in worlds:
            edited = edited_region_ids(world)
            masks = [by_id[region_id].mask for region_id in sorted(edited)]
            binary = union_mask(masks, self.config.output_size_hw)
            feathered = soft_mask(
                binary,
                dilation_px=self.config.mask_dilation_px,
                feather_px=self.config.mask_feather_px,
            )
            mask_path = artifact_dir / "world_masks" / f"{world.world_id}.png"
            atomic_mask(mask_path, feathered)
            world.mask_path = self._relative(mask_path)
            world.mask_sha256 = sha256_file(mask_path)

            composite = reference.copy()
            world_geometry = []
            for assignment_index, assignment in enumerate(
                sorted(world.assignments, key=lambda item: item.region_id)
            ):
                if assignment.region_id not in edited:
                    continue
                region = by_id[assignment.region_id]
                crop_box = roi_crop_box(
                    region.bbox_xyxy,
                    reference.size,
                    context_scale=context_scale,
                    target_size=pasd_target,
                )
                roi_control = roi_control_image(reference, crop_box, pasd_target)
                control_path = (
                    artifact_dir
                    / "pasd_roi_controls"
                    / world.world_id
                    / f"{assignment.region_id}.png"
                )
                atomic_png(control_path, roi_control)
                caption = (
                    "same pedestrian and same surveillance frame; edit only the target "
                    f"{assignment.region_id} region ({assignment.category}); clearly realize "
                    f"exactly this plausible hypothesis: {assignment.value}; make its defining "
                    "shape and boundaries visually legible; preserve all surrounding identity, "
                    "pose, clothing and anatomy"
                )
                seed_digest = hashlib.sha256(
                    f"{world.seed}:{assignment.region_id}:{assignment_index}".encode()
                ).hexdigest()
                realization_seed = int(seed_digest[:8], 16)
                generation = self.pasd.generate(
                    control_path,
                    [caption],
                    [realization_seed],
                    source.modality,
                    options=options,
                )
                validate_qri_pasd_generation(generation, pasd_target)
                if len(generation.images) != 1:
                    raise RuntimeError(
                        "localized PASD must return exactly one ROI image"
                    )
                generated_crop = generation.images[0].convert("RGB")
                generated_path = (
                    artifact_dir
                    / "pasd_roi_outputs"
                    / world.world_id
                    / f"{assignment.region_id}.png"
                )
                atomic_png(generated_path, generated_crop)
                region_mask = soft_mask(
                    region.mask,
                    dilation_px=self.config.mask_dilation_px,
                    feather_px=self.config.mask_feather_px,
                )
                composite = paste_roi_realization(
                    composite, generated_crop, crop_box, region_mask
                )
                realization = {
                    "region_id": assignment.region_id,
                    "category": assignment.category,
                    "state": assignment.state,
                    "value": assignment.value,
                    "seed": realization_seed,
                    "caption": caption,
                    "crop_box_xyxy": list(crop_box),
                    "context_scale": context_scale,
                    "control": self._relative(control_path),
                    "control_sha256": sha256_file(control_path),
                    "pasd_output": self._relative(generated_path),
                    "pasd_output_sha256": sha256_file(generated_path),
                    "sampling": options.manifest(),
                    "geometry": generation.geometry,
                }
                world.realizations.append(realization)
                world_geometry.append(
                    {
                        "region_id": assignment.region_id,
                        "crop_box_xyxy": list(crop_box),
                        "pasd": generation.geometry,
                    }
                )

            if edited:
                pasd_path = artifact_dir / "pasd_full" / f"{world.world_id}.png"
                atomic_png(pasd_path, composite)
                world.pasd_output = self._relative(pasd_path)
                world.pasd_output_sha256 = sha256_file(pasd_path)
                world.e_edit = edit_energy(composite, reference, feathered)
            else:
                world.e_edit = 0.0
            output_path = artifact_dir / "views" / f"{world.world_id}.png"
            atomic_png(output_path, composite)
            world.output = self._relative(output_path)
            world.output_sha256 = sha256_file(output_path)
            world.output_bytes = output_path.stat().st_size
            world.e_lr = lr_cycle_energy(composite, lr, source.modality)
            candidate_feature = self.identity.feature(composite, source.modality)
            world.e_id = cosine_identity_energy(reference_feature, candidate_feature)
            geometry_worlds[world.world_id] = world_geometry

        calibrate_world_weights(
            worlds,
            CalibrationWeights(
                alpha=self.config.calibration_alpha,
                beta=self.config.calibration_beta,
                gamma=self.config.calibration_gamma,
                delta=self.config.calibration_delta,
            ),
        )
        return {
            "mode": "roi_direct_rewrite",
            "canvas_size": list(reference.size),
            "pasd_target_size": list(pasd_target),
            "roi_context_scale": context_scale,
            "sampling": options.manifest(),
            "worlds": geometry_worlds,
        }

    def _fallback_record(
        self,
        source: SourceItem,
        artifact_dir: Path,
        reference: Image.Image,
        error: Exception,
    ) -> dict:
        world = fallback_world()
        output_path = artifact_dir / "views" / f"{world.world_id}.png"
        atomic_png(output_path, reference)
        world.output = self._relative(output_path)
        world.output_sha256 = sha256_file(output_path)
        world.output_bytes = output_path.stat().st_size
        return {
            "schema_version": self.config.schema_version,
            "plugin": self.config.plugin_id,
            "build_sha256": self.build_sha256,
            "source_key": source.source_key,
            "image": str(source.image),
            "identity": source.identity,
            "camera": source.camera,
            "modality": source.modality,
            "split": source.split,
            "selected_region_ids": [],
            "regions": [],
            "worlds": [world.manifest()],
            "pasd_geometry": None,
            "fallback": True,
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )[-8000:],
            },
        }

    def process(self, source: SourceItem, *, allow_fallback: bool = True) -> dict:
        artifact_dir = self._artifact_dir(source)
        lr = self._load_lr(source)
        reference = self.swin.restore(lr, source.modality).convert("RGB")
        expected = (self.config.output_size_hw[1], self.config.output_size_hw[0])
        if reference.size != expected:
            raise ValueError(f"SwinIR reference size {reference.size} != {expected}")
        try:
            reference, variants = restore_tta_set(
                self.swin, lr, source.modality, reference=reference
            )
            regions = self.roi.regions(reference, source.modality)
            self._score_regions(reference, variants, regions)
            selected = sorted(
                regions,
                key=lambda region: (
                    -region.u_swin_normalized,
                    -region.u_swin,
                    region.region_id,
                ),
            )[: self.config.selected_region_count]
            if len(selected) != self.config.selected_region_count:
                raise ValueError("ROI stack produced fewer than three regions")
            self._save_regions(artifact_dir, regions)
            proposals = self.reasoner.propose(lr, reference, selected)
            samples = sample_joint_worlds(
                self.reasoner,
                lr,
                reference,
                selected,
                proposals,
                self.config.qwen_sample_count,
                self.config.seed
                + int(hashlib.sha256(source.source_key.encode()).hexdigest()[:8], 16),
                coverage_first=self.config.coverage_sampling,
            )
            worlds = build_worlds(
                self.reasoner,
                lr,
                reference,
                selected,
                proposals,
                samples,
                max_worlds=self.config.max_worlds,
                seed=self.config.seed,
                ensure_editing_coverage=self.config.ensure_editing_world_per_region,
            )
            pasd_geometry = self._materialize_worlds(
                source, artifact_dir, lr, reference, selected, worlds
            )
            record = {
                "schema_version": self.config.schema_version,
                "plugin": self.config.plugin_id,
                "build_sha256": self.build_sha256,
                "source_key": source.source_key,
                "image": str(source.image),
                "identity": source.identity,
                "camera": source.camera,
                "modality": source.modality,
                "split": source.split,
                "selection_rule": "top-3-by-normalized-U_swin-no-hard-gate",
                "selected_region_ids": [region.region_id for region in selected],
                "regions": [region.manifest() for region in regions],
                "worlds": [world.manifest() for world in worlds],
                "pasd_geometry": pasd_geometry,
                "fallback": False,
            }
        except Exception as error:
            if not allow_fallback:
                raise
            record = self._fallback_record(source, artifact_dir, reference, error)
        save_source_record(self.config.output_root, record)
        return record


def category_statistics(records: list[dict]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for record in records:
        if record.get("split", "train") != "train":
            continue
        for region in record.get("regions", []):
            values.setdefault(region["category"], []).append(float(region["u_swin"]))
    stats = {}
    for category, category_values in sorted(values.items()):
        array = np.asarray(category_values, dtype=np.float64)
        q1, q3 = np.percentile(array, [25, 75])
        stats[category] = {
            "count": int(array.size),
            "median": float(np.median(array)),
            "iqr": float(q3 - q1),
        }
    return stats
