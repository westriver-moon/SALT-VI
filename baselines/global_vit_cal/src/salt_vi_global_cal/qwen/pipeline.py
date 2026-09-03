from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PIL import Image

from ..pose_posterior import PoseObservation
from .hypotheses import AtomSampler, build_regional_hypotheses
from .roi import MaskRefiner, pose_regions, score_region_uncertainty, select_uncertain_regions
from .sampling import sample_joint_worlds
from .schema import Region


@dataclass(frozen=True)
class QwenSamplingConfig:
    selected_region_count: int = 3
    max_selected_region_count: int = 8
    roi_board_size_px: int = 512
    atomic_sample_count: int = 8
    world_sample_count: int = 64
    max_worlds: int = 8
    similarity_threshold: float = 0.85
    max_attempts: int = 4


class QwenRegionalPipeline:
    """Final repeated regional-hypothesis and joint-world Qwen pipeline."""

    def __init__(
        self,
        backend: AtomSampler,
        config: QwenSamplingConfig | None = None,
    ):
        self.backend = backend
        self.config = config or QwenSamplingConfig()

    def run(
        self,
        swin: Image.Image,
        regions: Sequence[Region],
        *,
        modality: str,
        observed: str,
        source_key: str,
        seed: int,
    ) -> dict[str, object]:
        cfg = self.config
        regional = build_regional_hypotheses(
            self.backend,
            swin,
            regions,
            modality=modality,
            observed=observed,
            atomic_sample_count=cfg.atomic_sample_count,
            seed=seed,
            similarity_threshold=cfg.similarity_threshold,
            max_attempts=cfg.max_attempts,
            board_size_px=cfg.roi_board_size_px,
        )
        worlds = sample_joint_worlds(
            regional,
            sample_count=cfg.world_sample_count,
            max_worlds=cfg.max_worlds,
            seed=seed,
        )
        return {
            "version": "empirical-atomic-v1",
            "source_key": source_key,
            "source_scope": "source_image_or_track_anchor",
            "model_id": str(getattr(self.backend, "model_id", type(self.backend).__name__)),
            "weights_source": "repeated_atomic_samples_cluster_frequency",
            "contract": {
                "swinir_role": "qwen_roi_only_not_vit_input",
                "atomic_temperature": 0.75,
                "thinking": False,
                "self_reported_confidence": False,
                "cluster_linkage": "complete",
                "similarity_threshold": cfg.similarity_threshold,
                "regional_seed_namespace_stride": 1000003,
            },
            "regions": [item.to_dict() for item in regional],
            "worlds": worlds,
        }

    def run_from_pose(
        self,
        lr: Image.Image,
        swin: Image.Image,
        observation: PoseObservation,
        *,
        modality: str,
        observed: str,
        source_key: str,
        seed: int,
        parsing=None,
        mask_refiner: MaskRefiner | None = None,
        category_stats: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, object]:
        """Pose -> SCHP/SAM-compatible ROIs -> pre-scan -> Qwen worlds."""

        regions = pose_regions(
            observation,
            image_size_wh=swin.size,
            keypoint_confidence=0.30,
            allow_geometric_eye_fallback=False,
            image=swin,
            parsing=parsing,
            mask_refiner=mask_refiner,
        )
        scored = score_region_uncertainty(
            lr, swin, regions, category_stats=category_stats
        )
        selected = select_uncertain_regions(
            scored,
            selected_count=self.config.selected_region_count,
            max_selected_count=self.config.max_selected_region_count,
        )
        result = self.run(
            swin,
            selected,
            modality=modality,
            observed=observed,
            source_key=source_key,
            seed=seed,
        )
        result["roi_selection"] = [
            {
                "region_id": region.region_id,
                "category": region.category,
                "bbox_xyxy": list(region.bbox_xyxy),
                "uncertainty": region.uncertainty,
                "blur_uncertainty": region.blur_uncertainty,
            }
            for region in selected
        ]
        return result
