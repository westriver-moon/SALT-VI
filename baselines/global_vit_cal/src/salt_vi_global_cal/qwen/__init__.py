"""Backend-independent Qwen ROI annotation and empirical world sampling."""

from .hypotheses import (
    Atom,
    RegionHypotheses,
    build_regional_hypotheses,
    parse_atom,
)
from .pipeline import QwenRegionalPipeline
from .roi import pose_regions, score_region_uncertainty, select_uncertain_regions
from .sampling import sample_joint_worlds
from .schema import Region
from .zoom import expanded_bbox, swin_roi_board

__all__ = [
    "Atom",
    "QwenRegionalPipeline",
    "Region",
    "RegionHypotheses",
    "build_regional_hypotheses",
    "expanded_bbox",
    "parse_atom",
    "pose_regions",
    "sample_joint_worlds",
    "score_region_uncertainty",
    "select_uncertain_regions",
    "swin_roi_board",
]
