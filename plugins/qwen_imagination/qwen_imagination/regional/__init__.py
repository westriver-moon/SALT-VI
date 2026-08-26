"""Qwen Regional Imagination v1.

This package is intentionally parallel to the original atomic semantic sampler.
It owns regional uncertainty, joint semantic worlds, masked PASD realization,
and calibrated mixture weights without changing the original plugin contract.
"""

from .calibration import CalibrationWeights, calibrate_world_weights
from .config import RegionalConfig, load_regional_config
from .pipeline import RegionalImaginationPipeline
from .schema import Region, SourceItem, World

__all__ = [
    "CalibrationWeights",
    "Region",
    "RegionalConfig",
    "RegionalImaginationPipeline",
    "SourceItem",
    "World",
    "calibrate_world_weights",
    "load_regional_config",
]
