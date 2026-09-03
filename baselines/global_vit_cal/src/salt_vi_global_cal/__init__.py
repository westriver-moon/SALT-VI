"""SALT-VI global CAL baseline."""

from .losses import CenterAggregationLoss, SoftSuperellipseAttentionLoss
from .backbone import SALTGlobalVisionTransformer, VisionTransformerConfig
from .pose_posterior import PoseObservation, PoseSupportConfig, build_pose_posterior
from .token_dropout import (
    ConfidenceTokenDropout,
    LayerwiseTokenDropoutResult,
    TokenDropoutConfig,
    effective_support,
)

__all__ = [
    "CenterAggregationLoss",
    "ConfidenceTokenDropout",
    "LayerwiseTokenDropoutResult",
    "SALTGlobalVisionTransformer",
    "PoseObservation",
    "PoseSupportConfig",
    "SoftSuperellipseAttentionLoss",
    "TokenDropoutConfig",
    "VisionTransformerConfig",
    "build_pose_posterior",
    "effective_support",
]
