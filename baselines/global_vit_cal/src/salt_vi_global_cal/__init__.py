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
from .stage_b import (
    StageBHardMiningConfig,
    cross_modal_hard_triplet,
    stage_b_hard_weight,
    weighted_cross_modal_hard_loss,
)

__all__ = [
    "CenterAggregationLoss",
    "ConfidenceTokenDropout",
    "LayerwiseTokenDropoutResult",
    "SALTGlobalVisionTransformer",
    "PoseObservation",
    "PoseSupportConfig",
    "SoftSuperellipseAttentionLoss",
    "StageBHardMiningConfig",
    "TokenDropoutConfig",
    "VisionTransformerConfig",
    "build_pose_posterior",
    "cross_modal_hard_triplet",
    "effective_support",
    "stage_b_hard_weight",
    "weighted_cross_modal_hard_loss",
]
