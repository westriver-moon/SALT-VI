"""Dataset-scale, text-only Qwen annotation for SYSU-MM01."""

from .config import TextAnnotationConfig, load_text_annotation_config
from .pipeline import TextAnnotationPipeline
from .reasoner import TextAnnotationReasoner, sample_joint_text_worlds

__all__ = [
    "TextAnnotationConfig",
    "TextAnnotationPipeline",
    "TextAnnotationReasoner",
    "load_text_annotation_config",
    "sample_joint_text_worlds",
]
