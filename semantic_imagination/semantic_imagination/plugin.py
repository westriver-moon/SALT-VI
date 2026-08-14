"""Backward-compatible public facade for Semantic Imagination v2.

New code may import the specialized modules directly. Existing SALT/PASD callers can
keep importing these names from ``semantic_imagination.plugin``.
"""

from .clustering import cluster_hypothesis_samples
from .pasd import to_pasd_record
from .sampling import ImaginationBackend, build_hypothesis_manifest

__all__ = [
    "ImaginationBackend",
    "build_hypothesis_manifest",
    "cluster_hypothesis_samples",
    "to_pasd_record",
]
