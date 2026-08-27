"""Stable plugin entry point used by SALT orchestration code."""

from dataclasses import dataclass

from .pipeline import build, verify
from .store import PACTArtifactStore


@dataclass(frozen=True)
class PACTPlugin:
    plugin_id: str = "pact-v1"
    api_version: int = 1

    def store(self, person_assets_root, pact_root, mode, dataset):
        return PACTArtifactStore(person_assets_root, pact_root, mode, dataset)

    def build_assets(self, config, datasets, modes):
        return build(config, datasets, modes)

    def verify_assets(self, config, datasets, modes):
        return verify(config, datasets, modes)
