"""QRI-v6 adapter for the canonical semantic-imagination engine."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from semantic_imagination.v6 import (
    QRI_V6,
    RewriteBackend,
    SemanticEncoder,
    V6Pipeline,
    VLMBackend,
    load_v6_config,
)

from ..api import ImaginationRequest, ImaginationResult


class QRIv6Plugin:
    plugin_id = QRI_V6

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = load_v6_config(self.config_path)

    def bind(
        self,
        *,
        vlm: VLMBackend,
        encoder: SemanticEncoder,
        rewriter: RewriteBackend,
    ) -> V6Pipeline:
        return V6Pipeline(
            self.config,
            vlm=vlm,
            encoder=encoder,
            rewriter=rewriter,
        )

    def execute(self, request: ImaginationRequest) -> ImaginationResult:
        if request.config_path.expanduser().resolve() != self.config_path:
            raise ValueError(
                "request config_path does not match the loaded qri-v6 config"
            )
        if request.action != "preflight":
            raise ValueError("qri-v6 execution requires bind(vlm, encoder, rewriter)")
        return ImaginationResult(
            plugin_id=self.plugin_id,
            action=request.action,
            payload={
                "valid": True,
                "plugin_version": self.plugin_id,
                "config": str(self.config_path),
                "algorithm": self.config.algorithm_contract(),
                "runtime_backends": "injected",
            },
        )
