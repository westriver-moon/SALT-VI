from copy import deepcopy

import torch
import torch.nn as nn


class QuadruplePatchInput(nn.Module):
    """Four independent patch embeddings feeding one shared transformer trunk."""

    def __init__(self, patch_embed, branch_order):
        super().__init__()
        if not hasattr(patch_embed, "patch_size") or not hasattr(patch_embed, "stride_size"):
            raise TypeError("quadruple_patch requires a single PatchEmbedOverlap source")
        if isinstance(getattr(patch_embed, "proj", None), nn.ModuleList):
            raise ValueError("quadruple_patch cannot be combined with fused multi-branch patch embedding")
        self.branch_order = tuple(branch_order)
        if len(self.branch_order) != 4:
            raise ValueError("quadruple_patch requires exactly four branches")
        self.patch_embeds = nn.ModuleList(deepcopy(patch_embed) for _ in self.branch_order)

    @property
    def input_dtype(self):
        return self.patch_embeds[0].proj.weight.dtype

    def _grid_size(self, images):
        height, width = images.shape[-2:]
        patch_height, patch_width = self.patch_embeds[0].patch_size
        stride_height, stride_width = self.patch_embeds[0].stride_size
        grid_height = (height - patch_height) // stride_height + 1
        grid_width = (width - patch_width) // stride_width + 1
        if grid_height < 1 or grid_width < 1:
            raise ValueError(
                f"Input {height}x{width} is smaller than patch size "
                f"{patch_height}x{patch_width}"
            )
        return grid_height, grid_width

    def forward(self, views):
        if views.ndim != 5:
            raise ValueError(
                "quadruple_patch expects [B,4,3,H,W], "
                f"got {tuple(views.shape)}"
            )
        if views.shape[1] != 4 or views.shape[2] != 3:
            raise ValueError(
                "quadruple_patch expects exactly four three-channel views, "
                f"got {tuple(views.shape)}"
            )
        grid_size = self._grid_size(views[:, 0])
        branch_tokens = [
            patch_embed(views[:, branch_index])
            for branch_index, patch_embed in enumerate(self.patch_embeds)
        ]
        token_shapes = {tuple(tokens.shape[1:]) for tokens in branch_tokens}
        if len(token_shapes) != 1:
            raise RuntimeError(f"Quadruple patch branches disagree on token shape: {token_shapes}")
        # Branch-major layout makes labels.repeat(4) and reshape(4, B, ...) exact.
        return torch.cat(branch_tokens, dim=0), grid_size

    def forward_modality(self, images, modality):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "quadruple modality input expects [B,3,H,W], "
                f"got {tuple(images.shape)}"
            )
        normalized = str(modality).strip().lower()
        if normalized in {"rgb", "visible"}:
            branch_ids = (0, 1)
        elif normalized in {"ir", "infrared", "thermal"}:
            branch_ids = (2, 3)
        else:
            raise ValueError(f"Unsupported quadruple modality {modality!r}")
        grid_size = self._grid_size(images)
        tokens = [self.patch_embeds[branch_id](images) for branch_id in branch_ids]
        return torch.cat(tokens, dim=0), grid_size, branch_ids
