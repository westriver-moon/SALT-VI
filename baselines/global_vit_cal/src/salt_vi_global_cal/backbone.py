from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VisionTransformerConfig:
    image_size_hw: tuple[int, int] = (288, 144)
    patch_size: int = 16
    patch_stride: int = 10
    in_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    ellipse_attention_layer: int = 2
    apply_final_norm: bool = False

    @property
    def grid_size(self) -> tuple[int, int]:
        height, width = self.image_size_hw
        grid_h = (height - self.patch_size) // self.patch_stride + 1
        grid_w = (width - self.patch_size) // self.patch_stride + 1
        return grid_h, grid_w

    @property
    def patch_token_count(self) -> int:
        grid_h, grid_w = self.grid_size
        return grid_h * grid_w

    def validate(self) -> "VisionTransformerConfig":
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")
        if not 1 <= self.ellipse_attention_layer <= self.depth:
            raise ValueError("ellipse_attention_layer must be within the transformer")
        return self


def _zero_masked_patch_states(tokens: Tensor, keep_mask: Tensor) -> Tensor:
    if tokens.shape[1] != keep_mask.shape[1]:
        raise ValueError("token and keep-mask lengths differ")
    return tokens * keep_mask[:, :, None].to(tokens.dtype)


class MaskedSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self, tokens: Tensor, keep_mask: Tensor, *, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (q @ k.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~keep_mask[:, None, None, :], -torch.inf)
        attention = logits.softmax(dim=-1)
        output = (
            (self.attn_drop(attention) @ v)
            .transpose(1, 2)
            .reshape(batch, length, dim)
        )
        output = self.proj_drop(self.proj(output))
        cls_patch = attention[:, :, 0, 1:] if need_weights else None
        return output, cls_patch


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = MaskedSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, tokens: Tensor, keep_mask: Tensor, *, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        update, attention = self.attention(
            self.norm1(tokens), keep_mask, need_weights=need_weights
        )
        tokens = tokens + update
        tokens = tokens + self.mlp(self.norm2(tokens))
        return _zero_masked_patch_states(tokens, keep_mask), attention


class SALTGlobalVisionTransformer(nn.Module):
    """Single global ViT branch with fixed-position confidence masks."""

    def __init__(self, config: VisionTransformerConfig | None = None):
        super().__init__()
        self.config = (config or VisionTransformerConfig()).validate()
        cfg = self.config
        self.patch_embed = nn.Conv2d(
            cfg.in_channels,
            cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_stride,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.position = nn.Parameter(
            torch.zeros(1, cfg.patch_token_count + 1, cfg.embed_dim)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.embed_dim, cfg.heads, cfg.mlp_ratio, cfg.dropout
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def prepare_tokens(self, images: Tensor) -> Tensor:
        if tuple(images.shape[-2:]) != self.config.image_size_hw:
            raise ValueError(
                f"expected image size {self.config.image_size_hw}, "
                f"got {tuple(images.shape[-2:])}"
            )
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.class_token.expand(images.shape[0], -1, -1)
        return torch.cat((cls, patches), dim=1) + self.position

    @staticmethod
    def _with_cls_mask(patch_keep_mask: Tensor) -> Tensor:
        cls = torch.ones(
            patch_keep_mask.shape[0],
            1,
            dtype=torch.bool,
            device=patch_keep_mask.device,
        )
        return torch.cat((cls, patch_keep_mask.bool()), dim=1)

    def forward(
        self, images: Tensor, patch_keep_mask: Tensor | None = None
    ) -> dict[str, Tensor | tuple[int, int]]:
        tokens = self.prepare_tokens(images)
        batch = images.shape[0]
        patch_count = self.config.patch_token_count
        if patch_keep_mask is None:
            patch_keep_mask = torch.ones(
                batch, patch_count, dtype=torch.bool, device=images.device
            )
        if patch_keep_mask.shape == (batch, patch_count):
            layer_masks = patch_keep_mask[None].expand(
                self.config.depth, -1, -1
            )
        elif patch_keep_mask.shape == (self.config.depth, batch, patch_count):
            layer_masks = patch_keep_mask
        else:
            raise ValueError("patch_keep_mask must have shape [B,N] or [L,B,N]")

        captured_attention = None
        for block_index, block in enumerate(self.blocks):
            keep_mask = self._with_cls_mask(layer_masks[block_index])
            tokens, attention = block(
                tokens,
                keep_mask,
                need_weights=(
                    block_index + 1 == self.config.ellipse_attention_layer
                ),
            )
            if attention is not None:
                captured_attention = attention
        if captured_attention is None:
            raise RuntimeError("configured attention layer was not reached")
        output = self.norm(tokens) if self.config.apply_final_norm else tokens
        return {
            "tokens": output,
            "features": output[:, 0],
            "ellipse_attention": captured_attention,
            "grid_size": self.config.grid_size,
        }
