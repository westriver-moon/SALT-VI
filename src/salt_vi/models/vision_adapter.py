from __future__ import annotations

import os

import torch
import torch.nn as nn

from .vision_transformer import ViT, resize_pos_embed, to_2tuple
from .visual_inputs import build_visual_input_plugin, normalize_visual_input_backend


def _unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _normalize_checkpoint_key(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module.") :]
    return key


def _is_skipped_key(key: str) -> bool:
    return key.startswith("head.") or key.startswith("head_dist.") or key == "dist_token" or "dist" in key


def _is_core_backbone_key(key: str) -> bool:
    return key.startswith(("patch_embed", "blocks", "norm", "cls_token", "pos_embed"))


class PMTViTVisual(nn.Module):
    def __init__(
        self,
        input_resolution=(288, 144),
        patch_size=(16, 16),
        stride_size=(12, 12),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.03,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        output_dim=2048,
        pretrained_path=None,
        patch_embed_config=None,
        gradient_checkpointing=False,
        gradient_checkpoint_blocks=None,
        gradient_checkpoint_segments=None,
        attention_backend="manual",
        visual_input_backend="single",
        quadruple_branch_order=None,
        quadruple_template_trainable=False,
    ):
        super().__init__()
        self.input_resolution = to_2tuple(input_resolution)
        self.output_dim = output_dim
        self.vit = ViT(
            img_size=self.input_resolution,
            patch_size=patch_size,
            stride_size=stride_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_embed_config=patch_embed_config,
            attention_backend=attention_backend,
        )
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if gradient_checkpoint_blocks is None:
            self.gradient_checkpoint_blocks = (
                len(self.vit.blocks) if self.gradient_checkpointing else 0
            )
        else:
            self.gradient_checkpoint_blocks = int(gradient_checkpoint_blocks)
            if not 0 <= self.gradient_checkpoint_blocks <= len(self.vit.blocks):
                raise ValueError(
                    "gradient_checkpoint_blocks must be within [0, {}], got {}".format(
                        len(self.vit.blocks), self.gradient_checkpoint_blocks
                    )
                )
            if self.gradient_checkpoint_blocks and not self.gradient_checkpointing:
                raise ValueError(
                    "gradient_checkpoint_blocks requires gradient_checkpointing=true"
                )
        if gradient_checkpoint_segments is None:
            self.gradient_checkpoint_segments = self.gradient_checkpoint_blocks
        else:
            self.gradient_checkpoint_segments = int(gradient_checkpoint_segments)
            if self.gradient_checkpoint_segments < 1:
                raise ValueError("gradient_checkpoint_segments must be positive")
        if embed_dim == output_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(embed_dim, output_dim, bias=False)
            nn.init.normal_(self.projection.weight, std=embed_dim**-0.5)

        if pretrained_path:
            self.load_pretrained(pretrained_path)
        self.visual_input_backend = normalize_visual_input_backend(visual_input_backend)
        self.input_plugin = build_visual_input_plugin(
            self.visual_input_backend,
            self.vit.patch_embed,
            branch_order=quadruple_branch_order,
        )
        self.quadruple_template_trainable = bool(quadruple_template_trainable)
        if self.input_plugin is not None:
            self.vit.patch_embed.requires_grad_(self.quadruple_template_trainable)
            print(
                "Initialized four independent patch embeddings from the loaded PMT "
                "patch_embed; the original patch_embed is retained as a "
                + ("trainable warmup template" if self.quadruple_template_trainable else "frozen template")
            )

        self.ellipse_attention_layer = 0
        self.ellipse_attention_radius_x = 0.58
        self.ellipse_attention_radius_y = 0.55
        self.ellipse_attention_temperature = 0.12

    @property
    def input_dtype(self):
        if self.input_plugin is not None:
            return self.input_plugin.input_dtype
        proj = self.vit.patch_embed.proj
        if isinstance(proj, nn.ModuleList):
            return proj[0].weight.dtype
        return proj.weight.dtype

    def configure_ellipse_attention(
        self,
        *,
        layer=0,
        radius_x=0.58,
        radius_y=0.55,
        temperature=0.12,
    ):
        """Configure a 1-based shallow block for a broad soft ellipse prior."""
        layer = int(layer)
        radius_x = float(radius_x)
        radius_y = float(radius_y)
        temperature = float(temperature)
        if not 0 <= layer <= len(self.vit.blocks):
            raise ValueError(
                f"ellipse_attention_layer must be within [0, {len(self.vit.blocks)}]"
            )
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError("ellipse attention radii must be positive")
        if temperature <= 0:
            raise ValueError("ellipse_attention_temperature must be positive")
        self.ellipse_attention_layer = layer
        self.ellipse_attention_radius_x = radius_x
        self.ellipse_attention_radius_y = radius_y
        self.ellipse_attention_temperature = temperature

    def _ellipse_mask(self, grid_size, *, device):
        grid_height, grid_width = (int(value) for value in grid_size)
        y = (torch.arange(grid_height, device=device, dtype=torch.float32) + 0.5)
        x = (torch.arange(grid_width, device=device, dtype=torch.float32) + 0.5)
        y = y / grid_height
        x = x / grid_width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        distance = (
            ((xx - 0.5) / self.ellipse_attention_radius_x).square()
            + ((yy - 0.5) / self.ellipse_attention_radius_y).square()
        )
        return torch.sigmoid(
            (1.0 - distance) / self.ellipse_attention_temperature
        ).flatten()

    def _run_tokens(self, tokens, grid_size):
        checkpoint_blocks = (
            self.gradient_checkpoint_blocks if self.training else 0
        )
        layer = self.ellipse_attention_layer if self.training else 0
        if layer <= 0:
            return (
                self.run_blocks(
                    tokens,
                    0,
                    len(self.vit.blocks),
                    checkpoint_blocks=checkpoint_blocks,
                    checkpoint_segments=self.gradient_checkpoint_segments,
                ),
                None,
            )

        block_index = layer - 1
        tokens = self.run_blocks(
            tokens,
            0,
            block_index,
            checkpoint_blocks=checkpoint_blocks,
            checkpoint_segments=self.gradient_checkpoint_segments,
        )
        attention = self.vit.cls_patch_attention(tokens, block_index)
        if attention.shape[-1] != int(grid_size[0]) * int(grid_size[1]):
            raise RuntimeError(
                "Ellipse attention token count does not match the PMT patch grid"
            )
        tokens = self.run_blocks(
            tokens,
            block_index,
            len(self.vit.blocks),
            checkpoint_blocks=checkpoint_blocks,
            checkpoint_segments=self.gradient_checkpoint_segments,
        )
        auxiliary = {
            "ellipse_attention": attention,
            "ellipse_mask": self._ellipse_mask(grid_size, device=attention.device),
            "ellipse_layer": layer,
        }
        return tokens, auxiliary

    @staticmethod
    def _attach_auxiliary(packaged, auxiliary):
        if auxiliary is not None:
            packaged.update(auxiliary)
        return packaged

    def forward(self, x, mode=None):
        if self.input_plugin is not None:
            if mode == "shared_template":
                return self.forward_template(x)
            if x.ndim == 5:
                if mode is not None:
                    raise ValueError("quadruple training input does not accept a modality selector")
                return self.forward_quadruple(x)
            if mode is not None:
                return self.forward_modality(x, mode)
        del mode
        tokens, grid_size = self.prepare_tokens(x)
        tokens, auxiliary = self._run_tokens(tokens, grid_size)
        return self._attach_auxiliary(
            self.finalize_and_package(tokens), auxiliary
        )

    def forward_template(self, images):
        """Use the original shared patch embedding during phased PMT warmup."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"shared template input expects [B,3,H,W], got {tuple(images.shape)}"
            )
        tokens, grid_size = self.prepare_tokens(images)
        tokens, auxiliary = self._run_tokens(tokens, grid_size)
        return self._attach_auxiliary(
            self.finalize_and_package(tokens), auxiliary
        )

    @torch.no_grad()
    def sync_input_plugin_from_template(self):
        """Initialize all four branches from the Stage-A warmup patch embedding."""
        if self.input_plugin is None:
            raise RuntimeError("patch synchronization requires quadruple_patch")
        template_state = self.vit.patch_embed.state_dict()
        for patch_embed in self.input_plugin.patch_embeds:
            patch_embed.load_state_dict(template_state, strict=True)

    def _run_embedded_patches(self, patch_tokens, grid_size):
        tokens, _grid_size = self.vit.prepare_embedded_tokens(patch_tokens, grid_size)
        return self._run_tokens(tokens, grid_size)

    @staticmethod
    def _reshape_branch_major(tensor, branch_count, batch_size):
        return tensor.reshape(branch_count, batch_size, *tensor.shape[1:]).permute(
            1, 0, *range(2, tensor.ndim + 1)
        ).contiguous()

    def forward_quadruple(self, views):
        if self.input_plugin is None:
            raise RuntimeError("forward_quadruple requires visual_input_backend='quadruple_patch'")
        batch_size = views.shape[0]
        patch_tokens, grid_size = self.input_plugin(views)
        tokens, auxiliary = self._run_embedded_patches(patch_tokens, grid_size)
        packaged = self._attach_auxiliary(
            self.finalize_and_package(tokens), auxiliary)
        packaged.update(
            branch_tokens=self._reshape_branch_major(packaged["tokens"], 4, batch_size),
            branch_features=self._reshape_branch_major(packaged["features"], 4, batch_size),
            branch_raw_tokens=self._reshape_branch_major(packaged["raw_tokens"], 4, batch_size),
            branch_raw_features=self._reshape_branch_major(packaged["raw_features"], 4, batch_size),
            branch_order=self.input_plugin.branch_order,
        )
        return packaged

    def forward_modality(self, images, modality):
        if self.input_plugin is None:
            raise RuntimeError("forward_modality requires visual_input_backend='quadruple_patch'")
        batch_size = images.shape[0]
        patch_tokens, grid_size, branch_ids = self.input_plugin.forward_modality(images, modality)
        tokens, auxiliary = self._run_embedded_patches(patch_tokens, grid_size)
        packaged = self._attach_auxiliary(
            self.finalize_and_package(tokens), auxiliary)
        branch_tokens = self._reshape_branch_major(packaged["tokens"], 2, batch_size)
        branch_raw_tokens = self._reshape_branch_major(packaged["raw_tokens"], 2, batch_size)
        averaged_tokens = branch_tokens.mean(dim=1)
        averaged_raw_tokens = branch_raw_tokens.mean(dim=1)
        return {
            "tokens": averaged_tokens,
            "features": averaged_tokens[:, 0],
            "raw_tokens": averaged_raw_tokens,
            "raw_features": averaged_raw_tokens[:, 0],
            "branch_tokens": branch_tokens,
            "branch_features": branch_tokens[:, :, 0],
            "branch_raw_tokens": branch_raw_tokens,
            "branch_raw_features": branch_raw_tokens[:, :, 0],
            "branch_ids": branch_ids,
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        migrated = []
        if self.input_plugin is not None:
            for suffix in ("weight", "bias"):
                source_key = f"{prefix}vit.patch_embed.proj.{suffix}"
                source = state_dict.get(source_key)
                if source is None:
                    continue
                for branch_index in range(4):
                    target_key = (
                        f"{prefix}input_plugin.patch_embeds.{branch_index}.proj.{suffix}"
                    )
                    if target_key not in state_dict:
                        state_dict[target_key] = source.clone()
                        migrated.append(target_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if migrated:
            print(
                f"Migrated single patch_embed checkpoint into four independent "
                f"branches ({len(migrated)} tensors)"
            )

    def prepare_tokens(self, x):
        return self.vit.prepare_tokens(x)

    def run_blocks(
        self,
        tokens,
        start_index,
        end_index,
        checkpoint_blocks=False,
        checkpoint_segments=None,
    ):
        return self.vit.run_blocks(
            tokens,
            start_index,
            end_index,
            checkpoint_blocks=checkpoint_blocks,
            checkpoint_segments=checkpoint_segments,
        )

    def finalize_and_package(self, tokens):
        raw_tokens = self.vit.finalize_tokens(tokens)
        projected_tokens = self.projection(raw_tokens)
        return {
            "tokens": projected_tokens,
            "features": projected_tokens[:, 0],
            "raw_tokens": raw_tokens,
            "raw_features": raw_tokens[:, 0],
        }

    def load_pretrained(self, model_path, logger=print):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"PMT ImageNet checkpoint not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint = _unwrap_checkpoint(checkpoint)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"PMT checkpoint must resolve to a state dict, got {type(checkpoint)!r}")

        if hasattr(self.vit.patch_embed, "load_from_state_dict_fragment"):
            checkpoint = {
                _normalize_checkpoint_key(key): value
                for key, value in checkpoint.items()
            }
            self.vit.patch_embed.load_from_state_dict_fragment(checkpoint)

        state = {}
        skipped = []
        resized_pos_embed = None
        for original_key, value in checkpoint.items():
            key = _normalize_checkpoint_key(original_key)
            if _is_skipped_key(key):
                skipped.append(key)
                continue
            if key == "pos_embed" and value.shape != self.vit.pos_embed.shape:
                resized_pos_embed = (tuple(value.shape), tuple(self.vit.pos_embed.shape))
                value = resize_pos_embed(
                    value,
                    self.vit.pos_embed,
                    self.vit.patch_embed.num_y,
                    self.vit.patch_embed.num_x,
                )
            if key == "patch_embed.proj.weight" and len(value.shape) < 4:
                out_ch, in_ch, height, width = self.vit.patch_embed.proj.weight.shape
                value = value.reshape(out_ch, in_ch, height, width)
            state[key] = value

        result = self.vit.load_state_dict(state, strict=False)
        patch_embed_loaded_separately = hasattr(self.vit.patch_embed, "load_from_state_dict_fragment")
        missing_core = [
            key
            for key in result.missing_keys
            if _is_core_backbone_key(key)
            and not (patch_embed_loaded_separately and key.startswith("patch_embed."))
        ]
        allowed_missing_patch_embed = [
            key
            for key in result.missing_keys
            if patch_embed_loaded_separately and key.startswith("patch_embed.")
        ]
        unexpected_core = [key for key in result.unexpected_keys if _is_core_backbone_key(key)]

        logger(f"Loaded PMT ImageNet ViT weights from {model_path}")
        logger(f"Loaded keys: {len(state)}")
        logger(f"Missing keys: {len(result.missing_keys)}; Unexpected keys: {len(result.unexpected_keys)}")
        if allowed_missing_patch_embed:
            logger(
                "Allowed missing multi-branch patch keys: "
                f"{len(allowed_missing_patch_embed)}; initialized from single-branch patch_embed"
            )
        logger(f"Required missing core keys: {len(missing_core)}")
        logger(f"Skipped classifier/distillation keys: {len(skipped)}")
        if skipped:
            logger(f"Skipped keys: {skipped}")
        if resized_pos_embed:
            logger(f"Resized pos_embed from {resized_pos_embed[0]} to {resized_pos_embed[1]}")
        else:
            logger("Resized pos_embed: not needed")

        if missing_core:
            raise RuntimeError(f"Missing PMT core backbone keys while loading ImageNet weights: {missing_core}")
        if unexpected_core:
            raise RuntimeError(f"Unexpected PMT core backbone keys while loading ImageNet weights: {unexpected_core}")

        return result
