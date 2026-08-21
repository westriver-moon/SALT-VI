from __future__ import annotations

import math
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from salt_vi.utils.checkpointing import checkpoint_forward

from salt_vi.attention import normalize_attention_backend, run_scaled_dot_product_attention


def _ntuple(n):
    def parse(x):
        if isinstance(x, (tuple, list)):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def resize_pos_embed(posemb, posemb_new, height, width):
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([posemb_token, posemb_grid], dim=1)


def resize_pos_embed_grid(posemb, old_height, old_width, new_height, new_width):
    """Resize a rectangular ViT positional grid without changing the CLS token."""
    if (old_height, old_width) == (new_height, new_width):
        return posemb
    posemb_token, posemb_grid = posemb[:, :1], posemb[:, 1:]
    expected = int(old_height) * int(old_width)
    if posemb_grid.shape[1] != expected:
        raise ValueError(
            f"Positional grid has {posemb_grid.shape[1]} tokens, expected {expected} "
            f"for {old_height}x{old_width}"
        )
    posemb_grid = posemb_grid.reshape(1, old_height, old_width, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, new_height * new_width, -1)
    return torch.cat([posemb_token, posemb_grid], dim=1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attention_backend="manual",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.attention_backend = normalize_attention_backend(attention_backend)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = run_scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            dropout_p=self.attn_drop.p,
            training=self.training,
            backend=self.attention_backend,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attention_backend="manual",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attention_backend=attention_backend,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlap(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.num_patches = self.num_x * self.num_y
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        nn.init.normal_(self.proj.weight, 0, math.sqrt(2.0 / (patch_size[0] * patch_size[1] * embed_dim)))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def _resize_patch_kernel(weight, patch_size):
    if tuple(weight.shape[-2:]) == tuple(patch_size):
        return weight
    out_ch, in_ch, _height, _width = weight.shape
    flat = weight.reshape(out_ch * in_ch, 1, weight.shape[-2], weight.shape[-1])
    resized = F.interpolate(flat, size=patch_size, mode="bicubic", align_corners=False)
    return resized.reshape(out_ch, in_ch, patch_size[0], patch_size[1])


class MultiBranchPatchEmbedOverlap(nn.Module):
    """Multi-scale overlapping patch embedding fused on the anchor token grid."""

    def __init__(self, img_size=(288, 144), branches=None, anchor_branch=0, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        branches = list(branches or [])
        if not branches:
            raise ValueError("multi-branch patch embedding requires at least one branch")
        self.img_size = img_size
        self.anchor_branch = int(anchor_branch)
        if self.anchor_branch < 0 or self.anchor_branch >= len(branches):
            raise ValueError(f"anchor_branch {self.anchor_branch} is out of range for {len(branches)} branches")

        self.branch_configs = []
        self.proj = nn.ModuleList()
        for branch in branches:
            patch_size = to_2tuple(branch.get("patch_size", 16))
            stride_size = to_2tuple(branch.get("stride_size", branch.get("stride", patch_size)))
            self.branch_configs.append({"patch_size": patch_size, "stride_size": stride_size})
            conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
            nn.init.normal_(conv.weight, 0, math.sqrt(2.0 / (patch_size[0] * patch_size[1] * embed_dim)))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            self.proj.append(conv)

        anchor = self.branch_configs[self.anchor_branch]
        self.patch_size = anchor["patch_size"]
        self.stride_size = anchor["stride_size"]
        self.num_x = (img_size[1] - self.patch_size[1]) // self.stride_size[1] + 1
        self.num_y = (img_size[0] - self.patch_size[0]) // self.stride_size[0] + 1
        self.num_patches = self.num_x * self.num_y
        self.fuse = nn.Conv2d(embed_dim * len(self.proj), embed_dim, kernel_size=1, bias=True)
        self._init_fuse_as_anchor_identity(embed_dim)

    def _init_fuse_as_anchor_identity(self, embed_dim):
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.bias.zero_()
            start = self.anchor_branch * embed_dim
            for channel in range(embed_dim):
                self.fuse.weight[channel, start + channel, 0, 0] = 1.0

    def load_from_state_dict_fragment(self, state_dict):
        weight = state_dict.pop("patch_embed.proj.weight", None)
        bias = state_dict.pop("patch_embed.proj.bias", None)
        if weight is None:
            return
        if len(weight.shape) < 4:
            out_ch, in_ch = self.proj[self.anchor_branch].weight.shape[:2]
            height, width = self.proj[self.anchor_branch].weight.shape[-2:]
            weight = weight.reshape(out_ch, in_ch, height, width)
        with torch.no_grad():
            for conv, branch in zip(self.proj, self.branch_configs):
                conv.weight.copy_(_resize_patch_kernel(weight, branch["patch_size"]))
                if bias is not None and conv.bias is not None:
                    conv.bias.copy_(bias)

    def forward(self, x):
        feature_maps = [conv(x) for conv in self.proj]
        target_size = feature_maps[self.anchor_branch].shape[-2:]
        aligned = []
        for feature_map in feature_maps:
            if feature_map.shape[-2:] != target_size:
                feature_map = F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(feature_map)
        x = self.fuse(torch.cat(aligned, dim=1))
        return x.flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(
        self,
        img_size=(288, 144),
        patch_size=(16, 16),
        stride_size=(12, 12),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        patch_embed_config=None,
        norm_layer=nn.LayerNorm,
        attention_backend="manual",
    ):
        super().__init__()
        if patch_embed_config:
            self.patch_embed = MultiBranchPatchEmbedOverlap(
                img_size=img_size,
                branches=patch_embed_config.get("branches", []),
                anchor_branch=patch_embed_config.get("anchor_branch", 0),
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbedOverlap(img_size, patch_size, stride_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.base_grid_size = (self.patch_embed.num_y, self.patch_embed.num_x)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attention_backend=attention_backend,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def prepare_tokens(self, x):
        """Prepare pre-block tokens and return their actual rectangular grid."""
        if x.ndim != 4:
            raise ValueError(f"ViT expects BCHW input, got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_embed.patch_size
        stride_height, stride_width = self.patch_embed.stride_size
        grid_height = (height - patch_height) // stride_height + 1
        grid_width = (width - patch_width) // stride_width + 1
        if grid_height < 1 or grid_width < 1:
            raise ValueError(
                f"Input {height}x{width} is smaller than patch size "
                f"{patch_height}x{patch_width}"
            )
        x = self.patch_embed(x)
        return self.prepare_embedded_tokens(x, (grid_height, grid_width))

    def prepare_embedded_tokens(self, patch_tokens, grid_size):
        """Add the shared CLS and positional embeddings to externally embedded patches."""
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"Embedded PMT patches must have shape [B,N,D], got {tuple(patch_tokens.shape)}"
            )
        grid_height, grid_width = (int(value) for value in grid_size)
        expected_tokens = grid_height * grid_width
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Embedded PMT patches contain {patch_tokens.shape[1]} tokens, expected "
                f"{expected_tokens} for grid {grid_height}x{grid_width}"
            )
        batch = patch_tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, patch_tokens), dim=1)
        pos_embed = resize_pos_embed_grid(
            self.pos_embed,
            self.base_grid_size[0],
            self.base_grid_size[1],
            grid_height,
            grid_width,
        )
        if x.shape[1] != pos_embed.shape[1]:
            raise RuntimeError(
                f"Patch tokens ({x.shape[1]}) and positional tokens "
                f"({pos_embed.shape[1]}) disagree for {height}x{width}"
            )
        x = x + pos_embed
        x = self.pos_drop(x)
        return x, (grid_height, grid_width)

    def run_blocks(
        self,
        tokens,
        start_index,
        end_index,
        checkpoint_blocks=False,
        checkpoint_segments=None,
    ):
        """Run blocks[start_index:end_index] with an exclusive end index."""
        start_index = int(start_index)
        end_index = int(end_index)
        depth = len(self.blocks)
        if not 0 <= start_index <= end_index <= depth:
            raise ValueError(
                f"Invalid PMT block slice [{start_index}:{end_index}] for depth {depth}"
            )
        if tokens.ndim != 3:
            raise ValueError(f"PMT blocks expect [B,N,D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.pos_embed.shape[-1]:
            raise ValueError(
                f"PMT token dim {tokens.shape[-1]} does not match "
                f"backbone dim {self.pos_embed.shape[-1]}"
            )
        if isinstance(checkpoint_blocks, bool):
            checkpoint_count = depth if checkpoint_blocks else 0
        else:
            checkpoint_count = int(checkpoint_blocks)
            if not 0 <= checkpoint_count <= depth:
                raise ValueError(
                    f"checkpoint_blocks must be within [0, {depth}], "
                    f"got {checkpoint_count}"
                )
        checkpoint_end = min(end_index, checkpoint_count)
        checkpoint_start = min(max(start_index, 0), checkpoint_end)
        active_checkpoint_count = checkpoint_end - checkpoint_start
        if (
            active_checkpoint_count > 0
            and torch.is_grad_enabled()
            and tokens.requires_grad
        ):
            if checkpoint_segments is None:
                segment_count = active_checkpoint_count
            else:
                segment_count = int(checkpoint_segments)
                if segment_count < 1:
                    raise ValueError(
                        "checkpoint_segments must be positive when checkpointing "
                        f"is active, got {segment_count}"
                    )
                segment_count = min(segment_count, active_checkpoint_count)
            segment_size, remainder = divmod(
                active_checkpoint_count, segment_count
            )
            cursor = checkpoint_start
            for segment_index in range(segment_count):
                width = segment_size + int(segment_index < remainder)
                segment_blocks = tuple(self.blocks[cursor : cursor + width])

                def run_segment(value, blocks=segment_blocks):
                    for block in blocks:
                        value = block(value)
                    return value

                tokens = checkpoint_forward(run_segment, tokens)
                cursor += width
        else:
            checkpoint_end = start_index
        for block_index in range(checkpoint_end, end_index):
            tokens = self.blocks[block_index](tokens)
        return tokens

    def finalize_tokens(self, tokens):
        if tokens.ndim != 3:
            raise ValueError(f"PMT final norm expects [B,N,D], got {tuple(tokens.shape)}")
        return self.norm(tokens)

    def forward_features(self, x, return_tokens=False):
        x, _grid_size = self.prepare_tokens(x)
        x = self.run_blocks(x, 0, len(self.blocks))
        x = self.finalize_tokens(x)
        if return_tokens:
            return x
        return x[:, 0]

    def forward(self, x, return_tokens=False):
        return self.forward_features(x, return_tokens=return_tokens)
