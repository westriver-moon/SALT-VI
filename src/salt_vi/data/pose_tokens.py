from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional


class HumanTokenMaskStore:
    """Read exported pose/anatomy token masks without importing the PACT plugin."""

    def __init__(self, root, dataset, expected_grid=None):
        self.root = Path(root).expanduser() / str(dataset) / "anatomy"
        self.expected_grid = (
            None
            if expected_grid is None
            else tuple(int(value) for value in expected_grid)
        )
        if not self.root.is_dir():
            raise FileNotFoundError(f"CTI anatomy directory does not exist: {self.root}")
        contract_path = self.root.parent / "anatomy.contract.json"
        if not contract_path.is_file():
            # Backward compatibility for the original standalone PACT export.
            contract_path = self.root.parent / "contract.json"
        if contract_path.is_file() and self.expected_grid is not None:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            images = contract.get("person_pose_contract", {}).get("images", {})
            size_hw = images.get("size_hw")
            patch_hw = contract.get("patch_hw")
            stride_hw = contract.get("stride_hw")
            if size_hw and patch_hw and stride_hw:
                contract_grid = tuple(
                    (int(size) - int(patch)) // int(stride) + 1
                    for size, patch, stride in zip(size_hw, patch_hw, stride_hw)
                )
                if contract_grid != self.expected_grid:
                    raise ValueError(
                        f"CTI anatomy contract grid is {contract_grid}, "
                        f"expected {self.expected_grid}: {contract_path}"
                    )

    def path(self, source_key):
        relative = PurePosixPath(str(source_key).replace("\\", "/")).with_suffix(
            ".npz"
        )
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid CTI source key: {source_key!r}")
        return self.root.joinpath(*relative.parts)

    def require_keys(self, source_keys):
        keys = tuple(dict.fromkeys(str(key) for key in source_keys))
        missing_count = 0
        examples = []
        for source_key in keys:
            if not self.path(source_key).is_file():
                missing_count += 1
                if len(examples) < 5:
                    examples.append(source_key)
        if missing_count:
            raise FileNotFoundError(
                f"CTI anatomy coverage is incomplete: {missing_count}/{len(keys)} "
                f"source keys are missing; examples: {examples}"
            )
        return len(keys)

    @lru_cache(maxsize=2048)
    def load(self, source_key):
        path = self.path(source_key)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing CTI anatomy asset for {source_key!r}: {path}"
            )
        with np.load(path, allow_pickle=False) as payload:
            if "token_masks" not in payload:
                raise KeyError(f"CTI anatomy asset has no token_masks array: {path}")
            part_masks = np.asarray(payload["token_masks"], dtype=np.float32)
            if part_masks.ndim != 3:
                raise ValueError(
                    f"token_masks must have shape [parts,H,W], got {part_masks.shape}"
                )
            if "part_valid" in payload:
                part_valid = np.asarray(payload["part_valid"], dtype=bool)
                if part_valid.shape != (part_masks.shape[0],):
                    raise ValueError(
                        f"part_valid shape {part_valid.shape} does not match "
                        f"{part_masks.shape[0]} token-mask parts in {path}"
                    )
                part_masks = part_masks[part_valid]
            if part_masks.shape[0] == 0:
                human_mask = np.zeros(part_masks.shape[1:], dtype=np.float32)
            else:
                human_mask = part_masks.max(axis=0)
        if self.expected_grid is not None and human_mask.shape != self.expected_grid:
            raise ValueError(
                f"CTI token grid for {source_key!r} is {human_mask.shape}, "
                f"expected {self.expected_grid}"
            )
        return torch.from_numpy(np.ascontiguousarray(human_mask))


class SynchronizedTokenTransform:
    """Apply a Compose transform while mirroring every random horizontal flip."""

    _unsupported_geometry = (
        transforms.CenterCrop,
        transforms.FiveCrop,
        transforms.Pad,
        transforms.RandomAffine,
        transforms.RandomCrop,
        transforms.RandomPerspective,
        transforms.RandomResizedCrop,
        transforms.RandomRotation,
        transforms.TenCrop,
    )

    def __init__(self, image_transform):
        self.image_transform = image_transform
        self.steps = tuple(
            image_transform.transforms
            if isinstance(image_transform, transforms.Compose)
            else (image_transform,)
        )
        unsupported = [
            type(step).__name__
            for step in self.steps
            if isinstance(step, self._unsupported_geometry)
        ]
        if unsupported:
            raise ValueError(
                "CTI pose masks do not support crop/pad/warp transforms: "
                + ", ".join(unsupported)
            )

    def __call__(self, image, token_mask):
        mask = torch.as_tensor(token_mask, dtype=torch.float32)
        if mask.ndim != 2:
            raise ValueError(f"CTI token mask must be 2-D, got {tuple(mask.shape)}")
        for step in self.steps:
            if isinstance(step, transforms.RandomHorizontalFlip):
                if torch.rand(1).item() < float(step.p):
                    image = functional.hflip(image)
                    mask = torch.flip(mask, dims=(-1,))
            else:
                image = step(image)
        return image, mask.contiguous()
