from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SR_ARRAYS = {
    "rgb": "train_rgb_swinir_x2_img.npy",
    "ir": "train_ir_swinir_x2_img.npy",
}


class SYSUData(Dataset):
    """PMT training dataset backed by official SYSU npy caches."""

    def __init__(
        self,
        data_dir: str | Path,
        transform_visible=None,
        transform_ir=None,
        color_index=None,
        thermal_index=None,
        sr_data_dir: str | Path | None = None,
        sr_modalities=(),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sr_data_dir = Path(sr_data_dir) if sr_data_dir else None
        self.sr_modalities = frozenset(sr_modalities or ())
        unknown = self.sr_modalities.difference(SR_ARRAYS)
        if unknown:
            raise ValueError(f"Unsupported SR modalities: {sorted(unknown)}")
        if self.sr_modalities and self.sr_data_dir is None:
            raise ValueError("sr_data_dir is required when SR modalities are enabled")

        color_path = (
            self.sr_data_dir / SR_ARRAYS["rgb"]
            if "rgb" in self.sr_modalities
            else self.data_dir / "train_rgb_resized_img.npy"
        )
        thermal_path = (
            self.sr_data_dir / SR_ARRAYS["ir"]
            if "ir" in self.sr_modalities
            else self.data_dir / "train_ir_resized_img.npy"
        )
        self.train_color_image = np.load(color_path, mmap_mode="r")
        self.train_color_label = np.load(self.data_dir / "train_rgb_resized_label.npy")
        self.train_thermal_image = np.load(thermal_path, mmap_mode="r")
        self.train_thermal_label = np.load(self.data_dir / "train_ir_resized_label.npy")
        if len(self.train_color_image) != len(self.train_color_label):
            raise ValueError("visible SR image/label count mismatch")
        if len(self.train_thermal_image) != len(self.train_thermal_label):
            raise ValueError("IR SR image/label count mismatch")
        self.transform_visible = transform_visible
        self.transform_ir = transform_ir
        self.cIndex = color_index
        self.tIndex = thermal_index

    def set_indices(self, color_index, thermal_index) -> None:
        self.cIndex = np.asarray(color_index)
        self.tIndex = np.asarray(thermal_index)
        assert len(self.cIndex) == len(self.tIndex), "visible and IR index lengths differ"

    def __getitem__(self, index: int):
        assert self.cIndex is not None and self.tIndex is not None, "sampler indices are not set"
        color_idx = int(self.cIndex[index])
        thermal_idx = int(self.tIndex[index])
        img1 = self.train_color_image[color_idx]
        img2 = self.train_thermal_image[thermal_idx]
        target1 = int(self.train_color_label[color_idx])
        target2 = int(self.train_thermal_label[thermal_idx])
        if self.transform_visible is not None:
            img1 = self.transform_visible(img1)
        if self.transform_ir is not None:
            img2 = self.transform_ir(img2)
        return img1, img2, target1, target2

    def __len__(self) -> int:
        if self.cIndex is not None:
            return len(self.cIndex)
        return max(len(self.train_color_label), len(self.train_thermal_label))


class TestData(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        transform=None,
        source_root=None,
        sr_data_dir=None,
        modality=None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.labels = np.asarray(labels)
        self.transform = transform
        self.source_root = Path(source_root).resolve() if source_root else None
        self.sr_data_dir = Path(sr_data_dir).resolve() if sr_data_dir else None
        self.modality = modality
        if self.sr_data_dir is not None and (self.source_root is None or modality not in SR_ARRAYS):
            raise ValueError("Derived evaluation paths require source_root and modality")

    def _image_path(self, path):
        path = Path(path).resolve()
        if self.sr_data_dir is None:
            return path
        try:
            relative = path.relative_to(self.source_root)
        except ValueError as error:
            raise ValueError(f"SYSU evaluation path is outside source root: {path}") from error
        derived = self.sr_data_dir / "eval" / relative
        if not derived.is_file():
            raise FileNotFoundError(f"Missing derived {self.modality} evaluation image: {derived}")
        return derived

    def __getitem__(self, index: int):
        path = self._image_path(self.image_paths[index])
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image)
        if self.transform is not None:
            arr = self.transform(arr)
        return arr, int(self.labels[index])

    def __len__(self) -> int:
        return len(self.image_paths)
