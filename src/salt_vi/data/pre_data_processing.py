"""Prepare resized SYSU RGB/IR arrays for legacy image-only workflows.

This is an explicit command-line utility.  Importing it never accesses a dataset
or creates files.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

RGB_CAMERAS = ("cam1", "cam2", "cam4", "cam5")
IR_CAMERAS = ("cam3", "cam6")


def _read_split_ids(data_path: Path):
    identities = []
    for name in ("train_id.txt", "val_id.txt"):
        values = (data_path / "exp" / name).read_text(encoding="utf-8").splitlines()
        identities.extend(int(value) for value in values[0].split(",") if value)
    return [f"{identity:04d}" for identity in identities]


def _collect_images(data_path: Path, identities, cameras):
    images = []
    for identity in sorted(identities):
        for camera in cameras:
            image_dir = data_path / camera / identity
            if image_dir.is_dir():
                images.extend(sorted(path for path in image_dir.iterdir() if path.is_file()))
    return images


def _read_images(image_paths, pid_to_label, width: int, height: int):
    pixels, labels = [], []
    resampling = getattr(Image, "Resampling", Image)
    for image_path in image_paths:
        with Image.open(image_path) as image:
            pixels.append(np.asarray(image.resize((width, height), resampling.LANCZOS)))
        labels.append(pid_to_label[int(image_path.parent.name)])
    return np.asarray(pixels), np.asarray(labels)


def preprocess_images(data_path, width: int = 144, height: int = 288):
    """Write the legacy resized RGB/IR arrays and return their file paths."""
    data_path = Path(data_path).expanduser().resolve()
    identities = _read_split_ids(data_path)
    rgb_images = _collect_images(data_path, identities, RGB_CAMERAS)
    ir_images = _collect_images(data_path, identities, IR_CAMERAS)
    pid_to_label = {pid: index for index, pid in enumerate(sorted({int(path.parent.name) for path in ir_images}))}

    rgb_pixels, rgb_labels = _read_images(rgb_images, pid_to_label, width, height)
    ir_pixels, ir_labels = _read_images(ir_images, pid_to_label, width, height)
    outputs = {
        "train_rgb_resized_img.npy": rgb_pixels,
        "train_rgb_resized_label.npy": rgb_labels,
        "train_ir_resized_img.npy": ir_pixels,
        "train_ir_resized_label.npy": ir_labels,
    }
    for name, array in outputs.items():
        np.save(data_path / name, array)
    return [data_path / name for name in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="datasets/sysu")
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()
    for output in preprocess_images(args.data_path, args.width, args.height):
        print(output)


if __name__ == "__main__":
    main()
