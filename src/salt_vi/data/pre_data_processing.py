"""Prepare resized SYSU RGB/IR arrays for legacy image-only workflows.

This is an explicit command-line utility.  Importing it never accesses a dataset
or creates files.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from salt_vi.data.sysu_sources import collect_train_source_records, write_train_source_manifest


def _read_images(data_path, records, width: int, height: int):
    pixels, labels = [], []
    resampling = getattr(Image, "Resampling", Image)
    for record in records:
        image_path = data_path / record.source_key
        with Image.open(image_path) as image:
            pixels.append(np.asarray(image.resize((width, height), resampling.LANCZOS)))
        labels.append(record.label)
    return np.asarray(pixels), np.asarray(labels)


def preprocess_images(data_path, width: int = 144, height: int = 288):
    """Write the legacy resized RGB/IR arrays and return their file paths."""
    data_path = Path(data_path).expanduser().resolve()
    rgb_records = collect_train_source_records(data_path, "rgb")
    ir_records = collect_train_source_records(data_path, "ir")
    rgb_pixels, rgb_labels = _read_images(data_path, rgb_records, width, height)
    ir_pixels, ir_labels = _read_images(data_path, ir_records, width, height)
    outputs = {
        "train_rgb_resized_img.npy": rgb_pixels,
        "train_rgb_resized_label.npy": rgb_labels,
        "train_ir_resized_img.npy": ir_pixels,
        "train_ir_resized_label.npy": ir_labels,
    }
    for name, array in outputs.items():
        np.save(data_path / name, array)
    manifests = [
        write_train_source_manifest(data_path, "rgb", rgb_records),
        write_train_source_manifest(data_path, "ir", ir_records),
    ]
    return [data_path / name for name in outputs] + manifests


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
