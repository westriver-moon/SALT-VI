"""Export legacy SYSU RGB text augmentations as a NumPy array.

This is an explicit command-line utility.  Importing it never accesses a dataset
or creates files.
"""
import argparse
import json
from pathlib import Path

import numpy as np

RGB_CAMERAS = ("cam1", "cam2", "cam4", "cam5")


def _read_split_ids(data_path: Path):
    identities = []
    for name in ("train_id.txt", "val_id.txt"):
        values = (data_path / "exp" / name).read_text(encoding="utf-8").splitlines()
        identities.extend(int(value) for value in values[0].split(",") if value)
    return [f"{identity:04d}" for identity in identities]


def export_eaaug_text(data_path, captions_path, output_path=None):
    """Write ``ori_aug_description`` values aligned with sorted RGB training images."""
    data_path = Path(data_path).expanduser().resolve()
    captions_path = Path(captions_path).expanduser().resolve()
    with captions_path.open(encoding="utf-8") as handle:
        captions = json.load(handle)

    descriptions = []
    for identity in sorted(_read_split_ids(data_path)):
        for camera in RGB_CAMERAS:
            image_dir = data_path / camera / identity
            if image_dir.is_dir():
                for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
                    descriptions.append(captions[str(image_path)]["ori_aug_description"])

    output_path = Path(output_path) if output_path else data_path / "train_text_eaaug_list.npy"
    output_path = output_path.expanduser().resolve()
    np.save(output_path, np.asarray(descriptions))
    return output_path, len(descriptions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--captions-path", required=True)
    parser.add_argument("--output-path")
    args = parser.parse_args()
    output_path, count = export_eaaug_text(args.data_path, args.captions_path, args.output_path)
    print(f"wrote {count} descriptions to {output_path}")


if __name__ == "__main__":
    main()
