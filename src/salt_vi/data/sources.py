import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _weighted_view(store, index):
    weights = torch.as_tensor(store.weights(int(index)), dtype=torch.double)
    return int(torch.multinomial(weights, 1).item())


class ArrayVisualSource:
    def __init__(self, path):
        self.images = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def sample(self, index):
        return self.images[int(index)], None


class ImageTreeVisualSource:
    """Load one derived image per canonical SYSU source without materializing NPY arrays."""

    def __init__(self, root, source_keys):
        self.root = Path(root).expanduser().resolve()
        self.paths = []
        missing = []
        for source_key in source_keys:
            relative = Path(str(source_key).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid SYSU image-tree source key: {source_key}")
            path = (self.root / relative).with_suffix(".png").resolve()
            try:
                path.relative_to(self.root)
            except ValueError as error:
                raise ValueError(
                    f"SYSU image-tree path escapes its root: {source_key}"
                ) from error
            self.paths.append(path)
            if not path.is_file():
                missing.append(path)
        if missing:
            raise FileNotFoundError(
                f"SYSU image-tree is missing {len(missing)} derived images; "
                f"first missing path: {missing[0]}"
            )

    def __len__(self):
        return len(self.paths)

    def sample(self, index):
        path = self.paths[int(index)]
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB")).copy(), None


class MultiviewVisualSource:
    def __init__(self, store, views):
        self.store = store
        self.views = int(views)

    def __len__(self):
        return len(self.store)

    def sample(self, index):
        view = _weighted_view(self.store, index)
        return self.store.image(int(index), view), view


class NoCaptionSource:
    def sample(self, index, visual_view=None):
        return None


class ArrayCaptionSource:
    def __init__(self, captions, augmented=None, augmentation_probability=0.0):
        self.captions = captions
        self.augmented = augmented
        self.augmentation_probability = float(augmentation_probability)

    def sample(self, index, visual_view=None):
        index = int(index)
        if self.augmented is not None and random.random() < self.augmentation_probability:
            return self.augmented[index]
        return self.captions[index]


class MultiviewCaptionSource:
    def __init__(self, store, views, sampling, tokenize):
        self.store = store
        self.views = int(views)
        self.sampling = sampling
        self.tokenize = tokenize

    def sample(self, index, visual_view=None):
        view = visual_view
        if self.sampling == "independent":
            view = _weighted_view(self.store, index)
        return self.tokenize(self.store.caption(int(index), int(view)))
