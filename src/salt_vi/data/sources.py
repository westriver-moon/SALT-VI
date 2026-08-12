import random

import numpy as np
import torch


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
