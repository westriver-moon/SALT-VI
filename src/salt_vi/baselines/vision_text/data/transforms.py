from __future__ import annotations

import math
import random
from typing import Callable, Optional

from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomErasing:
    """Official PMT-style random erasing after tensor normalization."""

    def __init__(
        self,
        p: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: tuple[float, float, float] = tuple(IMAGENET_MEAN),
    ) -> None:
        self.p = p
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.p:
            return img
        for _ in range(100):
            area = img.size(1) * img.size(2)
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size(2) and h < img.size(1):
                x1 = random.randint(0, img.size(1) - h)
                y1 = random.randint(0, img.size(2) - w)
                for channel in range(min(img.size(0), len(self.mean))):
                    img[channel, x1 : x1 + h, y1 : y1 + w] = self.mean[channel]
                return img
        return img


class RectScale:
    def __init__(self, height: int, width: int, interpolation=Image.BILINEAR) -> None:
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if height == self.height and width == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class SourceTargetScale:
    """Materialize the common PMT LR source, then the experiment input size."""

    def __init__(
        self,
        source_height: int,
        source_width: int,
        target_height: int,
        target_width: int,
        source_interpolation=Image.BILINEAR,
        target_interpolation=Image.BICUBIC,
    ) -> None:
        self.source = RectScale(source_height, source_width, source_interpolation)
        self.target = RectScale(target_height, target_width, target_interpolation)

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.target(self.source(img))


class ExactScale:
    """Reject derived SR assets that do not already match the model input."""

    def __init__(self, height: int, width: int) -> None:
        self.height = int(height)
        self.width = int(width)

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.size != (self.width, self.height):
            raise ValueError(
                f"Derived SR image has size {img.size}, expected "
                f"{self.width}x{self.height}"
            )
        return img


def build_transforms(
    height: int,
    width: int,
    *,
    source_height: Optional[int] = None,
    source_width: Optional[int] = None,
    visible_is_sr: bool = False,
    ir_is_sr: bool = False,
) -> dict[str, Callable]:
    source_height = int(source_height or height)
    source_width = int(source_width or width)
    visible_scale = (
        ExactScale(height, width)
        if visible_is_sr
        else SourceTargetScale(source_height, source_width, height, width)
    )
    ir_scale = (
        ExactScale(height, width)
        if ir_is_sr
        else SourceTargetScale(source_height, source_width, height, width)
    )
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    mix_aug = [
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(21, sigma=(0.1, 3)),
    ]
    return {
        "rgb2gray": transforms.Compose(
            [
                transforms.ToPILImage(),
                visible_scale,
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                normalize,
                RandomErasing(p=0.5),
            ]
        ),
        "rgb": transforms.Compose(
            [
                transforms.ToPILImage(),
                visible_scale,
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                RandomErasing(p=0.5),
            ]
        ),
        "thermal": transforms.Compose(
            [
                transforms.ToPILImage(),
                ir_scale,
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice(mix_aug),
                transforms.ToTensor(),
                normalize,
                RandomErasing(p=0.5),
            ]
        ),
        "test_rgb": transforms.Compose(
            [
                transforms.ToPILImage(),
                visible_scale,
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "test_ir": transforms.Compose(
            [
                transforms.ToPILImage(),
                ir_scale,
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }
