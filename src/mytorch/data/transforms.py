"""Composable CPU image transforms used before GPU batch transfer."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np
from PIL import Image, ImageOps

_rng: ContextVar[random.Random | None] = ContextVar("mytorch_data_rng", default=None)


@contextmanager
def _use_sample_seed(seed: int):
    token = _rng.set(random.Random(seed))
    try:
        yield
    finally:
        _rng.reset(token)


def _random() -> random.Random:
    generator = _rng.get()
    return generator if generator is not None else random


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    result = (value, value) if isinstance(value, int) else tuple(value)
    if len(result) != 2 or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in result
    ):
        raise ValueError(f"{name} must be a positive integer or pair")
    return result


def _pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f":
        array = np.clip(array * 255 if array.max(initial=0) <= 1 else array, 0, 255)
        array = array.astype(np.uint8)
    return Image.fromarray(array)


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        if not isinstance(transforms, Sequence) or not all(
            callable(transform) for transform in transforms
        ):
            raise TypeError("transforms must be a sequence of callables")
        self.transforms = list(transforms)

    def __call__(self, value: Any) -> Any:
        for transform in self.transforms:
            value = transform(value)
        return value


class ToArray:
    """Convert a PIL image or array to CHW NumPy form."""

    def __init__(self, dtype: Any = np.float32, scale: bool = True) -> None:
        self.dtype = np.dtype(dtype)
        if not isinstance(scale, bool):
            raise TypeError("scale must be a bool")
        self.scale = scale

    def __call__(self, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise ValueError("ToArray expects a 2D grayscale or 3D image")
        if array.shape[-1] in {1, 3, 4}:
            array = np.moveaxis(array, -1, 0)
        result = array.astype(self.dtype, copy=False)
        if self.scale and array.dtype.kind in "ui":
            result = result / np.iinfo(array.dtype).max
        return np.ascontiguousarray(result)


class Resize:
    def __init__(
        self, size: int | Sequence[int], interpolation: str = "bilinear"
    ) -> None:
        self.size = _pair(size, "size")
        choices = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
        }
        if interpolation not in choices:
            raise ValueError("interpolation must be nearest, bilinear, or bicubic")
        self.interpolation = interpolation
        self._resample = choices[interpolation]

    def __call__(self, value: Any) -> Image.Image:
        height, width = self.size
        return _pil(value).resize((width, height), self._resample)


class CenterCrop:
    def __init__(self, size: int | Sequence[int]) -> None:
        self.size = _pair(size, "size")

    def __call__(self, value: Any) -> Image.Image:
        image = _pil(value)
        height, width = self.size
        if image.height < height or image.width < width:
            pad_width = max(0, width - image.width)
            pad_height = max(0, height - image.height)
            image = ImageOps.expand(
                image,
                border=(
                    pad_width // 2,
                    pad_height // 2,
                    pad_width - pad_width // 2,
                    pad_height - pad_height // 2,
                ),
            )
        left = (image.width - width) // 2
        top = (image.height - height) // 2
        return image.crop((left, top, left + width, top + height))


class RandomCrop:
    def __init__(self, size: int | Sequence[int], padding: int = 0) -> None:
        self.size = _pair(size, "size")
        if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        self.padding = padding

    def __call__(self, value: Any) -> Image.Image:
        image = _pil(value)
        if self.padding:
            image = ImageOps.expand(image, border=self.padding)
        height, width = self.size
        if image.height < height or image.width < width:
            raise ValueError("RandomCrop size exceeds the image dimensions")
        generator = _random()
        left = generator.randint(0, image.width - width)
        top = generator.randint(0, image.height - height)
        return image.crop((left, top, left + width, top + height))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        self.p = float(p)

    def __call__(self, value: Any) -> Any:
        if _random().random() >= self.p:
            return value
        if isinstance(value, Image.Image):
            return value.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
            axis = 1
        else:
            axis = -1 if array.ndim >= 2 else 0
        return np.flip(array, axis=axis).copy()


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if (
            self.mean.ndim != 1
            or self.std.shape != self.mean.shape
            or np.any(self.std <= 0)
        ):
            raise ValueError("mean and positive std must be matching 1D sequences")

    def __call__(self, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim < 2 or array.shape[0] != len(self.mean):
            raise ValueError("Normalize expects CHW data matching mean/std channels")
        shape = (len(self.mean),) + (1,) * (array.ndim - 1)
        return (array - self.mean.reshape(shape)) / self.std.reshape(shape)


__all__ = [
    "CenterCrop",
    "Compose",
    "Normalize",
    "RandomCrop",
    "RandomHorizontalFlip",
    "Resize",
    "ToArray",
]
