"""Map-style datasets and dataset composition helpers."""

from __future__ import annotations

import csv
import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mytorch.tensor import Tensor, tensor


class Dataset[SampleT](ABC):
    """Base class for finite, integer-indexed datasets."""

    @abstractmethod
    def __getitem__(self, index: int) -> SampleT:
        """Return one sample."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""


class IterableDataset[SampleT](ABC):
    """Base class for datasets that stream samples instead of indexing them."""

    @abstractmethod
    def __iter__(self) -> Iterator[SampleT]:
        """Return a fresh sample iterator."""


def _normalize_index(index: int, length: int) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError("dataset indices must be integers")
    normalized = index + length if index < 0 else index
    if normalized < 0 or normalized >= length:
        raise IndexError("dataset index out of range")
    return normalized


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    """Dataset that maps matching first dimensions across GPU Tensors."""

    def __init__(self, *tensors: Tensor) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one Tensor")
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("TensorDataset accepts only MyTorch Tensors")
        if any(value.ndim == 0 for value in tensors):
            raise ValueError("TensorDataset Tensors must have a sample dimension")
        length = len(tensors[0])
        if any(len(value) != length for value in tensors[1:]):
            raise ValueError("all TensorDataset Tensors must have equal lengths")
        devices = {value.device for value in tensors}
        if len(devices) != 1:
            raise ValueError("all TensorDataset Tensors must be on the same device")
        self.tensors = tuple(tensors)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        normalized = _normalize_index(index, len(self))
        return tuple(value[normalized] for value in self.tensors)

    def __len__(self) -> int:
        return len(self.tensors[0])

    def _get_batch(self, indices: Sequence[int]) -> tuple[Tensor, ...]:
        index = tensor(indices, dtype="int64", device=self.tensors[0].device)
        return tuple(value[index] for value in self.tensors)


class Subset[SampleT](Dataset[SampleT]):
    """View of selected sample indices from another Dataset."""

    def __init__(self, dataset: Dataset[SampleT], indices: Sequence[int]) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        if not isinstance(indices, Sequence):
            raise TypeError("indices must be a sequence of integers")
        self.dataset = dataset
        self.indices = tuple(_normalize_index(index, len(dataset)) for index in indices)

    def __getitem__(self, index: int) -> SampleT:
        normalized = _normalize_index(index, len(self))
        return self.dataset[self.indices[normalized]]

    def __len__(self) -> int:
        return len(self.indices)

    def _get_batch(self, indices: Sequence[int]):
        mapped = [self.indices[_normalize_index(index, len(self))] for index in indices]
        batch_getter = getattr(self.dataset, "_get_batch", None)
        if callable(batch_getter):
            return batch_getter(mapped)
        return None


def random_split[SampleT](
    dataset: Dataset[SampleT],
    lengths: Sequence[int],
    *,
    seed: int | None = None,
) -> tuple[Subset[SampleT], ...]:
    """Split a Dataset into non-overlapping, reproducibly shuffled subsets."""
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a Dataset")
    if not isinstance(lengths, Sequence) or not lengths:
        raise ValueError("lengths must be a non-empty sequence")
    if not all(
        isinstance(length, int) and not isinstance(length, bool) and length >= 0
        for length in lengths
    ):
        raise ValueError("split lengths must be non-negative integers")
    if sum(lengths) != len(dataset):
        raise ValueError("split lengths must sum to the dataset length")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    result: list[Subset[SampleT]] = []
    start = 0
    for length in lengths:
        result.append(Subset(dataset, indices[start : start + length]))
        start += length
    return tuple(result)


class ImageFolder(Dataset[tuple[Any, int]]):
    """Load RGB images arranged in ``root/class_name/image`` folders."""

    _DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        *,
        extensions: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"ImageFolder root is not a directory: {self.root}")
        self.transform = transform
        self.target_transform = target_transform
        allowed = (
            self._DEFAULT_EXTENSIONS
            if extensions is None
            else {str(value).lower() for value in extensions}
        )
        self.classes = sorted(
            path.name for path in self.root.iterdir() if path.is_dir()
        )
        if not self.classes:
            raise ValueError("ImageFolder found no class directories")
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = [
            (path, self.class_to_idx[class_name])
            for class_name in self.classes
            for path in sorted((self.root / class_name).rglob("*"))
            if path.is_file() and path.suffix.lower() in allowed
        ]
        if not self.samples:
            raise ValueError("ImageFolder found no supported image files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        from PIL import Image

        path, target = self.samples[_normalize_index(index, len(self))]
        with Image.open(path) as source:
            image = source.convert("RGB").copy()
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class CSVDataset(Dataset[tuple[np.ndarray, Any]]):
    """Index rows from a CSV file as numeric features and a target."""

    def __init__(
        self,
        path: str | Path,
        *,
        target_column: str | int,
        feature_columns: Sequence[str | int] | None = None,
        delimiter: str = ",",
        has_header: bool = True,
        feature_dtype: Any = np.float32,
        target_dtype: Any = np.int64,
        transform: Callable[[np.ndarray], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"CSV file does not exist: {self.path}")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("delimiter must be one character")
        with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle, delimiter=delimiter))
        if not rows:
            raise ValueError("CSV file is empty")
        header = rows.pop(0) if has_header else None
        width = len(header) if header is not None else len(rows[0])
        if not rows or any(len(row) != width for row in rows):
            raise ValueError("CSV rows must have a consistent, non-zero width")

        def column_index(value: str | int) -> int:
            if isinstance(value, str):
                if header is None:
                    raise ValueError("string columns require has_header=True")
                if value not in header:
                    raise ValueError(f"unknown CSV column: {value}")
                return header.index(value)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("CSV columns must be names or integer indices")
            normalized = value + width if value < 0 else value
            if normalized < 0 or normalized >= width:
                raise ValueError(f"CSV column index is out of range: {value}")
            return normalized

        self.target_index = column_index(target_column)
        if feature_columns is None:
            self.feature_indices = [
                index for index in range(width) if index != self.target_index
            ]
        else:
            self.feature_indices = [column_index(value) for value in feature_columns]
        if not self.feature_indices or self.target_index in self.feature_indices:
            raise ValueError("feature columns must be non-empty and exclude target")
        self.rows = rows
        self.feature_dtype = np.dtype(feature_dtype)
        self.target_dtype = np.dtype(target_dtype)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.rows[_normalize_index(index, len(self))]
        try:
            features = np.asarray(
                [row[position] for position in self.feature_indices],
                dtype=self.feature_dtype,
            )
            target = np.asarray(row[self.target_index], dtype=self.target_dtype).item()
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"CSV row {index} contains invalid numeric data"
            ) from error
        if self.transform is not None:
            features = self.transform(features)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return features, target


class NpyDataset(Dataset[Any]):
    """Memory-map feature and optional target NPY arrays."""

    def __init__(
        self,
        data_path: str | Path,
        target_path: str | Path | None = None,
        *,
        mmap_mode: str | None = "r",
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.data = np.load(Path(data_path), mmap_mode=mmap_mode, allow_pickle=False)
        if self.data.ndim == 0:
            raise ValueError("NpyDataset data must have a sample dimension")
        self.targets = (
            None
            if target_path is None
            else np.load(Path(target_path), mmap_mode=mmap_mode, allow_pickle=False)
        )
        if self.targets is not None and (
            self.targets.ndim == 0 or len(self.targets) != len(self.data)
        ):
            raise ValueError("NpyDataset data and targets must have equal lengths")
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        normalized = _normalize_index(index, len(self))
        value = np.asarray(self.data[normalized])
        if self.transform is not None:
            value = self.transform(value)
        if self.targets is None:
            return value
        target = np.asarray(self.targets[normalized])
        if target.ndim == 0:
            target = target.item()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return value, target


__all__ = [
    "CSVDataset",
    "Dataset",
    "ImageFolder",
    "IterableDataset",
    "NpyDataset",
    "Subset",
    "TensorDataset",
    "random_split",
]
