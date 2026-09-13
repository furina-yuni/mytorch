"""Map-style datasets and dataset composition helpers."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence

from mytorch.tensor import Tensor, tensor


class Dataset[SampleT](ABC):
    """Base class for finite, integer-indexed datasets."""

    @abstractmethod
    def __getitem__(self, index: int) -> SampleT:
        """Return one sample."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""


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


__all__ = ["Dataset", "Subset", "TensorDataset", "random_split"]
