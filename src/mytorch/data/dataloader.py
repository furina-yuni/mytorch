"""Single-process GPU-oriented batching and collation."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

from mytorch.tensor import Tensor, stack, tensor

from .dataset import Dataset

type CollateFn[SampleT, BatchT] = Callable[[list[SampleT]], BatchT]


def _find_device(value: Any) -> str | None:
    if isinstance(value, Tensor):
        return value.device
    if isinstance(value, Mapping):
        for child in value.values():
            if (device := _find_device(child)) is not None:
                return device
    elif isinstance(value, (tuple, list)):
        for child in value:
            if (device := _find_device(child)) is not None:
                return device
    return None


def default_collate(batch: Sequence[Any], *, device: str | int | None = None) -> Any:
    """Recursively combine equally structured samples into GPU batches."""
    if not isinstance(batch, Sequence) or not batch:
        raise ValueError("default_collate requires a non-empty sample sequence")
    resolved_device = device or _find_device(batch[0]) or "cuda:0"
    first = batch[0]

    if isinstance(first, Tensor):
        if not all(isinstance(value, Tensor) for value in batch):
            raise TypeError("batch elements must have matching types")
        return stack(batch, dim=0)
    if isinstance(first, np.ndarray):
        if not all(isinstance(value, np.ndarray) for value in batch):
            raise TypeError("batch elements must have matching types")
        try:
            values = np.stack(batch)
        except ValueError as error:
            raise ValueError(
                "NumPy batch elements must have matching shapes"
            ) from error
        if values.dtype.kind in "OUSV":
            raise TypeError("object and string NumPy arrays cannot be collated")
        return tensor(values, device=resolved_device)
    if isinstance(first, (bool, np.bool_)):
        if not all(isinstance(value, (bool, np.bool_)) for value in batch):
            raise TypeError("boolean batch elements must have matching types")
        return tensor(batch, dtype="bool", device=resolved_device)
    if isinstance(first, (Integral, np.integer)):
        if not all(
            isinstance(value, (Real, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in batch
        ):
            raise TypeError("numeric batch elements must have compatible types")
        dtype = (
            "int64"
            if all(isinstance(value, (Integral, np.integer)) for value in batch)
            else "float32"
        )
        return tensor(batch, dtype=dtype, device=resolved_device)
    if isinstance(first, (Real, np.floating)):
        if not all(
            isinstance(value, (Real, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in batch
        ):
            raise TypeError("numeric batch elements must have compatible types")
        return tensor(batch, dtype="float32", device=resolved_device)
    if isinstance(first, Mapping):
        keys = tuple(first)
        if not all(
            isinstance(value, Mapping) and set(value) == set(keys) for value in batch
        ):
            raise ValueError("mapping batch elements must have identical keys")
        return {
            key: default_collate(
                [value[key] for value in batch], device=resolved_device
            )
            for key in keys
        }
    if isinstance(first, tuple):
        if not all(
            isinstance(value, tuple) and len(value) == len(first) for value in batch
        ):
            raise ValueError("tuple batch elements must have matching lengths")
        return tuple(
            default_collate([value[index] for value in batch], device=resolved_device)
            for index in range(len(first))
        )
    if isinstance(first, list):
        if not all(
            isinstance(value, list) and len(value) == len(first) for value in batch
        ):
            raise ValueError("list batch elements must have matching lengths")
        return [
            default_collate([value[index] for value in batch], device=resolved_device)
            for index in range(len(first))
        ]
    if isinstance(first, str) and all(isinstance(value, str) for value in batch):
        return list(batch)
    raise TypeError(
        f"default_collate does not support samples of type {type(first).__name__}"
    )


class DataLoader[SampleT, BatchT]:
    """Re-iterable mini-batch loader for finite map-style Datasets."""

    def __init__(
        self,
        dataset: Dataset[SampleT],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: CollateFn[SampleT, BatchT] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise TypeError("shuffle and drop_last must be bool values")
        if collate_fn is not None and not callable(collate_fn):
            raise TypeError("collate_fn must be callable or None")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer or None")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        self.seed = seed
        self._random = random.Random(seed)

    def __iter__(self) -> Iterator[BatchT]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            yield self._load_batch(batch_indices)

    def __len__(self) -> int:
        length = len(self.dataset)
        if self.drop_last:
            return length // self.batch_size
        return (length + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> dict[str, Any]:
        """Capture shuffle state for exact continuation at an epoch boundary."""
        return {
            "version": 1,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "random_state": self._random.getstate(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        if not isinstance(state_dict, Mapping):
            raise TypeError("DataLoader state_dict must be a mapping")
        expected = {
            "version",
            "batch_size",
            "shuffle",
            "drop_last",
            "seed",
            "random_state",
        }
        if set(state_dict) != expected:
            raise ValueError("DataLoader state_dict has missing or unexpected fields")
        if state_dict["version"] != 1:
            raise ValueError(
                f"unsupported DataLoader state version: {state_dict['version']!r}"
            )
        for name in ("batch_size", "shuffle", "drop_last"):
            if state_dict[name] != getattr(self, name):
                raise ValueError(f"DataLoader {name} does not match the checkpoint")
        try:
            self._random.setstate(state_dict["random_state"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid DataLoader random state") from error
        self.seed = state_dict["seed"]

    def _load_batch(self, indices: Sequence[int]) -> BatchT:
        batch_getter = getattr(self.dataset, "_get_batch", None)
        if self.collate_fn is default_collate and callable(batch_getter):
            result = batch_getter(indices)
            if result is not None:
                return result
        samples = [self.dataset[index] for index in indices]
        return self.collate_fn(samples)


__all__ = ["DataLoader", "default_collate"]
