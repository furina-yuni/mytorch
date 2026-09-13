"""CPU-side sample ordering and mini-batch composition."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from typing import Any

import numpy as np


class Sampler(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[int]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping) or state_dict:
            raise ValueError("sampler state must be an empty mapping")


class SequentialSampler(Sampler):
    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(replacement, bool):
            raise TypeError("replacement must be a bool")
        if num_samples is None:
            num_samples = len(data_source)
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        if not replacement and num_samples > len(data_source):
            raise ValueError(
                "num_samples cannot exceed dataset length without replacement"
            )
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.seed = seed
        self._random = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            return iter(
                self._random.randrange(len(self.data_source))
                for _ in range(self.num_samples)
            )
        values = list(range(len(self.data_source)))
        self._random.shuffle(values)
        return iter(values[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "random_state": self._random.getstate()}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if (
            not isinstance(state_dict, Mapping)
            or set(state_dict) != {"version", "random_state"}
            or state_dict["version"] != 1
        ):
            raise ValueError("RandomSampler state is invalid")
        self._random.setstate(state_dict["random_state"])


class SubsetRandomSampler(RandomSampler):
    def __init__(self, indices: Sequence[int], *, seed: int | None = None) -> None:
        if not isinstance(indices, Sequence) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in indices
        ):
            raise TypeError("indices must be a sequence of integers")
        self.indices = tuple(indices)
        self.seed = seed
        self._random = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        values = list(self.indices)
        self._random.shuffle(values)
        return iter(values)

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    def __init__(
        self,
        weights: Sequence[float] | np.ndarray,
        num_samples: int,
        replacement: bool = True,
        *,
        seed: int | None = None,
    ) -> None:
        array = np.asarray(weights, dtype=np.float64)
        if (
            array.ndim != 1
            or array.size == 0
            or np.any(array < 0)
            or not np.all(np.isfinite(array))
            or array.sum() <= 0
        ):
            raise ValueError(
                "weights must be a finite, non-negative 1D sequence with positive sum"
            )
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        if not isinstance(replacement, bool):
            raise TypeError("replacement must be a bool")
        if not replacement and num_samples > array.size:
            raise ValueError(
                "num_samples cannot exceed weights length without replacement"
            )
        self.weights = array / array.sum()
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed
        self._random = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[int]:
        values = self._random.choice(
            len(self.weights),
            self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )
        return iter(int(value) for value in values)

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, Any]:
        state = self._random.bit_generator.state
        return {
            "version": 1,
            "bit_generator": state["bit_generator"],
            "state": state["state"],
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if (
            not isinstance(state_dict, Mapping)
            or set(state_dict) != {"version", "bit_generator", "state"}
            or state_dict["version"] != 1
        ):
            raise ValueError("WeightedRandomSampler state is invalid")
        current = self._random.bit_generator.state
        current["bit_generator"] = state_dict["bit_generator"]
        current["state"] = dict(state_dict["state"])
        self._random.bit_generator.state = current


class BatchSampler:
    def __init__(
        self, sampler: Iterable[int], batch_size: int, drop_last: bool
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        length = len(self.sampler)  # type: ignore[arg-type]
        return (
            length // self.batch_size
            if self.drop_last
            else (length + self.batch_size - 1) // self.batch_size
        )


__all__ = [
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]
