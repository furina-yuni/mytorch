"""GPU-oriented batching with CPU file workers and asynchronous transfer."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from numbers import Integral, Real
from typing import Any

import cupy as cp
import numpy as np

from mytorch._device import parse_device
from mytorch.tensor import Tensor, stack, tensor

from .dataset import Dataset, IterableDataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler
from .transforms import _use_sample_seed

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


def _host_collate(batch: Sequence[Any]) -> Any:
    if not isinstance(batch, Sequence) or not batch:
        raise ValueError("default_collate requires a non-empty sample sequence")
    first = batch[0]
    if isinstance(first, Tensor):
        if not all(isinstance(value, Tensor) for value in batch):
            raise TypeError("batch elements must have matching types")
        return stack(batch, dim=0)
    if isinstance(first, np.ndarray):
        if not all(isinstance(value, np.ndarray) for value in batch):
            raise TypeError("batch elements must have matching types")
        try:
            result = np.stack(batch)
        except ValueError as error:
            raise ValueError(
                "NumPy batch elements must have matching shapes"
            ) from error
        if result.dtype.kind in "OUSV":
            raise TypeError("object and string NumPy arrays cannot be collated")
        return result
    if isinstance(first, (bool, np.bool_)):
        return np.asarray(batch, dtype=np.bool_)
    if isinstance(first, (Integral, np.integer)) and not isinstance(
        first, (bool, np.bool_)
    ):
        if not all(
            isinstance(value, (Real, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in batch
        ):
            raise TypeError("numeric batch elements must have compatible types")
        dtype = (
            np.int64
            if all(isinstance(value, (Integral, np.integer)) for value in batch)
            else np.float32
        )
        return np.asarray(batch, dtype=dtype)
    if isinstance(first, (Real, np.floating)):
        if not all(
            isinstance(value, (Real, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in batch
        ):
            raise TypeError("numeric batch elements must have compatible types")
        return np.asarray(batch, dtype=np.float32)
    if isinstance(first, Mapping):
        keys = tuple(first)
        if not all(
            isinstance(value, Mapping) and set(value) == set(keys) for value in batch
        ):
            raise ValueError("mapping batch elements must have identical keys")
        return {key: _host_collate([value[key] for value in batch]) for key in keys}
    if isinstance(first, tuple):
        if not all(
            isinstance(value, tuple) and len(value) == len(first) for value in batch
        ):
            raise ValueError("tuple batch elements must have matching lengths")
        return tuple(
            _host_collate([value[index] for value in batch])
            for index in range(len(first))
        )
    if isinstance(first, list):
        if not all(
            isinstance(value, list) and len(value) == len(first) for value in batch
        ):
            raise ValueError("list batch elements must have matching lengths")
        return [
            _host_collate([value[index] for value in batch])
            for index in range(len(first))
        ]
    if isinstance(first, str) and all(isinstance(value, str) for value in batch):
        return list(batch)
    raise TypeError(
        f"default_collate does not support samples of type {type(first).__name__}"
    )


def _copy_array(
    array: np.ndarray,
    *,
    device: int,
    pin_memory: bool,
    stream: cp.cuda.Stream | None,
) -> Tensor:
    array = np.ascontiguousarray(array)
    if not pin_memory or stream is None or array.nbytes == 0:
        return tensor(array, device=device)
    memory = cp.cuda.alloc_pinned_memory(array.nbytes)
    pinned = np.frombuffer(memory, dtype=array.dtype, count=array.size).reshape(
        array.shape
    )
    np.copyto(pinned, array)
    with cp.cuda.Device(device), stream:
        gpu = cp.asarray(pinned, blocking=False)
        event = cp.cuda.Event()
        event.record(stream)
    with cp.cuda.Device(device):
        cp.cuda.get_current_stream().wait_event(event)
    result = Tensor._from_array(gpu)
    result._transfer_owner = (memory, event)
    return result


def _to_device(
    value: Any,
    *,
    device: int,
    pin_memory: bool = False,
    stream: cp.cuda.Stream | None = None,
) -> Any:
    if isinstance(value, Tensor):
        return value if value._device_index == device else value.to(device=device)
    if isinstance(value, np.ndarray):
        return _copy_array(value, device=device, pin_memory=pin_memory, stream=stream)
    if isinstance(value, Mapping):
        return {
            key: _to_device(item, device=device, pin_memory=pin_memory, stream=stream)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _to_device(item, device=device, pin_memory=pin_memory, stream=stream)
            for item in value
        )
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            return value
        return [
            _to_device(item, device=device, pin_memory=pin_memory, stream=stream)
            for item in value
        ]
    return value


def default_collate(batch: Sequence[Any], *, device: str | int | None = None) -> Any:
    """Recursively combine structured CPU or GPU samples into a GPU batch."""
    resolved = (device or _find_device(batch[0])) if batch else device
    return _to_device(_host_collate(batch), device=parse_device(resolved or "cuda:0"))


class DataLoader[SampleT, BatchT]:
    """Re-iterable loader with deterministic ordering and threaded prefetch."""

    def __init__(
        self,
        dataset: Dataset[SampleT] | IterableDataset[SampleT],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn: CollateFn[SampleT, BatchT] | None = None,
        *,
        sampler: Sampler | None = None,
        batch_sampler: BatchSampler | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        device: str | int = "cuda:0",
        seed: int | None = None,
    ) -> None:
        if not isinstance(dataset, (Dataset, IterableDataset)):
            raise TypeError("dataset must be a Dataset or IterableDataset")
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
        if (
            not isinstance(num_workers, int)
            or isinstance(num_workers, bool)
            or num_workers < 0
        ):
            raise ValueError("num_workers must be a non-negative integer")
        if (
            not isinstance(prefetch_factor, int)
            or isinstance(prefetch_factor, bool)
            or prefetch_factor <= 0
        ):
            raise ValueError("prefetch_factor must be a positive integer")
        if not all(
            isinstance(value, bool) for value in (persistent_workers, pin_memory)
        ):
            raise TypeError("persistent_workers and pin_memory must be bool values")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if isinstance(dataset, IterableDataset) and (
            shuffle or sampler is not None or batch_sampler is not None
        ):
            raise ValueError("IterableDataset does not support shuffle or samplers")
        if sampler is not None and shuffle:
            raise ValueError("shuffle cannot be combined with sampler")
        if batch_sampler is not None and (
            batch_size != 1 or shuffle or sampler is not None or drop_last
        ):
            raise ValueError(
                "batch_sampler conflicts with batch_size, shuffle, sampler, "
                "and drop_last"
            )
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError("sampler must be a Sampler")
        if batch_sampler is not None and not isinstance(batch_sampler, BatchSampler):
            raise TypeError("batch_sampler must be a BatchSampler")
        if isinstance(dataset, IterableDataset) and num_workers:
            raise ValueError("thread workers are not supported for IterableDataset")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or _host_collate
        self._uses_default_collate = collate_fn is None
        self.seed = seed
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.device = f"cuda:{parse_device(device)}"
        self._device_index = parse_device(device)
        self.epoch = 0
        self._executor: ThreadPoolExecutor | None = None
        self._stream = cp.cuda.Stream(non_blocking=True) if pin_memory else None
        if isinstance(dataset, Dataset):
            self.sampler = sampler or (
                RandomSampler(dataset, seed=seed)
                if shuffle
                else SequentialSampler(dataset)
            )
            self.batch_sampler = batch_sampler or BatchSampler(
                self.sampler, batch_size, drop_last
            )
        else:
            self.sampler = None
            self.batch_sampler = None

    def _executor_for_iteration(self) -> ThreadPoolExecutor:
        if self.persistent_workers:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
            return self._executor
        return ThreadPoolExecutor(max_workers=self.num_workers)

    def _sample(self, index: int) -> SampleT:
        base = 0 if self.seed is None else self.seed
        sample_seed = (base ^ (self.epoch * 0x9E3779B1) ^ (index * 0x85EBCA77)) & (
            (1 << 64) - 1
        )
        with _use_sample_seed(sample_seed):
            return self.dataset[index]  # type: ignore[index]

    def _samples(self, indices: Sequence[int]) -> list[SampleT]:
        try:
            return [self._sample(index) for index in indices]
        except Exception as error:
            raise RuntimeError(
                f"DataLoader worker failed for sample indices {list(indices)}"
            ) from error

    def _finish_batch(self, samples: list[SampleT]) -> BatchT:
        collated = self.collate_fn(samples)
        return _to_device(
            collated,
            device=self._device_index,
            pin_memory=self.pin_memory,
            stream=self._stream,
        )

    def _map_iterator(self) -> Iterator[BatchT]:
        assert self.batch_sampler is not None
        batches = iter(self.batch_sampler)
        batch_getter = getattr(self.dataset, "_get_batch", None)
        if self._uses_default_collate and callable(batch_getter):
            for indices in batches:
                result = batch_getter(indices)
                if result is not None:
                    yield result
            return
        if self.num_workers == 0:
            for indices in batches:
                yield self._finish_batch(self._samples(indices))
            return
        executor = self._executor_for_iteration()
        pending: deque[Future[list[SampleT]]] = deque()
        limit = self.num_workers * self.prefetch_factor
        try:
            for _ in range(limit):
                try:
                    pending.append(executor.submit(self._samples, next(batches)))
                except StopIteration:
                    break
            while pending:
                samples = pending.popleft().result()
                try:
                    pending.append(executor.submit(self._samples, next(batches)))
                except StopIteration:
                    pass
                yield self._finish_batch(samples)
        finally:
            for future in pending:
                future.cancel()
            if not self.persistent_workers:
                executor.shutdown(wait=True, cancel_futures=True)

    def _iterable_iterator(self) -> Iterator[BatchT]:
        batch: list[SampleT] = []
        for sample in self.dataset:  # type: ignore[union-attr]
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self._finish_batch(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._finish_batch(batch)

    def __iter__(self) -> Iterator[BatchT]:
        iterator = (
            self._iterable_iterator()
            if isinstance(self.dataset, IterableDataset)
            else self._map_iterator()
        )
        completed = False
        try:
            yield from iterator
            completed = True
        finally:
            if completed:
                self.epoch += 1

    def __len__(self) -> int:
        if isinstance(self.dataset, IterableDataset):
            if not hasattr(self.dataset, "__len__"):
                raise TypeError(
                    "DataLoader length is undefined for this IterableDataset"
                )
            length = len(self.dataset)  # type: ignore[arg-type]
            return (
                length // self.batch_size
                if self.drop_last
                else math.ceil(length / self.batch_size)
            )
        assert self.batch_sampler is not None
        return len(self.batch_sampler)

    def state_dict(self) -> dict[str, Any]:
        sampler_state = self.sampler.state_dict() if self.sampler is not None else {}
        return {
            "version": 2,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "epoch": self.epoch,
            "sampler_type": (
                None if self.sampler is None else type(self.sampler).__name__
            ),
            "sampler_state": sampler_state,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("DataLoader state_dict must be a mapping")
        expected = {
            "version",
            "batch_size",
            "shuffle",
            "drop_last",
            "seed",
            "epoch",
            "sampler_type",
            "sampler_state",
        }
        if set(state_dict) != expected or state_dict["version"] != 2:
            raise ValueError("DataLoader state_dict has an invalid format or version")
        for name in ("batch_size", "shuffle", "drop_last"):
            if state_dict[name] != getattr(self, name):
                raise ValueError(f"DataLoader {name} does not match the checkpoint")
        current_type = None if self.sampler is None else type(self.sampler).__name__
        if state_dict["sampler_type"] != current_type:
            raise ValueError("DataLoader sampler type does not match the checkpoint")
        epoch = state_dict["epoch"]
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("DataLoader epoch must be a non-negative integer")
        if self.sampler is not None:
            self.sampler.load_state_dict(state_dict["sampler_state"])
        self.seed = state_dict["seed"]
        self.epoch = epoch

    def close(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __del__(self) -> None:
        self.close()


__all__ = ["DataLoader", "default_collate"]
