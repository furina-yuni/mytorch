"""Safe, tensor-only NPZ state serialization."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .tensor import Tensor, tensor


def save(state_dict: Mapping[str, Tensor], path: str | Path) -> None:
    if not isinstance(state_dict, Mapping):
        raise TypeError("save expects a mapping of names to Tensors")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise TypeError("state_dict must map string names to Tensors")
        arrays[name] = value.numpy()
        metadata[name] = {"shape": value.shape, "dtype": str(value.dtype)}
    arrays["__mytorch_metadata__"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(Path(path), **arrays)


def load(path: str | Path, device: str | int = "cuda:0") -> OrderedDict[str, Tensor]:
    result: OrderedDict[str, Tensor] = OrderedDict()
    with np.load(Path(path), allow_pickle=False) as archive:
        if "__mytorch_metadata__" not in archive:
            raise ValueError("file is not a MyTorch state archive")
        metadata = json.loads(str(archive["__mytorch_metadata__"].item()))
        if set(metadata) != set(archive.files) - {"__mytorch_metadata__"}:
            raise ValueError("state archive metadata does not match its tensors")
        for name in metadata:
            expected = metadata[name]
            if (
                tuple(expected["shape"]) != archive[name].shape
                or str(archive[name].dtype) != expected["dtype"]
            ):
                raise ValueError(f"state archive tensor {name!r} failed validation")
            result[name] = tensor(archive[name], device=device)
    return result
