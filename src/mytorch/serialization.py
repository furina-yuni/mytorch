"""Safe NPZ serialization for model states and training checkpoints."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._random import get_rng_state, set_rng_state
from .tensor import Tensor, tensor

_MODEL_METADATA = "__mytorch_metadata__"
_CHECKPOINT_METADATA = "__mytorch_checkpoint_metadata__"
_CHECKPOINT_VERSION = 1

__all__ = [
    "get_rng_state",
    "load",
    "load_checkpoint",
    "save",
    "save_checkpoint",
    "set_rng_state",
]


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
    arrays[_MODEL_METADATA] = np.asarray(json.dumps(metadata))
    np.savez_compressed(Path(path), **arrays)


def load(path: str | Path, device: str | int = "cuda:0") -> OrderedDict[str, Tensor]:
    result: OrderedDict[str, Tensor] = OrderedDict()
    with np.load(Path(path), allow_pickle=False) as archive:
        if _MODEL_METADATA not in archive:
            raise ValueError("file is not a MyTorch state archive")
        metadata = json.loads(str(archive[_MODEL_METADATA].item()))
        if set(metadata) != set(archive.files) - {_MODEL_METADATA}:
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


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    """Save nested training state without using pickle."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("save_checkpoint expects a mapping")
    arrays: dict[str, np.ndarray] = {}
    tree = _encode_value(checkpoint, arrays)
    metadata = {
        "format": "mytorch-checkpoint",
        "version": _CHECKPOINT_VERSION,
        "tree": tree,
    }
    arrays[_CHECKPOINT_METADATA] = np.asarray(
        json.dumps(metadata, allow_nan=False, separators=(",", ":"))
    )
    target = Path(path)
    with target.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_checkpoint(
    path: str | Path, device: str | int = "cuda:0"
) -> OrderedDict[Any, Any]:
    """Load a checkpoint onto one explicit CUDA device."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if _CHECKPOINT_METADATA not in archive:
            raise ValueError("file is not a MyTorch checkpoint archive")
        try:
            metadata = json.loads(str(archive[_CHECKPOINT_METADATA].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint metadata is invalid") from error
        if not isinstance(metadata, dict) or set(metadata) != {
            "format",
            "version",
            "tree",
        }:
            raise ValueError("checkpoint metadata has missing or unexpected fields")
        if metadata["format"] != "mytorch-checkpoint":
            raise ValueError("file has an unknown checkpoint format")
        if metadata["version"] != _CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {metadata['version']!r}")
        referenced: set[str] = set()
        result = _decode_value(metadata["tree"], archive, device, referenced)
        stored = set(archive.files) - {_CHECKPOINT_METADATA}
        if referenced != stored:
            raise ValueError("checkpoint metadata does not match its tensors")
    if not isinstance(result, OrderedDict):
        raise ValueError("checkpoint root must be a mapping")
    return result


def _encode_value(value: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(value, Tensor):
        name = f"array_{len(arrays):08d}"
        array = value.numpy()
        arrays[name] = array
        return {
            "type": "tensor",
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "items": [
                [_encode_value(key, arrays), _encode_value(item, arrays)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_encode_value(item, arrays) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [_encode_value(item, arrays) for item in value],
        }
    raise TypeError(f"checkpoint cannot store {type(value).__name__} values")


def _decode_value(
    node: Any,
    archive: np.lib.npyio.NpzFile,
    device: str | int,
    referenced: set[str],
) -> Any:
    if not isinstance(node, dict) or not isinstance(node.get("type"), str):
        raise ValueError("checkpoint value metadata is invalid")
    kind = node["type"]
    if kind == "tensor":
        if set(node) != {"type", "name", "shape", "dtype"}:
            raise ValueError("checkpoint tensor metadata is invalid")
        name = node["name"]
        if not isinstance(name, str) or name not in archive or name in referenced:
            raise ValueError("checkpoint tensor reference is invalid")
        array = archive[name]
        if list(array.shape) != node["shape"] or str(array.dtype) != node["dtype"]:
            raise ValueError(f"checkpoint tensor {name!r} failed validation")
        referenced.add(name)
        return tensor(array, device=device)
    if kind == "none" and set(node) == {"type"}:
        return None
    scalar_types = {"bool": bool, "int": int, "float": float, "str": str}
    if kind in scalar_types and set(node) == {"type", "value"}:
        value = node["value"]
        expected = scalar_types[kind]
        if type(value) is not expected:
            raise ValueError(f"checkpoint {kind} metadata is invalid")
        return value
    if kind in {"tuple", "list"} and set(node) == {"type", "items"}:
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError(f"checkpoint {kind} metadata is invalid")
        values = [_decode_value(item, archive, device, referenced) for item in items]
        return tuple(values) if kind == "tuple" else values
    if kind == "mapping" and set(node) == {"type", "items"}:
        items = node["items"]
        if not isinstance(items, list):
            raise ValueError("checkpoint mapping metadata is invalid")
        result: OrderedDict[Any, Any] = OrderedDict()
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("checkpoint mapping entry is invalid")
            key = _decode_value(pair[0], archive, device, referenced)
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise ValueError("checkpoint mapping key is not a scalar")
            if key in result:
                raise ValueError("checkpoint mapping contains a duplicate key")
            result[key] = _decode_value(pair[1], archive, device, referenced)
        return result
    raise ValueError(f"unsupported checkpoint value type: {kind!r}")
