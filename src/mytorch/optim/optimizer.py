"""Shared optimizer parameter-group and state management."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any

import cupy as cp

from mytorch.nn.modules.base import Parameter
from mytorch.tensor import Tensor


def require_real(
    name: str, value: Any, *, minimum: float = 0.0, strict: bool = False
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (strict and value == minimum):
        qualifier = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {qualifier} {minimum}")
    return value


def require_betas(betas: Any) -> tuple[float, float]:
    if not isinstance(betas, tuple) or len(betas) != 2:
        raise TypeError("betas must be a tuple of two real numbers")
    first = require_real("beta1", betas[0])
    second = require_real("beta2", betas[1])
    if first >= 1 or second >= 1:
        raise ValueError("betas must be in [0, 1)")
    return first, second


class Optimizer:
    """Base class for dense GPU optimizers."""

    def __init__(self, params: Iterable[Any], defaults: dict[str, Any]) -> None:
        self.defaults = dict(defaults)
        self.param_groups: list[dict[str, Any]] = []
        self.state: dict[int, dict[str, Any]] = {}
        values = list(params)
        if not values:
            raise ValueError("optimizer received no parameters")
        if isinstance(values[0], dict):
            if not all(isinstance(value, dict) for value in values):
                raise TypeError("parameter groups cannot be mixed with bare Parameters")
            for group in values:
                self.add_param_group(group)
        else:
            self.add_param_group({"params": values})

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return tuple(
            parameter for group in self.param_groups for parameter in group["params"]
        )

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not isinstance(param_group, dict) or "params" not in param_group:
            raise TypeError("parameter group must be a dict containing 'params'")
        raw_params = param_group["params"]
        unknown = set(param_group) - {"params", *self.defaults}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown optimizer option(s): {names}")
        params = [raw_params] if isinstance(raw_params, Parameter) else list(raw_params)
        if not params:
            raise ValueError("parameter group cannot be empty")
        if not all(isinstance(parameter, Parameter) for parameter in params):
            raise TypeError("optimizer parameters must be Parameter objects")
        known = {id(parameter) for parameter in self.parameters}
        identities = [id(parameter) for parameter in params]
        if len(set(identities)) != len(identities) or known.intersection(identities):
            raise ValueError(
                "a Parameter cannot appear in more than one parameter group"
            )
        group = dict(self.defaults)
        group.update(
            {key: value for key, value in param_group.items() if key != "params"}
        )
        group["params"] = tuple(params)
        self._validate_group(group)
        self.param_groups.append(group)

    def _validate_group(self, group: dict[str, Any]) -> None:
        del group

    def _parameter_groups(self) -> Iterator[tuple[Parameter, dict[str, Any]]]:
        for group in self.param_groups:
            for parameter in group["params"]:
                yield parameter, group

    def zero_grad(self, set_to_none: bool = True) -> None:
        if not isinstance(set_to_none, bool):
            raise TypeError("set_to_none must be a bool")
        for parameter in self.parameters:
            parameter._clear_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state using stable parameter indices."""
        parameters = self.parameters
        indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        state = {
            indices[identity]: _copy_state_for_export(value)
            for identity, value in self.state.items()
            if identity in indices
        }
        groups = []
        for group in self.param_groups:
            exported = {
                name: deepcopy(value)
                for name, value in group.items()
                if name != "params"
            }
            exported["params"] = [
                indices[id(parameter)] for parameter in group["params"]
            ]
            groups.append(exported)
        return {"state": state, "param_groups": groups}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore state while retaining this optimizer's Parameter objects."""
        if not isinstance(state_dict, Mapping):
            raise TypeError("optimizer state_dict must be a mapping")
        if set(state_dict) != {"state", "param_groups"}:
            raise ValueError("optimizer state_dict has missing or unexpected fields")
        saved_state = state_dict["state"]
        saved_groups = state_dict["param_groups"]
        if not isinstance(saved_state, Mapping):
            raise TypeError("optimizer state must be a mapping")
        if not isinstance(saved_groups, Sequence) or isinstance(
            saved_groups, (str, bytes)
        ):
            raise TypeError("optimizer param_groups must be a sequence")
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "optimizer parameter-group count does not match the checkpoint"
            )

        index_to_parameter: dict[int, Parameter] = {}
        restored_groups: list[dict[str, Any]] = []
        expected_options = [set(group) - {"params"} for group in self.param_groups]
        for group_index, (saved, current) in enumerate(
            zip(saved_groups, self.param_groups, strict=True)
        ):
            if not isinstance(saved, Mapping) or "params" not in saved:
                raise TypeError("each optimizer parameter group must contain params")
            options = set(saved) - {"params"}
            if options != expected_options[group_index]:
                raise ValueError(
                    "optimizer parameter-group options do not match the checkpoint"
                )
            saved_indices = saved["params"]
            if not isinstance(saved_indices, Sequence) or isinstance(
                saved_indices, (str, bytes)
            ):
                raise TypeError("optimizer parameter indices must be a sequence")
            if len(saved_indices) != len(current["params"]):
                raise ValueError(
                    "optimizer parameter count does not match the checkpoint"
                )
            restored = {
                name: deepcopy(value)
                for name, value in saved.items()
                if name != "params"
            }
            restored["params"] = current["params"]
            self._validate_group(restored)
            restored_groups.append(restored)
            for index, parameter in zip(saved_indices, current["params"], strict=True):
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise ValueError(
                        "optimizer parameter indices must be non-negative integers"
                    )
                if index in index_to_parameter:
                    raise ValueError("optimizer parameter indices must be unique")
                index_to_parameter[index] = parameter

        if set(index_to_parameter) != set(range(len(self.parameters))):
            raise ValueError("optimizer parameter indices are incomplete")
        unknown = set(saved_state) - set(index_to_parameter)
        if unknown:
            raise ValueError(
                f"optimizer state contains unknown parameter indices: {unknown}"
            )

        restored_state: dict[int, dict[str, Any]] = {}
        for index, value in saved_state.items():
            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError("optimizer state keys must be parameter indices")
            if not isinstance(value, Mapping):
                raise TypeError("per-parameter optimizer state must be a mapping")
            parameter = index_to_parameter[index]
            restored_state[id(parameter)] = {
                name: _restore_state_value(item, parameter, name)
                for name, item in value.items()
            }

        self.param_groups = restored_groups
        self.state = restored_state

    def step(self) -> None:
        raise NotImplementedError


def _copy_state_for_export(value: Any) -> Any:
    if isinstance(value, cp.ndarray):
        with cp.cuda.Device(int(value.device.id)):
            return Tensor._from_array(cp.array(value, copy=True))
    if isinstance(value, Mapping):
        return {key: _copy_state_for_export(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_state_for_export(item) for item in value)
    if isinstance(value, list):
        return [_copy_state_for_export(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported optimizer state value: {type(value).__name__}")


def _restore_state_value(value: Any, parameter: Parameter, name: str) -> Any:
    if isinstance(value, Tensor):
        expected_shape = parameter.shape
        if name == "row_var":
            expected_shape = parameter.shape[:-1]
        elif name == "col_var":
            expected_shape = parameter.shape[:-2] + parameter.shape[-1:]
        if value.shape != expected_shape:
            raise ValueError(
                f"optimizer state {name!r} expected shape {expected_shape}, "
                f"got {value.shape}"
            )
        if value.dtype != parameter.dtype:
            raise ValueError(
                f"optimizer state {name!r} expected dtype {parameter.dtype}, "
                f"got {value.dtype}"
            )
        with cp.cuda.Device(parameter._device_index):
            return cp.array(value._array, copy=True)
    if isinstance(value, Mapping):
        return {
            key: _restore_state_value(item, parameter, str(key))
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_restore_state_value(item, parameter, name) for item in value)
    if isinstance(value, list):
        return [_restore_state_value(item, parameter, name) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported optimizer state value: {type(value).__name__}")
