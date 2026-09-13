"""Shared optimizer parameter-group and state management."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Any

from mytorch.nn._modules.base import Parameter


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

    def step(self) -> None:
        raise NotImplementedError
