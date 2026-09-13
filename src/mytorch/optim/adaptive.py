"""Classical adaptive dense optimizers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cupy as cp

from ._common import gradient_with_weight_decay, validate_common
from .optimizer import Optimizer, require_real


class Adagrad(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "lr_decay": lr_decay,
                "weight_decay": weight_decay,
                "initial_accumulator_value": initial_accumulator_value,
                "eps": eps,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group):
        validate_common(group)
        group["lr_decay"] = require_real("lr_decay", group["lr_decay"])
        group["initial_accumulator_value"] = require_real(
            "initial_accumulator_value", group["initial_accumulator_value"]
        )
        group["eps"] = require_real("eps", group["eps"], strict=True)

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = gradient_with_weight_decay(parameter, group)
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state["step"] = 0
                    state["sum"] = cp.full_like(
                        parameter._array, group["initial_accumulator_value"]
                    )
                state["step"] += 1
                state["sum"] += cp.square(gradient)
                learning_rate = group["lr"] / (
                    1 + (state["step"] - 1) * group["lr_decay"]
                )
                parameter._optimizer_update(
                    -learning_rate * gradient / (cp.sqrt(state["sum"]) + group["eps"])
                )


class RMSprop(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "alpha": alpha,
                "eps": eps,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "centered": centered,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group):
        validate_common(group)
        group["alpha"] = require_real("alpha", group["alpha"])
        if group["alpha"] >= 1:
            raise ValueError("alpha must be in [0, 1)")
        group["eps"] = require_real("eps", group["eps"], strict=True)
        group["momentum"] = require_real("momentum", group["momentum"])
        if not isinstance(group["centered"], bool):
            raise TypeError("centered must be a bool")

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = gradient_with_weight_decay(parameter, group)
                state = self.state.setdefault(id(parameter), {})
                square_avg = state.setdefault(
                    "square_avg", cp.zeros_like(parameter._array)
                )
                square_avg *= group["alpha"]
                square_avg += (1 - group["alpha"]) * cp.square(gradient)
                average = square_avg
                if group["centered"]:
                    grad_avg = state.setdefault(
                        "grad_avg", cp.zeros_like(parameter._array)
                    )
                    grad_avg *= group["alpha"]
                    grad_avg += (1 - group["alpha"]) * gradient
                    average = square_avg - cp.square(grad_avg)
                denominator = cp.sqrt(cp.maximum(average, 0)) + group["eps"]
                direction = gradient / denominator
                if group["momentum"]:
                    buffer = state.setdefault(
                        "momentum_buffer", cp.zeros_like(parameter._array)
                    )
                    buffer *= group["momentum"]
                    buffer += direction
                    direction = buffer
                parameter._optimizer_update(-group["lr"] * direction)


class Adadelta(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "rho": rho,
                "eps": eps,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group):
        validate_common(group)
        group["rho"] = require_real("rho", group["rho"])
        if group["rho"] >= 1:
            raise ValueError("rho must be in [0, 1)")
        group["eps"] = require_real("eps", group["eps"], strict=True)

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = gradient_with_weight_decay(parameter, group)
                state = self.state.setdefault(id(parameter), {})
                square_avg = state.setdefault(
                    "square_avg", cp.zeros_like(parameter._array)
                )
                acc_delta = state.setdefault(
                    "acc_delta", cp.zeros_like(parameter._array)
                )
                rho = group["rho"]
                square_avg *= rho
                square_avg += (1 - rho) * cp.square(gradient)
                delta = (
                    cp.sqrt(acc_delta + group["eps"])
                    / cp.sqrt(square_avg + group["eps"])
                    * gradient
                )
                acc_delta *= rho
                acc_delta += (1 - rho) * cp.square(delta)
                parameter._optimizer_update(-group["lr"] * delta)
