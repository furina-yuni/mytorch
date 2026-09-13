"""ASGD and resilient backpropagation optimizers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cupy as cp

from ._common import gradient as prepare_gradient
from ._common import validate_common
from .optimizer import Optimizer, require_real


class ASGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.01,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "lambd": lambd,
                "alpha": alpha,
                "t0": t0,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        validate_common(group)
        group["lambd"] = require_real("lambd", group["lambd"])
        group["alpha"] = require_real("alpha", group["alpha"])
        group["t0"] = require_real("t0", group["t0"])

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = prepare_gradient(parameter, group["maximize"])
                if group["weight_decay"]:
                    gradient = gradient + group["weight_decay"] * parameter._array
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state.update(step=0, ax=cp.zeros_like(parameter._array))
                state["step"] += 1
                step = state["step"]
                eta = group["lr"] / (
                    (1 + group["lambd"] * group["lr"] * step) ** group["alpha"]
                )
                update = -eta * gradient - eta * group["lambd"] * parameter._array
                parameter._optimizer_update(update)
                mu = 1 / max(1.0, step - group["t0"])
                state["ax"] += mu * (parameter._array - state["ax"])


class Rprop(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.01,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50.0),
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {"lr": lr, "etas": etas, "step_sizes": step_sizes, "maximize": maximize},
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        group["lr"] = require_real("lr", group["lr"], strict=True)
        if not isinstance(group["etas"], tuple) or len(group["etas"]) != 2:
            raise TypeError("etas must be a pair")
        decrease = require_real("etas[0]", group["etas"][0], strict=True)
        increase = require_real("etas[1]", group["etas"][1], strict=True)
        if decrease >= 1 or increase <= 1:
            raise ValueError("etas must satisfy 0 < etas[0] < 1 < etas[1]")
        group["etas"] = (decrease, increase)
        if not isinstance(group["step_sizes"], tuple) or len(group["step_sizes"]) != 2:
            raise TypeError("step_sizes must be a pair")
        minimum = require_real("step_sizes[0]", group["step_sizes"][0], strict=True)
        maximum = require_real("step_sizes[1]", group["step_sizes"][1], strict=True)
        if minimum > maximum:
            raise ValueError("step_sizes minimum cannot exceed maximum")
        group["step_sizes"] = (minimum, maximum)
        if not isinstance(group["maximize"], bool):
            raise TypeError("maximize must be a bool")

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = prepare_gradient(parameter, group["maximize"])
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state.update(
                        step=0,
                        prev=cp.zeros_like(parameter._array),
                        step_size=cp.full_like(parameter._array, group["lr"]),
                    )
                state["step"] += 1
                product = gradient * state["prev"]
                decrease, increase = group["etas"]
                factor = cp.where(
                    product > 0, increase, cp.where(product < 0, decrease, 1)
                )
                state["step_size"] *= factor
                cp.clip(
                    state["step_size"],
                    group["step_sizes"][0],
                    group["step_sizes"][1],
                    out=state["step_size"],
                )
                effective = cp.where(product < 0, 0, gradient)
                parameter._optimizer_update(-cp.sign(effective) * state["step_size"])
                state["prev"][...] = effective
