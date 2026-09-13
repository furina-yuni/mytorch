"""Stochastic gradient descent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cupy as cp

from .optimizer import Optimizer, require_real


class SGD(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.001,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "dampening": dampening,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        group["lr"] = require_real("lr", group["lr"], strict=True)
        group["momentum"] = require_real("momentum", group["momentum"])
        group["dampening"] = require_real("dampening", group["dampening"])
        group["weight_decay"] = require_real("weight_decay", group["weight_decay"])
        if not isinstance(group["nesterov"], bool) or not isinstance(
            group["maximize"], bool
        ):
            raise TypeError("nesterov and maximize must be bools")
        if group["nesterov"] and (group["momentum"] <= 0 or group["dampening"] != 0):
            raise ValueError(
                "Nesterov momentum requires momentum > 0 and dampening = 0"
            )

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                direction = parameter.grad._array
                if group["maximize"]:
                    direction = -direction
                if group["weight_decay"]:
                    direction = direction + group["weight_decay"] * parameter._array
                momentum = group["momentum"]
                if momentum:
                    state = self.state.setdefault(id(parameter), {})
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = cp.array(direction, copy=True)
                    else:
                        buffer = (
                            momentum * buffer + (1 - group["dampening"]) * direction
                        )
                    state["momentum_buffer"] = buffer
                    direction = (
                        direction + momentum * buffer if group["nesterov"] else buffer
                    )
                parameter._optimizer_update(-group["lr"] * direction)
