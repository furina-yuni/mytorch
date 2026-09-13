"""Stochastic gradient descent."""

from __future__ import annotations

from collections.abc import Iterable

import cupy as cp

from mytorch.nn.modules import Parameter


class SGD:
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if not isinstance(lr, (int, float)) or isinstance(lr, bool) or lr <= 0:
            raise ValueError("lr must be a positive real number")
        if (
            not isinstance(momentum, (int, float))
            or isinstance(momentum, bool)
            or momentum < 0
        ):
            raise ValueError("momentum must be a non-negative real number")
        if (
            not isinstance(weight_decay, (int, float))
            or isinstance(weight_decay, bool)
            or weight_decay < 0
        ):
            raise ValueError("weight_decay must be a non-negative real number")
        unique: list[Parameter] = []
        seen: set[int] = set()
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError("SGD parameters must be Parameter objects")
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique.append(parameter)
        if not unique:
            raise ValueError("SGD received no parameters")
        self.parameters = tuple(unique)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._momentum_buffers: dict[int, cp.ndarray] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        if not isinstance(set_to_none, bool):
            raise TypeError("set_to_none must be a bool")
        for parameter in self.parameters:
            parameter._clear_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                direction = parameter.grad._array
                if self.weight_decay:
                    direction = direction + self.weight_decay * parameter._array
                if self.momentum:
                    buffer = self._momentum_buffers.get(id(parameter))
                    if buffer is None:
                        buffer = cp.array(direction, copy=True)
                    else:
                        buffer = self.momentum * buffer + direction
                    self._momentum_buffers[id(parameter)] = buffer
                    direction = buffer
                parameter._optimizer_update(-self.lr * direction)
