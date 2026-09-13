"""Lion and Adafactor optimizers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import cupy as cp

from ._common import gradient as prepare_gradient
from ._common import validate_common
from .optimizer import Optimizer, require_betas, require_real


class Lion(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        validate_common(group)
        group["betas"] = require_betas(group["betas"])

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = prepare_gradient(parameter, group["maximize"])
                state = self.state.setdefault(id(parameter), {})
                momentum = state.setdefault("exp_avg", cp.zeros_like(parameter._array))
                beta1, beta2 = group["betas"]
                direction = cp.sign(beta1 * momentum + (1 - beta1) * gradient)
                update = -group["lr"] * direction
                if group["weight_decay"]:
                    update -= group["lr"] * group["weight_decay"] * parameter._array
                parameter._optimizer_update(update)
                momentum *= beta2
                momentum += (1 - beta2) * gradient


class Adafactor(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 0.01,
        beta2_decay: float = -0.8,
        eps: tuple[float | None, float] = (None, 0.001),
        d: float = 1.0,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "beta2_decay": beta2_decay,
                "eps": eps,
                "d": d,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        validate_common(group)
        if not isinstance(group["beta2_decay"], (int, float)) or isinstance(
            group["beta2_decay"], bool
        ):
            raise TypeError("beta2_decay must be a real number")
        group["beta2_decay"] = float(group["beta2_decay"])
        if group["beta2_decay"] > 0:
            raise ValueError("beta2_decay must be non-positive")
        if not isinstance(group["eps"], tuple) or len(group["eps"]) != 2:
            raise TypeError("eps must be a pair")
        eps1 = group["eps"][0]
        if eps1 is not None:
            eps1 = require_real("eps[0]", eps1)
        eps2 = require_real("eps[1]", group["eps"][1], strict=True)
        group["eps"] = (eps1, eps2)
        group["d"] = require_real("d", group["d"], strict=True)

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = prepare_gradient(parameter, group["maximize"])
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state["step"] = 0
                    if parameter.ndim >= 2:
                        state["row_var"] = cp.zeros(
                            parameter.shape[:-1], dtype=parameter.dtype
                        )
                        state["col_var"] = cp.zeros(
                            parameter.shape[:-2] + parameter.shape[-1:],
                            dtype=parameter.dtype,
                        )
                    else:
                        state["variance"] = cp.zeros_like(parameter._array)
                state["step"] += 1
                step = state["step"]
                beta2 = 1 - step ** group["beta2_decay"]
                squared = cp.square(gradient)
                eps1, eps2 = group["eps"]
                stabilizer = cp.finfo(parameter.dtype).tiny if eps1 is None else eps1
                if parameter.ndim >= 2:
                    row = state["row_var"]
                    col = state["col_var"]
                    row *= beta2
                    row += (1 - beta2) * squared.mean(axis=-1)
                    col *= beta2
                    col += (1 - beta2) * squared.mean(axis=-2)
                    row_mean = row.mean(axis=-1, keepdims=True)
                    estimate = (row / cp.maximum(row_mean, stabilizer))[..., :, None]
                    estimate = estimate * col[..., None, :]
                else:
                    estimate = state["variance"]
                    estimate *= beta2
                    estimate += (1 - beta2) * squared
                direction = gradient / cp.maximum(cp.sqrt(estimate), stabilizer)
                rms_update = float(cp.sqrt(cp.mean(cp.square(direction))).item())
                direction /= max(1.0, rms_update / group["d"])
                rms_parameter = float(
                    cp.sqrt(cp.mean(cp.square(parameter._array))).item()
                )
                alpha = max(eps2, rms_parameter) * min(group["lr"], 1 / math.sqrt(step))
                update = -alpha * direction
                if group["weight_decay"]:
                    update -= group["lr"] * group["weight_decay"] * parameter._array
                parameter._optimizer_update(update)
