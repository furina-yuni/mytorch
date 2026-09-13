"""Adam-family dense GPU optimizers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import cupy as cp

from .optimizer import Optimizer, require_betas, require_real


def _validate_adam(group: dict[str, Any]) -> None:
    group["lr"] = require_real("lr", group["lr"], strict=True)
    group["betas"] = require_betas(group["betas"])
    group["eps"] = require_real("eps", group["eps"], strict=True)
    group["weight_decay"] = require_real("weight_decay", group["weight_decay"])
    if not isinstance(group["maximize"], bool):
        raise TypeError("maximize must be a bool")


def _grad(parameter, maximize: bool):
    gradient = parameter.grad._array
    return -gradient if maximize else gradient


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        _validate_adam(group)
        if not isinstance(group["amsgrad"], bool):
            raise TypeError("amsgrad must be a bool")

    def _decoupled_weight_decay(self) -> bool:
        return False

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = _grad(parameter, group["maximize"])
                if group["weight_decay"] and not self._decoupled_weight_decay():
                    gradient = gradient + group["weight_decay"] * parameter._array
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state.update(
                        step=0,
                        exp_avg=cp.zeros_like(parameter._array),
                        exp_avg_sq=cp.zeros_like(parameter._array),
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = cp.zeros_like(parameter._array)
                state["step"] += 1
                beta1, beta2 = group["betas"]
                state["exp_avg"] *= beta1
                state["exp_avg"] += (1 - beta1) * gradient
                state["exp_avg_sq"] *= beta2
                state["exp_avg_sq"] += (1 - beta2) * cp.square(gradient)
                variance = state["exp_avg_sq"]
                if group["amsgrad"]:
                    cp.maximum(
                        state["max_exp_avg_sq"], variance, out=state["max_exp_avg_sq"]
                    )
                    variance = state["max_exp_avg_sq"]
                step = state["step"]
                first = 1 - beta1**step
                second = 1 - beta2**step
                direction = (state["exp_avg"] / first) / (
                    cp.sqrt(variance / second) + group["eps"]
                )
                update = -group["lr"] * direction
                if self._decoupled_weight_decay() and group["weight_decay"]:
                    update -= group["lr"] * group["weight_decay"] * parameter._array
                parameter._optimizer_update(update)


class AdamW(Adam):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, maximize)

    def _decoupled_weight_decay(self) -> bool:
        return True


class Adamax(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        _validate_adam(group)

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = _grad(parameter, group["maximize"])
                if group["weight_decay"]:
                    gradient = gradient + group["weight_decay"] * parameter._array
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state.update(
                        step=0,
                        exp_avg=cp.zeros_like(parameter._array),
                        exp_inf=cp.zeros_like(parameter._array),
                    )
                state["step"] += 1
                beta1, beta2 = group["betas"]
                state["exp_avg"] *= beta1
                state["exp_avg"] += (1 - beta1) * gradient
                cp.maximum(
                    beta2 * state["exp_inf"],
                    cp.abs(gradient) + group["eps"],
                    out=state["exp_inf"],
                )
                direction = state["exp_avg"] / (
                    (1 - beta1 ** state["step"]) * state["exp_inf"]
                )
                parameter._optimizer_update(-group["lr"] * direction)


class NAdam(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "momentum_decay": momentum_decay,
                "decoupled_weight_decay": decoupled_weight_decay,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        _validate_adam(group)
        group["momentum_decay"] = require_real(
            "momentum_decay", group["momentum_decay"]
        )
        if not isinstance(group["decoupled_weight_decay"], bool):
            raise TypeError("decoupled_weight_decay must be a bool")

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = _grad(parameter, group["maximize"])
                if group["weight_decay"] and not group["decoupled_weight_decay"]:
                    gradient = gradient + group["weight_decay"] * parameter._array
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state.update(
                        step=0,
                        mu_product=1.0,
                        exp_avg=cp.zeros_like(parameter._array),
                        exp_avg_sq=cp.zeros_like(parameter._array),
                    )
                state["step"] += 1
                step = state["step"]
                beta1, beta2 = group["betas"]
                decay = group["momentum_decay"]
                mu = beta1 * (1 - 0.5 * 0.96 ** (step * decay))
                mu_next = beta1 * (1 - 0.5 * 0.96 ** ((step + 1) * decay))
                state["mu_product"] *= mu
                state["exp_avg"] *= beta1
                state["exp_avg"] += (1 - beta1) * gradient
                state["exp_avg_sq"] *= beta2
                state["exp_avg_sq"] += (1 - beta2) * cp.square(gradient)
                product = state["mu_product"]
                momentum = mu_next * state["exp_avg"] / (1 - product * mu_next)
                momentum += (1 - mu) * gradient / (1 - product)
                variance = state["exp_avg_sq"] / (1 - beta2**step)
                update = -group["lr"] * momentum / (cp.sqrt(variance) + group["eps"])
                if group["decoupled_weight_decay"] and group["weight_decay"]:
                    update -= group["lr"] * group["weight_decay"] * parameter._array
                parameter._optimizer_update(update)


class RAdam(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "decoupled_weight_decay": decoupled_weight_decay,
                "maximize": maximize,
            },
        )

    def _validate_group(self, group: dict[str, Any]) -> None:
        _validate_adam(group)
        if not isinstance(group["decoupled_weight_decay"], bool):
            raise TypeError("decoupled_weight_decay must be a bool")

    def step(self) -> None:
        for parameter, group in self._parameter_groups():
            if parameter.grad is None:
                continue
            with cp.cuda.Device(parameter._device_index):
                gradient = _grad(parameter, group["maximize"])
                if group["weight_decay"] and not group["decoupled_weight_decay"]:
                    gradient = gradient + group["weight_decay"] * parameter._array
                state = self.state.setdefault(id(parameter), {})
                if not state:
                    state.update(
                        step=0,
                        exp_avg=cp.zeros_like(parameter._array),
                        exp_avg_sq=cp.zeros_like(parameter._array),
                    )
                state["step"] += 1
                step = state["step"]
                beta1, beta2 = group["betas"]
                state["exp_avg"] *= beta1
                state["exp_avg"] += (1 - beta1) * gradient
                state["exp_avg_sq"] *= beta2
                state["exp_avg_sq"] += (1 - beta2) * cp.square(gradient)
                bias1 = 1 - beta1**step
                rho_inf = 2 / (1 - beta2) - 1
                beta2_power = beta2**step
                rho = rho_inf - 2 * step * beta2_power / (1 - beta2_power)
                if rho > 5:
                    rect = math.sqrt(
                        (1 - beta2_power)
                        * (rho - 4)
                        / (rho_inf - 4)
                        * (rho - 2)
                        / rho
                        * rho_inf
                        / (rho_inf - 2)
                    )
                    direction = (
                        rect
                        * state["exp_avg"]
                        / (bias1 * (cp.sqrt(state["exp_avg_sq"]) + group["eps"]))
                    )
                else:
                    direction = state["exp_avg"] / bias1
                update = -group["lr"] * direction
                if group["decoupled_weight_decay"] and group["weight_decay"]:
                    update -= group["lr"] * group["weight_decay"] * parameter._array
                parameter._optimizer_update(update)
