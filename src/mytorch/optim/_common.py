"""Shared validation and gradient preparation for dense optimizers."""

from __future__ import annotations

from typing import Any

from .optimizer import require_betas, require_real


def validate_common(group: dict[str, Any]) -> None:
    group["lr"] = require_real("lr", group["lr"], strict=True)
    group["weight_decay"] = require_real("weight_decay", group["weight_decay"])
    if not isinstance(group["maximize"], bool):
        raise TypeError("maximize must be a bool")


def validate_adam(group: dict[str, Any]) -> None:
    validate_common(group)
    group["betas"] = require_betas(group["betas"])
    group["eps"] = require_real("eps", group["eps"], strict=True)


def gradient(parameter, maximize: bool):
    value = parameter.grad._array
    return -value if maximize else value


def gradient_with_weight_decay(parameter, group: dict[str, Any]):
    value = gradient(parameter, group["maximize"])
    if group["weight_decay"]:
        value = value + group["weight_decay"] * parameter._array
    return value
