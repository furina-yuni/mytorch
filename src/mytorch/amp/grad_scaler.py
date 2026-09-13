"""Dynamic loss scaling for float16 mixed-precision training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import cupy as cp

from mytorch.optim import Optimizer
from mytorch.tensor import Tensor


def _positive(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class GradScaler:
    """Scale losses dynamically and skip unsafe optimizer updates."""

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self._scale = _positive("init_scale", init_scale)
        self.growth_factor = _positive("growth_factor", growth_factor)
        self.backoff_factor = _positive("backoff_factor", backoff_factor)
        if self.growth_factor <= 1:
            raise ValueError("growth_factor must be greater than 1")
        if self.backoff_factor >= 1:
            raise ValueError("backoff_factor must be less than 1")
        if (
            not isinstance(growth_interval, int)
            or isinstance(growth_interval, bool)
            or growth_interval <= 0
        ):
            raise ValueError("growth_interval must be a positive integer")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self.growth_interval = growth_interval
        self.enabled = enabled
        self._growth_tracker = 0
        self._stage = "ready"
        self._found_inf = False

    def get_scale(self) -> float:
        return self._scale

    def scale(self, output: Tensor) -> Tensor:
        if not isinstance(output, Tensor) or output.dtype.kind != "f":
            raise TypeError("GradScaler.scale expects a floating-point Tensor")
        return output * self._scale if self.enabled else output

    def unscale_(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a mytorch.optim.Optimizer")
        if not self.enabled:
            return
        if self._stage != "ready":
            raise RuntimeError("unscale_() may be called only once before step()")
        inverse = 1.0 / self._scale
        flags: dict[int, cp.ndarray] = {}
        for parameter in optimizer.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            device = parameter._device_index
            with cp.cuda.Device(device):
                finite = cp.all(cp.isfinite(gradient._array))
                flags[device] = (
                    finite if device not in flags else flags[device] & finite
                )
                gradient._array[...] *= inverse
        self._found_inf = any(not bool(flag.item()) for flag in flags.values())
        self._stage = "unscaled"

    def step(self, optimizer: Optimizer) -> bool:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a mytorch.optim.Optimizer")
        if not self.enabled:
            optimizer.step()
            return True
        if self._stage == "stepped":
            raise RuntimeError("GradScaler.step() was already called before update()")
        if self._stage == "ready":
            self.unscale_(optimizer)
        updated = not self._found_inf
        if updated:
            optimizer.step()
        self._stage = "stepped"
        return updated

    def update(self, new_scale: float | None = None) -> None:
        if not self.enabled:
            return
        if self._stage != "stepped":
            raise RuntimeError("GradScaler.update() requires a preceding step()")
        if new_scale is not None:
            self._scale = _positive("new_scale", new_scale)
            self._growth_tracker = 0
        elif self._found_inf:
            self._scale *= self.backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self._scale *= self.growth_factor
                self._growth_tracker = 0
        self._stage = "ready"
        self._found_inf = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "scale": self._scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "growth_tracker": self._growth_tracker,
            "enabled": self.enabled,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != set(
            self.state_dict()
        ):
            raise ValueError("GradScaler state_dict is invalid")
        if state_dict["version"] != 1:
            raise ValueError("unsupported GradScaler state version")
        self._scale = _positive("scale", state_dict["scale"])
        self.growth_factor = _positive("growth_factor", state_dict["growth_factor"])
        self.backoff_factor = _positive("backoff_factor", state_dict["backoff_factor"])
        interval = state_dict["growth_interval"]
        tracker = state_dict["growth_tracker"]
        if not isinstance(interval, int) or interval <= 0:
            raise ValueError("invalid growth_interval in GradScaler state")
        if not isinstance(tracker, int) or tracker < 0:
            raise ValueError("invalid growth_tracker in GradScaler state")
        if not isinstance(state_dict["enabled"], bool):
            raise TypeError("GradScaler enabled state must be a bool")
        self.growth_interval = interval
        self._growth_tracker = tracker
        self.enabled = state_dict["enabled"]
        self._stage = "ready"
        self._found_inf = False


__all__ = ["GradScaler"]
