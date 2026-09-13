"""Learning-rate schedulers with checkpointable state."""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .optimizer import Optimizer, require_real


def _require_optimizer(optimizer: Optimizer) -> None:
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a mytorch.optim.Optimizer")


def _group_values(value: float | Sequence[float], count: int, name: str) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != count:
            raise ValueError(f"{name} must contain one value per parameter group")
        return [require_real(name, item) for item in value]
    return [require_real(name, value)] * count


class LRScheduler:
    """Base class for epoch- or step-driven learning-rate schedules."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        _require_optimizer(optimizer)
        if (
            not isinstance(last_epoch, int)
            or isinstance(last_epoch, bool)
            or last_epoch < -1
        ):
            raise ValueError(
                "last_epoch must be an integer greater than or equal to -1"
            )
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._last_lr = [float(group["lr"]) for group in optimizer.param_groups]

    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        elif not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer or None")
        else:
            self.last_epoch = epoch
        values = self.get_lr()
        if len(values) != len(self.optimizer.param_groups):
            raise RuntimeError("scheduler returned the wrong number of learning rates")
        for group, value in zip(self.optimizer.param_groups, values, strict=True):
            group["lr"] = require_real("lr", value)
        self._last_lr = [float(value) for value in values]

    def _extra_state(self) -> dict[str, Any]:
        excluded = {"optimizer", "base_lrs", "last_epoch", "_last_lr"}
        return {
            name: deepcopy(value)
            for name, value in self.__dict__.items()
            if name not in excluded
        }

    def _load_extra_state(self, state: Mapping[str, Any]) -> None:
        expected = set(self._extra_state())
        if set(state) != expected:
            raise ValueError("scheduler extra state has missing or unexpected fields")
        for name, value in state.items():
            setattr(self, name, deepcopy(value))

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "scheduler_type": type(self).__name__,
            "last_epoch": self.last_epoch,
            "base_lrs": list(self.base_lrs),
            "last_lr": list(self._last_lr),
            "extra": deepcopy(self._extra_state()),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("scheduler state_dict must be a mapping")
        expected = {
            "version",
            "scheduler_type",
            "last_epoch",
            "base_lrs",
            "last_lr",
            "extra",
        }
        if set(state_dict) != expected or state_dict["version"] != 1:
            raise ValueError("scheduler state_dict has an invalid format or version")
        if state_dict["scheduler_type"] != type(self).__name__:
            raise ValueError("scheduler type does not match the checkpoint")
        base_lrs = _group_values(
            state_dict["base_lrs"], len(self.optimizer.param_groups), "base_lrs"
        )
        last_lrs = _group_values(
            state_dict["last_lr"], len(self.optimizer.param_groups), "last_lr"
        )
        last_epoch = state_dict["last_epoch"]
        if (
            not isinstance(last_epoch, int)
            or isinstance(last_epoch, bool)
            or last_epoch < -1
        ):
            raise ValueError("scheduler checkpoint has an invalid last_epoch")
        extra = state_dict["extra"]
        if not isinstance(extra, Mapping):
            raise TypeError("scheduler extra state must be a mapping")
        self._load_extra_state(extra)
        self.base_lrs = base_lrs
        self.last_epoch = last_epoch
        self._last_lr = last_lrs
        for group, current_lr in zip(
            self.optimizer.param_groups, last_lrs, strict=True
        ):
            group["lr"] = current_lr


class StepLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if (
            not isinstance(step_size, int)
            or isinstance(step_size, bool)
            or step_size <= 0
        ):
            raise ValueError("step_size must be a positive integer")
        self.step_size = step_size
        self.gamma = require_real("gamma", gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [
            lr * self.gamma ** (self.last_epoch // self.step_size)
            for lr in self.base_lrs
        ]


class MultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Sequence[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(milestones, Sequence) or isinstance(milestones, (str, bytes)):
            raise TypeError("milestones must be a sequence of integers")
        self.milestones = list(milestones)
        if (
            not self.milestones
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in self.milestones
            )
            or self.milestones != sorted(self.milestones)
        ):
            raise ValueError("milestones must be a non-empty sorted sequence")
        self.gamma = require_real("gamma", gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        power = bisect.bisect_right(self.milestones, self.last_epoch)
        return [lr * self.gamma**power for lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(
        self, optimizer: Optimizer, gamma: float, last_epoch: int = -1
    ) -> None:
        self.gamma = require_real("gamma", gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [lr * self.gamma**self.last_epoch for lr in self.base_lrs]


class LinearLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        self.start_factor = require_real("start_factor", start_factor, strict=True)
        self.end_factor = require_real("end_factor", end_factor, strict=True)
        if self.start_factor > 1 or self.end_factor > 1:
            raise ValueError("start_factor and end_factor must be in (0, 1]")
        if (
            not isinstance(total_iters, int)
            or isinstance(total_iters, bool)
            or total_iters < 0
        ):
            raise ValueError("total_iters must be a non-negative integer")
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = max(self.last_epoch, 0)
        progress = 1.0 if self.total_iters == 0 else min(epoch / self.total_iters, 1.0)
        factor = self.start_factor + (self.end_factor - self.start_factor) * progress
        return [lr * factor for lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(T_max, int) or isinstance(T_max, bool) or T_max <= 0:
            raise ValueError("T_max must be a positive integer")
        self.T_max = T_max
        self.eta_min = require_real("eta_min", eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        factor = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return [self.eta_min + (lr - self.eta_min) * factor for lr in self.base_lrs]


class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(T_0, int) or isinstance(T_0, bool) or T_0 <= 0:
            raise ValueError("T_0 must be a positive integer")
        if not isinstance(T_mult, int) or isinstance(T_mult, bool) or T_mult < 1:
            raise ValueError("T_mult must be an integer of at least 1")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = require_real("eta_min", eta_min)
        super().__init__(optimizer, last_epoch)

    def _cycle(self) -> tuple[int, int]:
        position = self.last_epoch
        length = self.T_0
        while position >= length:
            position -= length
            length *= self.T_mult
        return position, length

    def get_lr(self) -> list[float]:
        position, length = self._cycle()
        factor = (1 + math.cos(math.pi * position / length)) / 2
        return [self.eta_min + (lr - self.eta_min) * factor for lr in self.base_lrs]


class SequentialLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: Sequence[LRScheduler],
        milestones: Sequence[int],
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(schedulers, Sequence) or len(schedulers) < 2:
            raise ValueError("schedulers must contain at least two schedulers")
        if any(scheduler.optimizer is not optimizer for scheduler in schedulers):
            raise ValueError("all schedulers must use the same optimizer")
        if len(milestones) != len(schedulers) - 1 or list(milestones) != sorted(
            milestones
        ):
            raise ValueError(
                "milestones must contain one sorted boundary per transition"
            )
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in milestones
        ):
            raise ValueError("SequentialLR milestones must be positive integers")
        self.schedulers = list(schedulers)
        self.milestones = list(milestones)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        index = bisect.bisect_right(self.milestones, self.last_epoch)
        start = 0 if index == 0 else self.milestones[index - 1]
        scheduler = self.schedulers[index]
        scheduler.base_lrs = list(self.base_lrs)
        scheduler.last_epoch = self.last_epoch - start
        values = scheduler.get_lr()
        scheduler._last_lr = list(values)
        return values

    def _extra_state(self) -> dict[str, Any]:
        return {
            "milestones": list(self.milestones),
            "children": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def _load_extra_state(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"milestones", "children"} or not isinstance(
            state["children"], Sequence
        ):
            raise ValueError("SequentialLR child state is invalid")
        if len(state["children"]) != len(self.schedulers):
            raise ValueError("SequentialLR child count does not match")
        for scheduler, child_state in zip(
            self.schedulers, state["children"], strict=True
        ):
            scheduler.load_state_dict(child_state)
        self.milestones = list(state["milestones"])


def _anneal(start: float, end: float, progress: float, strategy: str) -> float:
    if strategy == "linear":
        return start + (end - start) * progress
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


class OneCycleLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float | Sequence[float],
        total_steps: int | None = None,
        *,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        base_momentum: float | Sequence[float] = 0.85,
        max_momentum: float | Sequence[float] = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
    ) -> None:
        if total_steps is None:
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in (epochs, steps_per_epoch)
            ):
                raise ValueError(
                    "provide total_steps or positive epochs and steps_per_epoch"
                )
            total_steps = epochs * steps_per_epoch
        if (
            not isinstance(total_steps, int)
            or isinstance(total_steps, bool)
            or total_steps <= 0
        ):
            raise ValueError("total_steps must be a positive integer")
        if (
            not isinstance(pct_start, (int, float))
            or isinstance(pct_start, bool)
            or not 0 < pct_start < 1
        ):
            raise ValueError("pct_start must be between 0 and 1")
        if anneal_strategy not in {"cos", "linear"}:
            raise ValueError("anneal_strategy must be 'cos' or 'linear'")
        if not isinstance(cycle_momentum, bool) or not isinstance(three_phase, bool):
            raise TypeError("cycle_momentum and three_phase must be bool values")
        self.total_steps = total_steps
        self.pct_start = float(pct_start)
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.three_phase = three_phase
        count = len(optimizer.param_groups)
        self.max_lrs = _group_values(max_lr, count, "max_lr")
        divisor = require_real("div_factor", div_factor, strict=True)
        final_divisor = require_real("final_div_factor", final_div_factor, strict=True)
        self.initial_lrs = [value / divisor for value in self.max_lrs]
        self.min_lrs = [value / final_divisor for value in self.initial_lrs]
        self.base_momentums = _group_values(base_momentum, count, "base_momentum")
        self.max_momentums = _group_values(max_momentum, count, "max_momentum")
        for low, high in zip(self.base_momentums, self.max_momentums, strict=True):
            if low > high:
                raise ValueError("base_momentum cannot exceed max_momentum")
        super().__init__(optimizer, last_epoch)
        self.base_lrs = list(self.initial_lrs)
        if last_epoch == -1:
            self._set_values(self.initial_lrs, self.max_momentums)

    def _values(self) -> tuple[list[float], list[float]]:
        step = min(max(self.last_epoch + 1, 0), self.total_steps)
        rise_end = max(1, round(self.total_steps * self.pct_start))
        if self.three_phase:
            fall_end = min(self.total_steps, rise_end * 2)
            if step <= rise_end:
                progress = step / rise_end
                starts, ends = self.initial_lrs, self.max_lrs
                momentum_starts, momentum_ends = self.max_momentums, self.base_momentums
            elif step <= fall_end:
                progress = (step - rise_end) / max(1, fall_end - rise_end)
                starts, ends = self.max_lrs, self.initial_lrs
                momentum_starts, momentum_ends = self.base_momentums, self.max_momentums
            else:
                progress = (step - fall_end) / max(1, self.total_steps - fall_end)
                starts, ends = self.initial_lrs, self.min_lrs
                momentum_starts = momentum_ends = self.max_momentums
        elif step <= rise_end:
            progress = step / rise_end
            starts, ends = self.initial_lrs, self.max_lrs
            momentum_starts, momentum_ends = self.max_momentums, self.base_momentums
        else:
            progress = (step - rise_end) / max(1, self.total_steps - rise_end)
            starts, ends = self.max_lrs, self.min_lrs
            momentum_starts, momentum_ends = self.base_momentums, self.max_momentums
        lrs = [
            _anneal(a, b, progress, self.anneal_strategy)
            for a, b in zip(starts, ends, strict=True)
        ]
        momentums = [
            _anneal(a, b, progress, self.anneal_strategy)
            for a, b in zip(momentum_starts, momentum_ends, strict=True)
        ]
        return lrs, momentums

    def _set_values(self, lrs: Sequence[float], momentums: Sequence[float]) -> None:
        for group, lr, momentum in zip(
            self.optimizer.param_groups, lrs, momentums, strict=True
        ):
            group["lr"] = lr
            if self.cycle_momentum:
                if "momentum" in group:
                    group["momentum"] = momentum
                elif "betas" in group:
                    group["betas"] = (momentum, group["betas"][1])
                else:
                    raise ValueError(
                        "cycle_momentum requires momentum or betas in optimizer groups"
                    )
        self._last_lr = list(lrs)

    def get_lr(self) -> list[float]:
        if self.last_epoch + 1 > self.total_steps:
            raise ValueError("OneCycleLR stepped more than total_steps")
        return self._values()[0]

    def step(self, epoch: int | None = None) -> None:
        super().step(epoch)
        lrs, momentums = self._values()
        self._set_values(lrs, momentums)


class ReduceLROnPlateau:
    """Reduce learning rates when a monitored metric stops improving."""

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float | Sequence[float] = 0.0,
        eps: float = 1e-8,
    ) -> None:
        _require_optimizer(optimizer)
        if mode not in {"min", "max"} or threshold_mode not in {"rel", "abs"}:
            raise ValueError("invalid mode or threshold_mode")
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
            raise ValueError("patience must be a non-negative integer")
        if not isinstance(cooldown, int) or isinstance(cooldown, bool) or cooldown < 0:
            raise ValueError("cooldown must be a non-negative integer")
        self.optimizer = optimizer
        self.mode = mode
        self.factor = require_real("factor", factor, strict=True)
        if self.factor >= 1:
            raise ValueError("factor must be less than 1")
        self.patience = patience
        self.threshold = require_real("threshold", threshold)
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lrs = _group_values(min_lr, len(optimizer.param_groups), "min_lr")
        self.eps = require_real("eps", eps)
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = -1
        self._last_lr = [float(group["lr"]) for group in optimizer.param_groups]

    def _better(self, metric: float) -> bool:
        if self.mode == "min":
            boundary = (
                self.best * (1 - self.threshold)
                if self.threshold_mode == "rel"
                else self.best - self.threshold
            )
            return metric < boundary
        boundary = (
            self.best * (1 + self.threshold)
            if self.threshold_mode == "rel"
            else self.best + self.threshold
        )
        return metric > boundary

    def step(self, metrics: float) -> None:
        if hasattr(metrics, "item"):
            metrics = metrics.item()
        metric = float(metrics)
        if not math.isfinite(metric):
            metric = math.inf if self.mode == "min" else -math.inf
        self.last_epoch += 1
        if self._better(metric):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            for group, minimum in zip(
                self.optimizer.param_groups, self.min_lrs, strict=True
            ):
                old = float(group["lr"])
                new = max(old * self.factor, minimum)
                if old - new > self.eps:
                    group["lr"] = new
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "scheduler_type": type(self).__name__,
            "mode": self.mode,
            "factor": self.factor,
            "patience": self.patience,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lrs": list(self.min_lrs),
            "eps": self.eps,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
            "last_epoch": self.last_epoch,
            "last_lr": list(self._last_lr),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != set(
            self.state_dict()
        ):
            raise ValueError("ReduceLROnPlateau state_dict is invalid")
        if (
            state_dict["version"] != 1
            or state_dict["scheduler_type"] != type(self).__name__
        ):
            raise ValueError("ReduceLROnPlateau checkpoint type or version differs")
        current_groups = len(self.optimizer.param_groups)
        if (
            len(state_dict["last_lr"]) != current_groups
            or len(state_dict["min_lrs"]) != current_groups
        ):
            raise ValueError("scheduler parameter-group count does not match")
        for name in (
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "eps",
            "best",
            "num_bad_epochs",
            "cooldown_counter",
            "last_epoch",
        ):
            setattr(self, name, deepcopy(state_dict[name]))
        self.min_lrs = list(state_dict["min_lrs"])
        self._last_lr = list(state_dict["last_lr"])
        for group, lr in zip(self.optimizer.param_groups, self._last_lr, strict=True):
            group["lr"] = lr


__all__ = [
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "ExponentialLR",
    "LRScheduler",
    "LinearLR",
    "MultiStepLR",
    "OneCycleLR",
    "ReduceLROnPlateau",
    "SequentialLR",
    "StepLR",
]
