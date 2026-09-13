"""Normalization modules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, full, ones, zeros

from ..functional import layers as F
from .base import Module, Parameter
from .utils import positive, shape


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.normalized_shape = shape(normalized_shape)
        self.eps = eps
        self.weight = (
            Parameter(ones(self.normalized_shape, device=device, dtype=dtype))
            if elementwise_affine
            else None
        )
        self.bias = (
            Parameter(zeros(self.normalized_shape, device=device, dtype=dtype))
            if elementwise_affine and bias
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class RMSNorm(Module):
    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.normalized_shape = shape(normalized_shape)
        self.eps = eps
        self.weight = (
            Parameter(ones(self.normalized_shape, device=device, dtype=dtype))
            if elementwise_affine
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.rms_norm(input, self.normalized_shape, self.weight, self.eps)


class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.num_groups = positive("num_groups", num_groups)
        self.num_channels = positive("num_channels", num_channels)
        if num_channels % num_groups:
            raise ValueError("num_channels must be divisible by num_groups")
        self.eps = eps
        self.weight = (
            Parameter(ones(num_channels, device=device, dtype=dtype))
            if affine
            else None
        )
        self.bias = (
            Parameter(zeros(num_channels, device=device, dtype=dtype))
            if affine
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[1] != self.num_channels:
            raise ValueError("GroupNorm channel count does not match")
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)


class _BatchNorm(Module):
    expected_ndims: tuple[int, ...] = ()

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.num_features = positive("num_features", num_features)
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.weight = (
            Parameter(ones(num_features, device=device, dtype=dtype))
            if affine
            else None
        )
        self.bias = (
            Parameter(zeros(num_features, device=device, dtype=dtype))
            if affine
            else None
        )
        self.register_buffer(
            "running_mean",
            zeros(num_features, device=device, dtype=dtype)
            if track_running_stats
            else None,
        )
        self.register_buffer(
            "running_var",
            ones(num_features, device=device, dtype=dtype)
            if track_running_stats
            else None,
        )
        self.register_buffer(
            "num_batches_tracked",
            full((1,), 0, device=device, dtype=cp.int64)
            if track_running_stats
            else None,
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim not in self.expected_ndims or input.shape[1] != self.num_features:
            raise ValueError(
                f"BatchNorm expects rank {self.expected_ndims} input with "
                f"{self.num_features} channels"
            )
        axes = (0,) + tuple(range(2, input.ndim))
        count = math.prod(input.shape[axis] for axis in axes)
        shape = (1, self.num_features) + (1,) * (input.ndim - 2)
        if self.training or not self.track_running_stats:
            if count <= 1:
                raise ValueError(
                    "BatchNorm training expects more than one value per channel"
                )
            mean_value = input.mean(axes, keepdim=True)
            variance = ((input - mean_value) ** 2).mean(axes, keepdim=True)
            if self.track_running_stats:
                step = int(self.num_batches_tracked.item()) + 1
                self.num_batches_tracked._copy_from(
                    full((1,), step, device=input.device, dtype=cp.int64)
                )
                factor = 1 / step if self.momentum is None else self.momentum
                unbiased = variance._array.reshape(-1) * count / (count - 1)
                self.running_mean._copy_from(
                    (1 - factor) * self.running_mean._array
                    + factor * mean_value._array.reshape(-1)
                )
                self.running_var._copy_from(
                    (1 - factor) * self.running_var._array + factor * unbiased
                )
        else:
            mean_value = self.running_mean.reshape(shape)
            variance = self.running_var.reshape(shape)
        result = (input - mean_value) * (variance + self.eps).rsqrt()
        if self.weight is not None:
            result = result * self.weight.reshape(shape)
        if self.bias is not None:
            result = result + self.bias.reshape(shape)
        return result


class BatchNorm1d(_BatchNorm):
    expected_ndims = (2, 3)


class BatchNorm2d(_BatchNorm):
    expected_ndims = (4,)


class BatchNorm3d(_BatchNorm):
    expected_ndims = (5,)


class _InstanceNorm(Module):
    expected_ndim = 0

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.num_features = positive("num_features", num_features)
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.weight = (
            Parameter(ones(num_features, device=device, dtype=dtype))
            if affine
            else None
        )
        self.bias = (
            Parameter(zeros(num_features, device=device, dtype=dtype))
            if affine
            else None
        )
        self.register_buffer(
            "running_mean",
            zeros(num_features, device=device, dtype=dtype)
            if track_running_stats
            else None,
        )
        self.register_buffer(
            "running_var",
            ones(num_features, device=device, dtype=dtype)
            if track_running_stats
            else None,
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != self.expected_ndim or input.shape[1] != self.num_features:
            raise ValueError("InstanceNorm input shape is invalid")
        dims = tuple(range(2, input.ndim))
        shape = (1, self.num_features) + (1,) * (input.ndim - 2)
        if self.training or not self.track_running_stats:
            mean_value = input.mean(dims, keepdim=True)
            variance = ((input - mean_value) ** 2).mean(dims, keepdim=True)
            if self.track_running_stats:
                batch_mean = mean_value.mean(0).reshape(-1)
                batch_var = variance.mean(0).reshape(-1)
                self.running_mean._copy_from(
                    (1 - self.momentum) * self.running_mean._array
                    + self.momentum * batch_mean._array
                )
                self.running_var._copy_from(
                    (1 - self.momentum) * self.running_var._array
                    + self.momentum * batch_var._array
                )
        else:
            mean_value = self.running_mean.reshape(shape)
            variance = self.running_var.reshape(shape)
        result = (input - mean_value) * (variance + self.eps).rsqrt()
        if self.weight is not None:
            result = result * self.weight.reshape(shape)
        if self.bias is not None:
            result = result + self.bias.reshape(shape)
        return result


class InstanceNorm1d(_InstanceNorm):
    expected_ndim = 3


class InstanceNorm2d(_InstanceNorm):
    expected_ndim = 4


class InstanceNorm3d(_InstanceNorm):
    expected_ndim = 5
