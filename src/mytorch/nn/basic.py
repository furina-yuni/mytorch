"""Foundational trainable layers, normalization, and regularization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cupy as cp

from mytorch._device import parse_device
from mytorch.tensor import (
    Tensor,
    arange,
    full,
    maximum,
    ones,
    rand,
    stack,
    tensor,
    where,
    zeros,
)

from . import functional as F
from .modules import Linear, Module, Parameter


def _positive(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shape(value: int | Sequence[int]) -> tuple[int, ...]:
    result = (value,) if isinstance(value, int) else tuple(value)
    if not result or any(not isinstance(item, int) or item <= 0 for item in result):
        raise ValueError("normalized_shape must contain positive integers")
    return result


class Bilinear(Module):
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.in1_features = _positive("in1_features", in1_features)
        self.in2_features = _positive("in2_features", in2_features)
        self.out_features = _positive("out_features", out_features)
        bound = 1 / math.sqrt(in1_features)
        self.weight = Parameter(
            (
                rand(
                    out_features, in1_features, in2_features, device=device, dtype=dtype
                )
                * 2
                - 1
            )
            * bound
        )
        self.bias = (
            Parameter((rand(out_features, device=device, dtype=dtype) * 2 - 1) * bound)
            if bias
            else None
        )

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)


class LazyLinear(Module):
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.out_features = _positive("out_features", out_features)
        self.use_bias = bool(bias)
        self.device = device
        self.dtype = dtype
        self.in_features: int | None = None
        self.weight = None
        self.bias = None

    def _check_materialized(self) -> None:
        if self.weight is None:
            raise RuntimeError(
                "LazyLinear must receive an input before parameters are requested"
            )

    def _materialize(self, in_features: int) -> None:
        layer = Linear(
            in_features,
            self.out_features,
            self.use_bias,
            device=self.device,
            dtype=self.dtype,
        )
        self.in_features = in_features
        self.weight = layer.weight
        self.bias = layer.bias

    def to(self, device: str | int | None = None, dtype: Any = None) -> LazyLinear:
        if self.weight is None:
            if device is not None:
                self.device = device
            if dtype is not None:
                self.dtype = dtype
            return self
        return super().to(device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim < 1:
            raise ValueError("LazyLinear expects an input with at least one dimension")
        if self.weight is None:
            self._materialize(input.shape[-1])
        elif input.shape[-1] != self.in_features:
            raise ValueError(
                "LazyLinear input feature size changed after materialization"
            )
        return F.linear(input, self.weight, self.bias)


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.num_embeddings = _positive("num_embeddings", num_embeddings)
        self.embedding_dim = _positive("embedding_dim", embedding_dim)
        if padding_idx is not None:
            padding_idx = (
                padding_idx + num_embeddings if padding_idx < 0 else padding_idx
            )
            if padding_idx < 0 or padding_idx >= num_embeddings:
                raise ValueError("padding_idx is out of range")
        self.padding_idx = padding_idx
        with cp.cuda.Device(parse_device(device)):
            values = cp.random.normal(0, 1, (num_embeddings, embedding_dim)).astype(
                dtype
            )
            if padding_idx is not None:
                values[padding_idx] = 0
        self.weight = Parameter(values, device=device)

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype.kind not in "iu":
            raise TypeError("Embedding indices must be integer Tensors")
        if input.numel() and (
            bool((input._array.min() < 0).item())
            or bool((input._array.max() >= self.num_embeddings).item())
        ):
            raise IndexError("Embedding index is out of range")
        return F.embedding(input, self.weight, self.padding_idx)


class EmbeddingBag(Embedding):
    def __init__(self, *args: Any, mode: str = "mean", **kwargs: Any) -> None:
        if mode not in {"sum", "mean", "max"}:
            raise ValueError("EmbeddingBag mode must be 'sum', 'mean', or 'max'")
        self.mode = mode
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, offsets: Tensor) -> Tensor:
        if input.ndim != 1 or offsets.ndim != 1 or offsets.dtype.kind not in "iu":
            raise ValueError("EmbeddingBag expects 1D indices and integer offsets")
        if (
            not offsets.numel()
            or int(offsets[0].item()) != 0
            or bool(cp.any(offsets._array[1:] < offsets._array[:-1]).item())
            or bool(
                cp.any((offsets._array < 0) | (offsets._array > input.shape[0])).item()
            )
        ):
            raise ValueError("EmbeddingBag offsets must start at zero and be ordered")
        outputs = []
        values = F.embedding(input, self.weight, self.padding_idx)
        positions = arange(input.shape[0], dtype=cp.int64, device=input.device)
        for bag in range(offsets.shape[0]):
            end = (
                offsets[bag + 1]
                if bag + 1 < offsets.shape[0]
                else full((), input.shape[0], dtype=cp.int64, device=input.device)
            )
            mask = (positions >= offsets[bag]) * (positions < end)
            count = mask.sum()
            if self.mode == "sum":
                outputs.append((values * mask.unsqueeze(1)).sum(0))
            elif self.mode == "mean":
                outputs.append((values * mask.unsqueeze(1)).sum(0) / maximum(count, 1))
            else:
                masked = where(mask.unsqueeze(1), values, -cp.inf)
                outputs.append(
                    where(
                        count > 0,
                        masked.max(0),
                        zeros(
                            self.embedding_dim,
                            device=input.device,
                            dtype=self.weight.dtype,
                        ),
                    )
                )
        return stack(outputs)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training)


class _FeatureDropout(Dropout):
    def forward(self, input: Tensor) -> Tensor:
        return F.feature_dropout(input, self.p, self.training)


class Dropout1d(_FeatureDropout):
    pass


class Dropout2d(_FeatureDropout):
    pass


class Dropout3d(_FeatureDropout):
    pass


class StochasticDepth(Module):
    def __init__(self, p: float, mode: str = "row") -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        if mode not in {"row", "batch"}:
            raise ValueError("mode must be 'row' or 'batch'")
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return input
        if self.p == 1:
            return input * 0
        shape = (
            (1,) * input.ndim
            if self.mode == "batch"
            else (input.shape[0],) + (1,) * (input.ndim - 1)
        )
        with cp.cuda.Device(input._device_index):
            mask = tensor(cp.random.random(shape) >= self.p, device=input.device)
        return input * mask / (1 - self.p)


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
        self.normalized_shape = _shape(normalized_shape)
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
        self.normalized_shape = _shape(normalized_shape)
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
        self.num_groups = _positive("num_groups", num_groups)
        self.num_channels = _positive("num_channels", num_channels)
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
        self.num_features = _positive("num_features", num_features)
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
        self.num_features = _positive("num_features", num_features)
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
