"""N-dimensional convolution implementations."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import cupy as cp

from mytorch import _ops
from mytorch.tensor import Tensor, rand

from .base import Module, Parameter


def _tuple(value: int | Sequence[int], dims: int, name: str) -> tuple[int, ...]:
    result = (value,) * dims if isinstance(value, int) else tuple(value)
    if len(result) != dims or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in result
    ):
        raise ValueError(f"{name} must contain {dims} non-negative integers")
    return result


def _output_shape(
    spatial: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[int, ...]:
    result = tuple(
        (size + 2 * pad - dilation_value * (window - 1) - 1) // step + 1
        for size, window, step, pad, dilation_value in zip(
            spatial, kernel, stride, padding, dilation, strict=True
        )
    )
    if any(size <= 0 for size in result):
        raise ValueError("convolution output has a non-positive spatial dimension")
    return result


_COLUMN_KERNELS: dict[tuple[str, str], cp.RawKernel] = {}


def _column_kernel(dtype: cp.dtype, operation: str) -> cp.RawKernel:
    key = (str(dtype), operation)
    if key in _COLUMN_KERNELS:
        return _COLUMN_KERNELS[key]
    if dtype == cp.float16:
        declaration = "#include <cuda_fp16.h>\ntypedef half scalar_t;"
    elif dtype == cp.float64:
        declaration = "typedef double scalar_t;"
    else:
        declaration = "typedef float scalar_t;"
    if operation == "im2col":
        body = "columns[index] = valid ? input[input_index] : (scalar_t)0;"
        arguments = "const scalar_t* input, scalar_t* columns"
    else:
        body = "if (valid) atomicAdd(input + input_index, columns[index]);"
        arguments = "scalar_t* input, const scalar_t* columns"
    name = f"{operation}_{str(dtype).replace('float', 'f')}"
    code = f"""
    {declaration}
    extern "C" __global__ void {name}(
        {arguments}, const int* spatial, const int* kernel,
        const int* stride, const int* padding, const int* dilation,
        int dims, int batches, int channels, int output_size,
        int kernel_size, long long total) {{
      long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
      if (index >= total) return;
      int feature = index % (channels * kernel_size);
      int row = index / (channels * kernel_size);
      int channel = feature / kernel_size;
      int kernel_flat = feature - channel * kernel_size;
      int output_flat = row % output_size;
      int batch = row / output_size;
      int kernel_coord[3] = {{0, 0, 0}};
      int output_coord[3] = {{0, 0, 0}};
      for (int dim = dims - 1; dim >= 0; --dim) {{
        kernel_coord[dim] = kernel_flat % kernel[dim];
        kernel_flat /= kernel[dim];
        int out_dim = (spatial[dim] + 2 * padding[dim]
          - dilation[dim] * (kernel[dim] - 1) - 1) / stride[dim] + 1;
        output_coord[dim] = output_flat % out_dim;
        output_flat /= out_dim;
      }}
      bool valid = true;
      int input_index = batch * channels + channel;
      for (int dim = 0; dim < dims; ++dim) {{
        int coordinate = output_coord[dim] * stride[dim] - padding[dim]
          + kernel_coord[dim] * dilation[dim];
        valid = valid && coordinate >= 0 && coordinate < spatial[dim];
        input_index = input_index * spatial[dim] + max(coordinate, 0);
      }}
      {body}
    }}
    """
    result = cp.RawKernel(code, name, options=("--std=c++11",))
    _COLUMN_KERNELS[key] = result
    return result


def _column_metadata(
    spatial: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[cp.ndarray, ...]:
    return tuple(
        cp.asarray(value, dtype=cp.int32)
        for value in (spatial, kernel, stride, padding, dilation)
    )


def _im2col(
    source: cp.ndarray,
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    output_spatial: tuple[int, ...],
) -> cp.ndarray:
    rows = source.shape[0] * math.prod(output_spatial)
    kernel_elements = math.prod(kernel)
    columns = cp.empty((rows, source.shape[1] * kernel_elements), dtype=source.dtype)
    metadata = _column_metadata(source.shape[2:], kernel, stride, padding, dilation)
    total = columns.size
    _column_kernel(source.dtype, "im2col")(
        ((total + 255) // 256,),
        (256,),
        (
            cp.ascontiguousarray(source),
            columns,
            *metadata,
            len(kernel),
            source.shape[0],
            source.shape[1],
            math.prod(output_spatial),
            kernel_elements,
            total,
        ),
    )
    return columns


def _col2im(
    columns: cp.ndarray,
    input_shape: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    output_spatial: tuple[int, ...],
) -> cp.ndarray:
    result = cp.zeros(input_shape, dtype=columns.dtype)
    metadata = _column_metadata(input_shape[2:], kernel, stride, padding, dilation)
    total = columns.size
    _column_kernel(columns.dtype, "col2im")(
        ((total + 255) // 256,),
        (256,),
        (
            result,
            cp.ascontiguousarray(columns),
            *metadata,
            len(kernel),
            input_shape[0],
            input_shape[1],
            math.prod(output_spatial),
            math.prod(kernel),
            total,
        ),
    )
    return result


def convolution(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    groups: int,
) -> Tensor:
    dims = input.ndim - 2
    if dims not in {1, 2, 3} or weight.ndim != input.ndim:
        raise ValueError("convolution expects matching 3D, 4D, or 5D input/weight")
    steps = _tuple(stride, dims, "stride")
    pads = _tuple(padding, dims, "padding")
    dilations = _tuple(dilation, dims, "dilation")
    if any(value == 0 for value in steps + dilations):
        raise ValueError("stride and dilation values must be positive")
    if groups <= 0 or input.shape[1] % groups or weight.shape[0] % groups:
        raise ValueError("groups must divide input and output channels")
    if weight.shape[1] != input.shape[1] // groups:
        raise ValueError("convolution weight has the wrong input channel count")
    kernel = weight.shape[2:]
    output_spatial = _output_shape(input.shape[2:], kernel, steps, pads, dilations)
    channels_per_group = input.shape[1] // groups
    outputs_per_group = weight.shape[0] // groups
    kernel_elements = math.prod(kernel)
    state: dict[str, cp.ndarray] = {}

    def forward(source, matrix):
        columns = _im2col(source, kernel, steps, pads, dilations, output_spatial)
        state["columns"] = columns
        output_rows = cp.empty((columns.shape[0], matrix.shape[0]), dtype=source.dtype)
        for group in range(groups):
            in_slice = slice(
                group * channels_per_group * kernel_elements,
                (group + 1) * channels_per_group * kernel_elements,
            )
            out_slice = slice(
                group * outputs_per_group, (group + 1) * outputs_per_group
            )
            output_rows[:, out_slice] = (
                columns[:, in_slice]
                @ matrix[out_slice].reshape(outputs_per_group, -1).T
            )
        shaped = output_rows.reshape(source.shape[0], *output_spatial, matrix.shape[0])
        return cp.moveaxis(shaped, -1, 1)

    def backward(gradient, _result, arrays):
        source, matrix = arrays
        columns = state["columns"]
        grad_rows = cp.moveaxis(gradient, 1, -1).reshape(-1, matrix.shape[0])
        grad_columns = cp.zeros_like(columns)
        grad_weight = cp.zeros_like(matrix)
        for group in range(groups):
            in_slice = slice(
                group * channels_per_group * kernel_elements,
                (group + 1) * channels_per_group * kernel_elements,
            )
            out_slice = slice(
                group * outputs_per_group, (group + 1) * outputs_per_group
            )
            current_gradient = grad_rows[:, out_slice]
            current_weight = matrix[out_slice].reshape(outputs_per_group, -1)
            grad_columns[:, in_slice] = current_gradient @ current_weight
            grad_weight[out_slice] = (
                current_gradient.T @ columns[:, in_slice]
            ).reshape(matrix[out_slice].shape)
        grad_input = _col2im(
            grad_columns,
            source.shape,
            kernel,
            steps,
            pads,
            dilations,
            output_spatial,
        )
        return grad_input, grad_weight

    result = _ops.apply(
        forward,
        input,
        weight,
        backward=backward,
        name=f"conv{dims}d",
    )
    if bias is not None:
        result = result + bias.reshape((1, bias.shape[0]) + (1,) * dims)
    return result


def convolution_transpose(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    output_padding: int | Sequence[int],
    groups: int,
    dilation: int | Sequence[int],
) -> Tensor:
    dims = input.ndim - 2
    if dims not in {1, 2, 3} or weight.ndim != input.ndim:
        raise ValueError("transposed convolution expects matching input/weight rank")
    steps = _tuple(stride, dims, "stride")
    pads = _tuple(padding, dims, "padding")
    extra = _tuple(output_padding, dims, "output_padding")
    dilations = _tuple(dilation, dims, "dilation")
    if any(value == 0 for value in steps + dilations):
        raise ValueError("stride and dilation values must be positive")
    if any(value >= step for value, step in zip(extra, steps, strict=True)):
        raise ValueError("output_padding must be smaller than stride")
    if groups <= 0 or input.shape[1] % groups or weight.shape[0] != input.shape[1]:
        raise ValueError("invalid transposed convolution groups or weight")
    kernel = weight.shape[2:]
    output_channels = weight.shape[1] * groups
    output_spatial = tuple(
        (size - 1) * step - 2 * pad + dilation_value * (window - 1) + value + 1
        for size, step, pad, dilation_value, window, value in zip(
            input.shape[2:], steps, pads, dilations, kernel, extra, strict=True
        )
    )
    if any(size <= 0 for size in output_spatial):
        raise ValueError("transposed convolution output is non-positive")
    full_spatial = tuple(
        size + 2 * pad for size, pad in zip(output_spatial, pads, strict=True)
    )
    inputs_per_group = input.shape[1] // groups
    outputs_per_group = output_channels // groups
    crop = (slice(None), slice(None)) + tuple(
        slice(pad, pad + size) for pad, size in zip(pads, output_spatial, strict=True)
    )

    def forward(source, matrix):
        full = cp.zeros(
            (source.shape[0], output_channels, *full_spatial), dtype=source.dtype
        )
        for kernel_index in itertools.product(*(range(size) for size in kernel)):
            slices = tuple(
                slice(
                    offset * dilation_value,
                    offset * dilation_value + (size - 1) * step + 1,
                    step,
                )
                for offset, dilation_value, size, step in zip(
                    kernel_index, dilations, source.shape[2:], steps, strict=True
                )
            )
            for group in range(groups):
                in_slice = slice(
                    group * inputs_per_group, (group + 1) * inputs_per_group
                )
                out_slice = slice(
                    group * outputs_per_group, (group + 1) * outputs_per_group
                )
                full[(slice(None), out_slice, *slices)] += cp.einsum(
                    "ni...,io->no...",
                    source[:, in_slice],
                    matrix[(in_slice, slice(None), *kernel_index)],
                )
        return full[crop]

    def backward(gradient, _result, arrays):
        source, matrix = arrays
        full_gradient = cp.zeros(
            (source.shape[0], output_channels, *full_spatial), dtype=source.dtype
        )
        full_gradient[crop] = gradient
        grad_input = cp.zeros_like(source)
        grad_weight = cp.zeros_like(matrix)
        for kernel_index in itertools.product(*(range(size) for size in kernel)):
            slices = tuple(
                slice(
                    offset * dilation_value,
                    offset * dilation_value + (size - 1) * step + 1,
                    step,
                )
                for offset, dilation_value, size, step in zip(
                    kernel_index, dilations, source.shape[2:], steps, strict=True
                )
            )
            for group in range(groups):
                in_slice = slice(
                    group * inputs_per_group, (group + 1) * inputs_per_group
                )
                out_slice = slice(
                    group * outputs_per_group, (group + 1) * outputs_per_group
                )
                current = full_gradient[(slice(None), out_slice, *slices)]
                current_weight = matrix[(in_slice, slice(None), *kernel_index)]
                grad_input[:, in_slice] += cp.einsum(
                    "no...,io->ni...", current, current_weight
                )
                input_rows = cp.moveaxis(source[:, in_slice], 1, -1).reshape(
                    -1, inputs_per_group
                )
                grad_rows = cp.moveaxis(current, 1, -1).reshape(-1, outputs_per_group)
                grad_weight[(in_slice, slice(None), *kernel_index)] = (
                    input_rows.T @ grad_rows
                )
        return grad_input, grad_weight

    result = _ops.apply(
        forward,
        input,
        weight,
        backward=backward,
        name=f"conv_transpose{dims}d",
    )
    if bias is not None:
        result = result + bias.reshape((1, bias.shape[0]) + (1,) * dims)
    return result


class _ConvNd(Module):
    dims = 0
    transposed = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        output_padding: int | Sequence[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | Sequence[int] = 1,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or groups <= 0:
            raise ValueError("channel and group counts must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("groups must divide both channel counts")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _tuple(kernel_size, self.dims, "kernel_size")
        if any(value == 0 for value in self.kernel_size):
            raise ValueError("kernel_size values must be positive")
        self.stride = _tuple(stride, self.dims, "stride")
        self.padding = _tuple(padding, self.dims, "padding")
        self.output_padding = _tuple(output_padding, self.dims, "output_padding")
        self.dilation = _tuple(dilation, self.dims, "dilation")
        self.groups = groups
        weight_shape = (
            (in_channels, out_channels // groups, *self.kernel_size)
            if self.transposed
            else (out_channels, in_channels // groups, *self.kernel_size)
        )
        fan_in = (in_channels // groups) * math.prod(self.kernel_size)
        bound = 1 / math.sqrt(fan_in)
        self.weight = Parameter(
            (rand(weight_shape, device=device, dtype=dtype) * 2 - 1) * bound
        )
        self.bias = (
            Parameter((rand(out_channels, device=device, dtype=dtype) * 2 - 1) * bound)
            if bias
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != self.dims + 2 or input.shape[1] != self.in_channels:
            raise ValueError("convolution input has the wrong rank or channel count")
        if self.transposed:
            return convolution_transpose(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
        return convolution(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv1d(_ConvNd):
    dims = 1


class Conv2d(_ConvNd):
    dims = 2


class Conv3d(_ConvNd):
    dims = 3


class ConvTranspose1d(_ConvNd):
    dims = 1
    transposed = True


class ConvTranspose2d(_ConvNd):
    dims = 2
    transposed = True


class ConvTranspose3d(_ConvNd):
    dims = 3
    transposed = True
