"""Quantized and BitNet linear layers."""

from __future__ import annotations

import math
from typing import Any

import cupy as cp

from mytorch import _autograd, _ops
from mytorch.tensor import Tensor, rand, tensor, zeros

from ..functional import layers as F
from .base import Module, Parameter
from .linear import Linear
from .normalization import RMSNorm


def _reject_autograd(input: Tensor, name: str) -> None:
    if input.requires_grad and _autograd.is_grad_enabled():
        raise RuntimeError(f"{name} is inference-only and does not support autograd")


class Int8Linear(Module):
    def __init__(
        self,
        packed_weight: Tensor,
        weight_scale: Tensor,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", bias)
        self.out_features, self.in_features = packed_weight.shape

    @classmethod
    def from_float(cls, layer: Linear) -> Int8Linear:
        weight = layer.weight._array
        scale = cp.maximum(cp.max(cp.abs(weight), axis=1), 1e-12) / 127
        packed = cp.clip(cp.rint(weight / scale[:, None]), -127, 127).astype(cp.int8)
        bias = (
            None if layer.bias is None else Tensor(layer.bias, device=layer.bias.device)
        )
        return cls(
            tensor(packed, device=layer.weight.device),
            tensor(scale.astype(cp.float32), device=layer.weight.device),
            bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        _reject_autograd(input, "Int8Linear")
        if input.shape[-1] != self.in_features:
            raise ValueError("Int8Linear input feature size does not match")
        array = input._array
        activation_scale = (
            cp.maximum(cp.max(cp.abs(array), axis=-1, keepdims=True), 1e-12) / 127
        )
        quantized = cp.clip(cp.rint(array / activation_scale), -127, 127).astype(
            cp.int8
        )
        output = cp.matmul(
            quantized.astype(cp.int32), self.packed_weight._array.T.astype(cp.int32)
        )
        output = output * activation_scale * self.weight_scale._array
        if self.bias is not None:
            output = output + self.bias._array
        return Tensor._from_array(output.astype(input.dtype, copy=False))


class Int4WeightOnlyLinear(Module):
    def __init__(
        self,
        packed_weight: Tensor,
        weight_scale: Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

    @classmethod
    def from_float(cls, layer: Linear, group_size: int = 32) -> Int4WeightOnlyLinear:
        if group_size <= 0 or layer.in_features % group_size:
            raise ValueError("group_size must divide in_features")
        weight = layer.weight._array.reshape(
            layer.out_features, layer.in_features // group_size, group_size
        )
        scale = cp.maximum(cp.max(cp.abs(weight), axis=-1), 1e-12) / 7
        quantized = cp.clip(cp.rint(weight / scale[..., None]), -8, 7).astype(cp.int8)
        flat = (quantized.reshape(-1).astype(cp.int16) + 8).astype(cp.uint8)
        if flat.size % 2:
            flat = cp.pad(flat, (0, 1))
        packed = flat[::2] | (flat[1::2] << 4)
        bias = (
            None if layer.bias is None else Tensor(layer.bias, device=layer.bias.device)
        )
        return cls(
            tensor(packed, device=layer.weight.device),
            tensor(scale.astype(cp.float32), device=layer.weight.device),
            layer.in_features,
            layer.out_features,
            group_size,
            bias,
        )

    def _dequantize(self, dtype: Any) -> cp.ndarray:
        packed = self.packed_weight._array
        values = cp.empty(packed.size * 2, dtype=cp.int8)
        values[::2] = (packed & 0x0F).astype(cp.int8) - 8
        values[1::2] = ((packed >> 4) & 0x0F).astype(cp.int8) - 8
        values = values[: self.out_features * self.in_features].reshape(
            self.out_features, self.in_features // self.group_size, self.group_size
        )
        return (
            (values * self.weight_scale._array[..., None])
            .reshape(self.out_features, self.in_features)
            .astype(dtype)
        )

    def forward(self, input: Tensor) -> Tensor:
        _reject_autograd(input, "Int4WeightOnlyLinear")
        if input.shape[-1] != self.in_features:
            raise ValueError("Int4WeightOnlyLinear input feature size does not match")
        output = cp.matmul(input._array, self._dequantize(input.dtype).T)
        if self.bias is not None:
            output = output + self.bias._array
        return Tensor._from_array(output)


def _ste_ternary(input: Tensor) -> Tensor:
    def forward(weight):
        scale = cp.maximum(cp.mean(cp.abs(weight)), 1e-12)
        return cp.clip(cp.rint(weight / scale), -1, 1) * scale

    return _ops.apply(
        forward,
        input,
        backward=lambda gradient, _result, _arrays: (gradient,),
        name="ternary_ste",
    )


def _ste_int8(input: Tensor) -> Tensor:
    def forward(activation):
        scale = (
            cp.maximum(cp.max(cp.abs(activation), axis=-1, keepdims=True), 1e-12) / 127
        )
        return cp.clip(cp.rint(activation / scale), -127, 127) * scale

    return _ops.apply(
        forward,
        input,
        backward=lambda gradient, _result, _arrays: (gradient,),
        name="int8_ste",
    )


class BitLinear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_rms_norm: bool = True,
        eps: float = 1e-5,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_rms_norm = use_rms_norm
        self.eps = eps
        self.weight = Parameter(
            (rand(out_features, in_features, device=device, dtype=dtype) * 2 - 1)
            / math.sqrt(in_features)
        )
        self.bias = (
            Parameter(zeros(out_features, device=device, dtype=dtype)) if bias else None
        )
        self.rms_norm = (
            RMSNorm(in_features, eps, device=device, dtype=dtype)
            if use_rms_norm
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError("BitLinear input feature size does not match")
        normalized = self.rms_norm(input) if self.rms_norm is not None else input
        return F.linear(_ste_int8(normalized), _ste_ternary(self.weight), self.bias)

    def to_inference(self) -> PackedBitLinear:
        if self.training:
            raise RuntimeError("call eval() before converting BitLinear for inference")
        return PackedBitLinear.from_bitlinear(self)


_PACKED_KERNELS: dict[str, cp.RawKernel] = {}


def _activation_quant_kernel(dtype: cp.dtype) -> cp.RawKernel:
    name = "bitnet_quant_half" if dtype == cp.float16 else "bitnet_quant_float"
    if name in _PACKED_KERNELS:
        return _PACKED_KERNELS[name]
    if dtype == cp.float16:
        declaration = "#include <cuda_fp16.h>\ntypedef half scalar_t;"
        read = "__half2float(input[row * K + k])"
    else:
        declaration = "typedef float scalar_t;"
        read = "input[row * K + k]"
    code = f"""
    {declaration}
    extern "C" __global__ void {name}(
        const scalar_t* input, signed char* quantized, float* scales,
        int rows, int K) {{
      int row = blockDim.x * blockIdx.x + threadIdx.x;
      if (row >= rows) return;
      float maximum = 0.0f;
      for (int k = 0; k < K; ++k) maximum = fmaxf(maximum, fabsf({read}));
      float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
      scales[row] = scale;
      for (int k = 0; k < K; ++k) {{
        int value = (int)nearbyintf(({read}) / scale);
        quantized[row * K + k] = (signed char)max(-127, min(127, value));
      }}
    }}
    """
    result = cp.RawKernel(code, name, options=("--std=c++11",))
    _PACKED_KERNELS[name] = result
    return result


def _packed_kernel(dtype: cp.dtype) -> cp.RawKernel:
    name = "packed_bitlinear_half" if dtype == cp.float16 else "packed_bitlinear_float"
    if name in _PACKED_KERNELS:
        return _PACKED_KERNELS[name]
    if dtype == cp.float16:
        declaration = "#include <cuda_fp16.h>\ntypedef half scalar_t;"
        write = "__float2half(value)"
    else:
        declaration = "typedef float scalar_t;"
        write = "value"
    code = f"""
    {declaration}
    extern "C" __global__ void {name}(
        const signed char* x, const float* activation_scale,
        const unsigned char* packed, const float* weight_scale,
        const float* bias, scalar_t* output, int M, int K, int O,
        int has_bias) {{
      int local_o = threadIdx.x;
      int local_m = threadIdx.y;
      int o = blockIdx.x * 16 + local_o;
      int m = blockIdx.y * 16 + local_m;
      __shared__ signed char activation_tile[16][32];
      __shared__ signed char weight_tile[16][32];
      int accumulator = 0;
      int thread = local_m * 16 + local_o;
      for (int tile = 0; tile < K; tile += 32) {{
        for (int load = thread; load < 16 * 32; load += 256) {{
          int lane = load / 32;
          int offset = load - lane * 32;
          int k = tile + offset;
          int row = blockIdx.y * 16 + lane;
          int column = blockIdx.x * 16 + lane;
          activation_tile[lane][offset] =
            row < M && k < K ? x[row * K + k] : 0;
          if (column < O && k < K) {{
            int weight_index = column * K + k;
            unsigned char byte = packed[weight_index >> 2];
            weight_tile[lane][offset] =
              (signed char)(((byte >> ((weight_index & 3) * 2)) & 3) - 1);
          }} else weight_tile[lane][offset] = 0;
        }}
        __syncthreads();
        if (m < M && o < O) {{
          int limit = min(32, K - tile);
          for (int k = 0; k < limit; ++k)
            accumulator += (int)activation_tile[local_m][k]
              * (int)weight_tile[local_o][k];
        }}
        __syncthreads();
      }}
      if (m < M && o < O) {{
        float value = accumulator * activation_scale[m] * weight_scale[0];
        if (has_bias) value += bias[o];
        output[m * O + o] = {write};
      }}
    }}
    """
    kernel = cp.RawKernel(code, name, options=("--std=c++11",))
    _PACKED_KERNELS[name] = kernel
    return kernel


class PackedBitLinear(Module):
    def __init__(
        self,
        packed_weight: Tensor,
        weight_scale: Tensor,
        in_features: int,
        out_features: int,
        bias: Tensor | None = None,
        rms_weight: Tensor | None = None,
        eps: float = 1e-5,
        compute_dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", bias)
        self.register_buffer("rms_weight", rms_weight)
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.compute_dtype = cp.dtype(compute_dtype)
        self.last_kernel_used = False

    @classmethod
    def from_bitlinear(cls, layer: BitLinear) -> PackedBitLinear:
        weight = layer.weight._array
        scale = cp.maximum(cp.mean(cp.abs(weight)), 1e-12).astype(cp.float32)
        quantized = cp.clip(cp.rint(weight / scale), -1, 1).astype(cp.int8)
        values = (quantized.reshape(-1).astype(cp.int16) + 1).astype(cp.uint8)
        padding = (-values.size) % 4
        if padding:
            values = cp.pad(values, (0, padding), constant_values=1)
        packed = (
            values[0::4]
            | (values[1::4] << 2)
            | (values[2::4] << 4)
            | (values[3::4] << 6)
        )
        bias = None
        if layer.bias is not None:
            bias = tensor(
                layer.bias._array.astype(cp.float32), device=layer.bias.device
            )
        rms_weight = None
        if layer.rms_norm is not None:
            rms_weight = Tensor(
                layer.rms_norm.weight, device=layer.rms_norm.weight.device
            )
        return cls(
            tensor(packed, device=layer.weight.device),
            tensor(cp.asarray([scale], dtype=cp.float32), device=layer.weight.device),
            layer.in_features,
            layer.out_features,
            bias,
            rms_weight,
            layer.eps,
            layer.weight.dtype,
        )

    @property
    def compression_ratio(self) -> float:
        return self.in_features * self.out_features * 4 / self.packed_weight.numel()

    def to(self, device: str | int | None = None, dtype: Any = None) -> PackedBitLinear:
        target_dtype = self.compute_dtype if dtype is None else cp.dtype(dtype)
        if target_dtype not in {cp.dtype(cp.float16), cp.dtype(cp.float32)}:
            raise TypeError("PackedBitLinear compute dtype must be float16 or float32")
        with _autograd.no_grad():
            for name in ("packed_weight", "weight_scale", "bias", "rms_weight"):
                value = getattr(self, name)
                if value is None:
                    continue
                value_dtype = target_dtype if name == "rms_weight" else None
                converted = value.to(device=device, dtype=value_dtype)
                if converted is not value:
                    value._replace_array(converted._array)
        self.compute_dtype = target_dtype
        return self

    def forward(self, input: Tensor) -> Tensor:
        _reject_autograd(input, "PackedBitLinear")
        if input.dtype != self.compute_dtype:
            raise TypeError(
                f"PackedBitLinear expects {self.compute_dtype} input, got {input.dtype}"
            )
        if input.shape[-1] != self.in_features:
            raise ValueError("PackedBitLinear input feature size does not match")
        normalized = (
            F.rms_norm(input, (self.in_features,), self.rms_weight, self.eps)
            if self.rms_weight is not None
            else input
        ).contiguous()
        rows = normalized.numel() // self.in_features
        quantized = cp.empty((rows, self.in_features), dtype=cp.int8)
        activation_scale = cp.empty(rows, dtype=cp.float32)
        _activation_quant_kernel(input.dtype)(
            ((rows + 127) // 128,),
            (128,),
            (
                normalized._array,
                quantized,
                activation_scale,
                rows,
                self.in_features,
            ),
        )
        output = cp.empty((rows, self.out_features), dtype=input.dtype)
        bias = (
            self.bias._array
            if self.bias is not None
            else cp.empty((1,), dtype=cp.float32)
        )
        kernel = _packed_kernel(input.dtype)
        kernel(
            (
                (self.out_features + 15) // 16,
                (rows + 15) // 16,
            ),
            (16, 16),
            (
                quantized,
                activation_scale,
                self.packed_weight._array,
                self.weight_scale._array,
                bias,
                output,
                rows,
                self.in_features,
                self.out_features,
                int(self.bias is not None),
            ),
        )
        self.last_kernel_used = True
        return Tensor._from_array(
            output.reshape(input.shape[:-1] + (self.out_features,))
        )
