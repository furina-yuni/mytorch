"""Attention, rotary embeddings, and Transformer encoder modules."""

from __future__ import annotations

import math
from typing import Any

import cupy as cp

from mytorch import _ops
from mytorch._device import parse_device
from mytorch.tensor import Tensor, cat, tensor

from . import functional as F
from .basic import Dropout, LayerNorm
from .modules import Linear, Module, ModuleList


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("attention inputs must have at least two dimensions")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("attention input shapes are incompatible")
    if not 0 <= dropout_p <= 1:
        raise ValueError("dropout_p must be between 0 and 1")
    factor = 1 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    state: dict[str, cp.ndarray] = {}
    operands = (
        (query, key, value)
        if attn_mask is None
        else (
            query,
            key,
            value,
            attn_mask,
        )
    )

    def forward(*arrays):
        q, k, v = arrays[:3]
        scores = cp.matmul(q, cp.swapaxes(k, -1, -2)) * factor
        allowed = None
        if is_causal:
            allowed = cp.arange(q.shape[-2])[:, None] >= cp.arange(k.shape[-2])[None, :]
            scores = cp.where(allowed, scores, -cp.inf)
        if len(arrays) == 4:
            mask = arrays[3]
            if mask.dtype == cp.bool_:
                allowed = mask if allowed is None else cp.logical_and(allowed, mask)
                scores = cp.where(allowed, scores, -cp.inf)
            else:
                scores = scores + mask
        maximum = cp.max(scores, axis=-1, keepdims=True)
        finite_row = cp.isfinite(maximum)
        exponent = cp.where(finite_row, cp.exp(scores - maximum), 0)
        denominator = exponent.sum(axis=-1, keepdims=True)
        weights = cp.where(denominator > 0, exponent / cp.maximum(denominator, 1), 0)
        state["weights"] = weights
        if dropout_p:
            keep = cp.random.random(weights.shape) >= dropout_p
            state["dropout"] = keep
            used_weights = (
                weights * keep / (1 - dropout_p)
                if dropout_p < 1
                else cp.zeros_like(weights)
            )
        else:
            used_weights = weights
        state["used_weights"] = used_weights
        state["allowed"] = allowed
        return cp.matmul(used_weights, v)

    def backward(gradient, _result, arrays):
        q, k, v = arrays[:3]
        weights = state["weights"]
        used_weights = state["used_weights"]
        grad_value = cp.matmul(cp.swapaxes(used_weights, -1, -2), gradient)
        grad_weights = cp.matmul(gradient, cp.swapaxes(v, -1, -2))
        if dropout_p:
            keep = state["dropout"]
            grad_weights = (
                grad_weights * keep / (1 - dropout_p)
                if dropout_p < 1
                else cp.zeros_like(grad_weights)
            )
        grad_scores = weights * (
            grad_weights - (grad_weights * weights).sum(axis=-1, keepdims=True)
        )
        allowed = state["allowed"]
        if allowed is not None:
            grad_scores = cp.where(allowed, grad_scores, 0)
        grad_query = cp.matmul(grad_scores, k) * factor
        grad_key = cp.matmul(cp.swapaxes(grad_scores, -1, -2), q) * factor
        result = [grad_query, grad_key, grad_value]
        if len(arrays) == 4:
            result.append(None)
        return tuple(result)

    return _ops.apply(
        forward,
        *operands,
        backward=backward,
        name="scaled_dot_product_attention",
    )


class MultiheadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.q_proj = Linear(embed_dim, embed_dim, bias, device=device, dtype=dtype)
        self.k_proj = Linear(self.kdim, embed_dim, bias, device=device, dtype=dtype)
        self.v_proj = Linear(self.vdim, embed_dim, bias, device=device, dtype=dtype)
        self.out_proj = Linear(embed_dim, embed_dim, bias, device=device, dtype=dtype)

    def _heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        unbatched = query.ndim == 2
        if (
            query.ndim not in {2, 3}
            or key.ndim != query.ndim
            or value.ndim != query.ndim
        ):
            raise ValueError("MultiheadAttention inputs must all be 2D or all be 3D")
        if unbatched:
            query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)
        elif not self.batch_first:
            query, key, value = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )
        q, k, v = (
            self._heads(self.q_proj(query)),
            self._heads(self.k_proj(key)),
            self._heads(self.v_proj(value)),
        )
        mask = _merge_masks(attn_mask, key_padding_mask, q, k)
        attention = scaled_dot_product_attention(
            q,
            k,
            v,
            mask,
            self.dropout if self.training else 0.0,
            is_causal,
        )
        merged = attention.permute(0, 2, 1, 3).reshape(
            query.shape[0], query.shape[1], self.embed_dim
        )
        output = self.out_proj(merged)
        weights = None
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = (
                    scores.masked_fill(mask != True, -cp.inf)  # noqa: E712
                    if mask.dtype == cp.bool_
                    else scores + mask
                )
            if is_causal:
                causal = tensor(
                    cp.arange(q.shape[-2])[:, None] >= cp.arange(k.shape[-2])[None, :],
                    device=q.device,
                )
                scores = scores.masked_fill(causal != True, -cp.inf)  # noqa: E712
            weights = _safe_softmax(scores)
            if average_attn_weights:
                weights = weights.mean(1)
        if unbatched:
            output = output.squeeze(0)
            if weights is not None:
                weights = weights.squeeze(0)
        elif not self.batch_first:
            output = output.transpose(0, 1)
        return output, weights


def _safe_softmax(input: Tensor) -> Tensor:
    def forward(array):
        maximum = cp.max(array, axis=-1, keepdims=True)
        exponent = cp.where(cp.isfinite(maximum), cp.exp(array - maximum), 0)
        denominator = exponent.sum(axis=-1, keepdims=True)
        return cp.where(denominator > 0, exponent / cp.maximum(denominator, 1), 0)

    return _ops.apply(
        forward,
        input,
        backward=lambda gradient, result, _arrays: (
            result * (gradient - (gradient * result).sum(axis=-1, keepdims=True)),
        ),
        name="safe_softmax",
    )


def _merge_masks(
    attn_mask: Tensor | None,
    padding_mask: Tensor | None,
    query: Tensor,
    key: Tensor,
) -> Tensor | None:
    batch, heads, target, _ = query.shape
    source = key.shape[-2]
    arrays = []
    if attn_mask is not None:
        array = attn_mask._array
        if array.ndim == 2:
            array = array.reshape(1, 1, target, source)
        elif array.ndim == 3 and array.shape[0] == batch * heads:
            array = array.reshape(batch, heads, target, source)
        else:
            raise ValueError("attention mask has an unsupported shape")
        arrays.append(array)
    if padding_mask is not None:
        if padding_mask.shape != (batch, source):
            raise ValueError("key_padding_mask must have shape (batch, source)")
        arrays.append(padding_mask._array.reshape(batch, 1, 1, source))
    if not arrays:
        return None
    if all(array.dtype == cp.bool_ for array in arrays):
        combined = arrays[0]
        for array in arrays[1:]:
            combined = cp.logical_and(combined, cp.logical_not(array))
        if padding_mask is not None and len(arrays) == 1:
            combined = cp.logical_not(combined)
    else:
        combined = cp.zeros((batch, heads, target, source), dtype=query.dtype)
        for index, array in enumerate(arrays):
            if array.dtype == cp.bool_:
                is_padding = padding_mask is not None and index == len(arrays) - 1
                allowed = cp.logical_not(array) if is_padding else array
                combined = combined + cp.where(allowed, 0, -cp.inf)
            else:
                combined = combined + array
    return tensor(combined, device=query.device)


class TransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        ffn: str = "linear",
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"} or ffn not in {"linear", "swiglu"}:
            raise ValueError("unsupported Transformer activation or FFN")
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout,
            bias,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.linear1 = Linear(
            d_model,
            dim_feedforward * (2 if ffn == "swiglu" else 1),
            bias,
            device=device,
            dtype=dtype,
        )
        self.linear2 = Linear(
            dim_feedforward, d_model, bias, device=device, dtype=dtype
        )
        self.norm1 = LayerNorm(d_model, layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(d_model, layer_norm_eps, device=device, dtype=dtype)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first
        self.ffn = ffn
        self._config = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            ffn=ffn,
            device=device,
            dtype=dtype,
        )

    def _feedforward(self, input: Tensor) -> Tensor:
        hidden = self.linear1(input)
        if self.ffn == "swiglu":
            value, gate = hidden.chunk(2, -1)
            hidden = value * F.silu(gate)
        else:
            hidden = F.relu(hidden) if self.activation == "relu" else F.gelu(hidden)
        return self.linear2(F.dropout(hidden, self.dropout2.p, self.training))

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            normalized = self.norm1(src)
            attention, _ = self.self_attn(
                normalized,
                normalized,
                normalized,
                src_key_padding_mask,
                False,
                src_mask,
                is_causal=is_causal,
            )
            src = src + self.dropout1(attention)
            return src + self.dropout2(self._feedforward(self.norm2(src)))
        attention, _ = self.self_attn(
            src,
            src,
            src,
            src_key_padding_mask,
            False,
            src_mask,
            is_causal=is_causal,
        )
        src = self.norm1(src + self.dropout1(attention))
        return self.norm2(src + self.dropout2(self._feedforward(src)))

    def clone(self) -> TransformerEncoderLayer:
        result = TransformerEncoderLayer(**self._config)
        result.load_state_dict(self.state_dict())
        return result


class TransformerEncoder(Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Module | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = ModuleList(encoder_layer.clone() for _ in range(num_layers))
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, mask, src_key_padding_mask, is_causal)
        return output if self.norm is None else self.norm(output)


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("rotary dimension must be a positive even integer")
        self.dim = dim
        self.base = base
        self.dtype = dtype
        self.device = device
        self._build(max_seq_len)

    def _build(self, length: int) -> None:
        with cp.cuda.Device(parse_device(self.device)):
            inverse = 1 / (self.base ** (cp.arange(0, self.dim, 2) / self.dim))
            angles = cp.arange(length)[:, None] * inverse[None, :]
            angles = cp.repeat(angles, 2, axis=-1).astype(self.dtype)
        self.register_buffer(
            "cos_cached", tensor(cp.cos(angles), device=self.device), False
        )
        self.register_buffer(
            "sin_cached", tensor(cp.sin(angles), device=self.device), False
        )

    def to(self, device: str | int | None = None, dtype: Any = None) -> RotaryEmbedding:
        super().to(device=device, dtype=dtype)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def forward(
        self, input: Tensor, seq_len: int | None = None
    ) -> tuple[Tensor, Tensor]:
        length = input.shape[-2] if seq_len is None else seq_len
        if length > self.cos_cached.shape[0]:
            self._build(length)
        return self.cos_cached[:length], self.sin_cached[:length]


def apply_rotary_pos_emb(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor
) -> tuple[Tensor, Tensor]:
    def rotate(value: Tensor) -> Tensor:
        even, odd = value[..., ::2], value[..., 1::2]
        return stack_rotary(-odd, even)

    return query * cosine + rotate(query) * sine, key * cosine + rotate(key) * sine


def stack_rotary(even: Tensor, odd: Tensor) -> Tensor:
    return cat((even.unsqueeze(-1), odd.unsqueeze(-1)), -1).flatten(-2, -1)
