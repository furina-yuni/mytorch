"""Transformer encoder modules."""

from __future__ import annotations

from typing import Any

import cupy as cp

from mytorch.tensor import Tensor

from ..functional import activations as A
from ..functional import layers as L
from .attention import MultiheadAttention
from .base import Module
from .containers import ModuleList
from .linear import Linear
from .normalization import LayerNorm
from .regularization import Dropout


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
            hidden = value * A.silu(gate)
        else:
            hidden = A.relu(hidden) if self.activation == "relu" else A.gelu(hidden)
        return self.linear2(L.dropout(hidden, self.dropout2.p, self.training))

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
