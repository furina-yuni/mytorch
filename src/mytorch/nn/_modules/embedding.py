"""Embedding modules."""

from __future__ import annotations

from typing import Any

import cupy as cp

from mytorch._device import parse_device
from mytorch.tensor import Tensor, arange, full, maximum, stack, where, zeros

from .._functional import layers as F
from .base import Module, Parameter
from .utils import positive


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
        self.num_embeddings = positive("num_embeddings", num_embeddings)
        self.embedding_dim = positive("embedding_dim", embedding_dim)
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
