"""Sparse mixture-of-experts layers."""

from __future__ import annotations

from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, rand, tensor, zeros

from .._functional import activations as F
from .base import Module
from .containers import ModuleList
from .linear import Linear
from .low_rank import SwiGLUFeedForward


class TopKRouter(Module):
    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        top_k: int = 2,
        jitter_noise: float = 0.0,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError("top_k must be between one and num_experts")
        if jitter_noise < 0:
            raise ValueError("jitter_noise must be non-negative")
        self.projection = Linear(
            model_dim, num_experts, False, device=device, dtype=dtype
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        routed = input
        if self.training and self.jitter_noise:
            noise = (
                rand(input.shape, device=input.device, dtype=input.dtype) * 2 - 1
            ) * self.jitter_noise
            routed = input * (1 + noise)
        probabilities = F.softmax(self.projection(routed), -1)
        weights, indices = probabilities.topk(self.top_k, -1)
        weights = weights / weights.sum(-1, keepdim=True)
        return probabilities, weights, indices


class SparseMoE(Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        router_jitter: float = 0.0,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.router = TopKRouter(
            model_dim,
            num_experts,
            top_k,
            router_jitter,
            device=device,
            dtype=dtype,
        )
        self.experts = ModuleList(
            SwiGLUFeedForward(model_dim, hidden_dim, device=device, dtype=dtype)
            for _ in range(num_experts)
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        flat = input.reshape(-1, self.model_dim)
        probabilities, routing_weights, indices = self.router(flat)
        output = zeros(flat.shape, device=flat.device, dtype=flat.dtype)
        selected_load = cp.zeros(self.num_experts, dtype=cp.float32)
        for expert_index, expert in enumerate(self.experts):
            mask_array = indices._array == expert_index
            token_array = cp.where(cp.any(mask_array, axis=1))[0].astype(cp.int64)
            selected_load[expert_index] = mask_array.sum()
            if not token_array.size:
                continue
            token_indices = tensor(token_array, device=input.device)
            expert_input = flat[token_indices]
            expert_output = expert(expert_input)
            mask = tensor(mask_array, device=input.device)
            weights = (routing_weights * mask).sum(-1)[token_indices].unsqueeze(-1)
            scatter_index = token_indices.unsqueeze(-1).expand(
                token_indices.shape[0], self.model_dim
            )
            output = output.scatter_add(0, scatter_index, expert_output * weights)
        load = tensor(
            selected_load / cp.maximum(selected_load.sum(), 1), device=input.device
        )
        auxiliary = self.num_experts * (probabilities.mean(0) * load).sum()
        return output.reshape(input.shape), auxiliary
