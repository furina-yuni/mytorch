"""Educational GPU recurrent neural-network layers."""

from __future__ import annotations

import math
from typing import Any

import cupy as cp

from mytorch.tensor import Tensor, rand, stack, zeros

from . import functional as F
from .modules import Module, ModuleList, Parameter


class _RNNCellBase(Module):
    gates = 1

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        bound = 1 / math.sqrt(hidden_size)

        def parameter(*shape: int) -> Parameter:
            return Parameter((rand(shape, device=device, dtype=dtype) * 2 - 1) * bound)

        self.weight_ih = parameter(self.gates * hidden_size, input_size)
        self.weight_hh = parameter(self.gates * hidden_size, hidden_size)
        self.bias_ih = parameter(self.gates * hidden_size) if bias else None
        self.bias_hh = parameter(self.gates * hidden_size) if bias else None

    def _validate(self, input: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor, bool]:
        unbatched = input.ndim == 1
        if input.ndim not in {1, 2} or input.shape[-1] != self.input_size:
            raise ValueError("recurrent cell input has the wrong shape")
        if hidden.ndim != input.ndim or hidden.shape[-1] != self.hidden_size:
            raise ValueError("recurrent cell hidden state has the wrong shape")
        if unbatched:
            input, hidden = input.unsqueeze(0), hidden.unsqueeze(0)
        if input.shape[0] != hidden.shape[0]:
            raise ValueError("input and hidden batch sizes differ")
        return input, hidden, unbatched

    def _affine(self, input: Tensor, hidden: Tensor) -> Tensor:
        result = F.linear(input, self.weight_ih, self.bias_ih)
        return result + F.linear(hidden, self.weight_hh, self.bias_hh)


class RNNCell(_RNNCellBase):
    def __init__(self, *args: Any, nonlinearity: str = "tanh", **kwargs: Any) -> None:
        if nonlinearity not in {"tanh", "relu"}:
            raise ValueError("RNNCell nonlinearity must be 'tanh' or 'relu'")
        self.nonlinearity = nonlinearity
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if hx is None:
            hx = zeros(
                input.shape[:-1] + (self.hidden_size,),
                dtype=input.dtype,
                device=input.device,
            )
        input, hx, unbatched = self._validate(input, hx)
        result = self._affine(input, hx)
        result = F.tanh(result) if self.nonlinearity == "tanh" else F.relu(result)
        return result.squeeze(0) if unbatched else result


class LSTMCell(_RNNCellBase):
    gates = 4

    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        if hx is None:
            shape = input.shape[:-1] + (self.hidden_size,)
            hx = (
                zeros(shape, dtype=input.dtype, device=input.device),
                zeros(shape, dtype=input.dtype, device=input.device),
            )
        hidden, cell = hx
        input, hidden, unbatched = self._validate(input, hidden)
        if unbatched:
            cell = cell.unsqueeze(0)
        if cell.shape != hidden.shape:
            raise ValueError("LSTMCell cell state has the wrong shape")
        input_gate, forget_gate, candidate, output_gate = self._affine(
            input, hidden
        ).chunk(4, dim=-1)
        input_gate = F.sigmoid(input_gate)
        forget_gate = F.sigmoid(forget_gate)
        candidate = F.tanh(candidate)
        output_gate = F.sigmoid(output_gate)
        next_cell = forget_gate * cell + input_gate * candidate
        next_hidden = output_gate * F.tanh(next_cell)
        if unbatched:
            return next_hidden.squeeze(0), next_cell.squeeze(0)
        return next_hidden, next_cell


class GRUCell(_RNNCellBase):
    gates = 3

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if hx is None:
            hx = zeros(
                input.shape[:-1] + (self.hidden_size,),
                dtype=input.dtype,
                device=input.device,
            )
        input, hx, unbatched = self._validate(input, hx)
        input_gates = F.linear(input, self.weight_ih, self.bias_ih).chunk(3, -1)
        hidden_gates = F.linear(hx, self.weight_hh, self.bias_hh).chunk(3, -1)
        reset = F.sigmoid(input_gates[0] + hidden_gates[0])
        update = F.sigmoid(input_gates[1] + hidden_gates[1])
        candidate = F.tanh(input_gates[2] + reset * hidden_gates[2])
        result = (1 - update) * candidate + update * hx
        return result.squeeze(0) if unbatched else result


class _RNNBase(Module):
    cell_type = RNNCell

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        *,
        device: str | int = "cuda:0",
        dtype: Any = cp.float32,
        **cell_options: Any,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1
        cells = []
        for layer in range(num_layers):
            layer_input = input_size if layer == 0 else hidden_size * directions
            for _ in range(directions):
                cells.append(
                    self.cell_type(
                        layer_input,
                        hidden_size,
                        bias,
                        device=device,
                        dtype=dtype,
                        **cell_options,
                    )
                )
        self.cells = ModuleList(cells)

    def _initial(self, input: Tensor):
        batch = input.shape[0] if self.batch_first else input.shape[1]
        shape = (
            self.num_layers * (2 if self.bidirectional else 1),
            batch,
            self.hidden_size,
        )
        return zeros(shape, dtype=input.dtype, device=input.device)

    def _validate_input(self, input: Tensor) -> Tensor:
        if input.ndim != 3 or input.shape[-1] != self.input_size:
            raise ValueError("recurrent layer expects a 3D input")
        return input.transpose(0, 1) if self.batch_first else input

    def forward(self, input: Tensor, hx: Tensor | None = None):
        sequence = self._validate_input(input)
        if hx is None:
            hx = self._initial(input)
        directions = 2 if self.bidirectional else 1
        if hx.shape != (
            self.num_layers * directions,
            sequence.shape[1],
            self.hidden_size,
        ):
            raise ValueError("recurrent hidden state has the wrong shape")
        final = []
        layer_input = sequence
        for layer in range(self.num_layers):
            direction_outputs = []
            for direction in range(directions):
                state = hx[layer * directions + direction]
                order = range(sequence.shape[0])
                if direction:
                    order = range(sequence.shape[0] - 1, -1, -1)
                outputs = []
                cell = self.cells[layer * directions + direction]
                for step in order:
                    state = cell(layer_input[step], state)
                    outputs.append(state)
                if direction:
                    outputs.reverse()
                direction_outputs.append(stack(outputs))
                final.append(state)
            layer_input = (
                direction_outputs[0]
                if directions == 1
                else _concat_features(direction_outputs)
            )
            if layer + 1 < self.num_layers and self.dropout:
                layer_input = F.dropout(layer_input, self.dropout, self.training)
        output = layer_input.transpose(0, 1) if self.batch_first else layer_input
        return output, stack(final)


def _concat_features(values: list[Tensor]) -> Tensor:
    from mytorch.tensor import cat

    return cat(values, dim=-1)


class RNN(_RNNBase):
    cell_type = RNNCell

    def __init__(self, *args: Any, nonlinearity: str = "tanh", **kwargs: Any) -> None:
        super().__init__(*args, nonlinearity=nonlinearity, **kwargs)


class GRU(_RNNBase):
    cell_type = GRUCell


class LSTM(_RNNBase):
    cell_type = LSTMCell

    def _initial(self, input: Tensor):
        hidden = super()._initial(input)
        return hidden, zeros(hidden.shape, dtype=hidden.dtype, device=hidden.device)

    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        sequence = self._validate_input(input)
        if hx is None:
            hx = self._initial(input)
        hidden, cell_state = hx
        directions = 2 if self.bidirectional else 1
        expected = (
            self.num_layers * directions,
            sequence.shape[1],
            self.hidden_size,
        )
        if hidden.shape != expected or cell_state.shape != expected:
            raise ValueError("LSTM hidden or cell state has the wrong shape")
        final_hidden, final_cell = [], []
        layer_input = sequence
        for layer in range(self.num_layers):
            direction_outputs = []
            for direction in range(directions):
                index = layer * directions + direction
                h, c = hidden[index], cell_state[index]
                order = range(sequence.shape[0])
                if direction:
                    order = range(sequence.shape[0] - 1, -1, -1)
                outputs = []
                recurrent_cell = self.cells[index]
                for step in order:
                    h, c = recurrent_cell(layer_input[step], (h, c))
                    outputs.append(h)
                if direction:
                    outputs.reverse()
                direction_outputs.append(stack(outputs))
                final_hidden.append(h)
                final_cell.append(c)
            layer_input = (
                direction_outputs[0]
                if directions == 1
                else _concat_features(direction_outputs)
            )
            if layer + 1 < self.num_layers and self.dropout:
                layer_input = F.dropout(layer_input, self.dropout, self.training)
        output = layer_input.transpose(0, 1) if self.batch_first else layer_input
        return output, (stack(final_hidden), stack(final_cell))
