"""Small examples for Transformer, BitLinear, and state serialization."""

from __future__ import annotations

from pathlib import Path

import mytorch as mt


class TinyEncoder(mt.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = mt.nn.Embedding(128, 32)
        layer = mt.nn.TransformerEncoderLayer(
            32, 4, 64, dropout=0.0, batch_first=True, ffn="swiglu"
        )
        self.encoder = mt.nn.TransformerEncoder(layer, 2)
        self.output = mt.nn.Linear(32, 4)

    def forward(self, tokens: mt.Tensor) -> mt.Tensor:
        encoded = self.encoder(self.embedding(tokens), is_causal=True)
        return self.output(encoded.mean(1))


def main() -> None:
    mt.manual_seed(7)
    model = TinyEncoder()
    tokens = mt.tensor([[1, 2, 3], [4, 5, 6]], dtype=mt.int64)
    print("Transformer output:", model(tokens).numpy())

    bit = mt.nn.BitLinear(32, 16)
    bit.eval()
    packed = bit.to_inference()
    with mt.no_grad():
        output = packed(mt.randn(2, 32))
    print("Packed BitLinear:", output.shape, packed.compression_ratio)

    path = Path("modern-example-state.npz")
    mt.save(model.state_dict(), path)
    restored = TinyEncoder()
    restored.load_state_dict(mt.load(path))
    path.unlink()


if __name__ == "__main__":
    main()
