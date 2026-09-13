"""Informational RTX GPU benchmark; speed is not a correctness gate."""

from __future__ import annotations

import time

import cupy as cp

import mytorch as mt


def measure(operation, iterations: int = 20) -> float:
    for _ in range(10):
        operation()
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main() -> None:
    batch, in_features, out_features = 64, 512, 512
    value = mt.randn(batch, in_features)
    linear = mt.nn.Linear(in_features, out_features)
    bit = mt.nn.BitLinear(in_features, out_features)
    bit.eval()
    packed = bit.to_inference()
    with mt.no_grad():
        linear_ms = measure(lambda: linear(value))
        packed_ms = measure(lambda: packed(value))
    print(f"Linear: {linear_ms:.3f} ms")
    print(f"PackedBitLinear: {packed_ms:.3f} ms")
    print(f"Weight compression ratio: {packed.compression_ratio:.1f}x")


if __name__ == "__main__":
    main()
