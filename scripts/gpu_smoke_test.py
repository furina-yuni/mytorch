"""Exercise the CUDA components required by MyTorch."""

from __future__ import annotations

import cupy as cp
import numpy as np

import mytorch


def _device_name(properties: dict) -> str:
    name = properties.get("name", properties.get(b"name", "unknown"))
    return name.decode() if isinstance(name, bytes) else str(name)


def main() -> None:
    if not mytorch.cuda.is_available():
        raise RuntimeError(
            "No usable NVIDIA CUDA GPU was found; CPU fallback is disabled."
        )

    device = cp.cuda.Device(0)
    properties = cp.cuda.runtime.getDeviceProperties(0)

    print(f"CuPy: {cp.__version__}")
    print(f"CUDA runtime: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"GPU count: {mytorch.cuda.device_count()}")
    print(f"GPU 0: {_device_name(properties)}")
    print(f"Compute capability: {device.compute_capability}")
    cp.show_config()

    left = cp.arange(12, dtype=cp.float32).reshape(3, 4)
    right = cp.arange(8, dtype=cp.float32).reshape(4, 2)
    product = left @ right
    expected = np.arange(12, dtype=np.float32).reshape(3, 4) @ np.arange(
        8, dtype=np.float32
    ).reshape(4, 2)
    np.testing.assert_allclose(cp.asnumpy(product), expected)
    assert int(product.device.id) == 0

    increment = cp.RawKernel(
        r"""
        extern "C" __global__
        void increment(const float* input, float* output, const int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                output[index] = input[index] + 1.0f;
            }
        }
        """,
        "increment",
    )
    source = cp.arange(1024, dtype=cp.float32)
    output = cp.empty_like(source)
    increment(((source.size + 255) // 256,), (256,), (source, output, source.size))
    cp.testing.assert_allclose(output, source + 1)
    device.synchronize()

    print("GPU array, cuBLAS matrix multiplication, and NVRTC RawKernel: OK")


if __name__ == "__main__":
    main()
