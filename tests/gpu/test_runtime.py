from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import mytorch

pytestmark = pytest.mark.gpu


def test_cuda_interface_matches_cupy() -> None:
    assert mytorch.cuda.is_available()
    assert mytorch.cuda.device_count() == cp.cuda.runtime.getDeviceCount()
    assert mytorch.cuda.device_count() >= 1


def test_array_reduction_and_matrix_multiplication_stay_on_gpu() -> None:
    values = cp.arange(12, dtype=cp.float32).reshape(3, 4)
    weights = cp.arange(8, dtype=cp.float32).reshape(4, 2)

    result = values @ weights
    total = result.sum()

    assert isinstance(result, cp.ndarray)
    assert result.device.id == 0
    np.testing.assert_allclose(
        cp.asnumpy(result),
        np.arange(12, dtype=np.float32).reshape(3, 4)
        @ np.arange(8, dtype=np.float32).reshape(4, 2),
    )
    assert float(total) == pytest.approx(float(cp.asnumpy(result).sum()))


def test_nvrtc_raw_kernel() -> None:
    kernel = cp.RawKernel(
        r"""
        extern "C" __global__
        void square(const float* input, float* output, const int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                output[index] = input[index] * input[index];
            }
        }
        """,
        "square",
    )
    source = cp.arange(513, dtype=cp.float32)
    output = cp.empty_like(source)

    kernel(((source.size + 255) // 256,), (256,), (source, output, source.size))
    cp.cuda.Device(0).synchronize()

    cp.testing.assert_allclose(output, source**2)
