from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def test_tensor_defaults_python_data_to_float32_on_gpu() -> None:
    value = mt.tensor([[1, 2], [3, 4]])

    assert isinstance(value, mt.Tensor)
    assert value.shape == (2, 2)
    assert value.ndim == 2
    assert value.size() == (2, 2)
    assert value.size(-1) == 2
    assert value.numel() == 4
    assert value.dtype == cp.dtype(cp.float32)
    assert value.device == "cuda:0"
    assert value._array.device.id == 0


def test_numpy_input_preserves_explicit_floating_dtype() -> None:
    value = mt.tensor(np.array([1.0, 2.0], dtype=np.float64))

    assert value.dtype == cp.dtype(cp.float64)
    np.testing.assert_array_equal(value.numpy(), [1.0, 2.0])


def test_creation_functions_and_like_functions() -> None:
    factories = [
        mt.empty(2, 3),
        mt.zeros((2, 3)),
        mt.ones(2, 3),
        mt.full((2, 3), 7),
        mt.eye(3),
        mt.linspace(0, 1, 5),
        mt.arange(5),
    ]
    source = mt.ones(2, 3, dtype=cp.float64)
    factories.extend(
        [mt.empty_like(source), mt.zeros_like(source), mt.ones_like(source)]
    )

    assert all(isinstance(value, mt.Tensor) for value in factories)
    assert all(value.device == "cuda:0" for value in factories)
    assert mt.zeros_like(source).dtype == cp.dtype(cp.float64)
    np.testing.assert_array_equal(mt.full((2, 3), 7).numpy(), np.full((2, 3), 7))
    assert mt.tensor([1.0], dtype=mt.float16).dtype == cp.dtype(cp.float16)
    assert mt.tensor([1.0], dtype=mt.float32).dtype == cp.dtype(cp.float32)
    assert mt.tensor([1.0], dtype=mt.float64).dtype == cp.dtype(cp.float64)


def test_random_seed_is_reproducible() -> None:
    mt.manual_seed(123)
    first = mt.rand(2, 3)
    mt.manual_seed(123)
    second = mt.rand(2, 3)
    mt.manual_seed(321)
    normal = mt.randn(2, 3, dtype=cp.float64)

    np.testing.assert_array_equal(first.numpy(), second.numpy())
    assert normal.dtype == cp.dtype(cp.float64)


def test_cpu_transfer_must_be_explicit() -> None:
    value = mt.tensor([3.5])

    assert value.item() == pytest.approx(3.5)
    with pytest.raises(TypeError, match="implicit CPU conversion"):
        np.asarray(value)
    with pytest.raises(ValueError, match="requires one element"):
        mt.ones(2).item()


def test_repr_reports_metadata_without_values() -> None:
    representation = repr(mt.ones(2, 3))

    assert representation == "Tensor(shape=(2, 3), dtype=float32, device='cuda:0')"


def test_invalid_device_and_shape_fail_clearly() -> None:
    with pytest.raises(ValueError, match="GPU-only"):
        mt.tensor([1.0], device="cpu")
    with pytest.raises(ValueError, match="non-negative"):
        mt.zeros(2, -1)
    with pytest.raises(TypeError, match="integers"):
        mt.zeros(2, 1.5)
    with pytest.raises(mt.cuda.CudaUnavailableError, match="unavailable"):
        mt.zeros(1, device=10_000)
