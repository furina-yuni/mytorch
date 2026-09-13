from __future__ import annotations

import math

import cupy as cp
import numpy as np
import pytest

import mytorch as mt
from mytorch.nn import functional as functional

pytestmark = pytest.mark.gpu


def test_pointwise_activations_match_reference_values() -> None:
    source = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    value = mt.tensor(source)

    np.testing.assert_allclose(functional.relu(value).numpy(), np.maximum(source, 0))
    np.testing.assert_allclose(
        functional.leaky_relu(value, 0.2).numpy(),
        np.where(source >= 0, source, 0.2 * source),
    )
    np.testing.assert_allclose(
        functional.sigmoid(value).numpy(), 1 / (1 + np.exp(-source)), rtol=1e-6
    )
    np.testing.assert_allclose(
        functional.tanh(value).numpy(), np.tanh(source), rtol=1e-6
    )
    np.testing.assert_allclose(
        functional.silu(value).numpy(), source / (1 + np.exp(-source)), rtol=1e-6
    )


def test_gelu_exact_and_tanh_approximation() -> None:
    source = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
    value = mt.tensor(source)
    expected = 0.5 * source * (1 + np.vectorize(math.erf)(source / math.sqrt(2)))

    np.testing.assert_allclose(
        functional.gelu(value).numpy(), expected, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        functional.gelu(value, "tanh").numpy(), expected, rtol=3e-3, atol=3e-4
    )
    with pytest.raises(ValueError, match="approximate"):
        functional.gelu(value, "fast")


def test_softmax_and_log_softmax_are_stable() -> None:
    value = mt.tensor([[-1000.0, 0.0, 1000.0], [1000.0, 1000.0, 1000.0]])
    probabilities = functional.softmax(value, dim=-1)
    log_probabilities = functional.log_softmax(value, dim=-1)

    assert np.isfinite(probabilities.numpy()).all()
    assert np.isfinite(log_probabilities.numpy()).all()
    np.testing.assert_allclose(probabilities.sum(-1).numpy(), [1, 1], rtol=1e-6)
    np.testing.assert_allclose(
        mt.exp(log_probabilities).numpy(), probabilities.numpy(), rtol=1e-6
    )


def test_softplus_is_stable_for_extreme_values() -> None:
    source = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0], dtype=np.float32)
    result = functional.softplus(mt.tensor(source), beta=2.0)
    expected = np.where(2 * source > 20, source, np.logaddexp(0, 2 * source) / 2)

    assert np.isfinite(result.numpy()).all()
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="positive"):
        functional.softplus(mt.tensor(source), beta=0)


@pytest.mark.parametrize("dtype", [cp.float16, cp.float32, cp.float64])
def test_supported_float_dtypes(dtype: type) -> None:
    value = mt.tensor([-1.0, 0.0, 1.0], dtype=dtype)
    result = functional.sigmoid(value)
    tolerance = 2e-3 if dtype == cp.float16 else 1e-6

    assert result.dtype == cp.dtype(dtype)
    np.testing.assert_allclose(
        result.numpy(), [0.26894142, 0.5, 0.73105858], rtol=tolerance, atol=tolerance
    )


def test_tensor_activation_methods_share_functional_behavior() -> None:
    value = mt.tensor([-1.0, 0.0, 1.0])

    np.testing.assert_array_equal(value.relu().numpy(), functional.relu(value).numpy())
    np.testing.assert_allclose(
        value.sigmoid().numpy(), functional.sigmoid(value).numpy()
    )
    np.testing.assert_allclose(value.tanh().numpy(), functional.tanh(value).numpy())
    np.testing.assert_allclose(
        value.softmax().numpy(), functional.softmax(value).numpy()
    )
    np.testing.assert_allclose(
        value.log_softmax().numpy(), functional.log_softmax(value).numpy()
    )


def test_activation_rejects_non_floating_tensor() -> None:
    value = mt.tensor(np.array([1, 2, 3], dtype=np.int32))

    with pytest.raises(TypeError, match="floating-point"):
        functional.relu(value)
