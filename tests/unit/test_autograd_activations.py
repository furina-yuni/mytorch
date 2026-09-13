from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt
from mytorch.nn import functional as F
from tests.helpers import finite_difference

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "function",
    [
        F.relu,
        lambda value: F.leaky_relu(value, 0.2),
        F.sigmoid,
        F.tanh,
        lambda value: F.softmax(value, dim=-1).square(),
        lambda value: F.log_softmax(value, dim=-1).square(),
        F.gelu,
        lambda value: F.gelu(value, approximate="tanh"),
        F.silu,
        F.softplus,
    ],
)
def test_activation_gradients_match_finite_difference(
    function,
) -> None:
    source = np.array([[-1.3, -0.4, 0.7, 1.8]], dtype=np.float64)
    value = mt.tensor(source, requires_grad=True)
    function(value).sum().backward()

    expected = finite_difference(function, source)
    np.testing.assert_allclose(value.grad.numpy(), expected, rtol=2e-5, atol=2e-5)


def test_shape_cat_stack_and_reduction_gradients() -> None:
    first = mt.tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    second = mt.tensor(np.arange(6.0, 12.0).reshape(2, 3), requires_grad=True)
    combined = mt.stack([first.T, second.T], dim=0).permute(0, 2, 1)
    mt.cat([combined[0], combined[1]], dim=0).mean().backward()

    np.testing.assert_allclose(first.grad.numpy(), np.full((2, 3), 1 / 12))
    np.testing.assert_allclose(second.grad.numpy(), np.full((2, 3), 1 / 12))


def test_var_and_std_gradients_match_finite_difference() -> None:
    source = np.array([[0.2, 1.1, 2.4], [3.0, 4.2, 6.1]], dtype=np.float64)
    for function in (
        lambda value: value.var(dim=1).sum(),
        lambda value: value.std(dim=0, correction=1).sum(),
    ):
        value = mt.tensor(source, requires_grad=True)
        function(value).backward()
        expected = finite_difference(function, source)
        np.testing.assert_allclose(value.grad.numpy(), expected, rtol=2e-5, atol=2e-5)
