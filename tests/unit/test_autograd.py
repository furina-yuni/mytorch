from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def test_branching_and_broadcasting_accumulate_leaf_gradients() -> None:
    left = mt.tensor([[1.0], [2.0]], requires_grad=True)
    right = mt.tensor([[3.0, 4.0, 5.0]], requires_grad=True)

    loss = (left * right + left).sum()
    loss.backward()

    np.testing.assert_allclose(left.grad.numpy(), [[15.0], [15.0]])
    np.testing.assert_allclose(right.grad.numpy(), [[3.0, 3.0, 3.0]])
    assert loss.grad is None
    assert left.is_leaf
    assert not loss.is_leaf


def test_backward_requires_seed_for_non_scalar_and_validates_it() -> None:
    value = mt.tensor([1.0, 2.0], requires_grad=True)
    output = value.square()

    with pytest.raises(RuntimeError, match="explicit gradient"):
        output.backward()
    with pytest.raises(ValueError, match="shape"):
        output.backward(mt.ones(1))

    output.backward(mt.tensor([2.0, 3.0]))
    np.testing.assert_allclose(value.grad.numpy(), [4.0, 12.0])


def test_gradient_mode_detach_and_requires_grad_validation() -> None:
    value = mt.tensor([2.0], requires_grad=True)
    with mt.no_grad():
        disabled = value.square()
        assert not mt.is_grad_enabled()
        with mt.enable_grad():
            assert mt.is_grad_enabled()
    assert mt.is_grad_enabled()
    assert not disabled.requires_grad
    assert not value.detach().requires_grad

    with pytest.raises(TypeError, match="floating-point"):
        mt.tensor(np.array([1], dtype=np.int64), requires_grad=True)


def test_graph_release_retain_and_gradient_accumulation() -> None:
    value = mt.tensor([3.0], requires_grad=True)
    output = value.square().sum()
    output.backward(retain_graph=True)
    output.backward()
    np.testing.assert_allclose(value.grad.numpy(), [12.0])

    with pytest.raises(RuntimeError, match="freed"):
        output.backward()


def test_parameter_version_detects_update_before_backward() -> None:
    parameter = mt.nn.Parameter([2.0])
    optimizer = mt.optim.SGD([parameter], lr=0.1)
    old_graph = parameter.square().sum()
    parameter.square().sum().backward()
    optimizer.step()

    with pytest.raises(RuntimeError, match="modified"):
        old_graph.backward()


def test_prod_max_and_indexing_backward_rules() -> None:
    product_input = mt.tensor([0.0, 2.0, 3.0], requires_grad=True)
    product_input.prod().backward()
    np.testing.assert_allclose(product_input.grad.numpy(), [6.0, 0.0, 0.0])

    maximum_input = mt.tensor([2.0, 2.0, 1.0], requires_grad=True)
    maximum_input.max().backward()
    np.testing.assert_allclose(maximum_input.grad.numpy(), [0.5, 0.5, 0.0])

    indexed = mt.tensor([1.0, 2.0, 3.0], requires_grad=True)
    indices = mt.tensor(np.array([0, 0, 2], dtype=np.int64))
    indexed[indices].sum().backward()
    np.testing.assert_allclose(indexed.grad.numpy(), [2.0, 0.0, 1.0])


def test_math_extrema_and_clip_backward_rules() -> None:
    value = mt.tensor([0.5, 1.0, 2.0], requires_grad=True)
    expression = (
        mt.exp(value)
        + mt.log(value)
        + mt.sqrt(value)
        + mt.sin(value)
        + mt.cos(value)
        + value**2
    ).sum()
    expression.backward()
    source = value.numpy()
    expected = (
        np.exp(source)
        + 1 / source
        + 0.5 / np.sqrt(source)
        + np.cos(source)
        - np.sin(source)
        + 2 * source
    )
    np.testing.assert_allclose(value.grad.numpy(), expected, rtol=1e-6)

    clipped = mt.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    mt.clip(clipped, -1.0, 1.0).sum().backward()
    np.testing.assert_allclose(clipped.grad.numpy(), [0, 1, 1, 1, 0])

    left = mt.tensor([1.0, 2.0, 3.0], requires_grad=True)
    right = mt.tensor([2.0, 2.0, 1.0], requires_grad=True)
    (mt.maximum(left, right) + mt.minimum(left, right)).sum().backward()
    np.testing.assert_allclose(left.grad.numpy(), np.ones(3))
    np.testing.assert_allclose(right.grad.numpy(), np.ones(3))


def test_batched_matmul_backward_shapes_and_values() -> None:
    left = mt.tensor(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4), requires_grad=True
    )
    right = mt.tensor(
        np.arange(20, dtype=np.float64).reshape(1, 4, 5), requires_grad=True
    )
    (left @ right).sum().backward()

    expected_left = np.broadcast_to(right.numpy().sum(axis=-1)[:, None, :], left.shape)
    expected_right = left.numpy().sum(axis=(0, 1), keepdims=True)
    expected_right = np.repeat(expected_right.transpose(0, 2, 1), 5, axis=2)
    np.testing.assert_allclose(left.grad.numpy(), expected_left)
    np.testing.assert_allclose(right.grad.numpy(), expected_right)


@pytest.mark.parametrize(
    ("left_source", "right_source"),
    [
        (np.arange(3.0) + 1, np.arange(3.0) + 2),
        (np.arange(6.0).reshape(2, 3) + 1, np.arange(3.0) + 2),
        (np.arange(2.0) + 1, np.arange(6.0).reshape(2, 3) + 2),
        (np.arange(6.0).reshape(2, 3) + 1, np.arange(12.0).reshape(3, 4) + 2),
    ],
)
def test_matmul_vector_and_matrix_gradients_match_finite_difference(
    left_source: np.ndarray, right_source: np.ndarray
) -> None:
    left = mt.tensor(left_source, requires_grad=True)
    right = mt.tensor(right_source, requires_grad=True)
    (left @ right).sum().backward()
    epsilon = 1e-5

    expected_left = np.empty_like(left_source)
    for index in np.ndindex(left_source.shape):
        plus = left_source.copy()
        minus = left_source.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        expected_left[index] = (
            np.matmul(plus, right_source).sum() - np.matmul(minus, right_source).sum()
        ) / (2 * epsilon)

    expected_right = np.empty_like(right_source)
    for index in np.ndindex(right_source.shape):
        plus = right_source.copy()
        minus = right_source.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        expected_right[index] = (
            np.matmul(left_source, plus).sum() - np.matmul(left_source, minus).sum()
        ) / (2 * epsilon)

    np.testing.assert_allclose(left.grad.numpy(), expected_left, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(right.grad.numpy(), expected_right, rtol=1e-8, atol=1e-8)
