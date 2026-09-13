from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def test_arithmetic_and_reverse_scalar_operations() -> None:
    left = mt.tensor([[1.0], [2.0]])
    right = mt.tensor([[10.0, 20.0, 30.0]])

    np.testing.assert_allclose((left + right).numpy(), [[11, 21, 31], [12, 22, 32]])
    np.testing.assert_allclose((right - left).numpy(), [[9, 19, 29], [8, 18, 28]])
    np.testing.assert_allclose((left * 2).numpy(), [[2], [4]])
    np.testing.assert_allclose((2 * left).numpy(), [[2], [4]])
    np.testing.assert_allclose((6 / left).numpy(), [[6], [3]])
    np.testing.assert_allclose((left**2).numpy(), [[1], [4]])
    np.testing.assert_allclose((2**left).numpy(), [[2], [4]])
    np.testing.assert_allclose((-left).numpy(), [[-1], [-2]])
    np.testing.assert_allclose(abs(-left).numpy(), [[1], [2]])


def test_math_minimum_maximum_and_clip() -> None:
    value = mt.tensor([0.25, 1.0, 4.0])

    np.testing.assert_allclose(mt.exp(mt.log(value)).numpy(), value.numpy())
    np.testing.assert_allclose(mt.sqrt(value).numpy(), [0.5, 1.0, 2.0])
    np.testing.assert_allclose(mt.square(value).numpy(), [0.0625, 1.0, 16.0])
    np.testing.assert_allclose(mt.sin(value).numpy(), np.sin(value.numpy()))
    np.testing.assert_allclose(mt.cos(value).numpy(), np.cos(value.numpy()))
    np.testing.assert_allclose(mt.maximum(value, 1).numpy(), [1, 1, 4])
    np.testing.assert_allclose(mt.minimum(value, 1).numpy(), [0.25, 1, 1])
    np.testing.assert_allclose(mt.clip(value, 0.5, 2).numpy(), [0.5, 1, 2])


def test_comparison_and_boolean_indexing_remain_on_gpu() -> None:
    value = mt.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    mask = value > 0
    selected = value[mask]

    assert mask.dtype.kind == "b"
    assert selected.device == "cuda:0"
    np.testing.assert_array_equal(selected.numpy(), [1.0, 2.0])
    np.testing.assert_array_equal(
        (value == 0).numpy(), [False, False, True, False, False]
    )
    np.testing.assert_array_equal((value != 0).numpy(), [True, True, False, True, True])
    np.testing.assert_array_equal(
        (value <= 0).numpy(), [True, True, True, False, False]
    )
    np.testing.assert_array_equal(
        (value >= 0).numpy(), [False, False, True, True, True]
    )


def test_incompatible_broadcasting_raises() -> None:
    with pytest.raises(ValueError):
        _ = mt.ones(2, 3) + mt.ones(4)
    with pytest.raises(TypeError, match="wrapped"):
        _ = mt.ones(2) + np.ones(2)


def test_item_assignment_is_not_supported() -> None:
    value = mt.ones(2)
    with pytest.raises(TypeError):
        value[0] = 5  # type: ignore[index]
    with pytest.raises(TypeError, match="indices"):
        _ = value[np.array([0])]
