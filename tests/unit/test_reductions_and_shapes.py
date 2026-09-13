from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def test_reductions_support_dims_and_keepdim() -> None:
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    value = mt.tensor(source)

    np.testing.assert_allclose(value.sum().numpy(), source.sum())
    np.testing.assert_allclose(
        value.sum((0, 2), keepdim=True).numpy(), source.sum((0, 2), keepdims=True)
    )
    np.testing.assert_allclose(value.mean(-1).numpy(), source.mean(-1))
    np.testing.assert_allclose(value.prod(0).numpy(), source.prod(0))
    np.testing.assert_allclose(value.max(1).numpy(), source.max(1))
    np.testing.assert_allclose(value.min(1).numpy(), source.min(1))
    np.testing.assert_allclose(value.var((0, 2)).numpy(), source.var((0, 2)))
    np.testing.assert_allclose(
        value.std(2, correction=1).numpy(), source.std(2, ddof=1)
    )


def test_top_level_reductions_match_tensor_methods() -> None:
    value = mt.arange(6).reshape(2, 3)

    np.testing.assert_array_equal(mt.sum(value, dim=0).numpy(), value.sum(0).numpy())
    np.testing.assert_array_equal(mt.mean(value).numpy(), value.mean().numpy())
    np.testing.assert_array_equal(
        mt.prod(value + 1).numpy(), (value + 1).prod().numpy()
    )
    np.testing.assert_array_equal(mt.max(value).numpy(), value.max().numpy())
    np.testing.assert_array_equal(mt.min(value).numpy(), value.min().numpy())
    np.testing.assert_array_equal(mt.var(value).numpy(), value.var().numpy())
    np.testing.assert_array_equal(mt.std(value).numpy(), value.std().numpy())
    np.testing.assert_array_equal(mt.argmax(value).numpy(), value.argmax().numpy())
    np.testing.assert_array_equal(mt.argmin(value).numpy(), value.argmin().numpy())


def test_arg_reductions_return_integer_gpu_tensors() -> None:
    value = mt.tensor([[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]])

    np.testing.assert_array_equal(value.argmax(1).numpy(), [1, 0])
    np.testing.assert_array_equal(value.argmin(1, keepdim=True).numpy(), [[0], [1]])
    assert value.argmax().dtype.kind in {"i", "u"}

    batched = mt.arange(8).reshape(2, 2, 2)
    np.testing.assert_array_equal(batched.argmax((1, 2)).numpy(), [3, 3])
    assert batched.argmax((1, 2), keepdim=True).shape == (2, 1, 1)


def test_shape_operations_and_slicing() -> None:
    value = mt.arange(24).reshape(2, 3, 4)

    assert value.flatten().shape == (24,)
    assert value.flatten(1, 2).shape == (2, 12)
    assert value.transpose(0, 2).shape == (4, 3, 2)
    assert value.permute(2, 0, 1).shape == (4, 2, 3)
    assert value[1, :, 1:].shape == (3, 3)
    assert mt.ones(1, 2, 1).squeeze((0, 2)).shape == (2,)
    assert mt.ones(2, 3).unsqueeze(-1).shape == (2, 3, 1)
    assert mt.ones(2, 3).T.shape == (3, 2)


def test_cat_and_stack() -> None:
    first = mt.ones(2, 2)
    second = mt.zeros(2, 2)

    assert mt.cat([first, second], dim=0).shape == (4, 2)
    assert mt.stack([first, second], dim=-1).shape == (2, 2, 2)
    np.testing.assert_array_equal(mt.cat([first, second]).numpy()[2:], 0)


def test_invalid_shape_and_reduction_arguments() -> None:
    value = mt.ones(2, 3, 4)

    with pytest.raises(ValueError, match="out of range"):
        value.sum(3)
    with pytest.raises(ValueError, match="duplicates"):
        value.sum((1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        value.var(correction=-1)
    with pytest.raises(ValueError, match="start_dim"):
        value.flatten(2, 1)
    with pytest.raises(ValueError, match="exactly once"):
        value.permute(0, 0, 1)
    with pytest.raises(ValueError, match="only defined for 2D"):
        _ = value.T
    with pytest.raises(ValueError, match="non-empty"):
        mt.cat([])
    with pytest.raises(ValueError):
        mt.empty(0).max()
