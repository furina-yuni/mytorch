from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


def test_stable_log_ops_where_and_sign_backward() -> None:
    value = mt.tensor([-0.5, 0.0, 2.0], dtype=mt.float64, requires_grad=True)
    other = mt.tensor([1.0, 3.0, -2.0], dtype=mt.float64, requires_grad=True)
    selected = mt.where(value > 0, mt.log1p(value), other)
    selected.sum().backward()

    np.testing.assert_allclose(selected.numpy(), [1.0, 3.0, np.log(3.0)])
    np.testing.assert_allclose(value.grad.numpy(), [0.0, 0.0, 1 / 3])
    np.testing.assert_allclose(other.grad.numpy(), [1.0, 1.0, 0.0])
    np.testing.assert_allclose(mt.sign(value.detach()).numpy(), [-1, 0, 1])

    pair = mt.logaddexp(mt.tensor([1000.0]), mt.tensor([999.0]))
    assert np.isfinite(pair.item())
    assert pair.item() == pytest.approx(np.logaddexp(1000.0, 999.0), rel=1e-6)


def test_logsumexp_norm_normalize_split_and_chunk() -> None:
    source = np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 10
    value = mt.tensor(source, requires_grad=True)
    reduced = mt.logsumexp(value, dim=(1, 2), keepdim=True)
    expected = np.log(np.exp(source).sum(axis=(1, 2), keepdims=True))
    np.testing.assert_allclose(reduced.numpy(), expected, rtol=1e-12)

    vector = mt.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=mt.float64)
    np.testing.assert_allclose(mt.norm(vector, dim=1).numpy(), [5.0, 0.0])
    np.testing.assert_allclose(
        mt.normalize(vector, dim=1).numpy(), [[0.6, 0.8], [0.0, 0.0]]
    )

    pieces = mt.split(value, [1, 2], dim=1)
    assert [piece.shape for piece in pieces] == [(2, 1, 4), (2, 2, 4)]
    assert [piece.shape for piece in mt.chunk(value, 2, dim=2)] == [
        (2, 3, 2),
        (2, 3, 2),
    ]
    sum(piece.sum() for piece in pieces).backward()
    np.testing.assert_allclose(value.grad.numpy(), np.ones_like(source))


def test_norm_gradient_and_invalid_partition_arguments() -> None:
    value = mt.tensor([3.0, 4.0], dtype=mt.float64, requires_grad=True)
    mt.norm(value).backward()
    np.testing.assert_allclose(value.grad.numpy(), [0.6, 0.8])

    with pytest.raises(ValueError):
        mt.split(value, 0)
    with pytest.raises(ValueError):
        mt.chunk(value, 0)
    with pytest.raises(TypeError):
        mt.where(value, value, value)
