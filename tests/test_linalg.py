from __future__ import annotations

import numpy as np
import pytest

import mytorch as mt

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.arange(3, dtype=np.float32), np.arange(3, dtype=np.float32)),
        (np.arange(6, dtype=np.float32).reshape(2, 3), np.arange(3, dtype=np.float32)),
        (np.arange(2, dtype=np.float32), np.arange(6, dtype=np.float32).reshape(2, 3)),
        (
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(12, dtype=np.float32).reshape(3, 4),
        ),
        (
            np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            np.arange(20, dtype=np.float32).reshape(1, 4, 5),
        ),
    ],
)
def test_matmul_matches_numpy(left: np.ndarray, right: np.ndarray) -> None:
    result = mt.tensor(left) @ mt.tensor(right)

    assert result.device == "cuda:0"
    np.testing.assert_allclose(result.numpy(), np.matmul(left, right), rtol=1e-5)


def test_specialized_matrix_operations() -> None:
    vector = mt.tensor([1.0, 2.0, 3.0])
    matrix = mt.arange(6).reshape(2, 3)
    batch_left = mt.arange(24).reshape(2, 3, 4)
    batch_right = mt.arange(40).reshape(2, 4, 5)

    assert mt.dot(vector, vector).item() == pytest.approx(14)
    np.testing.assert_allclose(
        mt.outer(vector, vector).numpy(), np.outer([1, 2, 3], [1, 2, 3])
    )
    assert mt.mm(matrix, matrix.T).shape == (2, 2)
    assert mt.bmm(batch_left, batch_right).shape == (2, 3, 5)


def test_matrix_errors_include_operation_and_shapes() -> None:
    with pytest.raises(ValueError, match=r"mm.*\(2, 3\).*\(4, 2\)"):
        mt.mm(mt.ones(2, 3), mt.ones(4, 2))
    with pytest.raises(ValueError, match="bmm expects two 3D"):
        mt.bmm(mt.ones(2, 3), mt.ones(2, 3))
    with pytest.raises(ValueError, match="matching batch"):
        mt.bmm(mt.ones(2, 3, 4), mt.ones(3, 4, 5))
    with pytest.raises(ValueError, match="dot expects two 1D"):
        mt.dot(mt.ones(2, 2), mt.ones(2, 2))
