from __future__ import annotations

import cupy as cp
import pytest

import mytorch

pytestmark = pytest.mark.gpu


def test_cuda_interface_matches_cupy() -> None:
    assert mytorch.cuda.is_available()
    assert mytorch.cuda.device_count() == cp.cuda.runtime.getDeviceCount()
    assert mytorch.cuda.device_count() >= 1
