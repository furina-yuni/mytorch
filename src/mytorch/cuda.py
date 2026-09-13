"""CUDA availability helpers backed exclusively by CuPy."""

from __future__ import annotations


class CudaUnavailableError(RuntimeError):
    """Raised when the configured CUDA backend cannot be initialized."""


def _cupy():
    try:
        import cupy
    except Exception as exc:
        raise CudaUnavailableError(
            "MyTorch requires the CuPy CUDA backend, but CuPy could not be imported. "
            "Activate the 'mytorch-gpu' conda environment and verify its CUDA packages."
        ) from exc
    return cupy


def device_count() -> int:
    """Return the number of CUDA devices, or raise when CUDA cannot initialize."""
    cupy = _cupy()
    try:
        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise CudaUnavailableError(
            "MyTorch could not initialize an NVIDIA CUDA device. "
            "Check the NVIDIA driver and the 'mytorch-gpu' conda environment."
        ) from exc


def is_available() -> bool:
    """Return whether at least one usable NVIDIA CUDA device is available."""
    try:
        return device_count() > 0
    except CudaUnavailableError:
        return False
