"""CUDA availability helpers backed exclusively by CuPy."""

from __future__ import annotations

from threading import Lock

_peak_lock = Lock()
_peak_allocated: dict[int, int] = {}
_peak_reserved: dict[int, int] = {}


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


def current_device() -> int:
    """Return the index of the CUDA device active in the calling thread."""
    return int(_cupy().cuda.runtime.getDevice())


def synchronize(device: int | None = None) -> None:
    """Wait until all work queued on one CUDA device has completed."""
    cupy = _cupy()
    index = current_device() if device is None else int(device)
    with cupy.cuda.Device(index):
        cupy.cuda.Device(index).synchronize()


def _memory_values(device: int | None = None) -> tuple[int, int, int]:
    cupy = _cupy()
    index = current_device() if device is None else int(device)
    with cupy.cuda.Device(index):
        pool = cupy.get_default_memory_pool()
        allocated = int(pool.used_bytes())
        reserved = int(pool.total_bytes())
    return index, allocated, reserved


def _record_memory_snapshot(device: int | None = None) -> None:
    index, allocated, reserved = _memory_values(device)
    with _peak_lock:
        _peak_allocated[index] = max(_peak_allocated.get(index, 0), allocated)
        _peak_reserved[index] = max(_peak_reserved.get(index, 0), reserved)


def memory_allocated(device: int | None = None) -> int:
    """Return bytes currently occupied by arrays in CuPy's memory pool."""
    index, allocated, reserved = _memory_values(device)
    with _peak_lock:
        _peak_allocated[index] = max(_peak_allocated.get(index, 0), allocated)
        _peak_reserved[index] = max(_peak_reserved.get(index, 0), reserved)
    return allocated


def memory_reserved(device: int | None = None) -> int:
    """Return bytes currently reserved by CuPy's CUDA memory pool."""
    index, allocated, reserved = _memory_values(device)
    with _peak_lock:
        _peak_allocated[index] = max(_peak_allocated.get(index, 0), allocated)
        _peak_reserved[index] = max(_peak_reserved.get(index, 0), reserved)
    return reserved


def max_memory_allocated(device: int | None = None) -> int:
    """Return the largest observed allocated-byte count since the last reset."""
    index = current_device() if device is None else int(device)
    _record_memory_snapshot(index)
    return _peak_allocated[index]


def max_memory_reserved(device: int | None = None) -> int:
    """Return the largest observed reserved-byte count since the last reset."""
    index = current_device() if device is None else int(device)
    _record_memory_snapshot(index)
    return _peak_reserved[index]


def reset_peak_memory_stats(device: int | None = None) -> None:
    """Reset observed peak counters to the device's current pool usage."""
    index, allocated, reserved = _memory_values(device)
    with _peak_lock:
        _peak_allocated[index] = allocated
        _peak_reserved[index] = reserved


def mem_get_info(device: int | None = None) -> tuple[int, int]:
    """Return free and total device memory in bytes from the CUDA runtime."""
    cupy = _cupy()
    index = current_device() if device is None else int(device)
    with cupy.cuda.Device(index):
        free, total = cupy.cuda.runtime.memGetInfo()
    return int(free), int(total)


def memory_stats(device: int | None = None) -> dict[str, int]:
    """Return current pool, peak, and physical memory counters."""
    index, allocated, reserved = _memory_values(device)
    free, total = mem_get_info(index)
    _record_memory_snapshot(index)
    return {
        "device": index,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "max_allocated_bytes": _peak_allocated[index],
        "max_reserved_bytes": _peak_reserved[index],
        "free_bytes": free,
        "total_bytes": total,
    }


def _format_bytes(value: int) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.2f} {unit}"


def memory_summary(device: int | None = None) -> str:
    """Return a human-readable snapshot of CUDA and CuPy pool memory."""
    stats = memory_stats(device)
    return "\n".join(
        (
            f"MyTorch CUDA memory summary (cuda:{stats['device']})",
            f"Allocated: {_format_bytes(stats['allocated_bytes'])}",
            f"Reserved: {_format_bytes(stats['reserved_bytes'])}",
            f"Peak allocated: {_format_bytes(stats['max_allocated_bytes'])}",
            f"Peak reserved: {_format_bytes(stats['max_reserved_bytes'])}",
            f"Device free: {_format_bytes(stats['free_bytes'])}",
            f"Device total: {_format_bytes(stats['total_bytes'])}",
        )
    )


def empty_cache() -> None:
    """Release unused blocks held by the default device and pinned pools."""
    cupy = _cupy()
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    _record_memory_snapshot()
