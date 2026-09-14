"""Collect reproducible MyTorch, Python, CUDA, and GPU environment details."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from typing import Any


def _safe(call, default: Any = "unavailable") -> Any:
    try:
        return call()
    except Exception as exc:
        return f"{default} ({type(exc).__name__}: {exc})"


def _decode(value: Any) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def get_env_info() -> dict[str, Any]:
    """Return environment diagnostics without raising on a broken CUDA setup."""
    import mytorch

    info: dict[str, Any] = {
        "mytorch": mytorch.__version__,
        "installed_distribution": _safe(lambda: metadata.version("mytorch-gpu")),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import cupy as cp

        info["cupy"] = cp.__version__
        info["cuda_driver"] = _safe(cp.cuda.runtime.driverGetVersion)
        info["cuda_runtime"] = _safe(cp.cuda.runtime.runtimeGetVersion)
        count = _safe(cp.cuda.runtime.getDeviceCount, 0)
        info["device_count"] = count
        devices: list[dict[str, Any]] = []
        if isinstance(count, int):
            for index in range(count):
                properties = _safe(
                    lambda index=index: cp.cuda.runtime.getDeviceProperties(index), {}
                )
                if isinstance(properties, dict):
                    name = properties.get("name", properties.get(b"name", "unknown"))
                    major = properties.get("major", properties.get(b"major", "?"))
                    minor = properties.get("minor", properties.get(b"minor", "?"))
                    total = properties.get(
                        "totalGlobalMem", properties.get(b"totalGlobalMem", 0)
                    )
                    devices.append(
                        {
                            "index": index,
                            "name": _decode(name),
                            "compute_capability": f"{major}.{minor}",
                            "total_memory_bytes": int(total),
                        }
                    )
        info["devices"] = devices
    except Exception as exc:
        info["cupy"] = f"unavailable ({type(exc).__name__}: {exc})"
        info["cuda_driver"] = "unavailable"
        info["cuda_runtime"] = "unavailable"
        info["device_count"] = 0
        info["devices"] = []
    return info


def format_env_info(info: dict[str, Any] | None = None) -> str:
    """Format environment diagnostics as readable, issue-friendly text."""
    values = get_env_info() if info is None else info
    lines = [
        "MyTorch environment",
        f"- MyTorch: {values['mytorch']}",
        f"- Installed distribution: {values['installed_distribution']}",
        f"- Python: {values['python']}",
        f"- Executable: {values['executable']}",
        f"- Platform: {values['platform']}",
        f"- CuPy: {values['cupy']}",
        f"- CUDA driver: {values['cuda_driver']}",
        f"- CUDA runtime: {values['cuda_runtime']}",
        f"- CUDA devices: {values['device_count']}",
    ]
    for device in values.get("devices", []):
        gib = device["total_memory_bytes"] / (1024**3)
        lines.append(
            f"  - cuda:{device['index']} {device['name']} "
            f"(compute {device['compute_capability']}, {gib:.2f} GiB)"
        )
    return "\n".join(lines)


def main() -> None:
    """Print diagnostics; pass ``--json`` for machine-readable output."""
    info = get_env_info()
    if "--json" in sys.argv[1:]:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(format_env_info(info))


if __name__ == "__main__":
    main()
