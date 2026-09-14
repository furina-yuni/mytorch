"""Install a built or public wheel in isolation and exercise it on the GPU."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _install(target: Path, version: str, find_links: Path | None) -> None:
    requirement = "mytorch-gpu" if version == "local" else f"mytorch-gpu=={version}"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(target),
        requirement,
    ]
    if find_links is not None:
        command[4:4] = ["--no-index", "--find-links", str(find_links.resolve())]
    for attempt in range(6):
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            return
        if find_links is not None or attempt == 5:
            completed.check_returncode()
        time.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--find-links", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mytorch-wheel-") as temporary:
        target = Path(temporary).resolve()
        _install(target, args.version, args.find_links)
        sys.path.insert(0, str(target))
        for name in tuple(sys.modules):
            if name == "mytorch" or name.startswith("mytorch."):
                del sys.modules[name]
        mt = importlib.import_module("mytorch")
        module_path = Path(mt.__file__).resolve()
        if not module_path.is_relative_to(target):
            raise RuntimeError(f"import did not use isolated wheel: {module_path}")
        if args.version != "local" and mt.__version__ != args.version:
            raise RuntimeError(
                f"expected version {args.version}, imported {mt.__version__}"
            )

        left = mt.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        right = mt.tensor([[2.0], [1.0]], requires_grad=True)
        loss = (left @ right).sum()
        loss.backward()
        np.testing.assert_allclose(loss.numpy(), 14.0)
        np.testing.assert_allclose(left.grad.numpy(), [[2.0, 1.0], [2.0, 1.0]])
        np.testing.assert_allclose(right.grad.numpy(), [[4.0], [6.0]])
        print(f"Public wheel GPU smoke test OK: {mt.__version__} from {module_path}")


if __name__ == "__main__":
    main()
