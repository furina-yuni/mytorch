"""Validate MyTorch source metadata and built distribution artifacts."""

from __future__ import annotations

import argparse
import ast
import email.parser
import json
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_INIT = ROOT / "src" / "mytorch" / "__init__.py"
DOC_CATALOG = ROOT / "docs" / "content" / "modules.json"
EXPECTED_DISTRIBUTION = "mytorch-gpu"
EXPECTED_IMPORT = "mytorch"
FORBIDDEN_PARTS = {
    "__pycache__",
    ".coverage",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    "benchmarks",
    "docs",
    "examples",
    "tests",
}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _source_version() -> str:
    tree = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"), PACKAGE_INIT.name)
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    value = ast.literal_eval(statement.value)
                    if isinstance(value, str):
                        return value
    _fail("src/mytorch/__init__.py does not define a literal __version__")


def _normalise_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _validate_source() -> tuple[str, str]:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    distribution = project["name"]
    version = project["version"]
    source_version = _source_version()
    docs_version = json.loads(DOC_CATALOG.read_text(encoding="utf-8"))["site"][
        "version"
    ]

    if distribution != EXPECTED_DISTRIBUTION:
        _fail(f"distribution must be {EXPECTED_DISTRIBUTION!r}, got {distribution!r}")
    if len({version, source_version, docs_version}) != 1:
        _fail(
            "version mismatch: "
            f"pyproject={version}, package={source_version}, docs={docs_version}"
        )
    dependency_names = {
        re.split(r"[<>=!~;\[ ]", dependency, maxsplit=1)[0].lower()
        for dependency in project.get("dependencies", [])
    }
    if "cupy" in dependency_names:
        _fail("PyPI metadata must not depend on the source-only 'cupy' distribution")
    if "cupy-cuda13x" not in dependency_names:
        _fail("PyPI metadata must declare the CUDA 13 CuPy wheel")
    if not (ROOT / "LICENSE").is_file():
        _fail("LICENSE is missing")
    return distribution, version


def _metadata_from_wheel(path: Path) -> tuple[email.message.Message, list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            _fail(f"{path.name}: expected exactly one METADATA file")
        raw = archive.read(metadata_names[0]).decode("utf-8")
    return email.parser.Parser().parsestr(raw), names


def _validate_wheel(path: Path, distribution: str, version: str) -> None:
    metadata, names = _metadata_from_wheel(path)
    if metadata["Name"] != distribution or metadata["Version"] != version:
        _fail(
            f"{path.name}: metadata is {metadata['Name']} {metadata['Version']}, "
            f"expected {distribution} {version}"
        )
    requirements = metadata.get_all("Requires-Dist", [])
    if not any(item.lower().startswith("cupy-cuda13x[ctk]") for item in requirements):
        _fail(f"{path.name}: CUDA 13 CuPy runtime dependency is missing")
    package_prefix = f"{EXPECTED_IMPORT}/"
    if not any(name.startswith(package_prefix) for name in names):
        _fail(f"{path.name}: import package {EXPECTED_IMPORT!r} is missing")
    if not any(name.endswith(".dist-info/licenses/LICENSE") for name in names):
        _fail(f"{path.name}: MIT license file is missing")

    bad = []
    for name in names:
        parts = set(Path(name).parts)
        if parts & FORBIDDEN_PARTS or name.endswith((".pyc", ".pyo")):
            bad.append(name)
    if bad:
        _fail(f"{path.name}: forbidden wheel entries: {', '.join(sorted(bad))}")


def _validate_sdist(path: Path, distribution: str, version: str) -> None:
    expected_root = (
        f"{_normalise_distribution(distribution).replace('-', '_')}-{version}"
    )
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    roots = {Path(name).parts[0] for name in names if Path(name).parts}
    if len(roots) != 1:
        _fail(f"{path.name}: source archive must have one top-level directory")
    root = next(iter(roots))
    # Build backends may preserve hyphens or normalize them to underscores.
    if _normalise_distribution(root.removesuffix(f"-{version}")) != distribution:
        if root != expected_root:
            _fail(f"{path.name}: unexpected source archive root {root!r}")
    required_suffixes = {"/LICENSE", "/README.md", "/pyproject.toml"}
    for suffix in required_suffixes:
        if not any(name.endswith(suffix) for name in names):
            _fail(f"{path.name}: missing {suffix[1:]}")
    bad = [
        name
        for name in names
        if set(Path(name).parts[1:])
        & {"__pycache__", ".pytest_cache", ".ruff_cache", ".vscode"}
    ]
    if bad:
        _fail(f"{path.name}: forbidden source entries: {', '.join(sorted(bad))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=ROOT / "dist",
        help="directory containing one wheel and one source archive",
    )
    args = parser.parse_args()
    dist_dir = args.dist_dir.resolve()
    distribution, version = _validate_source()

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        _fail(
            f"{dist_dir}: expected one wheel and one sdist; "
            f"found {len(wheels)} wheel(s), {len(sdists)} sdist(s)"
        )
    _validate_wheel(wheels[0], distribution, version)
    _validate_sdist(sdists[0], distribution, version)
    print(
        f"Release artifacts OK: {distribution} {version} "
        f"({wheels[0].name}, {sdists[0].name})"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        RuntimeError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"release check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
