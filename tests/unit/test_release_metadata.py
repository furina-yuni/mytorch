from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _literal_version(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError("package __version__ is missing")


def test_distribution_metadata_and_versions_are_release_consistent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    docs = json.loads(
        (ROOT / "docs" / "content" / "modules.json").read_text(encoding="utf-8")
    )

    assert project["name"] == "mytorch-gpu"
    assert project["version"] == _literal_version(
        ROOT / "src" / "mytorch" / "__init__.py"
    )
    assert project["version"] == docs["site"]["version"]
    assert project["license"] == "MIT"
    assert any(
        requirement.startswith("cupy-cuda13x[ctk]")
        for requirement in project["dependencies"]
    )
    assert (ROOT / "LICENSE").is_file()
