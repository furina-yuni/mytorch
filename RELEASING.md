# Release guide

MyTorch is published under the distribution name `mytorch-gpu`; its Python
import remains `import mytorch`. The name `mytorch` cannot be used on PyPI
because it belongs to another project.

## 1. Configure the repository once

1. Create a GitHub repository, add it as `origin`, and add its Documentation,
   Issues, and Repository URLs to `[project.urls]` in `pyproject.toml`.
2. Enable private vulnerability reporting and branch protection.
3. Add a self-hosted Windows x64 runner with labels `self-hosted`, `Windows`,
   `X64`, `gpu`, and `nvidia`. Its `mytorch-gpu` Conda environment must be
   reproducible from `environment.yml`.
4. On TestPyPI and PyPI, create trusted publishers for the GitHub workflow
   `.github/workflows/publish.yml` and environments `testpypi` and `pypi`.

No repository URL is currently configured. This does not invalidate the wheel,
but adding verified project links before public publication is strongly
recommended.

## 2. Validate on the GPU runner

```powershell
conda env update -n mytorch-gpu -f environment.yml --prune
conda activate mytorch-gpu
python -m pip install -e . --no-deps
pytest --cov=mytorch --cov-branch --cov-report=term-missing
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python scripts/gpu_smoke_test.py
python scripts/build_docs.py
python scripts/validate_docs.py
```

## 3. Build and inspect

```powershell
python -m build
python -m twine check dist/*
python scripts/check_release.py --dist-dir dist
```

The distribution contains only `mytorch`, package metadata, README, and the
MIT license. Tests, docs, examples, cache files, and local paths must not be in
the wheel.

## 4. TestPyPI first

Run the **Publish distribution** workflow with target `testpypi`. In a clean
Python 3.12 virtual environment with a compatible NVIDIA driver:

```powershell
python -m pip install --index-url https://test.pypi.org/simple/ `
  --extra-index-url https://pypi.org/simple/ mytorch-gpu==0.12.0
python -c "import mytorch as mt; print(mt.__version__); print(mt.cuda.is_available())"
```

The extra production index is required because TestPyPI does not mirror CuPy,
NumPy, Pillow, or NVIDIA CUDA component wheels.

## 5. Production release

After the clean-install smoke test succeeds, run the same workflow with target
`pypi`. Create and push the signed tag `v0.12.0` only after publication. PyPI
files are immutable; fix mistakes with a new version instead of overwriting an
existing artifact.
