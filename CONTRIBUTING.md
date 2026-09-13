# Contributing

## Development environment

```powershell
conda env create -f environment.yml
conda activate mytorch-gpu
python -m pip install -e . --no-deps
```

The `--no-deps` flag is intentional: CuPy and the CUDA 13.2 user-space
libraries are owned by the Conda environment. Do not install `cupy` and a
`cupy-cudaXX` wheel into the same environment.

## Required checks

```powershell
pytest --cov=mytorch --cov-branch --cov-report=term-missing
ruff check src tests examples scripts
ruff format --check src tests examples scripts
python scripts/gpu_smoke_test.py
python scripts/build_docs.py
python scripts/validate_docs.py
python scripts/check_release.py --dist-dir dist
```

Add focused unit tests for new behavior and an integration test when a change
crosses Tensor, Module, optimizer, data, or serialization boundaries. GPU
tests must fail clearly when CUDA is unavailable; the library must not add a
CPU numerical fallback.

## Pull requests

- Keep public API changes documented in `CHANGELOG.md`.
- Preserve unrelated local changes.
- Never commit build output, caches, credentials, datasets, or trained weights.
- Run the complete check suite on an NVIDIA GPU before requesting review.
