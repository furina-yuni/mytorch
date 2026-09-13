# Changelog

MyTorch follows [Semantic Versioning](https://semver.org/). The project is in
beta, so minor releases may still refine APIs with an accompanying migration
note.

## 0.12.0 - 2026-09-13

### Added

- Epoch-, step-, composite-, and metric-based learning-rate schedulers.
- FP16 autocast and dynamic gradient scaling with overflow-aware optimizer
  updates.
- File-backed ImageFolder, CSV, NPY, and iterable datasets.
- Reproducible samplers, CPU image transforms, threaded prefetch, pinned host
  staging, and asynchronous GPU batch transfer.
- Exact epoch-boundary restoration of scheduler, scaler, loader, and GPU RNG
  state through safe checkpoints.

### Compatibility

- Python 3.12 only.
- NVIDIA CUDA GPU and a compatible NVIDIA display driver are required.
- Tested with CuPy 14.2 and CUDA runtime 13.2 on Windows x86-64.
- Numerical training operations do not provide a CPU fallback.

## 0.9.0

- Added checkpoint persistence and reproducible GPU random-number state.

## 0.7.0

- Added CNN, recurrent, attention, Transformer, LoRA, quantized, BitNet, and
  mixture-of-experts layers.
