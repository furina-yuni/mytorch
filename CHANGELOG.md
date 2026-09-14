# Changelog

MyTorch follows [Semantic Versioning](https://semver.org/). The project is in
beta, so minor releases may still refine APIs with an accompanying migration
note.

## 0.13.0 - 2026-09-14

### Added

- Evaluation-only `inference_mode` and non-accumulating `autograd.grad` VJPs.
- Backward anomaly detection with operation names, forward traces, and optional
  non-finite gradient checks.
- Dense gradient norm and value clipping in `mytorch.nn.utils`.
- CUDA pool/current/peak memory statistics, cache release, physical memory
  information, and synchronization helpers.
- A CUDA-event operator profiler with shape recording, aggregation, and text
  summaries.
- `python -m mytorch.utils` environment diagnostics for issue reports.
- Automated GitHub Pages documentation and tag-triggered GPU validation.

### Release quality

- Added tests for every new debugging, autograd, memory, and profiling API.
- Added a public-wheel GPU smoke-test procedure to the release workflow and
  documentation.

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
