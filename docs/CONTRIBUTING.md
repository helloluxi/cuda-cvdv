# Development Guide

## Setup

```bash
git clone https://github.com/your-org/cuda-cvdv.git
cd cuda-cvdv
pip install ".[torch,viz,dev]"
pre-commit install          # auto-runs tests on git commit
```

## Build

```bash
make build                  # compile build/libcvdv.so (required for cuda backend)
make clean                  # remove build artifacts
```

The build targets your GPU automatically (`CMAKE_CUDA_ARCHITECTURES=native`). To target a specific arch:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build
```

## Development Import

On first import each session, `libcvdv.so` is rebuilt if needed:

```python
from src import CVDV    # triggers make build
```

Skip the rebuild check when iterating on Python-only code:

```bash
CVDV_NO_REBUILD=1 python script.py
```

If the `.so` is missing and `CVDV_NO_REBUILD=1`, a `RuntimeError` is raised immediately.

## Tests

```bash
make test                   # run full test suite (requires CUDA GPU)
pytest tests/test_core.py   # CUDA-only tests
pytest tests/test_consistency.py  # CUDA vs Torch consistency
```

Key tolerances:
- Wigner closed-form: `ATOL=1e-2`, Husimi: `ATOL=5e-3` (`NQUBITS=8`, `BOUND=3.0`)
- Consistency (CUDA vs Torch): Wigner uses `VERY_LOOSE_ATOL`, Husimi uses `LOOSE_ATOL`

## Debug Logs

CUDA-side logs are written to `cuda.log` in the project root. The file is cleared each time `CVDV()` is instantiated.

## Adding a New Kernel

1. Add `__global__ void kernelFoo(...)` in `src/cvdv.cu`
2. Add a C-linkage wrapper `extern "C" void cvdvFoo(...)` in `src/cvdv.cu`
3. Add ctypes signature in `src/__init__.py` inside `_compile_and_load()`
4. Add the Python method to `CVDV` with dispatching on `self.backend`
5. Add a test in `tests/test_core.py`

## Project Structure

```
src/
  cvdv.cu              # All CUDA kernels + C API
  __init__.py          # Python ctypes wrapper + torch backend (CVDV class)
  separable.py         # SeparableState: per-register state initialization
tests/
  test_core.py         # CUDA backend correctness (inner product checks)
  test_consistency.py  # CUDA vs torch-cuda vs torch-cpu cross-validation
analysis/              # Standalone error analysis scripts (not part of the library)
benchmarks/            # Timing benchmarks vs bosonic-qiskit and per-gate
docs/
  api.md               # Full API reference
  CONTRIBUTING.md      # This file
```

## Profiling with Nsight

Profile first — optimize second. Quick start:

```bash
# Memory coalescing check on a specific kernel
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld,sm__warps_active.avg.pct_of_peak_sustained_active \
    python -c "
import os; os.environ['CVDV_NO_REBUILD'] = '1'
from src import CVDV
from src.separable import SeparableState
sep = SeparableState([10]); sep.setCoherent(0, 2+1j)
sim = CVDV([10]); sim.initStateVector(sep)
for _ in range(100): sim.d(0, 1+0.5j)
"
```

Roofline reference (RTX 4070 Laptop): **40 TFLOPS FP32 / 192 GB/s HBM**.
Phase-multiplication kernels are memory-bound; target ≥ 70% bandwidth utilization.
