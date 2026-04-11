# Refactor Plan: Split `src/cvdv.cu` into Multiple Translation Units

## Goal

Break `src/cvdv.cu` (2,340 lines) into focused translation units grouped by
concern, reorganise `src/` into language-separated subtrees, and clean up dead
code. Output is still a single `build/libcvdv.so`; all exported symbol names
and Python callers stay unchanged.

---

## API Design Decision

**Keep `CVDVContext*` C struct + C API. Do not export a C++ class. Do not move
context management to Python.**

### Why not export a C++ class?
- C++ ABI is not stable across compiler versions (mangled names, vtable layout).
  The `.so` would silently break on any toolchain update.
- `ctypes` cannot call C++ member functions directly. Replacing ctypes with
  pybind11 or SWIG adds a build dependency and couples the Python layer to the
  CUDA build.

### Why not manage context in Python?
- GPU allocations (`dState`, `cufftHandle`, `cutensorHandle_t`, Husimi plans)
  must live in CUDA-managed memory. Python cannot own or free these without a
  native handle.
- Moving them to Python would mean passing dozens of raw pointers through ctypes
  on every call — more fragile than the current approach.

### Why the current handle pattern is correct
`CVDVContext*` is exactly the opaque-handle pattern used by CUDA SDK libraries
themselves: `cublasHandle_t`, `cufftHandle`, `cutensorHandle_t`. It is
language-agnostic, ABI stable, and maps cleanly to a Python class wrapper.
The only improvement worth making is declaring `CVDVContext` as an **opaque
forward declaration** in `api.h` (callers see only `CVDVContext*`, not its
fields), while the full struct definition stays private in `types.h`.

---

## New File Structure

```
src/
├── cuda/                       # CUDA/C++ sources — CMake only, never imported by Python
│   ├── types.h                 # (NEW) CVDVContext, MeasurePlan, constants, checkCudaErrors
│   ├── api.h                   # (NEW) opaque forward-decl + extern "C" signatures
│   ├── kernels/
│   │   ├── utils.cuh           # (NEW) __device__ math/index helpers
│   │   ├── gates.cuh           # (NEW) __global__ gate kernels
│   │   ├── cv.cuh              # (NEW) __global__ Wigner + Husimi kernels
│   │   └── measure.cuh         # (NEW) __global__ reduction/norm/inner-product kernels
│   ├── context.cu              # (NEW) lifecycle: create, destroy, init, getters
│   ├── gates.cu                # (NEW) all gate C API (CV, DV, hybrid, two-mode, FFT)
│   ├── cv.cu                   # (NEW) Wigner + Husimi C API
│   └── measure.cu              # (NEW) measurement, fidelity, observables C API
│
└── cvdv/                       # Python package — importable as `cvdv`
    ├── __init__.py             # (UNCHANGED, move from src/)
    ├── cuda.py                 # (RENAME from cudaCvdv.py)
    ├── torch.py                # (RENAME from torchCvdv.py) — note: safe inside package
    └── separable.py            # (UNCHANGED, move from src/)

tests/                          # (UNCHANGED)
benchmarks/                     # (UNCHANGED, see step 14)
```

Files removed:
- `src/cvdv.cu` — deleted after migration
- `src/cudaCvdv.py` — replaced by `src/cvdv/cuda.py`
- `src/torchCvdv.py` — replaced by `src/cvdv/torch.py`

---

## Step-by-Step Tasks

### Step 1 — Scaffold directories

```bash
mkdir -p src/cuda/kernels
mkdir -p src/cvdv
```

---

### Step 2 — Create `src/cuda/types.h`

Move out of `cvdv.cu` (lines 1–98):
- All `#include` headers (`cuComplex.h`, `cuda_runtime.h`, `cufft.h`,
  `cutensor.h`, C++ stdlib)
- `#define checkCudaErrors(...)`
- Constants: `PI`, `SQRT2`, `PI_POW_NEG_QUARTER`, `CUDA_BLOCK_SIZE`
- `struct MeasurePlan` — full definition
- `typedef struct { ... } CVDVContext` — full definition (private to CUDA TUs)
- `typedef struct { ... } SeparableRegArrays`

Guard with `#pragma once`. This file is included only by CUDA TUs — never by
Python or `api.h`.

---

### Step 3 — Create `src/cuda/api.h`

Public-facing header. Callers see only the opaque pointer, not the struct
internals. Included by `measure.cu` (which calls gate functions across TUs) and
by `benchmarks/kernel_profiling/workload.cu`.

```c
// src/cuda/api.h
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle — callers hold CVDVContext* but cannot inspect fields.
typedef struct CVDVContext CVDVContext;

CVDVContext* cvdvCreate(int numReg, int* numQubits);
void         cvdvDestroy(CVDVContext* ctx);
void         cvdvFree(CVDVContext* ctx);
void         cvdvInitFromSeparable(CVDVContext* ctx, void** devicePtrs, int numReg);
void         cvdvSetStateFromDevicePtr(CVDVContext* ctx, void* d_src);
int          cvdvGetNumRegisters(CVDVContext* ctx);
size_t       cvdvGetTotalSize(CVDVContext* ctx);
void         cvdvGetRegisterInfo(CVDVContext* ctx, int* qubitCountsOut, double* gridStepsOut);
int          cvdvGetRegisterDim(CVDVContext* ctx, int regIdx);
double       cvdvGetRegisterDx(CVDVContext* ctx, int regIdx);

void   cvdvFtQ2P(CVDVContext* ctx, int regIdx);
void   cvdvFtP2Q(CVDVContext* ctx, int regIdx);
void   cvdvDisplacement(CVDVContext* ctx, int regIdx, double betaRe, double betaIm);
void   cvdvConditionalDisplacement(CVDVContext* ctx, int targetReg, int ctrlReg,
                                    int ctrlQubit, double betaRe, double betaIm);
void   cvdvPauliRotation(CVDVContext* ctx, int regIdx, int qubitIdx, int axis, double theta);
void   cvdvHadamard(CVDVContext* ctx, int regIdx, int qubitIdx);
void   cvdvParity(CVDVContext* ctx, int regIdx);
void   cvdvConditionalParity(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit);
void   cvdvSwapRegisters(CVDVContext* ctx, int reg1, int reg2);
void   cvdvPhaseSquare(CVDVContext* ctx, int regIdx, double t);
void   cvdvPhaseCubic(CVDVContext* ctx, int regIdx, double t);
void   cvdvRotation(CVDVContext* ctx, int regIdx, double theta);
void   cvdvConditionalRotation(CVDVContext* ctx, int targetReg, int ctrlReg,
                                int ctrlQubit, double theta);
void   cvdvSqueeze(CVDVContext* ctx, int regIdx, double r);
void   cvdvConditionalSqueeze(CVDVContext* ctx, int targetReg, int ctrlReg,
                               int ctrlQubit, double r);
void   cvdvBeamSplitter(CVDVContext* ctx, int reg1, int reg2, double theta);
void   cvdvConditionalBeamSplitter(CVDVContext* ctx, int reg1, int reg2,
                                    int ctrlReg, int ctrlQubit, double theta);
void   cvdvQ1Q2Gate(CVDVContext* ctx, int reg1, int reg2, double coeff);

void   cvdvGetWigner(CVDVContext* ctx, int regIdx, double* wignerOut);
void   cvdvGetHusimiQ(CVDVContext* ctx, int regIdx, double* husimiOut);

void   cvdvMeasureMultiple(CVDVContext* ctx, const int* regIdxs, int numRegs,
                            double* probsOut);
void   cvdvGetState(CVDVContext* ctx, double* realOut, double* imagOut);
double cvdvGetNorm(CVDVContext* ctx);
void   cvdvGetFidelity(CVDVContext* ctx, void** devicePtrs, int numReg, double* fidOut);
double cvdvGetPhotonNumber(CVDVContext* ctx, int regIdx);
void   cvdvFidelityStatevectors(CVDVContext* ctx1, CVDVContext* ctx2, double* fidOut);

#ifdef __cplusplus
}
#endif
```

---

### Step 4 — Create `src/cuda/kernels/utils.cuh`

Move from `cvdv.cu` (lines 102–186). Guard with `#pragma once`.
Include: `types.h`.

Functions:
- `gridX`, `phaseToZ`, `cmulPhase`, `conjMul`, `absSquare`, `cuCmul` (scalar overload)
- `getLocalIndex`, `warpReduceSum`, `blockReduceSum`, `qubitPairIndices`

---

### Step 5 — Create `src/cuda/kernels/gates.cuh`

Move from `cvdv.cu` (lines 188–607). Include: `utils.cuh`.

**Drop dead-code kernels** (never called from C API — state init is done in
Python via `SeparableState`):
- `kernelSetCoherent`, `kernelSetFock`, `kernelSetFocks`, `kernelSetCat`
- `kernelComputeRegisterNorm`, `kernelNormalizeRegister`

Kernels to keep:
- `kernelPhaseX`, `kernelCPhaseX`, `kernelCPhaseX2`
- `kernelPauliRotation`, `kernelHadamard`
- `kernelParity`, `kernelConditionalParity`
- `kernelSwapRegisters`
- `kernelPhaseX2`, `kernelPhaseX3`, `kernelPhaseXX`, `kernelCPhaseXX`
- `kernelGlobalScalar`, `kernelGlobalPhase`

---

### Step 6 — Create `src/cuda/kernels/cv.cuh`

Move from `cvdv.cu` (lines 611–747). Include: `utils.cuh`.

Kernels:
- `kernelBuildWignerRow`, `kernelFinalizeWigner`
- `kernelComputeHusimiG`, `kernelExtractChunkPsi`, `kernelFillHusimiABatched`
- `kernelFinalizeHusimi`, `kernelAccumHusimiPowerChunked`
- `kernelAbsSquareInPlace`

---

### Step 7 — Create `src/cuda/kernels/measure.cuh`

Move from `cvdv.cu` (lines 751–866). Include: `utils.cuh`.

Kernels:
- `kernelExpectX2`, `kernelComputeNorm`
- `kernelComputeInnerProduct`, `kernelInnerProductStatevectors`

---

### Step 8 — Create `src/cuda/context.cu`

Include: `types.h`.

C API functions (wrapped in `extern "C" { ... }`):
- `cvdvCreate` (lines 873–955)
- `cvdvDestroy` (lines 957–1030)
- `cvdvFree` (lines 1088–1108)
- `cvdvInitFromSeparable` (lines 1040–1086)
- `cvdvSetStateFromDevicePtr` (lines 2089–2097)
- `cvdvGetNumRegisters` (lines 2124–2127)
- `cvdvGetTotalSize` (lines 2129–2132)
- `cvdvGetRegisterInfo` (lines 2134–2142)
- `cvdvGetRegisterDim` (lines 2144–2151)
- `cvdvGetRegisterDx` (lines 2153–2160)

---

### Step 9 — Create `src/cuda/gates.cu`

Include: `types.h`, `kernels/gates.cuh`.

C API (wrapped in `extern "C" { ... }`), static helpers outside the block:
- **FFT**: `cvdvFtQ2P`, `cvdvFtP2Q` (lines 1112–1231)
- **CV**: `cvdvDisplacement`, `cvdvPhaseSquare`, `cvdvPhaseCubic`,
  `cvdvRotationSmall` (static), `cvdvRotation`, `cvdvSqueeze`,
  `cvdvBeamSplitterSmall` (static), `cvdvBeamSplitter`
- **DV**: `cvdvPauliRotation`, `cvdvHadamard`, `cvdvParity`, `cvdvSwapRegisters`
- **Hybrid**: `cvdvConditionalDisplacement`, `cvdvControlledPhaseSquare` (static),
  `cvdvConditionalRotationSmall` (static), `cvdvConditionalRotation`,
  `cvdvConditionalSqueeze`, `cvdvConditionalBeamSplitterSmall` (static),
  `cvdvConditionalBeamSplitter`, `cvdvConditionalParity`
- **Two-mode**: `cvdvQ1Q2Gate`

(All gate C API: lines 1112–1800)

---

### Step 10 — Create `src/cuda/cv.cu`

Include: `types.h`, `kernels/cv.cuh`.

C API (wrapped in `extern "C" { ... }`):
- `cvdvGetWigner` (lines 1804–1843)
- `cvdvGetHusimiQ` (lines 1845–1939)

---

### Step 11 — Create `src/cuda/measure.cu`

Include: `types.h`, `api.h` (for `cvdvFtQ2P`/`cvdvFtP2Q`),
`kernels/measure.cuh`, `kernels/cv.cuh` (for `kernelAbsSquareInPlace`).

Static helpers + C API (wrapped in `extern "C" { ... }`):
- `runInnerProductKernel` (static, lines 2166–2197)
- `computeExpectX2` (static, lines 2221–2242)
- `cvdvMeasureMultiple` (lines 1941–2087) — cuTENSOR plan caching
- `cvdvGetState` (lines 2099–2117)
- `cvdvGetNorm` (lines 2260–2295)
- `cvdvGetFidelity` (lines 2202–2218)
- `cvdvGetPhotonNumber` (lines 2247–2258) — calls `cvdvFtQ2P`/`cvdvFtP2Q`
- `cvdvFidelityStatevectors` (lines 2301–2338)

---

### Step 12 — Update `CMakeLists.txt`

```cmake
add_library(cvdv SHARED
    src/cuda/context.cu
    src/cuda/gates.cu
    src/cuda/cv.cu
    src/cuda/measure.cu
)

target_include_directories(cvdv PRIVATE
    src/cuda/
    ${CUTENSOR_INCLUDE_DIR}
)
```

`CUDA_SEPARABLE_COMPILATION ON` and `POSITION_INDEPENDENT_CODE ON` are already
set; keep them.

---

### Step 13 — Rename and move Python files

| Old path | New path | Notes |
|---|---|---|
| `src/__init__.py` | `src/cvdv/__init__.py` | Update imports (see below) |
| `src/cudaCvdv.py` | `src/cvdv/cuda.py` | Update class import in `__init__.py` |
| `src/torchCvdv.py` | `src/cvdv/torch.py` | Safe: `torch` here is a module inside the `cvdv` package, not the top-level PyTorch package |
| `src/separable.py` | `src/cvdv/separable.py` | No content change |

Update `src/cvdv/__init__.py`:

```python
from .cuda import CudaCvdv
from .torch import TorchCvdv
from .separable import SeparableState
```

Update relative imports inside each moved file:
- `src/cvdv/cuda.py`: `from .separable import SeparableState`
- `src/cvdv/torch.py`: `from .separable import SeparableState` (if imported)

The `project_dir` path calculation in `cuda.py` shifts by one level:

```python
# was: os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  → src/../
# becomes:
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

---

### Step 14 — Delete `src/cvdv.cu`, `src/cudaCvdv.py`, `src/torchCvdv.py`

After the build passes and tests pass.

---

### Step 15 — Update test imports

Tests currently do `from src.cudaCvdv import ...` or similar. Update to:

```python
from cvdv import CudaCvdv, TorchCvdv, SeparableState
# or
from cvdv.cuda import CudaCvdv
```

Confirm `pytest` discovers `src/cvdv/` as a package (add `src/` to
`PYTHONPATH` or configure `pyproject.toml`/`setup.cfg` if not already done).

---

### Step 16 — Update `benchmarks/kernel_profiling/workload.cu`

This file is the only non-Python C caller. Update its include path:

```cmake
# or in its compile command:
-I../../src/cuda
```

Include `api.h` instead of any old header it may have used. Symbol names are
unchanged.

---

### Step 17 — Update `ai/rules/project-overview.md`

```
Architecture:
- src/cuda/         — CUDA/C++ sources (CMake only)
  - types.h         — CVDVContext struct, MeasurePlan, constants
  - api.h           — opaque handle + extern "C" public API declarations
  - kernels/        — __global__ / __device__ kernel headers (.cuh)
  - context.cu      — context lifecycle (create, destroy, init)
  - gates.cu        — CV, DV, hybrid, and two-mode gate implementations
  - cv.cu        — Wigner and Husimi Q phase-space functions
  - measure.cu      — measurement, fidelity, and expectation values
- src/cvdv/         — Python package (importable as `cvdv`)
  - cuda.py         — ctypes wrapper (CudaCvdv class)
  - torch.py        — PyTorch backend (TorchCvdv class)
  - separable.py    — per-register state initialisation
```

---

## Validation Checklist

After each `.cu` step (8–11):

```bash
make build
make test
```

After step 14 (delete originals):

```bash
make clean && make build && make test
make bench-api
make bench-kernel
```

---

## Cross-TU Call Graph

One cross-module host call:

```
measure.cu::cvdvGetPhotonNumber
    → gates.cu::cvdvFtQ2P
    → gates.cu::cvdvFtP2Q
```

Resolved by `#include "api.h"` in `measure.cu`. No cross-TU device-code calls.

---

## Dead Code — Drop, Do Not Migrate

Verify with `grep -r "kernelSet\|kernelNormalize" .` before deleting.

| Kernel | Reason |
|---|---|
| `kernelSetCoherent` | State init moved to Python (`SeparableState`) |
| `kernelSetFock` | Same |
| `kernelSetFocks` | Same |
| `kernelSetCat` | Same |
| `kernelComputeRegisterNorm` | No C API caller |
| `kernelNormalizeRegister` | No C API caller |
