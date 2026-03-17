# Eliminating cuFFT Overhead in a CUDA Quantum Simulator

*March 2026 — NVIDIA RTX 4070 Laptop GPU*

## Background

[CUDA-CVDV](../README.md) is a CUDA library for simulating hybrid CV-DV (continuous-variable / discrete-variable) quantum circuits. The state vector lives on a grid of size `2^n` per bosonic register and the main computational primitive is a phase-space Fourier transform — an index-shifted QFT that maps between position and momentum representations, used on every bosonic gate (displacement, squeezing, rotation, beam splitter, ...).

The QFT is implemented as:

```
1. Phase kick exp(i·π(N-1)/N · j)   [kernelPhaseX]
2. cuFFT forward/inverse             [cufftExecZ2Z]
3. Phase kick exp(i·π(N-1)/N · k)   [kernelPhaseX]
4. Normalize 1/√N                    [kernelGlobalScalar]
5. Global phase correction           [kernelGlobalPhase]
```

## The Problem

The original code created and destroyed a cuFFT plan on **every single call**:

```c
void cvdvFtQ2P(CVDVContext* ctx, int regIdx) {
    // ...
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, batch);   // ← expensive every call
    cufftExecZ2Z(plan, ...);
    cufftDestroy(plan);                         // ← wasteful
}
```

And every step had an explicit `cudaDeviceSynchronize()`:

```c
kernelPhaseX<<<...>>>();
cudaDeviceSynchronize();   // stall 1
cufftExecZ2Z(...);
cudaDeviceSynchronize();   // stall 2
kernelPhaseX<<<...>>>();
cudaDeviceSynchronize();   // stall 3
kernelGlobalScalar<<<...>>>();
cudaDeviceSynchronize();   // stall 4
kernelGlobalPhase<<<...>>>();
cudaDeviceSynchronize();   // stall 5
```

That's **5 CPU↔GPU round-trips and 1 plan allocation per gate call**, for every bosonic gate in every circuit.

## Why This Doesn't Show Up in Kernel Profilers

When profiling per-kernel GPU execution time (Nsight, `ncu`), these optimizations are invisible — the kernels themselves do the same work. The overhead is **host-side**:

- `cufftPlan1d` triggers driver API calls, workspace allocation, and internal plan compilation. On an RTX 4070 Laptop this costs several microseconds per call even for small transforms, and scales with transform size.
- `cudaDeviceSynchronize` stalls the CPU until the entire GPU pipeline drains. Between sequential operations on the same (null) stream this is purely wasted latency — the hardware already serializes them.

The savings show up as **reduced end-to-end latency** per gate, which accumulates across hundreds of gate applications in a typical circuit.

## The Fix

### 1. Plan cache in `CVDVContext`

Since the FFT parameters (size, batch count, stride layout) are fully determined by register configuration at context creation time, plans can be created once and reused:

```c
typedef struct {
    // ... existing fields ...

    // Per-register FT plans — created at cvdvCreate, destroyed at cvdvDestroy.
    // One plan per register; same handle works for FORWARD and INVERSE
    // (direction is a runtime argument to cufftExecZ2Z).
    cufftHandle* ftPlans;

    // Wigner/Husimi plans — cached, recreated only if batch size changes.
    cufftHandle wsPlan;  int wsPlanCvDim;  int wsPlanBatch;  bool wsPlanValid;
    cufftHandle wfPlan;  int wfPlanCvDim;  int wfPlanBatch;  bool wfPlanValid;
    cufftHandle hPlan;   int hPlanCvDim;   int hPlanBatch;   bool hPlanValid;
} CVDVContext;
```

Plans are created eagerly in `cvdvCreate`:

```c
ctx->ftPlans = (cufftHandle*)malloc(numReg * sizeof(cufftHandle));
for (int i = 0; i < numReg; i++) {
    int n = 1 << ctx->gQbts[i];
    size_t regStride = (size_t)1 << ctx->gFlwQbts[i];
    if (regStride == 1) {
        cufftPlan1d(&ctx->ftPlans[i], n, CUFFT_Z2Z, totalSize / n);
    } else {
        int nembed[1] = {n * (int)regStride};
        cufftPlanMany(&ctx->ftPlans[i], 1, &n,
                      nembed, regStride, 1,
                      nembed, regStride, 1, CUFFT_Z2Z, regStride);
    }
}
```

The Wigner/Husimi plans can't be cached at creation time because the number of output samples (`wignerN`, `qN`) varies per call. They use a last-value cache instead — recreated only when parameters change:

```c
if (!ctx->wfPlanValid || ctx->wfPlanCvDim != cvDim || ctx->wfPlanBatch != wignerN) {
    if (ctx->wfPlanValid) cufftDestroy(ctx->wfPlan);
    cufftPlan1d(&ctx->wfPlan, cvDim, CUFFT_Z2Z, wignerN);
    ctx->wfPlanCvDim = cvDim; ctx->wfPlanBatch = wignerN; ctx->wfPlanValid = true;
}
cufftExecZ2Z(ctx->wfPlan, dBuf, dBuf, CUFFT_INVERSE);
```

### 2. Remove intermediate synchronizations

On CUDA's null stream (stream 0), all operations — kernel launches and cuFFT executions — are **already serialized by the hardware**. No explicit sync is needed between sequential steps.

The rule is simple: `cudaDeviceSynchronize()` is only needed when the **CPU needs to read results from the GPU**. Between GPU operations on the same stream, it is pure overhead.

```c
// After: 5 syncs → 1
kernelPhaseX<<<...>>>();
cufftExecZ2Z(plan, ...);       // serialized on null stream — no sync needed
kernelPhaseX<<<...>>>();
kernelGlobalScalar<<<...>>>();
kernelGlobalPhase<<<...>>>();
cudaDeviceSynchronize();       // only here, before returning to Python
```

## Results

Benchmarked on the CV-to-DV state transfer circuit (cat state, 4 DV qubits):

| System | CV dimension | Mean time |
|---|---|---|
| bosonic-qiskit (CPU, Fock basis) | 128 (7 qubits) | 1079 ms |
| CUDA-CVDV | 1024 (10 qubits) | 8.4 ms |
| CUDA-CVDV | 4096 (12 qubits) | 9.8 ms |
| CUDA-CVDV | **16384 (14 qubits)** | **12.4 ms** |

CUDA-CVDV runs a circuit at **128× larger dimension in 87× less time** than bosonic-qiskit at its practical limit. The near-flat scaling from 1024 to 16384 reflects memory bandwidth saturation — the GPU is fully utilized across the entire range.

The plan caching and sync removal contribute to the tight absolute numbers: on a circuit with many gate applications, each avoided plan creation and CPU stall accumulates into measurable wall-clock savings.
