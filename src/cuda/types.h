#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cutensor.h>

#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

// Modern error handling without file logging
#define checkCudaErrors(val)                                                           \
    do {                                                                                \
        cudaError_t err = (val);                                                        \
        if (err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880;
constexpr double PI_POW_NEG_QUARTER = 0.75112554446494248286;  // PI^(-0.25)
constexpr int CUDA_BLOCK_SIZE = 256;  // Default CUDA block size for RTX 4070 Laptop

// Cached plan for a specific register selection in MeasureMultiple.
// Keyed by the regIdxs vector; stored in a heap-allocated std::map.
//
// dSelQbts / dSelFlwQbts are precomputed at plan-build time from the managed
// gQbts / gFlwQbts arrays and stored as regular device memory.  This removes
// any managed-memory pointer from the kernel's hot path, eliminating the
// ~2 ms unified-memory page-migration overhead per call.
struct MeasurePlan {
    int*     dSelQbts;       // device: qbts[regIdxs[s]]    for s=0..numRegs-1
    int*     dSelFlwQbts;    // device: flwQbts[regIdxs[s]] for s=0..numRegs-1
    int64_t* dOutStrides;    // device: output strides [numRegs]
    size_t   outSize;        // product of selected register dimensions
};

// cuTENSOR-based plan for MeasureMultipleCT (baseline comparison path).
struct MeasurePlanCT {
    cutensorTensorDescriptor_t  descIn;
    cutensorTensorDescriptor_t  descOut;
    cutensorOperationDescriptor_t opDesc;
    cutensorPlanPreference_t    planPref;
    cutensorPlan_t              plan;
    void*                       dWorkspace;
    uint64_t                    workspaceSize;
    size_t                      outSize;
};

// Context structure to enable multiple instances
typedef struct CVDVContext {
    cuDoubleComplex* dState;
    int* gQbts;          // Device memory: number of qubits in each register
    int* gFlwQbts;       // Device memory: cumulative qubits after each register
    double* gGridSteps;  // Device memory: grid step (dx) for each register
    int* hQbts;          // Host mirror of gQbts (for CPU-side reads)
    int* hFlwQbts;       // Host mirror of gFlwQbts
    double* hGridSteps;  // Host mirror of gGridSteps
    int gNumReg;         // Total number of registers
    int gTotalQbt;       // Total number of qubits across all registers

    // Cached cuFFT plans for FT operations — one plan per register (host-only,
    // regular malloc). Each plan covers both FORWARD and INVERSE (direction is a
    // runtime arg to cufftExecZ2Z).
    cufftHandle* ftPlans;

    // Cached plans for Wigner/Husimi - simplified native grid only
    cufftHandle wPlan;
    int wPlanCvDim;
    bool wPlanValid;
    // Batched Husimi plans: forward FFT (batch=chunkSize) and inverse FFT (batch=chunkSize*N)
    cufftHandle hBatchFwdPlan;
    cufftHandle hBatchInvPlan;
    int hBatchCvDim;
    int hBatchChunkSize;
    bool hBatchPlanValid;

    // Cached analytic Gaussian kernel G[k] = FFT{g}[k] for Husimi
    cuDoubleComplex* dHusimiG;
    int hGCvDim;
    double hGDx;
    bool hGValid;

    // Cached buffers for MeasureMultiple (lazy-init)
    double* dMeasureOut;         // marginal output scratch, size = totalSize
    void*   measurePlanCache;    // heap-allocated std::map<std::vector<int>, MeasurePlan>*

    // cuTENSOR baseline path (cvdvMeasureMultipleCT)
    cutensorHandle_t ctHandle;
    bool ctHandleValid;
    double* dMeasureProbs;       // |ψ|² scratch, size = totalSize
    void*   measurePlanCacheCT;  // heap-allocated std::map<std::vector<int>, MeasurePlanCT>*
} CVDVContext;
#define CVDV_TYPES_INCLUDED

// Lightweight struct for passing separable-state device pointers into CUDA
// functions. Python fills this from
// SeparableState.register_arrays[i].data_ptr().
typedef struct {
    cuDoubleComplex** ptrs;  // host-side array of numReg device pointers
    int numReg;
} SeparableRegArrays;
