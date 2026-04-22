#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cutensor.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include "cudapp.cuh"

#define checkCudaErrors(val) CHECK_CUDA(val)

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
    // Device memory (RAII — automatic cleanup on ctx destroy)
    cudapp::CudaVector<cuDoubleComplex, cudapp::CudaDeviceArena> state;
    cudapp::CudaVector<int, cudapp::CudaDeviceArena> dQbts;
    cudapp::CudaVector<int, cudapp::CudaDeviceArena> dFlwQbts;
    cudapp::CudaVector<double, cudapp::CudaDeviceArena> dGridSteps;

    // Host mirrors (std::vector — automatic cleanup)
    std::vector<int> hQbts;
    std::vector<int> hFlwQbts;
    std::vector<double> hGridSteps;

    // cuFFT plans (RAII — automatic cleanup)
    std::vector<cudapp::CudaFftPlan> ftPlans;

    int gNumReg = 0;
    int gTotalQbt = 0;

    // Raw-pointer accessors for kernel calls
    cuDoubleComplex* dState() { return state.data(); }
    int* gpQbts() { return dQbts.data(); }
    int* gpFlwQbts() { return dFlwQbts.data(); }
    double* gpGridSteps() { return dGridSteps.data(); }
} CVDVContext;
#define CVDV_TYPES_INCLUDED

// Lightweight struct for passing separable-state device pointers into CUDA
// functions. Python fills this from
// SeparableState.register_arrays[i].data_ptr().
typedef struct {
    cuDoubleComplex** ptrs;  // host-side array of numReg device pointers
    int numReg;
} SeparableRegArrays;
