#pragma once
#include "../types.h"

// Position value at grid index: x_i = (i - (cvDim - 1)/2) * dx
__device__ __host__ inline double gridX(int idx, int cvDim, double dx) {
    return (idx - (cvDim - 1) * 0.5) * dx;
}

// Convert phase to complex exponential: e^{i*phase}
__device__ __host__ inline cuDoubleComplex phaseToZ(double phase) {
    return make_cuDoubleComplex(cos(phase), sin(phase));
}

// Multiply complex number by phase factor: z * e^{i*phase}
__device__ __host__ inline cuDoubleComplex cmulPhase(cuDoubleComplex z, double phase) {
    return cuCmul(phaseToZ(phase), z);
}

// Conjugate multiply: conj(a) * b
__device__ __host__ inline cuDoubleComplex conjMul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(cuConj(a), b);
}

__device__ __host__ inline double absSquare(cuDoubleComplex z) {
    return cuCreal(z) * cuCreal(z) + cuCimag(z) * cuCimag(z);
}

__device__ __host__ inline cuDoubleComplex cuCmul(cuDoubleComplex a, double b) {
    return make_cuDoubleComplex(cuCreal(a) * b, cuCimag(a) * b);
}

// Extract local index within a register from global index
// Returns the index for register regIdx given the global flat index
__device__ __host__ inline size_t getLocalIndex(size_t globalIdx, int flwQbtCount,
                                                int regQbtCount) {
    return (globalIdx >> flwQbtCount) & ((1 << regQbtCount) - 1);
}

// Warp-level reduction using shuffle instructions (modern CUDA reduction)
// Avoids shared memory and bank conflicts for intra-warp reduction
__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using warp reduction + minimal shared memory
// This is the modern standard for CUDA reductions (faster than tree-based
// shared memory)
__device__ __forceinline__ double blockReduceSum(double val) {
    static __shared__ double warpSums[32];  // Max 1024 threads / 32 = 32 warps

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    // Reduce within warp using shuffle
    val = warpReduceSum(val);

    // First thread in each warp writes to shared memory
    if (lane == 0) warpSums[wid] = val;
    __syncthreads();

    // First warp reduces the warp sums
    if (threadIdx.x < 32) {
        val = (threadIdx.x < blockDim.x / 32) ? warpSums[threadIdx.x] : 0.0;
        val = warpReduceSum(val);
    }

    return val;
}

// Expand a compressed pair-index into the two global state indices that differ
// only at the target qubit bit. Standard state-vector simulator pair-indexing.
//   pairIdx ∈ [0, totalSize/2)
//   globalBit = bit position of the target qubit in the flat global index
//             = flwQbts[r] + qbts[r] - 1 - qubitIdx  (qubit 0 is MSB of
//             register)
__device__ __forceinline__ void qubitPairIndices(size_t pairIdx, size_t globalBit, size_t& idx0,
                                                 size_t& idx1) {
    size_t mask = (size_t)1 << globalBit;
    idx0 = ((pairIdx & ~(mask - 1)) << 1) | (pairIdx & (mask - 1));
    idx1 = idx0 | mask;
}
