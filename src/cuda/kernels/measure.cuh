#pragma once
#include "utils.cuh"

// Element-wise |ψ|² for cuTENSOR reduction (used by cvdvMeasureMultipleCT)
__global__ void kernelAbsSquareInPlace(double* __restrict__ out,
                                        const cuDoubleComplex* __restrict__ in,
                                        size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = absSquare(in[i]);
    }
}

// Fused kernel: compute |ψ[i]|² and reduce over non-selected register modes
// in a single pass, eliminating the intermediate |ψ|² buffer.
//
// Two paths:
//   outSize == 1  → blockReduceSum + single atomicAdd (full norm)
//   outSize > 1   → block-private shared memory histogram + global atomicAdd
//
// The block-private histogram avoids per-element atomicAdd to global memory.
// Each block accumulates into its own shared memory histogram, then the final
// per-bin sums are atomically added to global output.  This reduces global
// atomicAdd count from totalSize to numBlocks * outSize.
//
// selQbts / selFlwQbts are precomputed per-plan device arrays (regular
// cudaMalloc, never managed memory).  Using const __restrict__ lets the
// compiler route reads through the L1 read-only cache, giving broadcast
// semantics when all warp threads read the same index — equivalent to
// constant memory but safe for multiple concurrent contexts.
constexpr size_t MEASURE_SHARED_HIST_MAX = 4096;  // 32 KB shared mem limit

__global__ void kernelAbsSquareReduce(
        double* __restrict__ output,
        const cuDoubleComplex* __restrict__ state,
        size_t totalSize,
        const int* __restrict__ selQbts,     // qbts[regIdxs[s]],    s=0..numSelRegs-1
        const int* __restrict__ selFlwQbts,  // flwQbts[regIdxs[s]], s=0..numSelRegs-1
        int numSelRegs,
        const int64_t* __restrict__ outStrides,
        size_t outSize) {
    extern __shared__ double sPartial[];

    if (outSize == 1) {
        // ── Full-norm path: blockReduceSum → single atomicAdd ───────────
        double localSum = 0.0;
        for (size_t gi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
             gi < totalSize;
             gi += (size_t)blockDim.x * gridDim.x) {
            localSum += absSquare(state[gi]);
        }
        localSum = blockReduceSum(localSum);
        if (threadIdx.x == 0) atomicAdd(output, localSum);
    } else if (outSize <= MEASURE_SHARED_HIST_MAX) {
        // ── Small-output path: block-private shared memory histogram ────
        for (size_t i = threadIdx.x; i < outSize; i += blockDim.x)
            sPartial[i] = 0.0;
        __syncthreads();

        {
        int lane = threadIdx.x & 31;
        for (size_t gi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
             gi < totalSize;
             gi += (size_t)blockDim.x * gridDim.x) {
            double prob = absSquare(state[gi]);
            int64_t oi = 0;
            for (int s = 0; s < numSelRegs; s++) {
                size_t li = getLocalIndex(gi, selFlwQbts[s], selQbts[s]);
                oi += (int64_t)li * outStrides[s];
            }

            // Warp-aggregated atomicAdd: group lanes targeting the same bin,
            // reduce within group, single atomic per unique bin per warp.
            uint32_t active = __activemask();
            int64_t lane0_oi = __shfl_sync(active, oi, __ffs(active) - 1);
            if (__all_sync(active, oi == lane0_oi)) {
                // Fast path: all active lanes → same bin (common when flwQbts ≥ 5)
                double sum = prob;
                for (int off = 16; off > 0; off >>= 1)
                    sum += __shfl_down_sync(active, sum, off);
                if (lane == (__ffs(active) - 1))
                    atomicAdd(&sPartial[oi], sum);
            } else {
                // General path: iterate unique bins via leader election
                uint32_t remaining = active;
                while (remaining != 0) {
                    int leader = __ffs(remaining) - 1;
                    int64_t leader_oi = __shfl_sync(active, oi, leader);
                    uint32_t peers = __ballot_sync(active, oi == leader_oi) & remaining;
                    remaining &= ~peers;
                    // Reduce prob across peer lanes — all active lanes participate
                    // in shuffles (required by CUDA), only peers accumulate.
                    double peer_sum = 0.0;
                    uint32_t tmp = peers;
                    while (tmp != 0) {
                        int src = __ffs(tmp) - 1;
                        double val = __shfl_sync(active, prob, src);
                        if (oi == leader_oi) peer_sum += val;
                        tmp &= tmp - 1;
                    }
                    if (lane == leader)
                        atomicAdd(&sPartial[leader_oi], peer_sum);
                }
            }
        }
        }
        __syncthreads();

        for (size_t i = threadIdx.x; i < outSize; i += blockDim.x)
            atomicAdd(&output[i], sPartial[i]);
    } else {
        // ── Large-output path: direct global atomicAdd ──────────────────
        {
        int lane = threadIdx.x & 31;
        for (size_t gi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
             gi < totalSize;
             gi += (size_t)blockDim.x * gridDim.x) {
            double prob = absSquare(state[gi]);
            int64_t oi = 0;
            for (int s = 0; s < numSelRegs; s++) {
                size_t li = getLocalIndex(gi, selFlwQbts[s], selQbts[s]);
                oi += (int64_t)li * outStrides[s];
            }
            uint32_t active = __activemask();
            int64_t lane0_oi = __shfl_sync(active, oi, __ffs(active) - 1);
            if (__all_sync(active, oi == lane0_oi)) {
                double sum = prob;
                for (int off = 16; off > 0; off >>= 1)
                    sum += __shfl_down_sync(active, sum, off);
                if (lane == (__ffs(active) - 1))
                    atomicAdd(&output[oi], sum);
            } else {
                uint32_t remaining = active;
                while (remaining != 0) {
                    int leader = __ffs(remaining) - 1;
                    int64_t leader_oi = __shfl_sync(active, oi, leader);
                    uint32_t peers = __ballot_sync(active, oi == leader_oi) & remaining;
                    remaining &= ~peers;
                    double peer_sum = 0.0;
                    uint32_t tmp = peers;
                    while (tmp != 0) {
                        int src = __ffs(tmp) - 1;
                        double val = __shfl_sync(active, prob, src);
                        if (oi == leader_oi) peer_sum += val;
                        tmp &= tmp - 1;
                    }
                    if (lane == leader)
                        atomicAdd(&output[leader_oi], peer_sum);
                }
            }
        }
        }
    }
}

// Kernel to compute <x²> expectation for a register: sum |state[i]|² *
// x_localIdx²
__global__ void kernelExpectX2(double* partialSums, const cuDoubleComplex* state, size_t totalSize,
                               int regIdx, const int* qbts, const double* gridSteps,
                               const int* flwQbts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localSum = 0.0;

    int regDim = 1 << qbts[regIdx];
    double dx = gridSteps[regIdx];

    for (size_t globalIdx = idx; globalIdx < totalSize; globalIdx += blockDim.x * gridDim.x) {
        size_t localIdx = getLocalIndex(globalIdx, flwQbts[regIdx], qbts[regIdx]);
        double x = gridX(localIdx, regDim, dx);
        localSum += absSquare(state[globalIdx]) * x * x;
    }

    // Modern warp-level reduction (faster, no bank conflicts)
    localSum = blockReduceSum(localSum);

    if (threadIdx.x == 0) partialSums[blockIdx.x] = localSum;
}

// Kernel to compute tensor product element and its contribution to inner
// product Each thread computes one element of the tensor product and
// accumulates <current_state | tensor_product>
__global__ void kernelComputeInnerProduct(cuDoubleComplex* partialSums,
                                          const cuDoubleComplex* state,
                                          cuDoubleComplex** dRegisterArrays, int numReg,
                                          const int* qbts, const int* flwQbts, size_t totalSize) {
    // Shared memory for reduction within block
    extern __shared__ cuDoubleComplex sdata[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    cuDoubleComplex localSum = make_cuDoubleComplex(0.0, 0.0);

    // Grid-stride loop
    for (size_t globalIdx = idx; globalIdx < totalSize; globalIdx += blockDim.x * gridDim.x) {
        // Compute tensor product element for this global index
        cuDoubleComplex tensorElement = make_cuDoubleComplex(1.0, 0.0);

        for (int r = 0; r < numReg; r++) {
            size_t localIdx = getLocalIndex(globalIdx, flwQbts[r], qbts[r]);
            tensorElement = cuCmul(tensorElement, dRegisterArrays[r][localIdx]);
        }

        // Accumulate <state | tensor_product> = sum conj(state) * tensor_product
        cuDoubleComplex stateVal = state[globalIdx];
        localSum = cuCadd(localSum, conjMul(stateVal, tensorElement));
    }

    // Store in shared memory
    sdata[threadIdx.x] = localSum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // Write block result
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// Kernel to compute inner product <psi1|psi2> = sum conj(psi1[i]) * psi2[i]
__global__ void kernelInnerProductStatevectors(cuDoubleComplex* partialSums,
                                               const cuDoubleComplex* psi1,
                                               const cuDoubleComplex* psi2, size_t totalSize) {
    extern __shared__ cuDoubleComplex sdata[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    cuDoubleComplex localSum = make_cuDoubleComplex(0.0, 0.0);

    for (size_t globalIdx = idx; globalIdx < totalSize; globalIdx += blockDim.x * gridDim.x) {
        localSum = cuCadd(localSum, conjMul(psi1[globalIdx], psi2[globalIdx]));
    }

    sdata[threadIdx.x] = localSum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = cuCadd(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}
