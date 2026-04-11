#pragma once
#include "utils.cuh"

// Kernel to compute element-wise |ψ|² for cuTENSOR reduction
__global__ void kernelAbsSquareInPlace(double* __restrict__ out,
                                        const cuDoubleComplex* __restrict__ in,
                                        size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = absSquare(in[i]);
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

// Kernel to compute norm (sum of |state[i]|^2)
__global__ void kernelComputeNorm(double* partialSums, const cuDoubleComplex* state,
                                  size_t totalSize) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localSum = 0.0;

    // Grid-stride loop
    for (size_t globalIdx = idx; globalIdx < totalSize; globalIdx += blockDim.x * gridDim.x) {
        localSum += absSquare(state[globalIdx]);
    }

    // Modern warp-level reduction (faster, no bank conflicts)
    localSum = blockReduceSum(localSum);

    // Write block result
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = localSum;
    }
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
