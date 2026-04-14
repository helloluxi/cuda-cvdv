// cvdv.cu - CUDA library for hybrid CV-DV quantum simulation

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

#pragma region Error Checking and Constants

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

#pragma endregion

#pragma region Global State

// Cached cuTENSOR plan for a specific register selection in MeasureMultiple.
// Keyed by the regIdxs vector; stored in a heap-allocated std::map.
struct MeasurePlan {
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
typedef struct {
    cuDoubleComplex* dState;
    int* gQbts;          // Managed memory: number of qubits in each register
    int* gFlwQbts;       // Managed memory: cumulative qubits after each register
    double* gGridSteps;  // Managed memory: grid step (dx) for each register
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
    cufftHandle hPlan;      // batch of N IFFTs (direction set at exec time)
    int hPlanCvDim;
    bool hPlanValid;
    cufftHandle hSinglePlan;  // single forward FFT for per-slice ψ
    bool hSinglePlanValid;

    // Cached analytic Gaussian kernel G[k] = FFT{g}[k] for Husimi
    cuDoubleComplex* dHusimiG;
    int hGCvDim;
    double hGDx;
    bool hGValid;

    // Cached cuTENSOR handle and buffers for MeasureMultiple (lazy-init)
    cutensorHandle_t ctHandle;
    bool ctHandleValid;
    double* dMeasureProbs;       // |ψ|² scratch, size = totalSize
    double* dMeasureOut;         // marginal output scratch, size = totalSize
    void*   measurePlanCache;    // heap-allocated std::map<std::vector<int>, MeasurePlan>*
} CVDVContext;

// Lightweight struct for passing separable-state device pointers into CUDA
// functions. Python fills this from
// SeparableState.register_arrays[i].data_ptr().
typedef struct {
    cuDoubleComplex** ptrs;  // host-side array of numReg device pointers
    int numReg;
} SeparableRegArrays;

#pragma endregion

#pragma region Device Helper Functions

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

#pragma endregion

#pragma region State Initialization Kernels

__global__ void kernelSetCoherent(cuDoubleComplex* state, int cvDim, double dx, double alphaRe,
                                  double alphaIm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;

    double x = gridX(idx, cvDim, dx);
    // Coherent state: |alpha> in position representation
    // q = sqrt(2) * Re(alpha), p = sqrt(2) * Im(alpha)
    double q = SQRT2 * alphaRe;
    double p = SQRT2 * alphaIm;

    double norm = PI_POW_NEG_QUARTER;
    double gauss = exp(-0.5 * (x - q) * (x - q));
    double phase = p * x - p * q / 2.0;

    cuDoubleComplex phaseFactor = phaseToZ(phase);
    double amplitude = norm * gauss * sqrt(dx);  // Normalize as qubit register

    state[idx] =
        make_cuDoubleComplex(amplitude * cuCreal(phaseFactor), amplitude * cuCimag(phaseFactor));
}

__global__ void kernelSetFock(cuDoubleComplex* state, int cvDim, double dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;

    double x = gridX(idx, cvDim, dx);

    // Fock state |n> using normalized Hermite function recurrence for numerical
    // stability: psi_0(x) = exp(-x^2/2) / pi^(1/4) * sqrt(dx) psi_n(x) =
    // sqrt(2/n) * x * psi_{n-1}(x) - sqrt((n-1)/n) * psi_{n-2}(x)
    double psiPrev = 0.0;
    double psiCurr = exp(-0.5 * x * x) * PI_POW_NEG_QUARTER * sqrt(dx);

    for (int k = 1; k <= n; k++) {
        double psiNext = sqrt(2.0 / k) * x * psiCurr - sqrt((k - 1.0) / k) * psiPrev;
        psiPrev = psiCurr;
        psiCurr = psiNext;
    }

    state[idx] = make_cuDoubleComplex(psiCurr, 0.0);
}

__global__ void kernelSetFocks(cuDoubleComplex* state, int cvDim, double dx,
                               const cuDoubleComplex* coeffs, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;
    double x = gridX(idx, cvDim, dx);

    // Fock state superposition using normalized Hermite function recurrence:
    // psi_0(x) = exp(-x^2/2) / pi^(1/4) * sqrt(dx)
    // psi_n(x) = sqrt(2/n) * x * psi_{n-1}(x) - sqrt((n-1)/n) * psi_{n-2}(x)
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    double psiPrev = 0.0;
    double psiCurr = exp(-0.5 * x * x) * PI_POW_NEG_QUARTER * sqrt(dx);

    for (int n = 0; n < length; n++) {
        if (n > 0) {
            double psiNext = sqrt(2.0 / n) * x * psiCurr - sqrt((n - 1.0) / n) * psiPrev;
            psiPrev = psiCurr;
            psiCurr = psiNext;
        }

        // Add coefficient * |n>
        cuDoubleComplex term = cuCmul(coeffs[n], make_cuDoubleComplex(psiCurr, 0.0));
        result = cuCadd(result, term);
    }
    state[idx] = result;
}

// Cat state: superposition of coherent states
// Takes array of [alpha_i, coeff_i] pairs
__global__ void kernelSetCat(cuDoubleComplex* state, int cvDim, double dx,
                             const cuDoubleComplex* alphas, const cuDoubleComplex* coeffs,
                             int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;

    double x = gridX(idx, cvDim, dx);
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    // Compute superposition of coherent states: sum_i coeff_i * |alpha_i>
    for (int i = 0; i < length; i++) {
        double alphaRe = cuCreal(alphas[i]);
        double alphaIm = cuCimag(alphas[i]);

        // Coherent state |alpha> in position representation
        // q = sqrt(2) * Re(alpha), p = sqrt(2) * Im(alpha)
        double q = SQRT2 * alphaRe;
        double p = SQRT2 * alphaIm;

        double norm = PI_POW_NEG_QUARTER;
        double gauss = exp(-0.5 * (x - q) * (x - q));
        double phase = p * x - p * q / 2.0;

        cuDoubleComplex phaseFactor = phaseToZ(phase);
        double amplitude = norm * gauss * sqrt(dx);

        cuDoubleComplex coherentState = make_cuDoubleComplex(amplitude * cuCreal(phaseFactor),
                                                             amplitude * cuCimag(phaseFactor));

        // Add coeff_i * |alpha_i>
        result = cuCadd(result, cuCmul(coeffs[i], coherentState));
    }

    state[idx] = result;
}

// Compute norm for a single register array (reduction within block)
__global__ void kernelComputeRegisterNorm(double* partialSums, const cuDoubleComplex* state,
                                          int cvDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localSum = 0.0;

    // Grid-stride loop
    for (int i = idx; i < cvDim; i += blockDim.x * gridDim.x) {
        localSum += absSquare(state[i]);
    }

    // Modern warp-level reduction (faster, no bank conflicts)
    localSum = blockReduceSum(localSum);

    // Write block result
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = localSum;
    }
}

// Normalize register array by dividing by a scalar
__global__ void kernelNormalizeRegister(cuDoubleComplex* state, int cvDim, double invNorm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;

    state[idx] = make_cuDoubleComplex(cuCreal(state[idx]) * invNorm, cuCimag(state[idx]) * invNorm);
}

#pragma endregion

#pragma region Gate Kernels

// Apply phase factor to a specific register: exp(i*phaseCoeff*x)
__global__ void kernelPhaseX(cuDoubleComplex* state, size_t totalSize, int regIdx, const int* qbts,
                             const double* gridSteps, const int* flwQbts, double phaseCoeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = 1 << qbts[regIdx];
    double dx = gridSteps[regIdx];
    size_t localIdx = getLocalIndex(idx, flwQbts[regIdx], qbts[regIdx]);

    double x = gridX(localIdx, regDim, dx);
    state[idx] = cmulPhase(state[idx], phaseCoeff * x);
}

// Apply controlled phase to a specific register with control from another
// register exp(i*phaseCoeff*Z*x) where Z acts on ctrlQubit in ctrlReg
__global__ void kernelCPhaseX(cuDoubleComplex* state, size_t totalSize, int targetReg, int ctrlReg,
                              int ctrlQubit, const int* qbts, const double* gridSteps,
                              const int* flwQbts, double phaseCoeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t targetDim = 1 << qbts[targetReg];
    double dx = gridSteps[targetReg];
    size_t targetLocalIdx = getLocalIndex(idx, flwQbts[targetReg], qbts[targetReg]);

    // Extract control qubit state
    size_t ctrlLocalIdx = getLocalIndex(idx, flwQbts[ctrlReg], qbts[ctrlReg]);
    int ctrlMask = 1 << (qbts[ctrlReg] - 1 - ctrlQubit);

    double x = gridX(targetLocalIdx, targetDim, dx);
    double phase = phaseCoeff * x;

    state[idx] = cmulPhase(state[idx], (ctrlLocalIdx & ctrlMask) ? -phase : phase);
}

// Apply controlled phase based on position squared: exp(i*t*Z*x^2)
// where Z acts on ctrlQubit in ctrlReg
__global__ void kernelCPhaseX2(cuDoubleComplex* state, size_t totalSize, int targetReg, int ctrlReg,
                               int ctrlQubit, const int* qbts, const double* gridSteps,
                               const int* flwQbts, double t) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t targetDim = 1 << qbts[targetReg];
    double dx = gridSteps[targetReg];
    size_t targetLocalIdx = getLocalIndex(idx, flwQbts[targetReg], qbts[targetReg]);

    // Extract control qubit state
    size_t ctrlLocalIdx = getLocalIndex(idx, flwQbts[ctrlReg], qbts[ctrlReg]);
    int ctrlMask = 1 << (qbts[ctrlReg] - 1 - ctrlQubit);

    double x = gridX(targetLocalIdx, targetDim, dx);
    double phase = t * x * x;

    state[idx] = cmulPhase(state[idx], (ctrlLocalIdx & ctrlMask) ? -phase : phase);
}

// Pauli rotation Rn(θ) on a single qubit: exp(-i·θ/2·σn), axis 0=X 1=Y 2=Z
__global__ void kernelPauliRotation(cuDoubleComplex* state, size_t totalSize, int regIdx,
                                    int qubitIdx, const int* qbts, const int* flwQbts, int axis,
                                    double theta) {
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= totalSize / 2) return;

    size_t idx0, idx1;
    qubitPairIndices(pairIdx, flwQbts[regIdx] + qbts[regIdx] - 1 - qubitIdx, idx0, idx1);

    cuDoubleComplex a = state[idx0];
    cuDoubleComplex b = state[idx1];
    double c = cos(theta / 2.0), s = sin(theta / 2.0);

    // Rx: [[c, -is], [-is, c]]   Ry: [[c, -s], [s, c]]   Rz: [[e^{-is}, 0], [0,
    // e^{is}]]
    if (axis == 0) {  // X: newA = c·a - i·s·b,  newB = -i·s·a + c·b
        state[idx0] =
            make_cuDoubleComplex(c * cuCreal(a) + s * cuCimag(b), c * cuCimag(a) - s * cuCreal(b));
        state[idx1] =
            make_cuDoubleComplex(s * cuCimag(a) + c * cuCreal(b), -s * cuCreal(a) + c * cuCimag(b));
    } else if (axis == 1) {  // Y: newA = c·a - s·b,  newB = s·a + c·b
        state[idx0] =
            make_cuDoubleComplex(c * cuCreal(a) - s * cuCreal(b), c * cuCimag(a) - s * cuCimag(b));
        state[idx1] =
            make_cuDoubleComplex(s * cuCreal(a) + c * cuCreal(b), s * cuCimag(a) + c * cuCimag(b));
    } else {  // Z: newA = e^{-is}·a,  newB = e^{is}·b
        state[idx0] = cuCmul(make_cuDoubleComplex(c, -s), a);
        state[idx1] = cuCmul(make_cuDoubleComplex(c, s), b);
    }
}

// Hadamard gate on a single qubit: H = 1/√2 · [[1,1],[1,-1]]
__global__ void kernelHadamard(cuDoubleComplex* state, size_t totalSize, int regIdx, int qubitIdx,
                               const int* qbts, const int* flwQbts) {
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= totalSize / 2) return;

    size_t idx0, idx1;
    qubitPairIndices(pairIdx, flwQbts[regIdx] + qbts[regIdx] - 1 - qubitIdx, idx0, idx1);

    cuDoubleComplex a = state[idx0], b = state[idx1];
    constexpr double INV_SQRT2 = 1.0 / SQRT2;
    state[idx0] = make_cuDoubleComplex(INV_SQRT2 * (cuCreal(a) + cuCreal(b)),
                                       INV_SQRT2 * (cuCimag(a) + cuCimag(b)));
    state[idx1] = make_cuDoubleComplex(INV_SQRT2 * (cuCreal(a) - cuCreal(b)),
                                       INV_SQRT2 * (cuCimag(a) - cuCimag(b)));
}

// Parity gate: flip all qubits of a register (X on each qubit = reverse index)
// This maps |j⟩ → |N-1-j⟩ for the target register
__global__ void kernelParity(cuDoubleComplex* state, size_t totalSize, int regIdx, const int* qbts,
                             const int* flwQbts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = 1 << qbts[regIdx];
    size_t localIdx = getLocalIndex(idx, flwQbts[regIdx], qbts[regIdx]);

    // Only process pairs where localIdx < flipped to avoid double-swap
    size_t flipped = (regDim - 1) - localIdx;
    if (localIdx >= flipped) return;

    // Compute partner global index by replacing local index
    size_t regStride = 1 << flwQbts[regIdx];
    size_t partnerIdx = idx + (flipped - localIdx) * regStride;

    cuDoubleComplex tmp = state[idx];
    state[idx] = state[partnerIdx];
    state[partnerIdx] = tmp;
}

__global__ void kernelConditionalParity(cuDoubleComplex* state, size_t totalSize, int targetReg,
                                        int ctrlReg, int ctrlQubit, const int* qbts,
                                        const int* flwQbts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    // Only act on |1⟩ control branch
    size_t ctrlLocalIdx = getLocalIndex(idx, flwQbts[ctrlReg], qbts[ctrlReg]);
    int ctrlMask = 1 << (qbts[ctrlReg] - 1 - ctrlQubit);
    if (!(ctrlLocalIdx & ctrlMask)) return;

    size_t regDim = 1 << qbts[targetReg];
    size_t localIdx = getLocalIndex(idx, flwQbts[targetReg], qbts[targetReg]);

    size_t flipped = (regDim - 1) - localIdx;
    if (localIdx >= flipped) return;

    size_t regStride = 1 << flwQbts[targetReg];
    size_t partnerIdx = idx + (flipped - localIdx) * regStride;

    cuDoubleComplex tmp = state[idx];
    state[idx] = state[partnerIdx];
    state[partnerIdx] = tmp;
}

// SWAP gate: swap the qubit contents of two registers (must have same
// numQubits) Maps |i⟩₁|j⟩₂ → |j⟩₁|i⟩₂
__global__ void kernelSwapRegisters(cuDoubleComplex* state, size_t totalSize, int reg1, int reg2,
                                    const int* qbts, const int* flwQbts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t local1 = getLocalIndex(idx, flwQbts[reg1], qbts[reg1]);
    size_t local2 = getLocalIndex(idx, flwQbts[reg2], qbts[reg2]);

    // Only process when local1 < local2 to avoid double-swap
    if (local1 >= local2) return;

    // Compute partner index: swap local1 and local2
    size_t stride1 = 1 << flwQbts[reg1];
    size_t stride2 = 1 << flwQbts[reg2];
    size_t partnerIdx = idx + (local2 - local1) * stride1  // reg1 gets local2
                        - (local2 - local1) * stride2;     // reg2 gets local1

    cuDoubleComplex tmp = state[idx];
    state[idx] = state[partnerIdx];
    state[partnerIdx] = tmp;
}

// Apply phase based on position squared: exp(i*t*x^2)
__global__ void kernelPhaseX2(cuDoubleComplex* state, size_t totalSize, int regIdx, const int* qbts,
                              const double* gridSteps, const int* flwQbts, double t) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = 1 << qbts[regIdx];
    double dx = gridSteps[regIdx];
    size_t localIdx = getLocalIndex(idx, flwQbts[regIdx], qbts[regIdx]);

    double x = gridX(localIdx, regDim, dx);
    state[idx] = cmulPhase(state[idx], t * x * x);
}

// Apply phase factor to a specific register: exp(i*phaseCoeff*x)
__global__ void kernelPhaseX3(cuDoubleComplex* state, size_t totalSize, int regIdx, const int* qbts,
                              const double* gridSteps, const int* flwQbts, double phaseCoeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = 1 << qbts[regIdx];
    double dx = gridSteps[regIdx];
    size_t localIdx = getLocalIndex(idx, flwQbts[regIdx], qbts[regIdx]);

    double x = gridX(localIdx, regDim, dx);
    state[idx] = cmulPhase(state[idx], phaseCoeff * x * x * x);
}

// Apply two-mode position coupling: exp(i*coeff*q1*q2)
// where q1 and q2 are position operators for two registers
__global__ void kernelPhaseXX(cuDoubleComplex* state, size_t totalSize, int reg1Idx, int reg2Idx,
                              const int* qbts, const double* gridSteps, const int* flwQbts,
                              double coeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t reg1Dim = 1 << qbts[reg1Idx];
    size_t reg2Dim = 1 << qbts[reg2Idx];
    double dx1 = gridSteps[reg1Idx];
    double dx2 = gridSteps[reg2Idx];

    size_t local1 = getLocalIndex(idx, flwQbts[reg1Idx], qbts[reg1Idx]);
    size_t local2 = getLocalIndex(idx, flwQbts[reg2Idx], qbts[reg2Idx]);

    double q1 = gridX(local1, reg1Dim, dx1);
    double q2 = gridX(local2, reg2Dim, dx2);

    state[idx] = cmulPhase(state[idx], coeff * q1 * q2);
}

// Apply controlled two-mode position coupling: exp(i*coeff*Z*q1*q2)
// where Z acts on ctrlQubit in ctrlReg
__global__ void kernelCPhaseXX(cuDoubleComplex* state, size_t totalSize, int reg1Idx, int reg2Idx,
                               int ctrlReg, int ctrlQubit, const int* qbts, const double* gridSteps,
                               const int* flwQbts, double coeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t reg1Dim = 1 << qbts[reg1Idx];
    size_t reg2Dim = 1 << qbts[reg2Idx];
    double dx1 = gridSteps[reg1Idx];
    double dx2 = gridSteps[reg2Idx];

    size_t local1 = getLocalIndex(idx, flwQbts[reg1Idx], qbts[reg1Idx]);
    size_t local2 = getLocalIndex(idx, flwQbts[reg2Idx], qbts[reg2Idx]);

    // Extract control qubit state
    size_t ctrlLocalIdx = getLocalIndex(idx, flwQbts[ctrlReg], qbts[ctrlReg]);
    int ctrlMask = 1 << (qbts[ctrlReg] - 1 - ctrlQubit);

    double q1 = gridX(local1, reg1Dim, dx1);
    double q2 = gridX(local2, reg2Dim, dx2);
    double phase = coeff * q1 * q2;

    // Z operator: |0⟩ gets +phase, |1⟩ gets -phase
    state[idx] = cmulPhase(state[idx], (ctrlLocalIdx & ctrlMask) ? -phase : phase);
}

#pragma endregion

#pragma region FFT Helper Kernels

// Apply scalar multiplication to register elements (not neccessarily normalized
// scalar)
__global__ void kernelGlobalScalar(cuDoubleComplex* data, size_t totalSize, int regIdx,
                                   const int* qbts, double scalar) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    data[idx] = make_cuDoubleComplex(cuCreal(data[idx]) * scalar, cuCimag(data[idx]) * scalar);
}

// Apply global complex phase exp(i*phase) to all elements of the state
__global__ void kernelGlobalPhase(cuDoubleComplex* data, size_t totalSize, double phase) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;
    data[idx] = cmulPhase(data[idx], phase);
}

#pragma endregion

#pragma region Utility Kernels

// ============ Native Grid Wigner Function ============
// Build integrand g_i[k] for each x-row on native grid, tracing out other
// registers
__global__ void kernelBuildWignerRow(cuDoubleComplex* dBuf, const cuDoubleComplex* state,
                                     int regIdx, int cvDim, double dx, int numReg, const int* qbts,
                                     const int* flwQbts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cvDim * cvDim) return;
    int i = tid / cvDim;
    int k = tid % cvDim;
    int kDisp = k - (cvDim - 1) / 2;
    int iPy = i + kDisp;
    int iMy = i - kDisp;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    if (iPy >= 0 && iPy < cvDim && iMy >= 0 && iMy < cvDim) {
        size_t regStride = 1 << flwQbts[regIdx];
        int totalQubits = qbts[0] + flwQbts[0];
        size_t otherSize = 1 << (totalQubits - qbts[regIdx]);
        for (size_t otherIdx = 0; otherIdx < otherSize; otherIdx++) {
            size_t baseIdx = 0, remainingIdx = otherIdx;
            for (int r = numReg - 1; r >= 0; r--) {
                if (r == regIdx) continue;
                size_t rDim = 1 << qbts[r];
                size_t rStride = 1 << flwQbts[r];
                baseIdx += (remainingIdx % rDim) * rStride;
                remainingIdx /= rDim;
            }
            sum = cuCadd(
                sum, conjMul(state[baseIdx + iPy * regStride], state[baseIdx + iMy * regStride]));
        }
    }
    dBuf[i * cvDim + k] = sum;
}

// Apply phase correction + fftshift, write N×N output
__global__ void kernelFinalizeWigner(double* wigner, const cuDoubleComplex* dFFTOut, int cvDim,
                                     double dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cvDim * cvDim) return;
    int jc = tid / cvDim;  // centered p-index (row of output)
    int i = tid % cvDim;   // x-index (col of output)
    int k_fft = (jc + cvDim / 2) % cvDim;
    cuDoubleComplex G = dFFTOut[i * cvDim + k_fft];
    double pj = (jc - cvDim / 2.0) * PI / ((double)cvDim * dx);
    double phase = -pj * (cvDim - 1) * dx;
    wigner[tid] = (cuCreal(G) * cos(phase) - cuCimag(G) * sin(phase)) * dx / PI;
}

// ============ Native Grid Husimi Q Function ============

// Compute G[k] = DFT{g_0}[k] analytically.
// g_0 is the Gaussian kernel with peak at index 0 (not centered at (N-1)/2),
// so that the circular convolution in kernelFillHusimiA gives the correct
// windowed FFT.  G[k] is real and uses the "fftfreq" signed-frequency so that
// negative-frequency bins (k > N/2) are handled correctly.
//   g_0[m] = π^{-1/4} sqrt(dx) exp(-½ (m·dx)²)
//   G[k]   = sqrt(2) π^{1/4} / sqrt(dx) · exp(-½ p_eff²)  (real, no phase)
// where p_eff = 2π·k_signed / (N·dx),  k_signed = k for k≤N/2, k-N otherwise.
__global__ void kernelComputeHusimiG(cuDoubleComplex* dG, int N, double dx) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    int k_signed = (k <= N / 2) ? k : k - N;  // fftfreq: centre the frequency axis
    double p_eff = 2.0 * M_PI * k_signed / ((double)N * dx);
    double mag   = pow(M_PI, 0.25) * sqrt(2.0 / dx) * exp(-0.5 * p_eff * p_eff);
    dG[k] = make_cuDoubleComplex(mag, 0.0);
}

// Extract ψ_s[xIdx] for one slice into a contiguous length-N buffer.
__global__ void kernelExtractSlicePsi(cuDoubleComplex* dPsi, const cuDoubleComplex* state,
                                      int regIdx, int cvDim, size_t sliceIdx,
                                      const int* qbts, const int* flwQbts) {
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIdx >= cvDim) return;
    int qbtsAfterCV = flwQbts[regIdx];
    size_t qbtsAfterCVMask = (1 << qbtsAfterCV) - 1;
    dPsi[xIdx] = state[(sliceIdx & qbtsAfterCVMask) | ((size_t)xIdx << flwQbts[regIdx]) |
                       ((sliceIdx & ~qbtsAfterCVMask) << qbts[regIdx])];
}

// Fill circulant A[m, k] = Psi_s[(m+k) % N] * G[k] for all (m, k).
// After batch IFFT over k, row m gives H_s[m, j] where j is the q-index.
__global__ void kernelFillHusimiA(cuDoubleComplex* dBuf, const cuDoubleComplex* dPsi,
                                   const cuDoubleComplex* dG, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * N) return;
    int m = tid / N;
    int k = tid % N;
    dBuf[tid] = cuCmul(dPsi[(m + k) & (N - 1)], dG[k]);
}

// fftshift + normalize accumulated power.
// dAccum layout: [m * cvDim + j] where m = FFT p-bin, j = q-index.
// Output layout: [jc * cvDim + qIdx] where jc = centered p-index.
// Divides by PI·N² to correct for cuFFT's unnormalized IFFT (factor N per IFFT
// call, squared because we accumulate |H|²).
__global__ void kernelFinalizeHusimi(double* outQ, const double* dAccum, int cvDim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cvDim * cvDim) return;
    int jc   = tid / cvDim;  // centered p-index (row of output)
    int qIdx = tid % cvDim;  // q-index (col of output)
    int m    = (jc + cvDim / 2) % cvDim;  // FFT bin (un-shifted p-index)
    double N2 = (double)cvDim * (double)cvDim;
    outQ[tid] = dAccum[m * cvDim + qIdx] / (PI * N2);
}

// Accumulate |H[k]|² from FFT output into accumulation buffer
__global__ void kernelAccumHusimiPower(
    double* dAccum,                  // [qN x cvDim], accumulated across slices
    const cuDoubleComplex* dFFTOut,  // [qN x cvDim], post-FFT for one slice
    int totalElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalElements) return;
    dAccum[tid] += absSquare(dFFTOut[tid]);
}

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

#pragma endregion

extern "C" {

#pragma region C API - Context Management

CVDVContext* cvdvCreate(int numReg, int* numQubits) {
    CVDVContext* ctx;
    checkCudaErrors(cudaMallocManaged(&ctx, sizeof(CVDVContext)));
    memset(ctx, 0, sizeof(CVDVContext));
    ctx->gQbts = nullptr;
    ctx->gFlwQbts = nullptr;
    ctx->gGridSteps = nullptr;
    ctx->gNumReg = 0;
    ctx->gTotalQbt = 0;
    ctx->ftPlans = nullptr;
    ctx->wPlanValid = false;
    ctx->hPlanValid = false;
    ctx->hSinglePlanValid = false;
    ctx->dHusimiG = nullptr;
    ctx->hGCvDim = 0;
    ctx->hGDx = 0.0;
    ctx->hGValid = false;
    ctx->ctHandleValid = false;
    ctx->dMeasureProbs = nullptr;
    ctx->dMeasureOut = nullptr;
    ctx->measurePlanCache = nullptr;

    // If no registers specified, return empty context
    if (numReg == 0 || numQubits == nullptr) {
        return ctx;
    }

    // Allocate registers
    ctx->gNumReg = numReg;

    // Allocate managed memory for register metadata
    checkCudaErrors(cudaMallocManaged(&ctx->gQbts, numReg * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&ctx->gFlwQbts, numReg * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&ctx->gGridSteps, numReg * sizeof(double)));

    // Initialize metadata for each register
    ctx->gTotalQbt = 0;
    for (int i = 0; i < numReg; i++) {
        ctx->gQbts[i] = numQubits[i];
        ctx->gTotalQbt += numQubits[i];

        // Calculate grid step using formula: dx = sqrt(2 * pi / regDim)
        size_t registerDim = 1 << numQubits[i];
        ctx->gGridSteps[i] = sqrt(2.0 * PI / registerDim);
    }

    // Compute following qubit counts (sum of qubits after each register)
    for (int i = 0; i < numReg; i++) {
        int followQubits = 0;
        for (int j = i + 1; j < numReg; j++) {
            followQubits += numQubits[j];
        }
        ctx->gFlwQbts[i] = followQubits;
    }

    // Build per-register cuFFT plans (parameters are fully determined now).
    ctx->ftPlans = (cufftHandle*)malloc(numReg * sizeof(cufftHandle));
    size_t totalSize = (size_t)1 << ctx->gTotalQbt;
    for (int i = 0; i < numReg; i++) {
        int n = 1 << ctx->gQbts[i];
        size_t regStride = (size_t)1 << ctx->gFlwQbts[i];
        cufftResult planRes;
        if (regStride == 1) {
            int batch = (int)(totalSize / n);
            planRes = cufftPlan1d(&ctx->ftPlans[i], n, CUFFT_Z2Z, batch);
        } else {
            int iStride = (int)regStride, oStride = (int)regStride;
            int iDist = 1, oDist = 1;
            int batch = (int)regStride;
            int nembed[1] = {n * (int)regStride};
            planRes = cufftPlanMany(&ctx->ftPlans[i], 1, &n, nembed, iStride, iDist, nembed,
                                    oStride, oDist, CUFFT_Z2Z, batch);
        }
        if (planRes != CUFFT_SUCCESS) {
            fprintf(stderr, "cvdvCreate: cuFFT plan creation failed for register %d: %d\n", i,
                    planRes);
            exit(EXIT_FAILURE);
        }
    }

    return ctx;
}

void cvdvDestroy(CVDVContext* ctx) {
    if (!ctx) return;

    // Free device memory
    if (ctx->dState != nullptr) {
        checkCudaErrors(cudaFree(ctx->dState));
        ctx->dState = nullptr;
    }

    // Free cuFFT plans
    if (ctx->ftPlans != nullptr) {
        for (int i = 0; i < ctx->gNumReg; i++) {
            if (ctx->ftPlans[i] != 0) {
                cufftDestroy(ctx->ftPlans[i]);
            }
        }
        free(ctx->ftPlans);
        ctx->ftPlans = nullptr;
    }

    // Free Wigner/Husimi plans
    if (ctx->wPlanValid) {
        cufftDestroy(ctx->wPlan);
    }
    if (ctx->hPlanValid) {
        cufftDestroy(ctx->hPlan);
    }
    if (ctx->hSinglePlanValid) {
        cufftDestroy(ctx->hSinglePlan);
    }
    if (ctx->dHusimiG != nullptr) {
        cudaFree(ctx->dHusimiG);
        ctx->dHusimiG = nullptr;
    }
    if (ctx->measurePlanCache != nullptr) {
        auto* cache = static_cast<std::map<std::vector<int>, MeasurePlan>*>(ctx->measurePlanCache);
        for (auto& [key, mp] : *cache) {
            if (mp.dWorkspace) cudaFree(mp.dWorkspace);
            cutensorDestroyPlan(mp.plan);
            cutensorDestroyPlanPreference(mp.planPref);
            cutensorDestroyOperationDescriptor(mp.opDesc);
            cutensorDestroyTensorDescriptor(mp.descIn);
            cutensorDestroyTensorDescriptor(mp.descOut);
        }
        delete cache;
        ctx->measurePlanCache = nullptr;
    }
    if (ctx->ctHandleValid) {
        cutensorDestroy(ctx->ctHandle);
        ctx->ctHandleValid = false;
    }
    if (ctx->dMeasureProbs != nullptr) {
        cudaFree(ctx->dMeasureProbs);
        ctx->dMeasureProbs = nullptr;
    }
    if (ctx->dMeasureOut != nullptr) {
        cudaFree(ctx->dMeasureOut);
        ctx->dMeasureOut = nullptr;
    }

    // Free managed memory
    if (ctx->gQbts != nullptr) {
        checkCudaErrors(cudaFree(ctx->gQbts));
        ctx->gQbts = nullptr;
    }
    if (ctx->gFlwQbts != nullptr) {
        checkCudaErrors(cudaFree(ctx->gFlwQbts));
        ctx->gFlwQbts = nullptr;
    }
    if (ctx->gGridSteps != nullptr) {
        checkCudaErrors(cudaFree(ctx->gGridSteps));
        ctx->gGridSteps = nullptr;
    }

    cudaFree(ctx);
}

#pragma endregion

#pragma region C API - Initialization and Cleanup

// Build the full tensor-product state from per-register device pointers.
// devicePtrs[i] must point to cuDoubleComplex device memory of size (1 <<
// gQbts[i]). This replaces the old cvdvInitStateVector +
// cvdvSetRegisterFromDevicePtr pattern.
void cvdvInitFromSeparable(CVDVContext* ctx, void** devicePtrs, int numReg) {
    if (!ctx) return;
    if (ctx->gNumReg == 0) {
        fprintf(stderr, "Error: Must call cvdvCreate with registers before cvdvInitFromSeparable\n");
        return;
    }
    if (numReg != ctx->gNumReg) {
        fprintf(stderr, "Error: numReg mismatch: context has %d, got %d\n", ctx->gNumReg, numReg);
        return;
    }

    size_t totalSize = 1 << ctx->gTotalQbt;

    // Download each register to host
    cuDoubleComplex** hTempRegs = new cuDoubleComplex*[numReg];
    for (int i = 0; i < numReg; i++) {
        size_t regDim = 1 << ctx->gQbts[i];
        hTempRegs[i] = new cuDoubleComplex[regDim];
        checkCudaErrors(cudaMemcpy(hTempRegs[i], reinterpret_cast<cuDoubleComplex*>(devicePtrs[i]),
                                   regDim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // Compute tensor product on host (last register varies fastest)
    cuDoubleComplex* hState = new cuDoubleComplex[totalSize];
    for (size_t globalIdx = 0; globalIdx < totalSize; globalIdx++) {
        cuDoubleComplex product = make_cuDoubleComplex(1.0, 0.0);
        size_t idx = globalIdx;
        for (int reg = numReg - 1; reg >= 0; reg--) {
            size_t regDim = 1 << ctx->gQbts[reg];
            size_t localIdx = idx % regDim;
            idx /= regDim;
            product = cuCmul(product, hTempRegs[reg][localIdx]);
        }
        hState[globalIdx] = product;
    }

    for (int i = 0; i < numReg; i++) delete[] hTempRegs[i];
    delete[] hTempRegs;

    if (ctx->dState != nullptr) cudaFree(ctx->dState);

    // Allocate device memory: %.3f GB
    checkCudaErrors(cudaMalloc(&ctx->dState, totalSize * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMemcpy(ctx->dState, hState, totalSize * sizeof(cuDoubleComplex),
                                    cudaMemcpyHostToDevice));
    delete[] hState;
}

void cvdvFree(CVDVContext* ctx) {
    if (!ctx) return;
    if (ctx->dState != nullptr) {
        cudaFree(ctx->dState);
        ctx->dState = nullptr;
    }
    if (ctx->gQbts != nullptr) {
        cudaFree(ctx->gQbts);
        ctx->gQbts = nullptr;
    }
    if (ctx->gFlwQbts != nullptr) {
        cudaFree(ctx->gFlwQbts);
        ctx->gFlwQbts = nullptr;
    }
    if (ctx->gGridSteps != nullptr) {
        cudaFree(ctx->gGridSteps);
        ctx->gGridSteps = nullptr;
    }
    ctx->gNumReg = 0;
    ctx->gTotalQbt = 0;
}

#pragma endregion

void cvdvFtQ2P(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    size_t totalSize = 1 << ctx->gTotalQbt;
    int grid = (totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    size_t regDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    // Step 1: Pre-phase correction: exp(i*π(N-1)/N * j)
    // In position representation: exp(i*π(N-1)/(N*dx) * x)
    double phaseCoeff = PI * (regDim - 1.0) / (regDim * dx);
    kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize, regIdx, ctx->gQbts,
                                            ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);

    // Step 2: Forward FFT — use cached plan
    size_t regStride = 1 << ctx->gFlwQbts[regIdx];
    cufftHandle plan = ctx->ftPlans[regIdx];

    if (regStride == 1) {
        cufftResult result = cufftExecZ2Z(plan, ctx->dState, ctx->dState, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT forward execution failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
    } else {
        // Strided case: loop over outer blocks (plan covers one block at a time)
        size_t regBlockSize = regStride << ctx->gQbts[regIdx];
        size_t outerDim = totalSize / regBlockSize;
        for (size_t o = 0; o < outerDim; o++) {
            cuDoubleComplex* blockStart =
                ctx->dState + (o << (ctx->gFlwQbts[regIdx] + ctx->gQbts[regIdx]));
            cufftResult result = cufftExecZ2Z(plan, blockStart, blockStart, CUFFT_FORWARD);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT forward execution failed: %d at block %zu\n", result, o);
                exit(EXIT_FAILURE);
            }
        }
    }

    // Step 3: Post-phase correction: exp(i*π(N-1)/N * k)
    // In momentum representation: exp(i*π(N-1)/(N*dx) * p)
    kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize, regIdx, ctx->gQbts,
                                            ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);

    // Step 4: Normalization (1/√N for unitary transform)
    double norm = 1.0 / sqrt((double)regDim);
    kernelGlobalScalar<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize, regIdx, ctx->gQbts, norm);

    // Step 5: Global phase correction: exp(i*π*(N-1)²/(2N))
    // Matches dvsim-code convention: dvsim_QFT = CVDV_QFT * exp(i*π*(N-1)²/(2N))
    double globalPhase = PI * (double)(regDim - 1) * (double)(regDim - 1) / (2.0 * (double)regDim);
    kernelGlobalPhase<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize, globalPhase);
}

void cvdvFtP2Q(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Inverse index-shifted QFT on specific register
    // Algorithm:
    // 1. Pre-phase correction: exp(-i*pi*j*(N-1)/N)
    // 2. Standard IFFT
    // 3. Post-phase correction: exp(-i*pi*k*(N-1)/N)
    // 4. Normalization: 1/√N

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    size_t regDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    // Step 1: Pre-phase correction (negative phase): exp(-i*π(N-1)/N * j)
    // In momentum representation: exp(-i*π(N-1)/(N*dx) * p)
    double phaseCoeff = -PI * (regDim - 1.0) / (regDim * dx);
    kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts,
                                            ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);

    // Step 2: Inverse FFT — use cached plan
    size_t regStride = 1 << ctx->gFlwQbts[regIdx];
    cufftHandle plan = ctx->ftPlans[regIdx];

    if (regStride == 1) {
        cufftResult result = cufftExecZ2Z(plan, ctx->dState, ctx->dState, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT inverse execution failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
    } else {
        size_t regBlockSize = regStride << ctx->gQbts[regIdx];
        size_t outerDim = (1 << ctx->gTotalQbt) / regBlockSize;
        for (size_t o = 0; o < outerDim; o++) {
            cuDoubleComplex* blockStart =
                ctx->dState + (o << (ctx->gFlwQbts[regIdx] + ctx->gQbts[regIdx]));
            cufftResult result = cufftExecZ2Z(plan, blockStart, blockStart, CUFFT_INVERSE);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT inverse execution failed: %d at block %zu\n", result, o);
                exit(EXIT_FAILURE);
            }
        }
    }

    // Step 3: Post-phase correction (negative phase): exp(-i*π(N-1)/N * k)
    // In position representation: exp(-i*π(N-1)/(N*dx) * x)
    kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts,
                                            ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);

    // Step 4: Normalization (1/√N for unitary transform)
    double norm = 1.0 / sqrt((double)regDim);
    kernelGlobalScalar<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx,
                                                  ctx->gQbts, norm);

    // Step 5: Global phase correction (conjugate of ftQ2P): exp(-i*π*(N-1)²/(2N))
    double globalPhase = -PI * (double)(regDim - 1) * (double)(regDim - 1) / (2.0 * (double)regDim);
    kernelGlobalPhase<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), globalPhase);
}

#pragma endregion

#pragma region C API - Gates

void cvdvDisplacement(CVDVContext* ctx, int regIdx, double betaRe, double betaIm) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // D(α) = exp(-i*Im(α)*Re(α)) * D(i*Im(α)) * D(Re(α))
    // D(i*p0) = exp(i*sqrt(2)*p0*q) - phase in position space
    // D(q0) = exp(-i*sqrt(2)*q0*p) - phase in momentum space

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    // Step 1: Apply D(i*Im(α)) = exp(i*sqrt(2)*Im(α)*q) in position space
    if (fabs(betaIm) > 1e-12) {
        kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx,
                                                ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                                SQRT2 * betaIm);
        checkCudaErrors(cudaGetLastError());
    }

    // Step 2: Apply D(Re(α)) = exp(-i*sqrt(2)*Re(α)*p) in momentum space
    if (fabs(betaRe) > 1e-12) {
        // Transform register to momentum space
        cvdvFtQ2P(ctx, regIdx);

        // Apply phase in momentum space
        kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx,
                                                ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                                -SQRT2 * betaRe);
        checkCudaErrors(cudaGetLastError());

        // Transform back to position space
        cvdvFtP2Q(ctx, regIdx);
    }

    // Note: Global phase exp(-i*Im(α)*Re(α)) is ignored
}

void cvdvConditionalDisplacement(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit,
                                 double alphaRe, double alphaIm) {
    if (!ctx) return;
    if (targetReg < 0 || targetReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid target register index: %d\n", targetReg);
        return;
    }
    if (ctrlReg < 0 || ctrlReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid control register index: %d\n", ctrlReg);
        return;
    }

    // Conditional displacement: CD(α) = CD(i*Im(α)) CD(Re(α))
    // CD(i*p0) = exp(i*sqrt(2)*p0*Z*q) - controlled phase in position space
    // CD(q0) = F^{-1} exp(-i*sqrt(2)*q0*Z*p) F - controlled phase in momentum
    // space where Z = |0⟩⟨0| - |1⟩⟨1|, so |0⟩ gets +α and |1⟩ gets -α

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    // Step 1: Apply CD(i*Im(α)) = exp(i√2 Im(α) Z q) in position space
    if (fabs(alphaIm) > 1e-12) {
        kernelCPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), targetReg,
                                                 ctrlReg, ctrlQubit, ctx->gQbts, ctx->gGridSteps,
                                                 ctx->gFlwQbts, SQRT2 * alphaIm);
        checkCudaErrors(cudaGetLastError());
    }

    // Step 2: Apply CD(Re(α)) = F^{-1} exp(-i√2 Re(α) Z p) F
    if (fabs(alphaRe) > 1e-12) {
        // Transform target register to momentum space
        cvdvFtQ2P(ctx, targetReg);

        // Apply exp(-i√2 Re(α) Z p) in momentum space
        kernelCPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), targetReg,
                                                 ctrlReg, ctrlQubit, ctx->gQbts, ctx->gGridSteps,
                                                 ctx->gFlwQbts, -SQRT2 * alphaRe);
        checkCudaErrors(cudaGetLastError());

        // Transform back to position space
        cvdvFtP2Q(ctx, targetReg);
    }
}

void cvdvPauliRotation(CVDVContext* ctx, int regIdx, int qubitIdx, int axis, double theta) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) / 2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelPauliRotation<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx,
                                                   qubitIdx, ctx->gQbts, ctx->gFlwQbts, axis,
                                                   theta);
    checkCudaErrors(cudaGetLastError());
}

void cvdvHadamard(CVDVContext* ctx, int regIdx, int qubitIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) / 2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelHadamard<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, qubitIdx,
                                              ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(cudaGetLastError());
}

void cvdvParity(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelParity<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts,
                                            ctx->gFlwQbts);
    checkCudaErrors(cudaGetLastError());
}

void cvdvConditionalParity(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit) {
    if (!ctx) return;
    if (targetReg < 0 || targetReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid target register index: %d\n", targetReg);
        return;
    }
    if (ctrlReg < 0 || ctrlReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid control register index: %d\n", ctrlReg);
        return;
    }
    if (ctrlQubit < 0 || ctrlQubit >= ctx->gQbts[ctrlReg]) {
        fprintf(stderr, "Invalid control qubit index: %d\n", ctrlQubit);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelConditionalParity<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                       targetReg, ctrlReg, ctrlQubit, ctx->gQbts,
                                                       ctx->gFlwQbts);
    checkCudaErrors(cudaGetLastError());
}

void cvdvSwapRegisters(CVDVContext* ctx, int reg1, int reg2) {
    if (!ctx) return;
    if (reg1 < 0 || reg1 >= ctx->gNumReg || reg2 < 0 || reg2 >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register indices: %d, %d\n", reg1, reg2);
        return;
    }
    if (ctx->gQbts[reg1] != ctx->gQbts[reg2]) {
        fprintf(stderr, "SWAP requires registers with same number of qubits\n");
        return;
    }
    if (reg1 == reg2) return;

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelSwapRegisters<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                                   ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(cudaGetLastError());
}

void cvdvPhaseSquare(CVDVContext* ctx, int regIdx, double t) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelPhaseX2<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts,
                                             ctx->gGridSteps, ctx->gFlwQbts, t);
    checkCudaErrors(cudaGetLastError());
}

void cvdvPhaseCubic(CVDVContext* ctx, int regIdx, double t) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelPhaseX3<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts,
                                             ctx->gGridSteps, ctx->gFlwQbts, t);
    checkCudaErrors(cudaGetLastError());
}

// Internal: small-angle rotation |θ| ≤ π/4
static void cvdvRotationSmall(CVDVContext* ctx, int regIdx, double theta) {
    if (fabs(theta) < 1e-15) return;

    // R(θ) = exp(-i/2 tan(θ/2) q^2) exp(-i/2 sin(θ) p^2) exp(-i/2 tan(θ/2) q^2)
    double tanHalfTheta = tan(theta / 2.0);
    double sinTheta = sin(theta);

    cvdvPhaseSquare(ctx, regIdx, -0.5 * tanHalfTheta);
    cvdvFtQ2P(ctx, regIdx);
    cvdvPhaseSquare(ctx, regIdx, -0.5 * sinTheta);
    cvdvFtP2Q(ctx, regIdx);
    cvdvPhaseSquare(ctx, regIdx, -0.5 * tanHalfTheta);
}

void cvdvRotation(CVDVContext* ctx, int regIdx, double theta) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // For |θ| > π/4, decompose R(θ) = R(θ₀) R(θ-θ₀)
    // where θ₀ ∈ (π/2)Z is chosen so |θ-θ₀| ≤ π/4
    // R(π/2) = FT, R(π) = Parity, R(-π/2) = FT†

    // Find nearest multiple of π/2 using round-half-to-even (matches Python's
    // np.round)
    double theta0 = rint(theta / (PI / 2.0)) * (PI / 2.0);
    double remainder = theta - theta0;

    // Apply R(θ₀) for the integer-multiple part
    // theta0 / (π/2) gives the number of quarter-turns
    int quarterTurns = (int)rint(theta0 / (PI / 2.0));
    // Normalize to [0,4) since R(2π) = identity
    quarterTurns = ((quarterTurns % 4) + 4) % 4;

    switch (quarterTurns) {
        case 0:
            break;  // identity
        case 1:
            cvdvFtQ2P(ctx, regIdx);
            break;  // R(π/2) = FT
        case 2:
            cvdvParity(ctx, regIdx);
            break;  // R(π) = Parity
        case 3:
            cvdvFtP2Q(ctx, regIdx);
            break;  // R(-π/2) = R(3π/2) = FT†
    }

    // Apply small-angle remainder
    cvdvRotationSmall(ctx, regIdx, remainder);
}

// Internal helper: apply controlled phase square exp(i*t*Z*q^2)
static void cvdvControlledPhaseSquare(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit,
                                      double t) {
    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelCPhaseX2<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), targetReg,
                                              ctrlReg, ctrlQubit, ctx->gQbts, ctx->gGridSteps,
                                              ctx->gFlwQbts, t);
    checkCudaErrors(cudaGetLastError());
}

// Internal: small-angle conditional rotation |θ| ≤ π/4
static void cvdvConditionalRotationSmall(CVDVContext* ctx, int targetReg, int ctrlReg,
                                         int ctrlQubit, double theta) {
    if (fabs(theta) < 1e-15) return;

    // CR(θ) = exp(-i/2 Z tan(θ/2) q^2) exp(-i/2 Z sin(θ) p^2) exp(-i/2 Z tan(θ/2)
    // q^2)
    double tanHalfTheta = tan(theta / 2.0);
    double sinTheta = sin(theta);

    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * tanHalfTheta);
    cvdvFtQ2P(ctx, targetReg);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * sinTheta);
    cvdvFtP2Q(ctx, targetReg);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * tanHalfTheta);
}

void cvdvConditionalRotation(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit,
                             double theta) {
    if (!ctx) return;
    if (targetReg < 0 || targetReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid target register index: %d\n", targetReg);
        return;
    }
    if (ctrlReg < 0 || ctrlReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid control register index: %d\n", ctrlReg);
        return;
    }
    if (ctrlQubit < 0 || ctrlQubit >= ctx->gQbts[ctrlReg]) {
        fprintf(stderr, "Invalid control qubit index: %d\n", ctrlQubit);
        return;
    }

    // For |θ| > π/4, decompose R(θ) = R(θ₀) R(θ-θ₀)
    // where θ₀ ∈ (π/2)Z is chosen so |θ-θ₀| ≤ π/4
    // R(π/2) = FT, R(π) = Parity, R(-π/2) = FT†

    // Find nearest multiple of π/2 using round-half-to-even (matches Python's
    // np.round)
    double theta0 = rint(theta / (PI / 2.0)) * (PI / 2.0);
    double remainder = theta - theta0;

    // Apply R(θ₀) for the integer-multiple part
    // theta0 / (π/2) gives the number of quarter-turns
    int quarterTurns = (int)rint(theta0 / (PI / 2.0));
    // Normalize to [0,4) since R(2π) = identity
    quarterTurns = ((quarterTurns % 4) + 4) % 4;

    switch (quarterTurns) {
        case 0:
            break;  // identity
        case 1:
            cvdvFtQ2P(ctx, targetReg);
            cvdvConditionalParity(ctx, targetReg, ctrlReg, ctrlQubit);
            cvdvPauliRotation(ctx, ctrlReg, ctrlQubit, 2, PI / 2);
            break;
        case 2:
            cvdvParity(ctx, targetReg);
            cvdvPauliRotation(ctx, ctrlReg, ctrlQubit, 2, PI);
            break;
        case 3:
            cvdvFtP2Q(ctx, targetReg);
            cvdvConditionalParity(ctx, targetReg, ctrlReg, ctrlQubit);
            cvdvPauliRotation(ctx, ctrlReg, ctrlQubit, 2, -PI / 2);
            break;
    }

    // Apply small-angle remainder
    cvdvConditionalRotationSmall(ctx, targetReg, ctrlReg, ctrlQubit, remainder);
}

void cvdvSqueeze(CVDVContext* ctx, int regIdx, double r) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    double expHalfR = exp(0.5 * r);
    double expMinusHalfR = exp(-0.5 * r);
    double sqrtExpRMinus1 = sqrt(fabs(exp(r) - 1.0));
    double sign = (r >= 0) ? 1.0 : -1.0;

    // First
    cvdvFtQ2P(ctx, regIdx);
    cvdvPhaseSquare(ctx, regIdx, -0.5 * expHalfR * sqrtExpRMinus1);
    cvdvFtP2Q(ctx, regIdx);

    // Second
    cvdvPhaseSquare(ctx, regIdx, 0.5 * expMinusHalfR * sqrtExpRMinus1 * sign);

    // Third
    cvdvFtQ2P(ctx, regIdx);
    cvdvPhaseSquare(ctx, regIdx, 0.5 * expMinusHalfR * sqrtExpRMinus1);
    cvdvFtP2Q(ctx, regIdx);

    // Fourth
    cvdvPhaseSquare(ctx, regIdx, -0.5 * expHalfR * sqrtExpRMinus1 * sign);
}

void cvdvConditionalSqueeze(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double r) {
    if (!ctx) return;
    if (targetReg < 0 || targetReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid target register index: %d\n", targetReg);
        return;
    }
    if (ctrlReg < 0 || ctrlReg >= ctx->gNumReg) {
        fprintf(stderr, "Invalid control register index: %d\n", ctrlReg);
        return;
    }
    if (ctrlQubit < 0 || ctrlQubit >= ctx->gQbts[ctrlReg]) {
        fprintf(stderr, "Invalid control qubit index: %d\n", ctrlQubit);
        return;
    }

    // CS(r) = conditional version of S(r), replacing cvdvPhaseSquare with
    // cvdvControlledPhaseSquare
    double chR = cosh(r);
    double shR = sinh(r);
    double s = sqrt(2.0 * abs(sinh(0.5 * r)));

    // First
    cvdvPhaseSquare(ctx, targetReg, 0.5 * s * chR);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * s * shR);
    // Second
    cvdvFtQ2P(ctx, targetReg);
    cvdvPhaseSquare(ctx, targetReg, 0.5 * (chR - 1) / s);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, 0.5 * shR / s);
    cvdvFtP2Q(ctx, targetReg);
    // Third
    cvdvPhaseSquare(ctx, targetReg, -0.5 * s);
    // Fourth
    cvdvFtQ2P(ctx, targetReg);
    cvdvPhaseSquare(ctx, targetReg, 0.5 * (chR - 1) / s);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * shR / s);
    cvdvFtP2Q(ctx, targetReg);
}

// Internal: small-angle beam splitter |θ| ≤ π/2
static void cvdvBeamSplitterSmall(CVDVContext* ctx, int reg1, int reg2, double theta) {
    if (fabs(theta) < 1e-15) return;

    double coeff_q = -tan(0.25 * theta);
    double coeff_p = -sin(0.5 * theta);

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelPhaseXX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                             ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, coeff_q);

    cvdvFtQ2P(ctx, reg1);
    cvdvFtQ2P(ctx, reg2);

    kernelPhaseXX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                             ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, coeff_p);

    cvdvFtP2Q(ctx, reg1);
    cvdvFtP2Q(ctx, reg2);

    kernelPhaseXX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                             ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, coeff_q);
}

void cvdvBeamSplitter(CVDVContext* ctx, int reg1, int reg2, double theta) {
    if (!ctx) return;
    if (reg1 < 0 || reg1 >= ctx->gNumReg || reg2 < 0 || reg2 >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register indices: %d, %d\n", reg1, reg2);
        return;
    }
    if (reg1 == reg2) {
        fprintf(stderr, "Beam splitter requires two different registers\n");
        return;
    }

    // For |θ| > π/2, decompose BS(θ) = BS(θ₀) BS(θ-θ₀)
    // where θ₀ ∈ πZ is chosen so |θ-θ₀| ≤ π/2
    // BS(π) = FT₁ FT₂ SWAP,  BS(2π) = Parity₁ Parity₂,  BS(-π) = FT₁† FT₂† SWAP

    double theta0 = round(theta / PI) * PI;
    double remainder = theta - theta0;

    // Apply BS(θ₀) for the integer-multiple-of-π part
    int halfTurns = (int)round(theta0 / PI);
    // Normalize to [0,4) since BS(4π) = identity (BS(2π) = Par₁Par₂, applied
    // twice = id)
    halfTurns = ((halfTurns % 4) + 4) % 4;

    switch (halfTurns) {
        case 0:
            break;  // identity
        case 1:     // BS(π) = FT₁ FT₂ SWAP
            cvdvFtQ2P(ctx, reg1);
            cvdvFtQ2P(ctx, reg2);
            cvdvSwapRegisters(ctx, reg1, reg2);
            break;
        case 2:  // BS(2π) = Parity₁ Parity₂
            cvdvParity(ctx, reg1);
            cvdvParity(ctx, reg2);
            break;
        case 3:  // BS(-π) = BS(3π) = FT₁† FT₂† SWAP
            cvdvFtP2Q(ctx, reg1);
            cvdvFtP2Q(ctx, reg2);
            cvdvSwapRegisters(ctx, reg1, reg2);
            break;
    }

    // Apply small-angle remainder
    cvdvBeamSplitterSmall(ctx, reg1, reg2, remainder);
}

// Internal: small-angle conditional beam splitter |θ| ≤ π/2
// |0⟩ gets BS(θ), |1⟩ gets BS(-θ)
static void cvdvConditionalBeamSplitterSmall(CVDVContext* ctx, int reg1, int reg2, int ctrlReg,
                                             int ctrlQubit, double theta) {
    if (fabs(theta) < 1e-15) return;

    double coeff_q = -tan(0.25 * theta);
    double coeff_p = -sin(0.5 * theta);

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelCPhaseXX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                              ctrlReg, ctrlQubit, ctx->gQbts, ctx->gGridSteps,
                                              ctx->gFlwQbts, coeff_q);

    cvdvFtQ2P(ctx, reg1);
    cvdvFtQ2P(ctx, reg2);

    kernelCPhaseXX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                              ctrlReg, ctrlQubit, ctx->gQbts, ctx->gGridSteps,
                                              ctx->gFlwQbts, coeff_p);

    cvdvFtP2Q(ctx, reg1);
    cvdvFtP2Q(ctx, reg2);

    kernelCPhaseXX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                              ctrlReg, ctrlQubit, ctx->gQbts, ctx->gGridSteps,
                                              ctx->gFlwQbts, coeff_q);
}

void cvdvConditionalBeamSplitter(CVDVContext* ctx, int reg1, int reg2, int ctrlReg, int ctrlQubit,
                                 double theta) {
    if (!ctx) return;
    if (reg1 < 0 || reg1 >= ctx->gNumReg || reg2 < 0 || reg2 >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register indices: %d, %d\n", reg1, reg2);
        return;
    }
    if (reg1 == reg2) {
        fprintf(stderr, "Conditional beam splitter requires two different registers\n");
        return;
    }

    // For |θ| > π/2, decompose CBS(θ) = BS(θ₀) CBS(θ-θ₀)
    // BS(θ₀) applied unconditionally (correct for even half-turns; odd half-turns
    // are approximate)
    double theta0 = round(theta / PI) * PI;
    double remainder = theta - theta0;

    int halfTurns = (int)round(theta0 / PI);
    halfTurns = ((halfTurns % 4) + 4) % 4;

    switch (halfTurns) {
        case 0:
            break;
        case 1:
            cvdvFtQ2P(ctx, reg1);
            cvdvFtQ2P(ctx, reg2);
            cvdvConditionalParity(ctx, reg1, ctrlReg, ctrlQubit);
            cvdvConditionalParity(ctx, reg2, ctrlReg, ctrlQubit);
            cvdvSwapRegisters(ctx, reg1, reg2);
            break;
        case 2:
            cvdvParity(ctx, reg1);
            cvdvParity(ctx, reg2);
            break;
        case 3:
            cvdvFtP2Q(ctx, reg1);
            cvdvFtP2Q(ctx, reg2);
            cvdvConditionalParity(ctx, reg1, ctrlReg, ctrlQubit);
            cvdvConditionalParity(ctx, reg2, ctrlReg, ctrlQubit);
            cvdvSwapRegisters(ctx, reg1, reg2);
            break;
    }

    cvdvConditionalBeamSplitterSmall(ctx, reg1, reg2, ctrlReg, ctrlQubit, remainder);
}

void cvdvQ1Q2Gate(CVDVContext* ctx, int reg1, int reg2, double coeff) {
    if (!ctx) return;
    if (reg1 < 0 || reg1 >= ctx->gNumReg || reg2 < 0 || reg2 >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register indices: reg1=%d, reg2=%d\n", reg1, reg2);
        return;
    }

    if (reg1 == reg2) {
        fprintf(stderr, "Cannot apply Q1Q2 gate to the same register\n");
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelPhaseXX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), reg1, reg2,
                                             ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, coeff);
}

#pragma endregion

#pragma region C API - State Access

void cvdvGetWigner(CVDVContext* ctx, int regIdx, double* wignerOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    size_t cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    cuDoubleComplex* dBuf;
    checkCudaErrors(cudaMalloc(&dBuf, cvDim * cvDim * sizeof(cuDoubleComplex)));

    int grid1 = (cvDim * cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelBuildWignerRow<<<grid1, CUDA_BLOCK_SIZE>>>(dBuf, ctx->dState, regIdx, cvDim, dx,
                                                     ctx->gNumReg, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(cudaDeviceSynchronize());

    // Fixed plan: batch = cvDim rows (only recreated when cvDim changes)
    if (!ctx->wPlanValid || ctx->wPlanCvDim != (int)cvDim) {
        if (ctx->wPlanValid) cufftDestroy(ctx->wPlan);
        cufftResult planRes = cufftPlan1d(&ctx->wPlan, cvDim, CUFFT_Z2Z, cvDim);
        if (planRes != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT wPlan creation failed: %d\n", planRes);
            cudaFree(dBuf);
            ctx->wPlanValid = false;
            return;
        }
        ctx->wPlanCvDim = (int)cvDim;
        ctx->wPlanValid = true;
    }
    cufftExecZ2Z(ctx->wPlan, dBuf, dBuf, CUFFT_INVERSE);
    checkCudaErrors(cudaDeviceSynchronize());

    double* dWigner;
    checkCudaErrors(cudaMalloc(&dWigner, cvDim * cvDim * sizeof(double)));
    kernelFinalizeWigner<<<grid1, CUDA_BLOCK_SIZE>>>(dWigner, dBuf, cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wignerOut, dWigner, cvDim * cvDim * sizeof(double),
                                    cudaMemcpyDeviceToHost));
    cudaFree(dWigner);
    cudaFree(dBuf);
}

void cvdvGetHusimiQ(CVDVContext* ctx, int regIdx, double* husimiOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    int cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];
    size_t sliceCount = 1 << (ctx->gTotalQbt - ctx->gQbts[regIdx]);

    // --- Lazy plan: batch of N IFFTs (used for inverse direction now) ---
    if (!ctx->hPlanValid || ctx->hPlanCvDim != cvDim) {
        if (ctx->hPlanValid) cufftDestroy(ctx->hPlan);
        cufftResult r = cufftPlan1d(&ctx->hPlan, cvDim, CUFFT_Z2Z, cvDim);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT hPlan creation failed: %d\n", r);
            return;
        }
        ctx->hPlanCvDim = cvDim;
        ctx->hPlanValid = true;
    }

    // --- Lazy plan: single forward FFT for per-slice ψ ---
    if (!ctx->hSinglePlanValid || ctx->hPlanCvDim != cvDim) {
        if (ctx->hSinglePlanValid) cufftDestroy(ctx->hSinglePlan);
        cufftResult r = cufftPlan1d(&ctx->hSinglePlan, cvDim, CUFFT_Z2Z, 1);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT hSinglePlan creation failed: %d\n", r);
            return;
        }
        ctx->hSinglePlanValid = true;
    }

    // --- Lazy G[k]: analytic Gaussian kernel in frequency domain ---
    if (!ctx->hGValid || ctx->hGCvDim != cvDim || ctx->hGDx != dx) {
        if (ctx->dHusimiG != nullptr) cudaFree(ctx->dHusimiG);
        checkCudaErrors(cudaMalloc(&ctx->dHusimiG, cvDim * sizeof(cuDoubleComplex)));
        int gGrid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        kernelComputeHusimiG<<<gGrid, CUDA_BLOCK_SIZE>>>(ctx->dHusimiG, cvDim, dx);
        checkCudaErrors(cudaDeviceSynchronize());
        ctx->hGCvDim = cvDim;
        ctx->hGDx    = dx;
        ctx->hGValid = true;
    }

    // --- Per-call buffers ---
    cuDoubleComplex* dPsi;   // length-N slice buffer
    cuDoubleComplex* dBuf;   // N×N working buffer: A[m,k] then H[m,j]
    double* dAccum;          // N×N accumulator layout: [m*N + j]
    checkCudaErrors(cudaMalloc(&dPsi,   cvDim * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMalloc(&dBuf,   (size_t)cvDim * cvDim * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMalloc(&dAccum, (size_t)cvDim * cvDim * sizeof(double)));
    checkCudaErrors(cudaMemset(dAccum, 0, (size_t)cvDim * cvDim * sizeof(double)));

    int gridN  = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridN2 = ((size_t)cvDim * cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    // IMPORTANT TODO: batch kernel
    for (size_t sliceIdx = 0; sliceIdx < sliceCount; sliceIdx++) {
        // Step 1: extract ψ_s into contiguous dPsi
        kernelExtractSlicePsi<<<gridN, CUDA_BLOCK_SIZE>>>(
            dPsi, ctx->dState, regIdx, cvDim, sliceIdx, ctx->gQbts, ctx->gFlwQbts);
        // Step 2: forward FFT → Ψ_s in dPsi
        cufftExecZ2Z(ctx->hSinglePlan, dPsi, dPsi, CUFFT_FORWARD);
        // Step 3: fill circulant A[m,k] = Ψ_s[(m+k)%N] * G[k]
        kernelFillHusimiA<<<gridN2, CUDA_BLOCK_SIZE>>>(dBuf, dPsi, ctx->dHusimiG, cvDim);
        // Step 4: batch IFFT over k for each row m → H_s[m, j]
        cufftExecZ2Z(ctx->hPlan, dBuf, dBuf, CUFFT_INVERSE);
        // Step 5: accumulate |H_s|²
        kernelAccumHusimiPower<<<gridN2, CUDA_BLOCK_SIZE>>>(dAccum, dBuf, cvDim * cvDim);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    double* dHusimiQ;
    checkCudaErrors(cudaMalloc(&dHusimiQ, (size_t)cvDim * cvDim * sizeof(double)));
    kernelFinalizeHusimi<<<gridN2, CUDA_BLOCK_SIZE>>>(dHusimiQ, dAccum, cvDim);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(husimiOut, dHusimiQ, (size_t)cvDim * cvDim * sizeof(double),
                               cudaMemcpyDeviceToHost));
    cudaFree(dHusimiQ);
    cudaFree(dAccum);
    cudaFree(dBuf);
    cudaFree(dPsi);
}

void cvdvMeasureMultiple(CVDVContext* ctx, const int* regIdxs, int numRegs, double* probsOut) {
    if (!ctx || !regIdxs || numRegs <= 0) return;
    for (int i = 0; i < numRegs; i++) {
        if (regIdxs[i] < 0 || regIdxs[i] >= ctx->gNumReg) {
            fprintf(stderr, "Invalid register index: %d\n", regIdxs[i]);
            return;
        }
    }

    size_t totalSize = (size_t)1 << ctx->gTotalQbt;

    // Lazy-init cuTENSOR handle (once per context lifetime)
    if (!ctx->ctHandleValid) {
        cutensorStatus_t initStatus = cutensorCreate(&ctx->ctHandle);
        if (initStatus != CUTENSOR_STATUS_SUCCESS) {
            fprintf(stderr, "cuTENSOR init failed: %d\n", initStatus);
            return;
        }
        ctx->ctHandleValid = true;
    }

    // Lazy-init scratch buffers (both sized to totalSize — worst-case output)
    if (ctx->dMeasureProbs == nullptr) {
        checkCudaErrors(cudaMalloc(&ctx->dMeasureProbs, totalSize * sizeof(double)));
        checkCudaErrors(cudaMalloc(&ctx->dMeasureOut, totalSize * sizeof(double)));
    }

    // Lazy-init plan cache
    if (ctx->measurePlanCache == nullptr) {
        ctx->measurePlanCache = new std::map<std::vector<int>, MeasurePlan>();
    }
    auto* cache = static_cast<std::map<std::vector<int>, MeasurePlan>*>(ctx->measurePlanCache);

    // Look up cached plan by regIdxs key, build on miss
    std::vector<int> key(regIdxs, regIdxs + numRegs);
    auto it = cache->find(key);
    if (it == cache->end()) {
        // Fetch qbts/flwQbts via explicit memcpy — managed memory may have
        // migrated to device after GPU kernels ran; direct CPU reads are unsafe.
        std::vector<int> qbts(ctx->gNumReg), flwQbts(ctx->gNumReg);
        checkCudaErrors(cudaMemcpy(qbts.data(), ctx->gQbts,
                                   ctx->gNumReg * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(flwQbts.data(), ctx->gFlwQbts,
                                   ctx->gNumReg * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int64_t> extents(ctx->gNumReg), strides(ctx->gNumReg);
        for (int r = 0; r < ctx->gNumReg; r++) {
            extents[r] = 1LL << qbts[r];
            strides[r] = 1LL << flwQbts[r];
        }
        std::vector<int64_t> outExtents(numRegs), outStrides(numRegs);
        size_t outSize = 1;
        for (int i = 0; i < numRegs; i++) {
            outExtents[i] = extents[regIdxs[i]];
            outSize *= outExtents[i];
        }
        // Column-major strides: first selected register varies fastest (stride 1).
        // The Python wrapper does reshape(shape[::-1]).T expecting this layout.
        outStrides[0] = 1;
        for (int i = 1; i < numRegs; i++)
            outStrides[i] = outStrides[i - 1] * outExtents[i - 1];

        std::vector<int32_t> modesIn(ctx->gNumReg), modesOut(numRegs);
        for (int i = 0; i < ctx->gNumReg; i++) modesIn[i] = i;
        for (int i = 0; i < numRegs; i++) modesOut[i] = regIdxs[i];

        MeasurePlan mp{};
        mp.outSize = outSize;
        cutensorHandle_t h = ctx->ctHandle;
        cutensorStatus_t status;
        uint32_t alignment = 256;
        cutensorComputeDescriptor_t computeDesc = CUTENSOR_COMPUTE_DESC_64F;

        status = cutensorCreateTensorDescriptor(h, &mp.descIn, (uint32_t)ctx->gNumReg,
                                                extents.data(), strides.data(),
                                                CUDA_R_64F, alignment);
        if (status != CUTENSOR_STATUS_SUCCESS) {
            fprintf(stderr, "cuTENSOR input desc failed: %d\n", status); return;
        }
        status = cutensorCreateTensorDescriptor(h, &mp.descOut, (uint32_t)numRegs,
                                                outExtents.data(), outStrides.data(),
                                                CUDA_R_64F, alignment);
        if (status != CUTENSOR_STATUS_SUCCESS) {
            fprintf(stderr, "cuTENSOR output desc failed: %d\n", status);
            cutensorDestroyTensorDescriptor(mp.descIn); return;
        }
        status = cutensorCreateReduction(h, &mp.opDesc,
                                         mp.descIn,  modesIn.data(),  CUTENSOR_OP_IDENTITY,
                                         mp.descOut, modesOut.data(), CUTENSOR_OP_IDENTITY,
                                         mp.descOut, modesOut.data(),
                                         CUTENSOR_OP_ADD, computeDesc);
        if (status != CUTENSOR_STATUS_SUCCESS) {
            fprintf(stderr, "cuTENSOR reduction desc failed: %d\n", status);
            cutensorDestroyTensorDescriptor(mp.descIn);
            cutensorDestroyTensorDescriptor(mp.descOut); return;
        }
        status = cutensorCreatePlanPreference(h, &mp.planPref,
                                              CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
        if (status != CUTENSOR_STATUS_SUCCESS) {
            fprintf(stderr, "cuTENSOR plan pref failed: %d\n", status);
            cutensorDestroyOperationDescriptor(mp.opDesc);
            cutensorDestroyTensorDescriptor(mp.descIn);
            cutensorDestroyTensorDescriptor(mp.descOut); return;
        }
        mp.workspaceSize = 0;
        cutensorEstimateWorkspaceSize(h, mp.opDesc, mp.planPref,
                                      CUTENSOR_WORKSPACE_DEFAULT, &mp.workspaceSize);
        status = cutensorCreatePlan(h, &mp.plan, mp.opDesc, mp.planPref, mp.workspaceSize);
        if (status != CUTENSOR_STATUS_SUCCESS) {
            fprintf(stderr, "cuTENSOR plan failed: %d\n", status);
            cutensorDestroyPlanPreference(mp.planPref);
            cutensorDestroyOperationDescriptor(mp.opDesc);
            cutensorDestroyTensorDescriptor(mp.descIn);
            cutensorDestroyTensorDescriptor(mp.descOut); return;
        }
        mp.dWorkspace = nullptr;
        if (mp.workspaceSize > 0)
            checkCudaErrors(cudaMalloc(&mp.dWorkspace, mp.workspaceSize));

        (*cache)[key] = mp;
        it = cache->find(key);
    }
    const MeasurePlan& mp = it->second;

    // Step 1: Compute |ψ|² into scratch buffer
    int gridSize = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    kernelAbsSquareInPlace<<<gridSize, CUDA_BLOCK_SIZE>>>(ctx->dMeasureProbs, ctx->dState, totalSize);

    // Zero output before reduction: cuTENSOR beta=0 only writes positions with
    // non-zero contributions, leaving stale values in zero-probability slots.
    checkCudaErrors(cudaMemset(ctx->dMeasureOut, 0, mp.outSize * sizeof(double)));

    // Step 2: cuTENSOR reduction
    const double one = 1.0, zero = 0.0;
    cutensorStatus_t status = cutensorReduce(ctx->ctHandle, mp.plan,
                                             &one, ctx->dMeasureProbs,
                                             &zero, ctx->dMeasureOut, ctx->dMeasureOut,
                                             mp.dWorkspace, mp.workspaceSize, 0);
    if (status != CUTENSOR_STATUS_SUCCESS)
        fprintf(stderr, "cuTENSOR reduction failed: %d\n", status);

    checkCudaErrors(cudaStreamSynchronize(0));
    checkCudaErrors(cudaMemcpy(probsOut, ctx->dMeasureOut, mp.outSize * sizeof(double),
                               cudaMemcpyDeviceToHost));
}

// Copy a full flat state from an existing CUDA device pointer (e.g. a torch
// tensor). d_src must be cuDoubleComplex* on the GPU with totalSize elements.
// Allocates dState if not yet allocated.
void cvdvSetStateFromDevicePtr(CVDVContext* ctx, void* d_src) {
    if (!ctx || !d_src) return;
    size_t totalSize = 1 << ctx->gTotalQbt;
    if (ctx->dState == nullptr) {
        checkCudaErrors(cudaMalloc(&ctx->dState, totalSize * sizeof(cuDoubleComplex)));
    }
    checkCudaErrors(cudaMemcpy(ctx->dState, d_src, totalSize * sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToDevice));
}

void cvdvGetState(CVDVContext* ctx, double* realOut, double* imagOut) {
    if (!ctx) return;
    size_t totalSize = 1 << ctx->gTotalQbt;
    if (totalSize == 0) {
        fprintf(stderr, "Error: State not initialized. Call cvdvInitStateVector first.\n");
        return;
    }

    cuDoubleComplex* hState = new cuDoubleComplex[totalSize];
    checkCudaErrors(cudaMemcpy(hState, ctx->dState, totalSize * sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < totalSize; i++) {
        realOut[i] = cuCreal(hState[i]);
        imagOut[i] = cuCimag(hState[i]);
    }

    delete[] hState;
}


#pragma endregion

#pragma region C API - Getters

int cvdvGetNumRegisters(CVDVContext* ctx) {
    if (!ctx) return 0;
    return ctx->gNumReg;
}

size_t cvdvGetTotalSize(CVDVContext* ctx) {
    if (!ctx) return 0;
    return 1 << ctx->gTotalQbt;
}

void cvdvGetRegisterInfo(CVDVContext* ctx, int* qubitCountsOut, double* gridStepsOut) {
    if (!ctx) return;
    if (ctx->gNumReg == 0) return;

    checkCudaErrors(cudaMemcpy(qubitCountsOut, ctx->gQbts, ctx->gNumReg * sizeof(int),
                                    cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(gridStepsOut, ctx->gGridSteps, ctx->gNumReg * sizeof(double),
                                    cudaMemcpyDeviceToHost));
}

int cvdvGetRegisterDim(CVDVContext* ctx, int regIdx) {
    if (!ctx) return -1;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return -1;

    int qubit_count;
    checkCudaErrors(cudaMemcpy(&qubit_count, ctx->gQbts + regIdx, sizeof(int), cudaMemcpyDeviceToHost));
    return 1 << qubit_count;
}

double cvdvGetRegisterDx(CVDVContext* ctx, int regIdx) {
    if (!ctx) return -1.0;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return -1.0;

    double dx;
    checkCudaErrors(cudaMemcpy(&dx, ctx->gGridSteps + regIdx, sizeof(double), cudaMemcpyDeviceToHost));
    return dx;
}


// Helper: build device pointer-of-pointers from a host array of numReg device
// ptrs, run kernelComputeInnerProduct, reduce, free, and return the complex
// sum.
static cuDoubleComplex runInnerProductKernel(CVDVContext* ctx, void** devicePtrs, int numReg) {
    // Upload pointer array to device
    cuDoubleComplex** dRegArrayPtrs;
    checkCudaErrors(cudaMalloc(&dRegArrayPtrs, numReg * sizeof(cuDoubleComplex*)));
    // Copy host-side pointer values (each is a device ptr) to device memory
    checkCudaErrors(cudaMemcpy(dRegArrayPtrs, devicePtrs, numReg * sizeof(cuDoubleComplex*),
                                    cudaMemcpyHostToDevice));

    size_t totalSize = 1 << ctx->gTotalQbt;
    int numBlocks = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    numBlocks = min(numBlocks, 1024);

    cuDoubleComplex* dPartialSums;
    checkCudaErrors(cudaMalloc(&dPartialSums, numBlocks * sizeof(cuDoubleComplex)));

    size_t sharedMemSize = CUDA_BLOCK_SIZE * sizeof(cuDoubleComplex);
    kernelComputeInnerProduct<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemSize>>>(
        dPartialSums, ctx->dState, dRegArrayPtrs, numReg, ctx->gQbts, ctx->gFlwQbts, totalSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cuDoubleComplex* hPartialSums = new cuDoubleComplex[numBlocks];
    checkCudaErrors(cudaMemcpy(hPartialSums, dPartialSums, numBlocks * sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost));
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < numBlocks; i++) result = cuCadd(result, hPartialSums[i]);

    delete[] hPartialSums;
    checkCudaErrors(cudaFree(dPartialSums));
    checkCudaErrors(cudaFree(dRegArrayPtrs));
    return result;
}

// Compute fidelity |⟨sep|ψ⟩|² where sep is a SeparableState passed as device
// pointers. devicePtrs[i] points to cuDoubleComplex device memory of size (1 <<
// gQbts[i]).
void cvdvGetFidelity(CVDVContext* ctx, void** devicePtrs, int numReg, double* fidOut) {
    if (!ctx) return;
    if (ctx->dState == nullptr || ctx->gNumReg == 0) {
        fprintf(stderr, "State not initialized\n");
        *fidOut = 0.0;
        return;
    }
    if (numReg != ctx->gNumReg) {
        fprintf(stderr, "numReg mismatch: context has %d, got %d\n", ctx->gNumReg, numReg);
        *fidOut = 0.0;
        return;
    }

    cuDoubleComplex ip = runInnerProductKernel(ctx, devicePtrs, numReg);
    double re = cuCreal(ip), im = cuCimag(ip);
    *fidOut = re * re + im * im;
}

// Internal: compute <x²> for a register using parallel reduction
static double computeExpectX2(CVDVContext* ctx, int regIdx) {
    size_t totalSize = 1ULL << ctx->gTotalQbt;
    int numBlocks = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    numBlocks = min(numBlocks, 1024);

    double* dPartialSums;
    checkCudaErrors(cudaMalloc(&dPartialSums, numBlocks * sizeof(double)));

    kernelExpectX2<<<numBlocks, CUDA_BLOCK_SIZE>>>(dPartialSums, ctx->dState, totalSize, regIdx,
                                                   ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    double* hPartialSums = new double[numBlocks];
    checkCudaErrors(cudaMemcpy(hPartialSums, dPartialSums, numBlocks * sizeof(double),
                                    cudaMemcpyDeviceToHost));
    double result = 0.0;
    for (int i = 0; i < numBlocks; i++) result += hPartialSums[i];
    delete[] hPartialSums;
    checkCudaErrors(cudaFree(dPartialSums));
    return result;
}

// Compute mean photon number <n> = (<q²> + <p²> - 1) / 2 for register regIdx.
// Temporarily applies ftQ2P / ftP2Q to measure <p²> in momentum basis (dp =
// dx).
double cvdvGetPhotonNumber(CVDVContext* ctx, int regIdx) {
    if (!ctx || ctx->dState == nullptr) return -1.0;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return -1.0;

    double q2 = computeExpectX2(ctx, regIdx);

    cvdvFtQ2P(ctx, regIdx);
    double p2 = computeExpectX2(ctx, regIdx);
    cvdvFtP2Q(ctx, regIdx);

    return (q2 + p2 - 1.0) / 2.0;
}

double cvdvGetNorm(CVDVContext* ctx) {
    if (!ctx) return 0.0;
    if (ctx->dState == nullptr) {
        fprintf(stderr, "State not initialized\n");
        return 0.0;
    }

    // Allocate partial sums for reduction
    size_t totalSize = 1 << ctx->gTotalQbt;
    int numBlocks = (totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);  // Cap at 1024 blocks

    double* dPartialSums;
    checkCudaErrors(cudaMalloc(&dPartialSums, numBlocks * sizeof(double)));

    // Launch kernel with warp-level reduction
    kernelComputeNorm<<<numBlocks, CUDA_BLOCK_SIZE>>>(dPartialSums, ctx->dState, totalSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Download partial sums and reduce on host
    double* hPartialSums = new double[numBlocks];
    checkCudaErrors(cudaMemcpy(hPartialSums, dPartialSums, numBlocks * sizeof(double),
                                    cudaMemcpyDeviceToHost));

    double result = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        result += hPartialSums[i];
    }

    // Cleanup
    delete[] hPartialSums;
    checkCudaErrors(cudaFree(dPartialSums));

    return result;
}

#pragma endregion

// Compute fidelity |⟨psi1|psi2⟩|² between two CUDA statevectors of the same
// size.
void cvdvFidelityStatevectors(CVDVContext* ctx1, CVDVContext* ctx2, double* fidOut) {
    *fidOut = 0.0;
    if (!ctx1 || !ctx2) return;
    if (ctx1->dState == nullptr || ctx2->dState == nullptr) {
        fprintf(stderr, "cvdvFidelityStatevectors: state not initialized\n");
        return;
    }
    if (ctx1->gTotalQbt != ctx2->gTotalQbt) {
        fprintf(stderr, "cvdvFidelityStatevectors: total qubit count mismatch (%d vs %d)\n",
                 ctx1->gTotalQbt, ctx2->gTotalQbt);
        return;
    }

    size_t totalSize = 1ULL << ctx1->gTotalQbt;
    int numBlocks = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    numBlocks = min(numBlocks, 1024);

    cuDoubleComplex* dPartialSums;
    checkCudaErrors(cudaMalloc(&dPartialSums, numBlocks * sizeof(cuDoubleComplex)));

    size_t sharedMemSize = CUDA_BLOCK_SIZE * sizeof(cuDoubleComplex);
    kernelInnerProductStatevectors<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemSize>>>(
        dPartialSums, ctx1->dState, ctx2->dState, totalSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cuDoubleComplex* hPartialSums = new cuDoubleComplex[numBlocks];
    checkCudaErrors(cudaMemcpy(hPartialSums, dPartialSums,
                                     numBlocks * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < numBlocks; i++) result = cuCadd(result, hPartialSums[i]);

    delete[] hPartialSums;
    checkCudaErrors(cudaFree(dPartialSums));

    double re = cuCreal(result), im = cuCimag(result);
    *fidOut = re * re + im * im;
}

}  // extern "C"
