// cvdv.cu - CUDA library for hybrid CV-DV quantum simulation

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <ctime>

#pragma region Error Checking and Constants

// Global log file (filename from LOG_FILENAME constexpr)
static FILE* gLogFile = nullptr;

#define logInfo(ctx, ...) do { \
    if (gLogFile) { \
        fprintf(gLogFile, "[INFO] "); \
        fprintf(gLogFile, __VA_ARGS__); \
        fprintf(gLogFile, "\n"); \
        fflush(gLogFile); \
    } \
} while(0)

#define logDebug(ctx, ...) do { \
    if (gLogFile) { \
        fprintf(gLogFile, "[DEBUG] "); \
        fprintf(gLogFile, __VA_ARGS__); \
        fprintf(gLogFile, "\n"); \
        fflush(gLogFile); \
    } \
} while(0)

#define logError(ctx, ...) do { \
    fprintf(stderr, "[ERROR] "); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    if (gLogFile) { \
        fprintf(gLogFile, "[ERROR] "); \
        fprintf(gLogFile, __VA_ARGS__); \
        fprintf(gLogFile, "\n"); \
        fflush(gLogFile); \
    } \
} while(0)

#define checkCudaErrors(ctx, val) do { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        const char* msg = cudaGetErrorString(err); \
        fprintf(stderr, "[CUDA ERROR] %s at %s:%d\n", msg, __FILE__, __LINE__); \
        if (gLogFile) { \
            fprintf(gLogFile, "[CUDA ERROR] %s at %s:%d\n", msg, __FILE__, __LINE__); \
            fflush(gLogFile); \
        } \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880;
constexpr double PI_POW_NEG_QUARTER = 0.75112554446494248286;  // PI^(-0.25)
constexpr int CUDA_BLOCK_SIZE = 256;  // Default CUDA block size for RTX 4070 Laptop
constexpr const char* LOG_FILENAME = "cuda.log";

#pragma endregion

#pragma region Global State

// Context structure to enable multiple instances
typedef struct {
    cuDoubleComplex* dState;
    int* gQbts;                // Managed memory: number of qubits in each register
    int* gFlwQbts;             // Managed memory: cumulative qubits after each register
    double* gGridSteps;        // Managed memory: grid step (dx) for each register
    int gNumReg;               // Total number of registers
    int gTotalQbt;             // Total number of qubits across all registers
    cuDoubleComplex** dRegisterArrays;  // Device arrays for each register's coefficients
} CVDVContext;

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

#pragma endregion

#pragma region Register-Based Indexing Helpers

// Extract local index within a register from global index
// Returns the index for register regIdx given the global flat index
__device__ __host__ inline size_t getLocalIndex(size_t globalIdx, int flwQbtCount, int regQbtCount) {
    return (globalIdx >> flwQbtCount) & ((1 << regQbtCount) - 1);
}

#pragma endregion

#pragma region State Initialization Kernels

__global__ void kernelSetCoherent(cuDoubleComplex* state, int cvDim, double dx,
                                   double alphaRe, double alphaIm) {
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
    double amplitude = norm * gauss * sqrt(dx); // Normalize as qubit register

    state[idx] = make_cuDoubleComplex(amplitude * cuCreal(phaseFactor), 
                                      amplitude * cuCimag(phaseFactor));
}

__global__ void kernelSetFock(cuDoubleComplex* state, int cvDim, double dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;

    double x = gridX(idx, cvDim, dx);

    // Fock state |n> using normalized Hermite function recurrence for numerical stability:
    // psi_0(x) = exp(-x^2/2) / pi^(1/4) * sqrt(dx)
    // psi_n(x) = sqrt(2/n) * x * psi_{n-1}(x) - sqrt((n-1)/n) * psi_{n-2}(x)
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
                              const cuDoubleComplex* alphas, const cuDoubleComplex* coeffs, int length) {
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

        cuDoubleComplex coherentState = make_cuDoubleComplex(
            amplitude * cuCreal(phaseFactor),
            amplitude * cuCimag(phaseFactor)
        );

        // Add coeff_i * |alpha_i>
        result = cuCadd(result, cuCmul(coeffs[i], coherentState));
    }

    state[idx] = result;
}

// Compute norm for a single register array (reduction within block)
__global__ void kernelComputeRegisterNorm(double* partialSums, const cuDoubleComplex* state, int cvDim) {
    extern __shared__ double sdataReg[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localSum = 0.0;

    // Grid-stride loop
    for (int i = idx; i < cvDim; i += blockDim.x * gridDim.x) {
        localSum += absSquare(state[i]);
    }

    // Store in shared memory
    sdataReg[threadIdx.x] = localSum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdataReg[threadIdx.x] += sdataReg[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write block result
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sdataReg[0];
    }
}

// Normalize register array by dividing by a scalar
__global__ void kernelNormalizeRegister(cuDoubleComplex* state, int cvDim, double invNorm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;

    state[idx] = make_cuDoubleComplex(
        cuCreal(state[idx]) * invNorm,
        cuCimag(state[idx]) * invNorm
    );
}

#pragma endregion

#pragma region Gate Kernels

// Apply phase factor to a specific register: exp(i*phaseCoeff*x)
__global__ void kernelApplyOneModeQ(cuDoubleComplex* state, size_t totalSize,
                                             int regIdx,
                                             const int* qbts, const double* gridSteps,
                                             const int* flwQbts,
                                             double phaseCoeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = 1 << qbts[regIdx];
    double dx = gridSteps[regIdx];
    size_t localIdx = getLocalIndex(idx, flwQbts[regIdx], qbts[regIdx]);

    double x = gridX(localIdx, regDim, dx);
    state[idx] = cmulPhase(state[idx], phaseCoeff * x);
}

// Apply controlled phase to a specific register with control from another register
// exp(i*phaseCoeff*Z*x) where Z acts on ctrlQubit in ctrlReg
__global__ void kernelApplyConditionalOneModeQ(cuDoubleComplex* state, size_t totalSize,
                                                         int targetReg, int ctrlReg, int ctrlQubit,
                                                         const int* qbts, const double* gridSteps,
                                                         const int* flwQbts,
                                                         double phaseCoeff) {
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

    // Z operator: |0⟩ gets +phase, |1⟩ gets -phase
    if (ctrlLocalIdx & ctrlMask) {
        state[idx] = cmulPhase(state[idx], -phase);
    } else {
        state[idx] = cmulPhase(state[idx], phase);
    }
}

// Apply controlled phase based on position squared: exp(i*t*Z*x^2)
// where Z acts on ctrlQubit in ctrlReg
__global__ void kernalApplyConditionalOneModeQ2(cuDoubleComplex* state, size_t totalSize,
                                                        int targetReg, int ctrlReg, int ctrlQubit,
                                                        const int* qbts, const double* gridSteps,
                                                        const int* flwQbts,
                                                        double t) {
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

    // Z operator: |0⟩ gets +phase, |1⟩ gets -phase
    if (ctrlLocalIdx & ctrlMask) {
        state[idx] = cmulPhase(state[idx], -phase);
    } else {
        state[idx] = cmulPhase(state[idx], phase);
    }
}

// Apply Pauli rotation to a specific qubit within a register
__global__ void kernelPauliRotation(cuDoubleComplex* state, size_t totalSize,
                                                int regIdx, int qubitIdx,
                                                const int* qbts, const int* flwQbts,
                                                int axis, double theta) {
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= totalSize / 2) return;

    size_t regStride = 1 << flwQbts[regIdx];

    // Target qubit mask within the register's local index space
    // Qubit 0 is the most significant bit
    size_t targetBit = 1 << (qbts[regIdx] - 1 - qubitIdx);

    // Decompose pairIdx into compressed space
    size_t regBlockSize = regStride << qbts[regIdx];
    size_t regBlockSizeCompressed = regStride << (qbts[regIdx] - 1);
    
    size_t outerIdx = pairIdx / regBlockSizeCompressed;
    size_t innerCompressed = pairIdx % regBlockSizeCompressed;
    
    size_t localCompressed = innerCompressed / regStride;
    size_t strideOffset = innerCompressed % regStride;
    
    // Expand localCompressed by inserting the target bit
    size_t lowerBits = localCompressed & (targetBit - 1);
    size_t upperBits = localCompressed & ~(targetBit - 1);
    
    size_t local0 = (upperBits << 1) | lowerBits;
    size_t local1 = local0 | targetBit;
    
    size_t idx0 = (outerIdx * regBlockSize) + (local0 << flwQbts[regIdx]) + strideOffset;
    size_t idx1 = (outerIdx * regBlockSize) + (local1 << flwQbts[regIdx]) + strideOffset;

    cuDoubleComplex a = state[idx0];
    cuDoubleComplex b = state[idx1];

    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);

    cuDoubleComplex newA, newB;

    if (axis == 0) {  // X
        newA = make_cuDoubleComplex(c * cuCreal(a) + s * cuCimag(b),
                                      c * cuCimag(a) - s * cuCreal(b));
        newB = make_cuDoubleComplex(s * cuCimag(a) + c * cuCreal(b),
                                      -s * cuCreal(a) + c * cuCimag(b));
    } else if (axis == 1) {  // Y
        newA = make_cuDoubleComplex(c * cuCreal(a) - s * cuCreal(b),
                                      c * cuCimag(a) - s * cuCimag(b));
        newB = make_cuDoubleComplex(s * cuCreal(a) + c * cuCreal(b),
                                      s * cuCimag(a) + c * cuCimag(b));
    } else {  // Z
        cuDoubleComplex phase0 = make_cuDoubleComplex(c, -s);
        cuDoubleComplex phase1 = make_cuDoubleComplex(c, s);
        newA = cuCmul(phase0, a);
        newB = cuCmul(phase1, b);
    }

    state[idx0] = newA;
    state[idx1] = newB;
}

// Hadamard gate on a specific qubit within a register
__global__ void kernelHadamard(cuDoubleComplex* state, size_t totalSize,
                                          int regIdx, int qubitIdx,
                                          const int* qbts, const int* flwQbts) {
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= totalSize / 2) return;

    size_t regStride = 1 << flwQbts[regIdx];

    // Target qubit mask within the register's local index space
    // Qubit 0 is the most significant bit
    size_t targetBit = 1 << (qbts[regIdx] - 1 - qubitIdx);

    // Strategy: pairIdx indexes the "compressed" space where target qubit is factored out
    // We need to expand it to get both global indices (with qubit=0 and qubit=1)
    
    // First, decompose pairIdx into:
    // - indices for registers other than regIdx
    // - index within regIdx with target qubit bit removed (compressed local index)
    
    size_t regBlockSize = regStride << qbts[regIdx];  // Total elements in one "block" of this register
    size_t regBlockSizeCompressed = regStride << (qbts[regIdx] - 1);
    
    // Index in the space outside this register's block
    size_t outerIdx = pairIdx / regBlockSizeCompressed;
    // Index within the compressed register block
    size_t innerCompressed = pairIdx % regBlockSizeCompressed;
    
    // Decompose innerCompressed into local register index (compressed) and stride offset
    size_t localCompressed = innerCompressed / regStride;
    size_t strideOffset = innerCompressed % regStride;
    
    // Expand localCompressed by inserting the target bit
    // Split into bits below and above the target bit position
    size_t lowerBits = localCompressed & (targetBit - 1);  // Bits below target
    size_t upperBits = localCompressed & ~(targetBit - 1);  // Bits above target (need to shift)
    
    // Construct local indices with target qubit = 0 and = 1
    size_t local0 = (upperBits << 1) | lowerBits;  // Insert 0 at target position
    size_t local1 = local0 | targetBit;              // Set target bit to 1
    
    // Construct global indices
    size_t idx0 = outerIdx * regBlockSize + local0 * regStride + strideOffset;
    size_t idx1 = outerIdx * regBlockSize + local1 * regStride + strideOffset;

    // Read amplitudes
    cuDoubleComplex a = state[idx0];
    cuDoubleComplex b = state[idx1];

    // Apply Hadamard: H = 1/√2 * [[1, 1], [1, -1]]
    // |0⟩ -> 1/√2(|0⟩ + |1⟩), |1⟩ -> 1/√2(|0⟩ - |1⟩)
    double invSqrt2 = 1.0 / SQRT2;

    cuDoubleComplex newA = make_cuDoubleComplex(
        invSqrt2 * (cuCreal(a) + cuCreal(b)),
        invSqrt2 * (cuCimag(a) + cuCimag(b))
    );
    cuDoubleComplex newB = make_cuDoubleComplex(
        invSqrt2 * (cuCreal(a) - cuCreal(b)),
        invSqrt2 * (cuCimag(a) - cuCimag(b))
    );

    state[idx0] = newA;
    state[idx1] = newB;
}

// Parity gate: flip all qubits of a register (X on each qubit = reverse index)
// This maps |j⟩ → |N-1-j⟩ for the target register
__global__ void kernelParity(cuDoubleComplex* state, size_t totalSize,
                             int regIdx, const int* qbts, const int* flwQbts) {
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

__global__ void kernelConditionalParity(cuDoubleComplex* state, size_t totalSize,
                             int targetReg, int ctrlReg, int ctrlQubit,
                             const int* qbts, const int* flwQbts) {
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

// SWAP gate: swap the qubit contents of two registers (must have same numQubits)
// Maps |i⟩₁|j⟩₂ → |j⟩₁|i⟩₂
__global__ void kernelSwapRegisters(cuDoubleComplex* state, size_t totalSize,
                                    int reg1, int reg2,
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
    size_t partnerIdx = idx
        + (local2 - local1) * stride1   // reg1 gets local2
        - (local2 - local1) * stride2;  // reg2 gets local1

    cuDoubleComplex tmp = state[idx];
    state[idx] = state[partnerIdx];
    state[partnerIdx] = tmp;
}

// Apply phase based on position squared: exp(i*t*x^2)
__global__ void kernelApplyOneModeQ2(cuDoubleComplex* state, size_t totalSize,
                                                    int regIdx,
                                                    const int* qbts, const double* gridSteps,
                                                    const int* flwQbts,
                                                    double t) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = 1 << qbts[regIdx];
    double dx = gridSteps[regIdx];
    size_t localIdx = getLocalIndex(idx, flwQbts[regIdx], qbts[regIdx]);

    double x = gridX(localIdx, regDim, dx);
    state[idx] = cmulPhase(state[idx], t * x * x);
}

// Apply phase factor to a specific register: exp(i*phaseCoeff*x)
__global__ void kernelApplyOneModeQ3(cuDoubleComplex* state, size_t totalSize,
                                             int regIdx,
                                             const int* qbts, const double* gridSteps,
                                             const int* flwQbts,
                                             double phaseCoeff) {
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
__global__ void kernelApplyTwoModeQQ(cuDoubleComplex* state, size_t totalSize,
                                          int reg1Idx, int reg2Idx,
                                          const int* qbts, const double* gridSteps,
                                          const int* flwQbts,
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
__global__ void kernelApplyConditionalTwoModeQQ(cuDoubleComplex* state, size_t totalSize,
                                          int reg1Idx, int reg2Idx,
                                          int ctrlReg, int ctrlQubit,
                                          const int* qbts, const double* gridSteps,
                                          const int* flwQbts,
                                          double coeff) {
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
    if (ctrlLocalIdx & ctrlMask) {
        state[idx] = cmulPhase(state[idx], -phase);
    } else {
        state[idx] = cmulPhase(state[idx], phase);
    }
}

#pragma endregion

#pragma region FFT Helper Kernels

// Apply scalar multiplication to register elements (not neccessarily normalized scalar)
__global__ void kernelApplyScalarRegister(cuDoubleComplex* data, size_t totalSize,
                                              int regIdx,
                                              const int* qbts, double scalar) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    data[idx] = make_cuDoubleComplex(cuCreal(data[idx]) * scalar,
                                      cuCimag(data[idx]) * scalar);
}

#pragma endregion

#pragma region Utility Kernels
// Compute Wigner function W(x,p) = (1/π) ∫ ψ*(x+y)ψ(x-y)e^(2ipy) dy for a register slice
__global__ void kernelComputeWignerSingleSlice(double* wigner, const cuDoubleComplex* state,
                                                    int regIdx, const int* sliceIndices,
                                                    int cvDim, double dx,
                                                    int wignerN, double wXMax, double wPMax,
                                                    const int* qbts, const int* flwQbts) {
    int wIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (wIdx >= wignerN * wignerN) return;

    int wxIdx = wIdx % wignerN;
    int wpIdx = wIdx / wignerN;

    double wDx = 2.0 * wXMax / (wignerN - 1);
    double wDp = 2.0 * wPMax / (wignerN - 1);

    double wx = -wXMax + wxIdx * wDx;
    double wp = -wPMax + wpIdx * wDp;

    // Compute base index for slice: sum over all registers except target
    size_t baseIdx = 0;
    // Count registers to know when to stop
    for (int r = 0; ; r++) {
        if (flwQbts[r] == 0 && r > 0) {
            break;
        }
        if (r != regIdx) {
            baseIdx += sliceIndices[r] << flwQbts[r];
        }
    }
    size_t regStride = 1 << flwQbts[regIdx];

    // Integrate over y: W(x,p) = (1/π) ∫ ψ*(x+y)ψ(x-y)e^(2ipy) dy
    double realSum = 0.0;
    for (int yIdx = 0; yIdx < cvDim; yIdx++) {
        double y = gridX(yIdx, cvDim, dx);

        // Grid indices for x±y
        int xpyIdx = (int)round((wx + y) / dx) + cvDim / 2;
        int xmyIdx = (int)round((wx - y) / dx) + cvDim / 2;

        if (xpyIdx >= 0 && xpyIdx < cvDim && xmyIdx >= 0 && xmyIdx < cvDim) {
            // Direct index computation: globalIdx = baseIdx + localIdx * regStride
            size_t idxPy = baseIdx + xpyIdx * regStride;
            size_t idxMy = baseIdx + xmyIdx * regStride;

            cuDoubleComplex psiXpy = state[idxPy];
            cuDoubleComplex psiXmy = state[idxMy];
            cuDoubleComplex prod = conjMul(psiXpy, psiXmy);

            double phase = 2.0 * wp * y;
            realSum += cuCreal(prod) * cos(phase) - cuCimag(prod) * sin(phase);
        }
    }

    wigner[wIdx] = realSum * dx / PI;
}

// Compute Wigner function W(x,p) summed over all other registers
// This computes the reduced Wigner function by tracing out all registers except regIdx
__global__ void kernelComputeWignerFullMode(double* wigner, const cuDoubleComplex* state,
                                                 int regIdx, int cvDim, double dx,
                                                 int wignerN, double wXMax, double wPMax,
                                                 int numReg, const int* qbts, const int* flwQbts) {
    // Use grid-stride loop for better occupancy
    int wIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWignerPoints = wignerN * wignerN;
    
    // Pre-compute constants
    double wDx = 2.0 * wXMax / (wignerN - 1);
    double wDp = 2.0 * wPMax / (wignerN - 1);
    size_t regStride = 1 << flwQbts[regIdx];
    
    // Total size of all other registers combined = 2^(total qubits - this register's qubits)
    int totalQubits = qbts[0] + flwQbts[0];
    size_t otherSize = 1 << (totalQubits - qbts[regIdx]);
    
    // Grid-stride loop to process multiple Wigner points per thread if needed
    for (int w = wIdx; w < totalWignerPoints; w += blockDim.x * gridDim.x) {
        int wxIdx = w % wignerN;
        int wpIdx = w / wignerN;

        double wx = -wXMax + wxIdx * wDx;
        double wp = -wPMax + wpIdx * wDp;

        // Sum over all configurations of other registers
        double realSum = 0.0;
        
        // Loop over all possible states of other registers
        for (size_t otherIdx = 0; otherIdx < otherSize; otherIdx++) {
            // Reconstruct global base index for this configuration of other registers
            
            // Extract base index from compressed representation
            size_t baseIdx = 0;
            size_t remainingIdx = otherIdx;
            
            for (int r = numReg - 1; r >= 0; r--) {
                if (r == regIdx) continue;
                
                size_t rDim = 1 << qbts[r];
                size_t rStride = 1 << flwQbts[r];
                size_t rLocalIdx = remainingIdx % rDim;
                remainingIdx /= rDim;
                
                baseIdx += rLocalIdx * rStride;
            }
            
            // Now compute Wigner for this slice - use register variables for accumulation
            for (int yIdx = 0; yIdx < cvDim; yIdx++) {
                double y = gridX(yIdx, cvDim, dx);

                // Grid indices for x±y
                int xpyIdx = (int)round((wx + y) / dx) + cvDim / 2;
                int xmyIdx = (int)round((wx - y) / dx) + cvDim / 2;

                if (xpyIdx >= 0 && xpyIdx < cvDim && xmyIdx >= 0 && xmyIdx < cvDim) {
                    size_t idxPy = baseIdx + xpyIdx * regStride;
                    size_t idxMy = baseIdx + xmyIdx * regStride;

                    cuDoubleComplex psiXpy = state[idxPy];
                    cuDoubleComplex psiXmy = state[idxMy];
                    cuDoubleComplex prod = conjMul(psiXpy, psiXmy);

                    double phase = 2.0 * wp * y;
                    realSum += cuCreal(prod) * cos(phase) - cuCimag(prod) * sin(phase);
                }
            }
        }

        wigner[w] = realSum * dx / PI;
    }
}

// TODO: Use Convolution to speed up computation
__global__ void kernelComputeHusimiQFullMode(double* outHusimiQ, const cuDoubleComplex* state,
                                             int regIdx, double dx,
                                             int qN, double qMax,
                                             int numReg, const int* qbts, const int* flwQbts) {
    // Thread ID in flattened grid
    int husimiGridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (husimiGridIdx >= qN * qN) return;
    
    // Pre-compute grid parameters
    double dq = 2.0 * qMax / (qN - 1);
    int qIdx = husimiGridIdx % qN;
    int pIdx = husimiGridIdx / qN;
    double sample_q = -qMax + qIdx * dq;
    double sample_p = -qMax + pIdx * dq;
    
    double res = 0;
    cuDoubleComplex localOverlapSum;
    size_t cvDim = 1 << qbts[regIdx];
    size_t sliceCount = 1 << (qbts[0] + flwQbts[0] - qbts[regIdx]);
    int qbtsAfterCV = flwQbts[regIdx];
    size_t qbtsAfterCVMask = (1 << qbtsAfterCV) - 1;
    for (size_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
        localOverlapSum = make_cuDoubleComplex(0.0, 0.0);
        for (int xIdx = 0; xIdx < cvDim; ++xIdx) {
            double x = gridX(xIdx, cvDim, dx);
            double amplitude = PI_POW_NEG_QUARTER * exp(-0.5 * (x - sample_q) * (x - sample_q)) * sqrt(dx); // Normalize as qubit regsiter
            cuDoubleComplex phaseFactor = phaseToZ(sample_p * (x - 0.5 * sample_q));
            localOverlapSum = cuCadd(localOverlapSum, cuCmul(conjMul(phaseFactor,
                state[(sliceIdx & qbtsAfterCVMask) | (xIdx << flwQbts[regIdx]) | ((sliceIdx & ~qbtsAfterCVMask) << (qbts[regIdx]))]
            ), amplitude));
        }
        res += absSquare(localOverlapSum);
    }
        
    outHusimiQ[husimiGridIdx] = res / PI;
}

// Compute probability of joint measurement for two registers by summing over all other registers
__global__ void kernelComputeJointMeasure(double* outJointProb, const cuDoubleComplex* state,
                                          int reg1Idx, int reg2Idx,
                                          size_t totalSize, int numReg,
                                          const int* qbts, const int* flwQbts) {
    // Each thread computes one joint probability P(local1, local2)
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalPairs = 1 << (qbts[reg1Idx] + qbts[reg2Idx]);
    
    if (pairIdx >= totalPairs) return;
    
    // Extract local indices for the two registers from pairIdx
    size_t local1 = pairIdx / (1 << qbts[reg2Idx]);
    size_t local2 = pairIdx % (1 << qbts[reg2Idx]);
    
    // Sum over all configurations of other registers
    double prob = 0.0;
    
    // Total dimension of all other registers
    size_t otherSize = totalSize >> (qbts[reg1Idx] + qbts[reg2Idx]);
    
    for (size_t otherIdx = 0; otherIdx < otherSize; otherIdx++) {
        // Reconstruct global index from local indices and otherIdx
        size_t globalIdx = 0;
        size_t remainingOther = otherIdx;
        
        // Build global index by processing registers from first to last
        for (int r = 0; r < numReg; r++) {
            size_t localIdx;
            if (r == reg1Idx) {
                localIdx = local1;
            } else if (r == reg2Idx) {
                localIdx = local2;
            } else {
                // Extract this register's contribution from remainingOther
                size_t regDim = 1 << qbts[r];
                localIdx = remainingOther % regDim;
                remainingOther /= regDim;
            }
            
            // Add contribution to global index (last register varies fastest)
            globalIdx += localIdx << flwQbts[r];
        }
        
        prob += absSquare(state[globalIdx]);
    }
    
    outJointProb[pairIdx] = prob;
}

// Kernel to compute norm (sum of |state[i]|^2)
__global__ void kernelComputeNorm(double* partialSums, const cuDoubleComplex* state, size_t totalSize) {
    // Shared memory for reduction within block
    extern __shared__ double sdataNorm[];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localSum = 0.0;

    // Grid-stride loop
    for (size_t globalIdx = idx; globalIdx < totalSize; globalIdx += blockDim.x * gridDim.x) {
        localSum += absSquare(state[globalIdx]);
    }

    // Store in shared memory
    sdataNorm[threadIdx.x] = localSum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdataNorm[threadIdx.x] += sdataNorm[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write block result
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = sdataNorm[0];
    }
}

// Kernel to compute tensor product element and its contribution to inner product
// Each thread computes one element of the tensor product and accumulates <current_state | tensor_product>
__global__ void kernelComputeInnerProduct(cuDoubleComplex* partialSums, const cuDoubleComplex* state,
                                          cuDoubleComplex** dRegisterArrays,
                                          int numReg, const int* qbts, const int* flwQbts,
                                          size_t totalSize) {
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

#pragma endregion

extern "C" {

#pragma region C API - Context Management

CVDVContext* cvdvCreate(int numReg, int* numQubits) {
    // Open global log file (overwrites existing)
    if (gLogFile == nullptr) {
        gLogFile = fopen(LOG_FILENAME, "w");
        if (gLogFile) {
            time_t now = time(nullptr);
            fprintf(gLogFile, "=== CVDV Session Started: %s", ctime(&now));
            fflush(gLogFile);
        }
    }
    
    CVDVContext* ctx;
    checkCudaErrors(nullptr, cudaMallocManaged(&ctx, sizeof(CVDVContext)));
    ctx->dState = nullptr;
    ctx->gQbts = nullptr;
    ctx->gFlwQbts = nullptr;
    ctx->gGridSteps = nullptr;
    ctx->gNumReg = 0;
    ctx->gTotalQbt = 0;
    ctx->dRegisterArrays = nullptr;
    
    // If no registers specified, return empty context
    if (numReg == 0 || numQubits == nullptr) {
        logInfo(ctx, "Created empty context");
        return ctx;
    }
    
    logInfo(ctx, "Allocating %d registers", numReg);
    ctx->gNumReg = numReg;

    // Allocate managed memory for register metadata
    checkCudaErrors(ctx, cudaMallocManaged(&ctx->gQbts, numReg * sizeof(int)));
    checkCudaErrors(ctx, cudaMallocManaged(&ctx->gFlwQbts, numReg * sizeof(int)));
    checkCudaErrors(ctx, cudaMallocManaged(&ctx->gGridSteps, numReg * sizeof(double)));

    // Allocate device register arrays
    ctx->dRegisterArrays = new cuDoubleComplex*[numReg];

    // Initialize metadata and allocate space for each register
    ctx->gTotalQbt = 0;
    for (int i = 0; i < numReg; i++) {
        ctx->gQbts[i] = numQubits[i];
        ctx->gTotalQbt += numQubits[i];
        
        // Calculate grid step using formula: dx = sqrt(2 * pi / regDim)
        size_t registerDim = 1 << numQubits[i];
        ctx->gGridSteps[i] = sqrt(2.0 * PI / registerDim);
        
        // Allocate device array for this register
        checkCudaErrors(ctx, cudaMalloc(&ctx->dRegisterArrays[i], registerDim * sizeof(cuDoubleComplex)));
        checkCudaErrors(ctx, cudaMemset(ctx->dRegisterArrays[i], 0, registerDim * sizeof(cuDoubleComplex)));

        logDebug(ctx, "Register %d: qubits=%d, dx=%.6f, dim=%zu, x_bound=%.6f", 
                  i, numQubits[i], ctx->gGridSteps[i], registerDim, 
                  sqrt(2.0 * PI * registerDim));
    }
    
    // Compute following qubit counts (sum of qubits after each register)
    for (int i = 0; i < numReg; i++) {
        int followQubits = 0;
        for (int j = i + 1; j < numReg; j++) {
            followQubits += numQubits[j];
        }
        ctx->gFlwQbts[i] = followQubits;
    }
    
    logInfo(ctx, "Registers allocated successfully: %d total qubits", ctx->gTotalQbt);
    return ctx;
}

void cvdvDestroy(CVDVContext* ctx) {
    if (!ctx) return;
    
    // Free device memory
    if (ctx->dState != nullptr) {
        checkCudaErrors(ctx, cudaFree(ctx->dState));
        ctx->dState = nullptr;
    }
    
    // Free managed memory
    if (ctx->gQbts != nullptr) {
        checkCudaErrors(ctx, cudaFree(ctx->gQbts));
        ctx->gQbts = nullptr;
    }
    if (ctx->gFlwQbts != nullptr) {
        checkCudaErrors(ctx, cudaFree(ctx->gFlwQbts));
        ctx->gFlwQbts = nullptr;
    }
    if (ctx->gGridSteps != nullptr) {
        checkCudaErrors(ctx, cudaFree(ctx->gGridSteps));
        ctx->gGridSteps = nullptr;
    }
    
    // Free device register arrays
    if (ctx->dRegisterArrays != nullptr) {
        for (int i = 0; i < ctx->gNumReg; i++) {
            if (ctx->dRegisterArrays[i] != nullptr) {
                checkCudaErrors(ctx, cudaFree(ctx->dRegisterArrays[i]));
            }
        }
        delete[] ctx->dRegisterArrays;
        ctx->dRegisterArrays = nullptr;
    }
    
    // Close global log file (if this was the last instance using it)
    if (gLogFile != nullptr) {
        time_t now = time(nullptr);
        fprintf(gLogFile, "=== CVDV Session Ended: %s", ctime(&now));
        fclose(gLogFile);
        gLogFile = nullptr;
    }
    
    cudaFree(ctx);
}

#pragma endregion

#pragma region C API - Initialization and Cleanup

void cvdvInitStateVector(CVDVContext* ctx) {
    if (!ctx) return;
    if (ctx->gNumReg == 0 || ctx->dRegisterArrays == nullptr) {
        logError(ctx, "Must call cvdvCreate with registers before cvdvInitStateVector");
        return;
    }

    logInfo(ctx, "Initializing state vector from register data");

    // Compute total state size from qubit counts
    size_t totalSize = 1 << ctx->gTotalQbt;

    logDebug(ctx, "Total state size: %zu", totalSize);

    // Allocate host memory for full state and download register arrays
    cuDoubleComplex* hState = new cuDoubleComplex[totalSize];
    cuDoubleComplex** hTempRegs = new cuDoubleComplex*[ctx->gNumReg];
    
    for (int i = 0; i < ctx->gNumReg; i++) {
        size_t regDim = 1 << ctx->gQbts[i];
        hTempRegs[i] = new cuDoubleComplex[regDim];
        checkCudaErrors(ctx, cudaMemcpy(hTempRegs[i], ctx->dRegisterArrays[i], 
                                   regDim * sizeof(cuDoubleComplex), 
                                   cudaMemcpyDeviceToHost));
    }

    // Compute tensor product
    // New layout: last register varies fastest
    // globalIdx = localIdx_R0 * (D1*...*D_{n-1}) + localIdx_R1 * (D2*...*D_{n-1}) + ... + localIdx_R_{n-1}
    for (size_t globalIdx = 0; globalIdx < totalSize; globalIdx++) {
        cuDoubleComplex product = make_cuDoubleComplex(1.0, 0.0);

        size_t idx = globalIdx;
        // Iterate from last register to first (last varies fastest)
        for (int reg = ctx->gNumReg - 1; reg >= 0; reg--) {
            size_t regDim = 1 << ctx->gQbts[reg];
            size_t localIdx = idx % regDim;
            idx /= regDim;

            cuDoubleComplex regVal = hTempRegs[reg][localIdx];
            product = cuCmul(product, regVal);
        }

        hState[globalIdx] = product;
    }
    
    // Free temporary host arrays
    for (int i = 0; i < ctx->gNumReg; i++) {
        delete[] hTempRegs[i];
    }
    delete[] hTempRegs;

    // Free old device state if it exists
    if (ctx->dState != nullptr) {
        cudaFree(ctx->dState);
    }

    // Allocate and copy device state
    logDebug(ctx, "Allocating device memory: %.3f GB", totalSize * sizeof(cuDoubleComplex) / (1024.0 * 1024.0 * 1024.0));
    checkCudaErrors(ctx, cudaMalloc(&ctx->dState, totalSize * sizeof(cuDoubleComplex)));
    checkCudaErrors(ctx, cudaMemcpy(ctx->dState, hState, totalSize * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));

    // Cleanup
    delete[] hState;

    logInfo(ctx, "State vector initialized: %d registers, total size: %zu", ctx->gNumReg, totalSize);
    // printf("Initialized state with %d registers, total size: %zu\n", ctx->gNumReg, totalSize);
}

void cvdvFree(CVDVContext* ctx) {
    if (!ctx) return;
    // Free device memory
    if (ctx->dState != nullptr) {
        cudaFree(ctx->dState);
        ctx->dState = nullptr;
    }
    
    // Free managed memory for register metadata
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

    // Free device register arrays
    if (ctx->dRegisterArrays != nullptr) {
        for (int i = 0; i < ctx->gNumReg; i++) {
            if (ctx->dRegisterArrays[i] != nullptr) {
                checkCudaErrors(ctx, cudaFree(ctx->dRegisterArrays[i]));
            }
        }
        delete[] ctx->dRegisterArrays;
        ctx->dRegisterArrays = nullptr;
    }

    ctx->gNumReg = 0;
    ctx->gTotalQbt = 0;
}

#pragma endregion

#pragma region C API - State Initialization

void cvdvSetZero(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }

    logInfo(ctx, "Setting register %d to |0> state", regIdx);

    size_t dim = 1 << ctx->gQbts[regIdx];

    // Set to zero first
    checkCudaErrors(ctx, cudaMemset(ctx->dRegisterArrays[regIdx], 0, dim * sizeof(cuDoubleComplex)));
    
    // Set first element to 1
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    checkCudaErrors(ctx, cudaMemcpy(ctx->dRegisterArrays[regIdx], &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    logDebug(ctx, "Register %d set to |0> state", regIdx);
}

void cvdvSetCoherent(CVDVContext* ctx, int regIdx, double alphaRe, double alphaIm) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }

    logInfo(ctx, "Setting register %d to coherent state |(%.3f, %.3f)>",
             regIdx, alphaRe, alphaIm);

    size_t cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    if (fabs(dx) < 1e-15) {
        logError(ctx, "Coherent state requires CV mode (dx > 0)");
        return;
    }

    // Compute coherent state directly in dRegisterArrays
    int grid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelSetCoherent<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dRegisterArrays[regIdx], cvDim, dx, alphaRe, alphaIm);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    logDebug(ctx, "Register %d set to coherent state", regIdx);
}

void cvdvSetFock(CVDVContext* ctx, int regIdx, int n) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }

    logInfo(ctx, "Setting register %d to Fock state |%d>", regIdx, n);

    size_t cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    if (fabs(dx) < 1e-15) {
        logError(ctx, "Fock state requires CV mode (dx > 0)");
        return;
    }

    // Compute Fock state directly in dRegisterArrays
    int grid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelSetFock<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dRegisterArrays[regIdx], cvDim, dx, n);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    logDebug(ctx, "Register %d set to Fock state |%d>", regIdx, n);
}

void cvdvSetUniform(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }

    size_t cvDim = 1 << ctx->gQbts[regIdx];
    
    logInfo(ctx, "Setting register %d to uniform superposition (dim=%zu)", regIdx, cvDim);

    // Create uniform state: all elements = 1/sqrt(N)
    double amplitude = 1.0 / sqrt(cvDim);
    
    cuDoubleComplex* hTemp = new cuDoubleComplex[cvDim];
    for (int i = 0; i < cvDim; i++) {
        hTemp[i] = make_cuDoubleComplex(amplitude, 0.0);
    }
    
    checkCudaErrors(ctx, cudaMemcpy(ctx->dRegisterArrays[regIdx], hTemp, 
                               cvDim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    delete[] hTemp;

    logDebug(ctx, "Register %d set to uniform superposition", regIdx);
}

void cvdvSetFocks(CVDVContext* ctx, int regIdx, double* coeffs, int length) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }

    logInfo(ctx, "Setting register %d to Fock superposition with %d terms", regIdx, length);

    size_t cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    if (fabs(dx) < 1e-15) {
        logError(ctx, "Fock superposition requires CV mode (dx > 0)");
        return;
    }

    // Copy coefficients to device (interleaved: Re,Im,Re,Im,...)
    cuDoubleComplex* hCoeffs = new cuDoubleComplex[length];
    for (int i = 0; i < length; i++) {
        hCoeffs[i] = make_cuDoubleComplex(coeffs[2*i], coeffs[2*i+1]);
    }

    cuDoubleComplex* dCoeffs;
    checkCudaErrors(ctx, cudaMalloc(&dCoeffs, length * sizeof(cuDoubleComplex)));
    checkCudaErrors(ctx, cudaMemcpy(dCoeffs, hCoeffs, length * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));

    // Compute Fock superposition directly in dRegisterArrays
    int grid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelSetFocks<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dRegisterArrays[regIdx], cvDim, dx, dCoeffs, length);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    cudaFree(dCoeffs);
    delete[] hCoeffs;

    logDebug(ctx, "Register %d set to Fock superposition", regIdx);
}

void cvdvSetCoeffs(CVDVContext* ctx, int regIdx, double* coeffs, int length) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }

    logInfo(ctx, "Setting register %d to custom coefficients with %d terms", regIdx, length);

    size_t cvDim = 1 << ctx->gQbts[regIdx];

    if (length != cvDim) {
        logError(ctx, "Coefficient array length (%d) must match register dimension (%zu)", length, cvDim);
        return;
    }

    // Copy coefficients to device (interleaved: Re,Im,Re,Im,...)
    cuDoubleComplex* hCoeffs = new cuDoubleComplex[length];
    for (int i = 0; i < length; i++) {
        hCoeffs[i] = make_cuDoubleComplex(coeffs[2*i], coeffs[2*i+1]);
    }

    checkCudaErrors(ctx, cudaMemcpy(ctx->dRegisterArrays[regIdx], hCoeffs,
                               length * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    delete[] hCoeffs;

    logDebug(ctx, "Register %d set to custom coefficients", regIdx);
}

void cvdvSetCat(CVDVContext* ctx, int regIdx, double* data, int length) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }

    logInfo(ctx, "Setting register %d to cat state with %d coherent states", regIdx, length);

    size_t cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    if (fabs(dx) < 1e-15) {
        logError(ctx, "Cat state requires CV mode (dx > 0)");
        return;
    }

    // Parse data array: [alphaRe, alphaIm, coeffRe, coeffIm, ...] for each coherent state
    cuDoubleComplex* hAlphas = new cuDoubleComplex[length];
    cuDoubleComplex* hCoeffs = new cuDoubleComplex[length];

    for (int i = 0; i < length; i++) {
        hAlphas[i] = make_cuDoubleComplex(data[4*i], data[4*i+1]);
        hCoeffs[i] = make_cuDoubleComplex(data[4*i+2], data[4*i+3]);
    }

    // Copy to device
    cuDoubleComplex* dAlphas;
    cuDoubleComplex* dCoeffs;
    checkCudaErrors(ctx, cudaMalloc(&dAlphas, length * sizeof(cuDoubleComplex)));
    checkCudaErrors(ctx, cudaMalloc(&dCoeffs, length * sizeof(cuDoubleComplex)));
    checkCudaErrors(ctx, cudaMemcpy(dAlphas, hAlphas, length * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(ctx, cudaMemcpy(dCoeffs, hCoeffs, length * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));

    // Compute cat state (unnormalized)
    int grid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelSetCat<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dRegisterArrays[regIdx], cvDim, dx,
                                             dAlphas, dCoeffs, length);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Compute norm using reduction
    int numBlocks = min(grid, 1024);
    double* dPartialSums;
    checkCudaErrors(ctx, cudaMalloc(&dPartialSums, numBlocks * sizeof(double)));

    size_t sharedMemSize = CUDA_BLOCK_SIZE * sizeof(double);
    kernelComputeRegisterNorm<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemSize>>>(
        dPartialSums, ctx->dRegisterArrays[regIdx], cvDim);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Download partial sums and reduce on host
    double* hPartialSums = new double[numBlocks];
    checkCudaErrors(ctx, cudaMemcpy(hPartialSums, dPartialSums,
                               numBlocks * sizeof(double), cudaMemcpyDeviceToHost));

    double norm2 = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        norm2 += hPartialSums[i];
    }
    double norm = sqrt(norm2);

    logDebug(ctx, "Cat state norm before normalization: %.10f", norm);

    // Normalize the state
    if (norm > 1e-15) {
        double invNorm = 1.0 / norm;
        kernelNormalizeRegister<<<grid, CUDA_BLOCK_SIZE>>>(
            ctx->dRegisterArrays[regIdx], cvDim, invNorm);
        checkCudaErrors(ctx, cudaGetLastError());
        checkCudaErrors(ctx, cudaDeviceSynchronize());
    } else {
        logError(ctx, "Cat state has near-zero norm, cannot normalize");
    }

    // Cleanup
    delete[] hAlphas;
    delete[] hCoeffs;
    delete[] hPartialSums;
    cudaFree(dAlphas);
    cudaFree(dCoeffs);
    cudaFree(dPartialSums);

    logDebug(ctx, "Register %d set to normalized cat state", regIdx);
}

#pragma endregion

#pragma region C API - Fourier Transforms

void cvdvFtQ2P(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    logDebug(ctx, "ftQ2P called for register %d", regIdx);

    // Index-shifted QFT on specific register
    // Algorithm:
    // 1. Pre-phase correction: exp(i*pi*k*(N-1)/N)
    // 2. Standard FFT
    // 3. Post-phase correction: exp(i*pi*j*(N-1)/N)
    // 4. Normalization: 1/√N

    size_t totalSize = 1 << ctx->gTotalQbt;
    int grid = (totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    size_t regDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    // Step 1: Pre-phase correction: exp(i*π(N-1)/N * j)
    // In position representation: exp(i*π(N-1)/(N*dx) * x)
    double phaseCoeff = PI * (regDim - 1.0) / (regDim * dx);
    logDebug(ctx, "Applying pre-phase correction: phaseCoeff=%.6f", phaseCoeff);
    kernelApplyOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize,
                                          regIdx,
                                          ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Step 2: Forward FFT using cuFFTPlanMany
    // New layout: last register is contiguous (stride 1), first register has largest stride
    // Stride for regIdx = product of dimensions after regIdx = 2^(sum of qubits after regIdx)
    size_t regStride = 1 << ctx->gFlwQbts[regIdx];
    
    // Number of complete FFTs to perform
    size_t numFFTs = (1 << ctx->gTotalQbt) / regDim;
    
    cufftHandle plan;
    int n = regDim;
    
    if (regStride == 1) {
        // Contiguous case: FFT dimension is contiguous, batch FFTs together
        int batch = numFFTs;
        logDebug(ctx, "Contiguous FFT: n=%d, batch=%d", n, batch);
        cufftResult result = cufftPlan1d(&plan, n, CUFFT_Z2Z, batch);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        
        logDebug(ctx, "Executing FFT...");
        result = cufftExecZ2Z(plan, ctx->dState, ctx->dState, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT forward execution failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        logDebug(ctx, "FFT completed successfully");
    } else {
        // Strided case: FFT dimension is strided  
        // Layout: elements of one FFT are at stride regStride apart
        // Consecutive FFTs start at distance 1 (consecutive in inner dimension)
        int iStride = regStride, oStride = regStride;
        int iDist = 1, oDist = 1;
        int batch = regStride;  
        int inembed[1] = {n * (int)regStride};  // Logical size of input array
        int onembed[1] = {n * (int)regStride};  // Logical size of output array
        
        logDebug(ctx, "FFT strided case: regIdx=%d, n=%d, stride=%zu, batch=%d, nembed=%d", 
                 regIdx, n, regStride, batch, inembed[0]);
        
        cufftResult result = cufftPlanMany(&plan, 1, &n, inembed, iStride, iDist,
                                            onembed, oStride, oDist, CUFFT_Z2Z, batch);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        
        // Number of outer blocks to process
        size_t regBlockSize = regStride << ctx->gQbts[regIdx];
        size_t outerDim = totalSize / regBlockSize;
        logDebug(ctx, "Processing %zu outer blocks", outerDim);
        
        // Process each outer block
        for (size_t o = 0; o < outerDim; o++) {
            // Start of this outer block
            cuDoubleComplex* blockStart = ctx->dState + (o << (ctx->gFlwQbts[regIdx] + ctx->gQbts[regIdx]));
            result = cufftExecZ2Z(plan, blockStart, blockStart, CUFFT_FORWARD);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT forward execution failed: %d at block %zu\n", result, o);
                exit(EXIT_FAILURE);
            }
        }
        logDebug(ctx, "Strided FFT completed successfully");
    }
    
    checkCudaErrors(ctx, cudaDeviceSynchronize());
    cufftDestroy(plan);

    // Step 3: Post-phase correction: exp(i*π(N-1)/N * k)
    // In momentum representation: exp(i*π(N-1)/(N*dx) * p)
    logDebug(ctx, "Applying post-phase correction: phaseCoeff=%.6f", phaseCoeff);
    kernelApplyOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize,
                                          regIdx,
                                          ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Step 4: Normalization (1/√N for unitary transform)
    double norm = 1.0 / sqrt((double)regDim);
    logDebug(ctx, "Applying normalization: norm=%.6f", norm);
    kernelApplyScalarRegister<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize,
                                                   regIdx,
                                                   ctx->gQbts, norm);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    logDebug(ctx, "ftQ2P completed for register %d", regIdx);
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
    kernelApplyOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                          regIdx,
                                          ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Step 2: Inverse FFT using cuFFTPlanMany
    // New layout: last register is contiguous (stride 1), first register has largest stride
    // Stride for regIdx = product of dimensions after regIdx = 2^(sum of qubits after regIdx)
    size_t regStride = 1 << ctx->gFlwQbts[regIdx];
    
    // Number of complete FFTs to perform
    size_t numFFTs = (1 << ctx->gTotalQbt) / regDim;
    
    cufftHandle plan;
    int n = regDim;
    
    if (regStride == 1) {
        // Contiguous case: FFT dimension is contiguous, batch FFTs together
        int batch = numFFTs;
        cufftResult result = cufftPlan1d(&plan, n, CUFFT_Z2Z, batch);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        
        result = cufftExecZ2Z(plan, ctx->dState, ctx->dState, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT inverse execution failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
    } else {
        // Strided case: FFT dimension is strided
        int iStride = regStride, oStride = regStride;
        int iDist = 1, oDist = 1;
        int batch = regStride;
        int inembed[1] = {n * (int)regStride};
        int onembed[1] = {n * (int)regStride};
        
        cufftResult result = cufftPlanMany(&plan, 1, &n, inembed, iStride, iDist,
                                            onembed, oStride, oDist, CUFFT_Z2Z, batch);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        
        // Number of outer blocks
        size_t regBlockSize = regStride << ctx->gQbts[regIdx];
        size_t outerDim = (1 << ctx->gTotalQbt) / regBlockSize;
        for (size_t o = 0; o < outerDim; o++) {
            cuDoubleComplex* blockStart = ctx->dState + (o << (ctx->gFlwQbts[regIdx] + ctx->gQbts[regIdx]));
            result = cufftExecZ2Z(plan, blockStart, blockStart, CUFFT_INVERSE);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT inverse execution failed: %d at block %zu\n", result, o);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    checkCudaErrors(ctx, cudaDeviceSynchronize());
    cufftDestroy(plan);

    // Step 3: Post-phase correction (negative phase): exp(-i*π(N-1)/N * k)
    // In position representation: exp(-i*π(N-1)/(N*dx) * x)
    kernelApplyOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                          regIdx,
                                          ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Step 4: Normalization (1/√N for unitary transform)
    double norm = 1.0 / sqrt((double)regDim);
    kernelApplyScalarRegister<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                   regIdx,
                                                   ctx->gQbts, norm);
    checkCudaErrors(ctx, cudaDeviceSynchronize());
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
        kernelApplyOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                       regIdx,
                                                       ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                                       SQRT2 * betaIm);
        checkCudaErrors(ctx, cudaGetLastError());
        checkCudaErrors(ctx, cudaDeviceSynchronize());
    }

    // Step 2: Apply D(Re(α)) = exp(-i*sqrt(2)*Re(α)*p) in momentum space
    if (fabs(betaRe) > 1e-12) {
        // Transform register to momentum space
        cvdvFtQ2P(ctx, regIdx);

        // Apply phase in momentum space
        kernelApplyOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                       regIdx,
                                                       ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                                       -SQRT2 * betaRe);
        checkCudaErrors(ctx, cudaGetLastError());
        checkCudaErrors(ctx, cudaDeviceSynchronize());

        // Transform back to position space
        cvdvFtP2Q(ctx, regIdx);
    }

    // Note: Global phase exp(-i*Im(α)*Re(α)) is ignored
}

void cvdvConditionalDisplacement(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double alphaRe, double alphaIm) {
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
    // CD(q0) = F^{-1} exp(-i*sqrt(2)*q0*Z*p) F - controlled phase in momentum space
    // where Z = |0⟩⟨0| - |1⟩⟨1|, so |0⟩ gets +α and |1⟩ gets -α

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    // Step 1: Apply CD(i*Im(α)) = exp(i√2 Im(α) Z q) in position space
    if (fabs(alphaIm) > 1e-12) {
        kernelApplyConditionalOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                targetReg, ctrlReg, ctrlQubit,
                                                ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                                SQRT2 * alphaIm);
        checkCudaErrors(ctx, cudaGetLastError());
        checkCudaErrors(ctx, cudaDeviceSynchronize());
    }

    // Step 2: Apply CD(Re(α)) = F^{-1} exp(-i√2 Re(α) Z p) F
    if (fabs(alphaRe) > 1e-12) {
        // Transform target register to momentum space
        cvdvFtQ2P(ctx, targetReg);

        // Apply exp(-i√2 Re(α) Z p) in momentum space
        kernelApplyConditionalOneModeQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                targetReg, ctrlReg, ctrlQubit,
                                                ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                                -SQRT2 * alphaRe);
        checkCudaErrors(ctx, cudaGetLastError());
        checkCudaErrors(ctx, cudaDeviceSynchronize());

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

    kernelPauliRotation<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                      regIdx, qubitIdx,
                                                      ctx->gQbts, ctx->gFlwQbts,
                                                      axis, theta);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
}

void cvdvHadamard(CVDVContext* ctx, int regIdx, int qubitIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) / 2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelHadamard<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, qubitIdx, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
}

void cvdvParity(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelParity<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                            regIdx, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
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
                                            targetReg, ctrlReg, ctrlQubit,
                                            ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
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
    kernelSwapRegisters<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                    reg1, reg2,
                                                    ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
}

void cvdvPhaseSquare(CVDVContext* ctx, int regIdx, double t) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelApplyOneModeQ2<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, t);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
}

void cvdvPhaseCubic(CVDVContext* ctx, int regIdx, double t) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelApplyOneModeQ3<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, t);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
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

    // Find nearest multiple of π/2
    double theta0 = round(theta / (PI / 2.0)) * (PI / 2.0);
    double remainder = theta - theta0;

    // Apply R(θ₀) for the integer-multiple part
    // theta0 / (π/2) gives the number of quarter-turns
    int quarterTurns = (int)round(theta0 / (PI / 2.0));
    // Normalize to [0,4) since R(2π) = identity
    quarterTurns = ((quarterTurns % 4) + 4) % 4;

    switch (quarterTurns) {
        case 0: break;  // identity
        case 1: cvdvFtQ2P(ctx, regIdx); break;          // R(π/2) = FT
        case 2: cvdvParity(ctx, regIdx); break;          // R(π) = Parity
        case 3: cvdvFtP2Q(ctx, regIdx); break;           // R(-π/2) = R(3π/2) = FT†
    }

    // Apply small-angle remainder
    cvdvRotationSmall(ctx, regIdx, remainder);
}

// Internal helper: apply controlled phase square exp(i*t*Z*q^2)
static void cvdvControlledPhaseSquare(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double t) {
    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernalApplyConditionalOneModeQ2<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                                targetReg, ctrlReg, ctrlQubit,
                                                ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts, t);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
}

// Internal: small-angle conditional rotation |θ| ≤ π/4
static void cvdvConditionalRotationSmall(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double theta) {
    if (fabs(theta) < 1e-15) return;

    // CR(θ) = exp(-i/2 Z tan(θ/2) q^2) exp(-i/2 Z sin(θ) p^2) exp(-i/2 Z tan(θ/2) q^2)
    double tanHalfTheta = tan(theta / 2.0);
    double sinTheta = sin(theta);

    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * tanHalfTheta);
    cvdvFtQ2P(ctx, targetReg);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * sinTheta);
    cvdvFtP2Q(ctx, targetReg);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * tanHalfTheta);
}

void cvdvConditionalRotation(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double theta) {
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

    // Find nearest multiple of π/2
    double theta0 = round(theta / (PI / 2.0)) * (PI / 2.0);
    double remainder = theta - theta0;

    // Apply R(θ₀) for the integer-multiple part
    // theta0 / (π/2) gives the number of quarter-turns
    int quarterTurns = (int)round(theta0 / (PI / 2.0));
    // Normalize to [0,4) since R(2π) = identity
    quarterTurns = ((quarterTurns % 4) + 4) % 4;

    switch (quarterTurns) {
        case 0: break;  // identity
        case 1: cvdvFtQ2P(ctx, targetReg); cvdvConditionalParity(ctx, targetReg, ctrlReg, ctrlQubit); cvdvPauliRotation(ctx, ctrlReg, ctrlQubit, 2, PI/2); break;
        case 2: cvdvParity(ctx, targetReg); cvdvPauliRotation(ctx, ctrlReg, ctrlQubit, 2, PI); break;
        case 3: cvdvFtP2Q(ctx, targetReg); cvdvConditionalParity(ctx, targetReg, ctrlReg, ctrlQubit); cvdvPauliRotation(ctx, ctrlReg, ctrlQubit, 2, -PI/2); break;
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

    // S(r) = exp(i(e^{-r}-1)/(2te^r) p^2) exp(-i(te^r)/2 q^2) exp(i(1-e^{-r})/(2t) p^2) exp(i(t)/2 q^2)
    // where t = e^{-r/2} * sqrt(|1-e^{-r}|) minimizes sum of coefficients
    double expR = exp(r);
    double expMinusR = exp(-r);
    double t = exp(-r / 2.0) * sqrt(fabs(1.0 - expMinusR));

    // First: exp(i(t)/2 q^2) in position space
    cvdvPhaseSquare(ctx, regIdx, 0.5 * t);

    // Second: exp(i(1-e^{-r})/(2t) p^2) in momentum space
    cvdvFtQ2P(ctx, regIdx);
    cvdvPhaseSquare(ctx, regIdx, (1.0 - expMinusR) / (2.0 * t));
    cvdvFtP2Q(ctx, regIdx);

    // Third: exp(-i(te^r)/2 q^2) in position space
    cvdvPhaseSquare(ctx, regIdx, -0.5 * t * expR);

    // Fourth: exp(i(e^{-r}-1)/(2te^r) p^2) in momentum space
    cvdvFtQ2P(ctx, regIdx);
    cvdvPhaseSquare(ctx, regIdx, (expMinusR - 1.0) / (2.0 * t * expR));
    cvdvFtP2Q(ctx, regIdx);
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

    // CS(r) = conditional version of S(r), replacing cvdvPhaseSquare with cvdvControlledPhaseSquare
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
    double coeff_p = -sin(0.5  * theta);

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelApplyTwoModeQQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                        reg1, reg2,
                                        ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                        coeff_q);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    cvdvFtQ2P(ctx, reg1);
    cvdvFtQ2P(ctx, reg2);

    kernelApplyTwoModeQQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                        reg1, reg2,
                                        ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                        coeff_p);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    cvdvFtP2Q(ctx, reg1);
    cvdvFtP2Q(ctx, reg2);

    kernelApplyTwoModeQQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                        reg1, reg2,
                                        ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                        coeff_q);
    checkCudaErrors(ctx, cudaDeviceSynchronize());
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
    // Normalize to [0,4) since BS(4π) = identity (BS(2π) = Par₁Par₂, applied twice = id)
    halfTurns = ((halfTurns % 4) + 4) % 4;

    switch (halfTurns) {
        case 0: break;  // identity
        case 1:  // BS(π) = FT₁ FT₂ SWAP
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
static void cvdvConditionalBeamSplitterSmall(CVDVContext* ctx, int reg1, int reg2, int ctrlReg, int ctrlQubit, double theta) {
    if (fabs(theta) < 1e-15) return;

    double coeff_q = -tan(0.25 * theta);
    double coeff_p = -sin(0.5  * theta);

    int grid = ((1 << ctx->gTotalQbt) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelApplyConditionalTwoModeQQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                        reg1, reg2,
                                        ctrlReg, ctrlQubit,
                                        ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                        coeff_q);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    cvdvFtQ2P(ctx, reg1);
    cvdvFtQ2P(ctx, reg2);

    kernelApplyConditionalTwoModeQQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                        reg1, reg2,
                                        ctrlReg, ctrlQubit,
                                        ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                        coeff_p);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    cvdvFtP2Q(ctx, reg1);
    cvdvFtP2Q(ctx, reg2);

    kernelApplyConditionalTwoModeQQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                        reg1, reg2,
                                        ctrlReg, ctrlQubit,
                                        ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                        coeff_q);
    checkCudaErrors(ctx, cudaDeviceSynchronize());
}

void cvdvConditionalBeamSplitter(CVDVContext* ctx, int reg1, int reg2, int ctrlReg, int ctrlQubit, double theta) {
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
    // BS(θ₀) applied unconditionally (correct for even half-turns; odd half-turns are approximate)
    double theta0 = round(theta / PI) * PI;
    double remainder = theta - theta0;

    int halfTurns = (int)round(theta0 / PI);
    halfTurns = ((halfTurns % 4) + 4) % 4;

    switch (halfTurns) {
        case 0: break;
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
    
    kernelApplyTwoModeQQ<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt),
                                        reg1, reg2,
                                        ctx->gQbts, ctx->gGridSteps, ctx->gFlwQbts,
                                        coeff);
    checkCudaErrors(ctx, cudaDeviceSynchronize());
}

#pragma endregion

#pragma region C API - State Access

void cvdvGetWignerSingleSlice(CVDVContext* ctx, int regIdx, int* sliceIndices, double* wignerOut,
                                   int wignerN, double wXMax, double wPMax) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Get register dimension and dx from managed memory (direct CPU access)
    size_t cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    // Copy slice indices to device
    int* dSliceIndices;
    checkCudaErrors(ctx, cudaMalloc(&dSliceIndices, ctx->gNumReg * sizeof(int)));
    checkCudaErrors(ctx, cudaMemcpy(dSliceIndices, sliceIndices, ctx->gNumReg * sizeof(int),
                               cudaMemcpyHostToDevice));

    double* dWigner;
    checkCudaErrors(ctx, cudaMalloc(&dWigner, wignerN * wignerN * sizeof(double)));

    int grid = (wignerN * wignerN + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelComputeWignerSingleSlice<<<grid, CUDA_BLOCK_SIZE>>>(dWigner, ctx->dState, regIdx, dSliceIndices,
                                                         cvDim, dx, wignerN,
                                                         wXMax, wPMax, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    checkCudaErrors(ctx, cudaMemcpy(wignerOut, dWigner, wignerN * wignerN * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(dWigner);
    cudaFree(dSliceIndices);
}

void cvdvGetWignerFullMode(CVDVContext* ctx, int regIdx, double* wignerOut,
                                int wignerN, double wXMax, double wPMax) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Get register dimension and dx from managed memory (direct CPU access)
    size_t cvDim = 1 << ctx->gQbts[regIdx];
    double dx = ctx->gGridSteps[regIdx];

    double* dWigner;
    checkCudaErrors(ctx, cudaMalloc(&dWigner, wignerN * wignerN * sizeof(double)));

    int totalPoints = wignerN * wignerN;
    int grid = (totalPoints + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelComputeWignerFullMode<<<grid, CUDA_BLOCK_SIZE>>>(dWigner, ctx->dState, regIdx,
                                                      cvDim, dx, wignerN,
                                                      wXMax, wPMax, ctx->gNumReg, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    checkCudaErrors(ctx, cudaMemcpy(wignerOut, dWigner, wignerN * wignerN * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(dWigner);
}

void cvdvGetHusimiQFullMode(CVDVContext* ctx, int regIdx, double* outHusimiQ, int qN, double qMax) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Get register dimension and dx from managed memory (direct CPU access)
    double dx = ctx->gGridSteps[regIdx];

    double* dHusimiQ;
    checkCudaErrors(ctx, cudaMalloc(&dHusimiQ, qN * qN * sizeof(double)));

    int totalPoints = qN * qN;
    int grid = (totalPoints + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelComputeHusimiQFullMode<<<grid, CUDA_BLOCK_SIZE>>>(dHusimiQ, ctx->dState, regIdx, dx, qN,
                                                   qMax, ctx->gNumReg, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    checkCudaErrors(ctx, cudaMemcpy(outHusimiQ, dHusimiQ, qN * qN * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(dHusimiQ);
}

void cvdvJointMeasure(CVDVContext* ctx, int reg1Idx, int reg2Idx, double* jointProbsOut) {
    if (!ctx) return;
    if (reg1Idx < 0 || reg1Idx >= ctx->gNumReg || reg2Idx < 0 || reg2Idx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register indices: %d, %d\n", reg1Idx, reg2Idx);
        return;
    }
    if (reg1Idx == reg2Idx) {
        fprintf(stderr, "Register indices must be different for joint measurement\n");
        return;
    }

    // Get register dimensions from managed memory (direct CPU access)
    size_t totalPairs = 1 << (ctx->gQbts[reg1Idx] + ctx->gQbts[reg2Idx]);

    // Allocate device memory for joint probabilities
    double* dJointProb;
    checkCudaErrors(ctx, cudaMalloc(&dJointProb, totalPairs * sizeof(double)));

    // Launch kernel
    int grid = min((int)((totalPairs + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE), CUDA_BLOCK_SIZE);

    kernelComputeJointMeasure<<<grid, CUDA_BLOCK_SIZE>>>(dJointProb, ctx->dState,
                                                reg1Idx, reg2Idx,
                                                (1 << ctx->gTotalQbt), ctx->gNumReg,
                                                ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Copy result to CPU
    checkCudaErrors(ctx, cudaMemcpy(jointProbsOut, dJointProb, totalPairs * sizeof(double),
                               cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(dJointProb);
}

void cvdvGetState(CVDVContext* ctx, double* realOut, double* imagOut) {
    if (!ctx) return;
    size_t totalSize = 1 << ctx->gTotalQbt;
    if (totalSize == 0) {
        fprintf(stderr, "Error: State not initialized. Call cvdvInitStateVector first.\n");
        return;
    }

    cuDoubleComplex* hState = new cuDoubleComplex[totalSize];
    checkCudaErrors(ctx, cudaMemcpy(hState, ctx->dState, totalSize * sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < totalSize; i++) {
        realOut[i] = cuCreal(hState[i]);
        imagOut[i] = cuCimag(hState[i]);
    }

    delete[] hState;
}

// CUDA kernel to compute marginal probabilities for a register
// Sums |amplitude|^2 over all other registers
__global__ void kernelComputeRegisterProbabilities(double* probabilities, const cuDoubleComplex* state,
                                                    int regIdx, int numReg,
                                                    const int* qbts, const int* flwQbts) {
    // Each thread computes probability for one basis state of the target register
    size_t regLocalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t regDim = 1 << qbts[regIdx];
    
    if (regLocalIdx >= regDim) return;
    
    // Compute total elements in all other registers = 2^(total qubits - this register's qubits)
    int totalQubits = qbts[0] + flwQbts[0];
    size_t otherSize = 1 << (totalQubits - qbts[regIdx]);
    
    // Sum |amplitude|^2 over all indices where target register = regLocalIdx
    double prob = 0.0;
    
    // Iterate over all possible values of other registers
    for (size_t otherIdx = 0; otherIdx < otherSize; otherIdx++) {
        // Reconstruct global index from regLocalIdx and otherIdx
        // We need to insert regLocalIdx at the correct position
        
        size_t globalIdx = 0;
        size_t tempOtherIdx = otherIdx;
        size_t currentStride = 1;
        
        // Build global index from right to left (last register varies fastest)
        for (int r = numReg - 1; r >= 0; r--) {
            size_t localIdx;
            if (r == regIdx) {
                localIdx = regLocalIdx;
            } else {
                localIdx = tempOtherIdx % (1 << qbts[r]);
                tempOtherIdx /= (1 << qbts[r]);
            }
            globalIdx += localIdx << flwQbts[r];
            currentStride <<= qbts[r];
        }
        
        cuDoubleComplex amp = state[globalIdx];
        prob += cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
    }
    
    probabilities[regLocalIdx] = prob;
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

    checkCudaErrors(ctx, cudaMemcpy(qubitCountsOut, ctx->gQbts,
                               ctx->gNumReg * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(ctx, cudaMemcpy(gridStepsOut, ctx->gGridSteps,
                               ctx->gNumReg * sizeof(double), cudaMemcpyDeviceToHost));
}

int cvdvGetRegisterDim(CVDVContext* ctx, int regIdx) {
    if (!ctx) return -1;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return -1;

    int qubit_count;
    checkCudaErrors(ctx, cudaMemcpy(&qubit_count, ctx->gQbts + regIdx,
                               sizeof(int), cudaMemcpyDeviceToHost));
    return 1 << qubit_count;
}

double cvdvGetRegisterDx(CVDVContext* ctx, int regIdx) {
    if (!ctx) return -1.0;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return -1.0;

    double dx;
    checkCudaErrors(ctx, cudaMemcpy(&dx, ctx->gGridSteps + regIdx,
                               sizeof(double), cudaMemcpyDeviceToHost));
    return dx;
}

void cvdvMeasure(CVDVContext* ctx, int regIdx, double* probabilitiesOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        logError(ctx, "Invalid register index: %d", regIdx);
        return;
    }
    if (ctx->dState == nullptr) {
        logError(ctx, "State not initialized");
        return;
    }

    logDebug(ctx, "Computing marginal probabilities for register %d", regIdx);

    size_t regDim = 1 << ctx->gQbts[regIdx];
    
    // Allocate device memory for probabilities
    double* dProbs;
    checkCudaErrors(ctx, cudaMalloc(&dProbs, regDim * sizeof(double)));
    
    // Launch kernel to compute probabilities
    int grid = (regDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelComputeRegisterProbabilities<<<grid, CUDA_BLOCK_SIZE>>>(dProbs, ctx->dState, regIdx, ctx->gNumReg,
                                                         ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());
    
    // Copy results back to host
    checkCudaErrors(ctx, cudaMemcpy(probabilitiesOut, dProbs, regDim * sizeof(double), cudaMemcpyDeviceToHost));
}

void cvdvInnerProduct(CVDVContext* ctx, double* realOut, double* imagOut) {
    if (!ctx) return;
    if (ctx->dState == nullptr || ctx->dRegisterArrays == nullptr || ctx->gNumReg == 0) {
        logError(ctx, "State or register arrays not initialized");
        *realOut = 0.0;
        *imagOut = 0.0;
        return;
    }

    logInfo(ctx, "Computing inner product between state and register tensor product");

    // Prepare device array of register array pointers
    cuDoubleComplex** dRegArrayPtrs;
    checkCudaErrors(ctx, cudaMalloc(&dRegArrayPtrs, ctx->gNumReg * sizeof(cuDoubleComplex*)));
    checkCudaErrors(ctx, cudaMemcpy(dRegArrayPtrs, ctx->dRegisterArrays, 
                               ctx->gNumReg * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));

    // Allocate partial sums for reduction
    size_t totalSize = 1 << ctx->gTotalQbt;
    int numBlocks = (totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);  // Cap at 1024 blocks
    
    cuDoubleComplex* dPartialSums;
    checkCudaErrors(ctx, cudaMalloc(&dPartialSums, numBlocks * sizeof(cuDoubleComplex)));

    // Launch kernel with shared memory for reduction
    size_t sharedMemSize = CUDA_BLOCK_SIZE * sizeof(cuDoubleComplex);
    kernelComputeInnerProduct<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemSize>>>(
        dPartialSums, ctx->dState, dRegArrayPtrs, ctx->gNumReg, ctx->gQbts, ctx->gFlwQbts, totalSize);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Download partial sums and reduce on host
    cuDoubleComplex* hPartialSums = new cuDoubleComplex[numBlocks];
    checkCudaErrors(ctx, cudaMemcpy(hPartialSums, dPartialSums, 
                               numBlocks * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < numBlocks; i++) {
        result = cuCadd(result, hPartialSums[i]);
    }

    *realOut = cuCreal(result);
    *imagOut = cuCimag(result);

    // Cleanup
    delete[] hPartialSums;
    checkCudaErrors(ctx, cudaFree(dPartialSums));
    checkCudaErrors(ctx, cudaFree(dRegArrayPtrs));

    logInfo(ctx, "Inner product: (%.10f, %.10f)", *realOut, *imagOut);
}

double cvdvGetNorm(CVDVContext* ctx) {
    if (!ctx) return 0.0;
    if (ctx->dState == nullptr) {
        logError(ctx, "State not initialized");
        return 0.0;
    }

    logInfo(ctx, "Computing state norm");

    // Allocate partial sums for reduction
    size_t totalSize = 1 << ctx->gTotalQbt;
    int numBlocks = (totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    numBlocks = min(numBlocks, 1024);  // Cap at 1024 blocks

    double* dPartialSums;
    checkCudaErrors(ctx, cudaMalloc(&dPartialSums, numBlocks * sizeof(double)));

    // Launch kernel with shared memory for reduction
    size_t sharedMemSize = CUDA_BLOCK_SIZE * sizeof(double);
    kernelComputeNorm<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemSize>>>(
        dPartialSums, ctx->dState, totalSize);
    checkCudaErrors(ctx, cudaGetLastError());
    checkCudaErrors(ctx, cudaDeviceSynchronize());

    // Download partial sums and reduce on host
    double* hPartialSums = new double[numBlocks];
    checkCudaErrors(ctx, cudaMemcpy(hPartialSums, dPartialSums,
                               numBlocks * sizeof(double), cudaMemcpyDeviceToHost));

    double result = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        result += hPartialSums[i];
    }

    // Cleanup
    delete[] hPartialSums;
    checkCudaErrors(ctx, cudaFree(dPartialSums));

    logInfo(ctx, "Norm: %.10f", result);
    return result;
}

#pragma endregion

}  // extern "C"
