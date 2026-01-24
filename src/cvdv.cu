// cvdv.cu - CUDA library for hybrid CV-DV quantum simulation

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <ctime>

#pragma region Error Checking and Constants

static FILE* gLogFile = nullptr;

static void initLogFile() {
    if (!gLogFile) {
        gLogFile = fopen("cuda.log", "a");
        if (gLogFile) {
            time_t now = time(nullptr);
            fprintf(gLogFile, "\n=== CUDA Session Started: %s ===", ctime(&now));
            fflush(gLogFile);
        }
    }
}

static void clearLogFile() {
    if (gLogFile) {
        fclose(gLogFile);
        gLogFile = nullptr;
    }
    gLogFile = fopen("cuda.log", "w");
    if (gLogFile) {
        time_t now = time(nullptr);
        fprintf(gLogFile, "=== CUDA Session Started: %s ===\n", ctime(&now));
        fflush(gLogFile);
    }
}

#define logInfo(...) do { \
    initLogFile(); \
    if (gLogFile) { \
        fprintf(gLogFile, "[INFO] "); \
        fprintf(gLogFile, __VA_ARGS__); \
        fprintf(gLogFile, "\n"); \
        fflush(gLogFile); \
    } \
} while(0)

#define logDebug(...) do { \
    initLogFile(); \
    if (gLogFile) { \
        fprintf(gLogFile, "[DEBUG] "); \
        fprintf(gLogFile, __VA_ARGS__); \
        fprintf(gLogFile, "\n"); \
        fflush(gLogFile); \
    } \
} while(0)

#define logError(...) do { \
    initLogFile(); \
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

#define checkCudaErrors(val) do { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        initLogFile(); \
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

#pragma endregion

#pragma region Global State

static cuDoubleComplex* dState = nullptr;

// New register-based model with managed memory
static int* gQubitCounts = nullptr;         // Managed memory: number of qubits in each register
static size_t* gFollowQubitCounts = nullptr; // Managed memory: cumulative qubits after each register
static double* gGridSteps = nullptr;        // Managed memory: grid step (dx) for each register
static size_t* gRegisterDims = nullptr;     // Managed memory: dimension (2^numQubits) for each register
static int gNumReg = 0;                     // Total number of registers
static size_t gTotalSize = 0;               // Total state vector size

// Temporary storage for register initialization and inner products
static cuDoubleComplex** dRegisterArrays = nullptr;  // Device arrays for each register's coefficients

#pragma endregion

#pragma region Device Helper Functions

// Position value at grid index: x_i = (i - cvDim/2) * dx
__device__ __host__ inline double gridX(int idx, int cvDim, double dx) {
    return (idx - cvDim / 2) * dx;
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

/**
 * Memory Layout: LAST REGISTER VARIES FASTEST (contiguous)
 * 
 * For registers [R0, R1, R2, ..., R_{n-1}] with dimensions [D0, D1, D2, ..., D_{n-1}]:
 * globalIdx = localIdx_R0 * (D1*D2*...*D_{n-1}) + localIdx_R1 * (D2*...*D_{n-1}) + ... + localIdx_R_{n-1}
 * 
 * Register strides:
 * - R_{n-1} (last):  stride = 1 (contiguous, optimal for FFT)
 * - R_{n-2}:         stride = D_{n-1}
 * - R_i:             stride = D_{i+1} * D_{i+2} * ... * D_{n-1}
 * - R_0 (first):     stride = D1 * D2 * ... * D_{n-1} (slowest varying)
 */

// Get dimension of a register
__device__ __host__ inline size_t getRegisterDim(int regIdx, const int* qubitCounts) {
    return 1 << qubitCounts[regIdx];
}

// Compute stride for a register (product of dimensions of all registers AFTER it)
// Last register has stride 1 (varies fastest, contiguous in memory)
// Simplified using precomputed following qubit counts
__device__ __host__ inline size_t getRegisterStride(int regIdx, const size_t* followQubitCounts) {
    return 1 << followQubitCounts[regIdx];
}

// Extract local index within a register from global index
// Returns the index for register regIdx given the global flat index
__device__ __host__ inline size_t getLocalIndex(size_t globalIdx, int followingQubitCount, int regQubitCount) {
    return (globalIdx >> followingQubitCount) & ((1 << regQubitCount) - 1);
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
    double gauss = exp(-(x - q) * (x - q) / 2.0);
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

    // Fock state |n>: psi_n(x) = pi^(-1/4) / sqrt(2^n n!) * H_n(x) * exp(-x^2/2)
    // Hermite polynomial via recurrence: H_0 = 1, H_1 = 2x, H_{n+1} = 2x H_n - 2n H_{n-1}
    double hCurr = 1.0;
    double factorial = 1.0;

    if (n >= 1) {
        hCurr = 2.0 * x;
        factorial = 1.0;
        
        double hPrev = 1.0;
        for (int k = 1; k < n; k++) {
            double hNext = 2.0 * x * hCurr - 2.0 * k * hPrev;
            hPrev = hCurr;
            hCurr = hNext;
            factorial *= (k + 1);
        }
    }

    // Normalization: pi^(-1/4) / sqrt(2^n * n!)
    // Use bit shift for 2^n: (1 << n) = 2^n
    double norm = PI_POW_NEG_QUARTER / sqrt((1 << n) * factorial);

    double val = norm * hCurr * exp(-x * x / 2.0) * sqrt(dx); // Normalize as qubit register
    state[idx] = make_cuDoubleComplex(val, 0.0);
}

__global__ void kernelSetFocks(cuDoubleComplex* state, int cvDim, double dx,
                                  const cuDoubleComplex* coeffs, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cvDim) return;
    double x = gridX(idx, cvDim, dx);
    
    // Pre-compute exponential term
    double expTerm = exp(-x * x / 2.0) * sqrt(dx);
    
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    
    // Cache for Hermite recurrence
    double hLast2 = 1.0;  // H_0
    double hLast = 2.0 * x;  // H_1
    
    double factorial = 1.0;
    for (int n = 0; n < length; n++) {
        if (n > 0) factorial *= n;
        
        // Get current Hermite polynomial
        double hCurrent;
        if (n == 0) {
            hCurrent = hLast2;
        } else if (n == 1) {
            hCurrent = hLast;
        } else {
            // H_n = 2x * H_{n-1} - 2(n-1) * H_{n-2}
            hCurrent = 2.0 * x * hLast - 2.0 * (n-1) * hLast2;
            hLast2 = hLast;
            hLast = hCurrent;
        }
        
        // Normalization: pi^(-1/4) / sqrt(2^n * n!)
        double norm = PI_POW_NEG_QUARTER / sqrt((1 << n) * factorial);
        
        // Fock state wavefunction
        double psiN = norm * hCurrent * expTerm;
        
        // Add coefficient * |n>
        cuDoubleComplex term = cuCmul(coeffs[n], make_cuDoubleComplex(psiN, 0.0));
        result = cuCadd(result, term);
    }
    state[idx] = result;
}

#pragma endregion

#pragma region Gate Kernels

// Apply phase factor to a specific register: exp(i*phaseCoeff*x)
__global__ void kernelApplyOneModeQ(cuDoubleComplex* state, size_t totalSize,
                                             int regIdx,
                                             const int* qubitCounts, const double* gridSteps,
                                             const size_t* followQubitCounts,
                                             double phaseCoeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = getRegisterDim(regIdx, qubitCounts);
    double dx = gridSteps[regIdx];
    size_t localIdx = getLocalIndex(idx, followQubitCounts[regIdx], qubitCounts[regIdx]);

    double x = gridX(localIdx, regDim, dx);
    state[idx] = cmulPhase(state[idx], phaseCoeff * x);
}

// Apply controlled phase to a specific register with control from another register
// exp(i*phaseCoeff*Z*x) where Z acts on ctrlQubit in ctrlReg
__global__ void kernelApplyControlledQ(cuDoubleComplex* state, size_t totalSize,
                                                         int targetReg, int ctrlReg, int ctrlQubit,
                                                         const int* qubitCounts, const double* gridSteps,
                                                         const size_t* followQubitCounts,
                                                         double phaseCoeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t target_dim = getRegisterDim(targetReg, qubitCounts);
    double dx = gridSteps[targetReg];
    size_t targetLocalIdx = getLocalIndex(idx, followQubitCounts[targetReg], qubitCounts[targetReg]);

    // Extract control qubit state
    size_t ctrlLocalIdx = getLocalIndex(idx, followQubitCounts[ctrlReg], qubitCounts[ctrlReg]);
    int ctrlMask = 1 << (qubitCounts[ctrlReg] - 1 - ctrlQubit);

    double x = gridX(targetLocalIdx, target_dim, dx);
    double phase = phaseCoeff * x;

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
                                                const int* qubitCounts, const size_t* followQubitCounts,
                                                int axis, double theta) {
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= totalSize / 2) return;

    size_t regDim = getRegisterDim(regIdx, qubitCounts);
    size_t regStride = getRegisterStride(regIdx, followQubitCounts);

    // Target qubit mask within the register's local index space
    // Qubit 0 is the most significant bit
    size_t targetBit = 1 << (qubitCounts[regIdx] - 1 - qubitIdx);

    // Decompose pairIdx into compressed space
    size_t regDimCompressed = regDim / 2;
    size_t regBlockSize = regStride * regDim;
    size_t regBlockSizeCompressed = regStride * regDimCompressed;
    
    size_t outerIdx = pairIdx / regBlockSizeCompressed;
    size_t innerCompressed = pairIdx % regBlockSizeCompressed;
    
    size_t localCompressed = innerCompressed / regStride;
    size_t strideOffset = innerCompressed % regStride;
    
    // Expand localCompressed by inserting the target bit
    size_t lowerBits = localCompressed & (targetBit - 1);
    size_t upperBits = localCompressed & ~(targetBit - 1);
    
    size_t local0 = (upperBits << 1) | lowerBits;
    size_t local1 = local0 | targetBit;
    
    size_t idx0 = outerIdx * regBlockSize + local0 * regStride + strideOffset;
    size_t idx1 = outerIdx * regBlockSize + local1 * regStride + strideOffset;

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
                                          const int* qubitCounts, const size_t* followQubitCounts) {
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= totalSize / 2) return;

    size_t regDim = getRegisterDim(regIdx, qubitCounts);
    size_t regStride = getRegisterStride(regIdx, followQubitCounts);

    // Target qubit mask within the register's local index space
    // Qubit 0 is the most significant bit
    size_t targetBit = 1 << (qubitCounts[regIdx] - 1 - qubitIdx);

    // Strategy: pairIdx indexes the "compressed" space where target qubit is factored out
    // We need to expand it to get both global indices (with qubit=0 and qubit=1)
    
    // First, decompose pairIdx into:
    // - indices for registers other than regIdx
    // - index within regIdx with target qubit bit removed (compressed local index)
    
    size_t regDimCompressed = regDim / 2;  // Register dimension with one qubit factored out
    size_t regBlockSize = regStride * regDim;  // Total elements in one "block" of this register
    size_t regBlockSizeCompressed = regStride * regDimCompressed;
    
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

// Apply phase based on position squared: exp(i*t*x^2)
__global__ void kernelApplyOneModeQ2(cuDoubleComplex* state, size_t totalSize,
                                                    int regIdx,
                                                    const int* qubitCounts, const double* gridSteps,
                                                    const size_t* followQubitCounts,
                                                    double t) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t regDim = getRegisterDim(regIdx, qubitCounts);
    double dx = gridSteps[regIdx];
    size_t localIdx = getLocalIndex(idx, followQubitCounts[regIdx], qubitCounts[regIdx]);

    double x = gridX(localIdx, regDim, dx);
    state[idx] = cmulPhase(state[idx], t * x * x);
}

// Apply two-mode position coupling: exp(i*coeff*q1*q2)
// where q1 and q2 are position operators for two registers
__global__ void kernelApplyTwoModeQQ(cuDoubleComplex* state, size_t totalSize,
                                          int reg1Idx, int reg2Idx,
                                          const int* qubitCounts, const double* gridSteps,
                                          const size_t* followQubitCounts,
                                          double coeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    size_t reg1Dim = getRegisterDim(reg1Idx, qubitCounts);
    size_t reg2Dim = getRegisterDim(reg2Idx, qubitCounts);
    double dx1 = gridSteps[reg1Idx];
    double dx2 = gridSteps[reg2Idx];
    
    size_t local1 = getLocalIndex(idx, followQubitCounts[reg1Idx], qubitCounts[reg1Idx]);
    size_t local2 = getLocalIndex(idx, followQubitCounts[reg2Idx], qubitCounts[reg2Idx]);

    double q1 = gridX(local1, reg1Dim, dx1);
    double q2 = gridX(local2, reg2Dim, dx2);
    
    state[idx] = cmulPhase(state[idx], coeff * q1 * q2);
}

#pragma endregion

#pragma region FFT Helper Kernels

// Apply scalar multiplication to register elements (not neccessarily normalized scalar)
__global__ void kernelApplyScalarRegister(cuDoubleComplex* data, size_t totalSize,
                                              int regIdx,
                                              const int* qubitCounts, double scalar) {
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
                                                    const int* qubitCounts, const size_t* followQubitCounts) {
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
        if (followQubitCounts[r] == 0 && r > 0) {
            break;
        }
        if (r != regIdx) {
            size_t stride = getRegisterStride(r, followQubitCounts);
            baseIdx += sliceIndices[r] * stride;
        }
    }
    size_t regStride = getRegisterStride(regIdx, followQubitCounts);

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
                                                 int numReg, const int* qubitCounts, const size_t* followQubitCounts) {
    // Use grid-stride loop for better occupancy
    int wIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWignerPoints = wignerN * wignerN;
    
    // Pre-compute constants
    double wDx = 2.0 * wXMax / (wignerN - 1);
    double wDp = 2.0 * wPMax / (wignerN - 1);
    size_t regStride = getRegisterStride(regIdx, followQubitCounts);
    size_t regDim = getRegisterDim(regIdx, qubitCounts);
    
    // Total size of all other registers combined
    size_t otherSize = 1;
    for (int r = 0; r < numReg; r++) {
        if (r != regIdx) {
            otherSize *= getRegisterDim(r, qubitCounts);
        }
    }
    
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
                
                size_t rDim = getRegisterDim(r, qubitCounts);
                size_t rStride = getRegisterStride(r, followQubitCounts);
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

__global__ void kernelComputeHusimiQFullMode(double* outHusimiQ, const cuDoubleComplex* state,
                                             int regIdx, double dx,
                                             int qN, double qMax,
                                             int numReg, const int* qubitCounts, const size_t* followQubitCounts) {
    // Thread ID in flattened grid
    int husimiGridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (husimiGridIdx >= qN * qN) return;
    
    // Pre-compute grid parameters
    double dq = 2.0 * qMax / (qN - 1);
    double sample_q = ((husimiGridIdx % qN) - qN / 2) * dq;
    double sample_p = ((husimiGridIdx / qN) - qN / 2) * dq;
    size_t regStride = getRegisterStride(regIdx, followQubitCounts);
    size_t regDim = getRegisterDim(regIdx, qubitCounts);
    
    double res = 0;
    cuDoubleComplex localOverlapSum;
    size_t cvDim = 1 << qubitCounts[regIdx];
    size_t sliceCount = 1 << (qubitCounts[0] + followQubitCounts[0] - qubitCounts[regIdx]);
    int qbtsAfterCV = followQubitCounts[regIdx];
    size_t qbtsAfterCVMask = (1 << qbtsAfterCV) - 1;
    for (size_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
        localOverlapSum = make_cuDoubleComplex(0.0, 0.0);
        for (int xIdx = 0; xIdx < cvDim; ++xIdx) {
            double x = gridX(xIdx, cvDim, dx);
            double amplitude = PI_POW_NEG_QUARTER * exp(-0.5 * (x - sample_q) * (x - sample_q)) * sqrt(dx); // Normalize as qubit regsiter
            cuDoubleComplex phaseFactor = phaseToZ(sample_p * (x - 0.5 * sample_q));
            localOverlapSum = cuCadd(localOverlapSum, cuCmul(conjMul(phaseFactor,
                state[(sliceIdx & qbtsAfterCVMask) | (xIdx << followQubitCounts[regIdx]) | ((sliceIdx & ~qbtsAfterCVMask) << (qubitCounts[regIdx]))]
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
                                          const int* qubitCounts, const size_t* followQubitCounts) {
    // Each thread computes one joint probability P(local1, local2)
    size_t pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalPairs = (1 << qubitCounts[reg1Idx]) * (1 << qubitCounts[reg2Idx]);
    
    if (pairIdx >= totalPairs) return;
    
    // Extract local indices for the two registers from pairIdx
    size_t local1 = pairIdx / (1 << qubitCounts[reg2Idx]);
    size_t local2 = pairIdx % (1 << qubitCounts[reg2Idx]);
    
    // Sum over all configurations of other registers
    double prob = 0.0;
    
    // Total dimension of all other registers
    size_t otherSize = totalSize / ((1 << qubitCounts[reg1Idx]) * (1 << qubitCounts[reg2Idx]));
    
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
                size_t regDim = 1 << qubitCounts[r];
                localIdx = remainingOther % regDim;
                remainingOther /= regDim;
            }
            
            // Add contribution to global index (last register varies fastest)
            size_t stride = getRegisterStride(r, followQubitCounts);
            globalIdx += localIdx * stride;
        }
        
        prob += absSquare(state[globalIdx]);
    }
    
    outJointProb[pairIdx] = prob;
}

// Kernel to compute tensor product element and its contribution to inner product
// Each thread computes one element of the tensor product and accumulates <current_state | tensor_product>
__global__ void kernelComputeInnerProduct(cuDoubleComplex* partialSums, const cuDoubleComplex* state,
                                          cuDoubleComplex** dRegisterArrays,
                                          int numReg, const int* qubitCounts, const size_t* followQubitCounts,
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
            size_t localIdx = getLocalIndex(globalIdx, followQubitCounts[r], qubitCounts[r]);
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

#pragma region C API - Initialization and Cleanup

void cvdvAllocateRegisters(int numReg, int* numQubits) {
    // Clear and reinitialize log file
    clearLogFile();
    logInfo("Allocating %d registers", numReg);
    
    // Free any existing temporary storage
    if (dRegisterArrays != nullptr) {
        for (int i = 0; i < gNumReg; i++) {
            if (dRegisterArrays[i] != nullptr) {
                checkCudaErrors(cudaFree(dRegisterArrays[i]));
            }
        }
        delete[] dRegisterArrays;
    }
    if (gQubitCounts != nullptr) checkCudaErrors(cudaFree(gQubitCounts));
    if (gFollowQubitCounts != nullptr) checkCudaErrors(cudaFree(gFollowQubitCounts));
    if (gGridSteps != nullptr) checkCudaErrors(cudaFree(gGridSteps));
    if (gRegisterDims != nullptr) checkCudaErrors(cudaFree(gRegisterDims));

    // Store register information
    gNumReg = numReg;

    // Allocate managed memory for register metadata
    checkCudaErrors(cudaMallocManaged(&gQubitCounts, numReg * sizeof(int)));
    checkCudaErrors(cudaMallocManaged(&gFollowQubitCounts, numReg * sizeof(size_t)));
    checkCudaErrors(cudaMallocManaged(&gGridSteps, numReg * sizeof(double)));
    checkCudaErrors(cudaMallocManaged(&gRegisterDims, numReg * sizeof(size_t)));

    // Allocate device register arrays
    dRegisterArrays = new cuDoubleComplex*[numReg];

    // Initialize metadata and allocate space for each register
    size_t prevQubits = 0;
    for (int i = 0; i < numReg; i++) {
        gQubitCounts[i] = numQubits[i];
        prevQubits += numQubits[i];
        
        // Calculate grid step using formula: dx = sqrt(2 * pi / regDim)
        size_t registerDim = 1 << numQubits[i];
        gRegisterDims[i] = registerDim;
        gGridSteps[i] = sqrt(2.0 * PI / registerDim);
        
        // Allocate device array for this register
        checkCudaErrors(cudaMalloc(&dRegisterArrays[i], registerDim * sizeof(cuDoubleComplex)));
        checkCudaErrors(cudaMemset(dRegisterArrays[i], 0, registerDim * sizeof(cuDoubleComplex)));

        logDebug("Register %d: qubits=%d, dx=%.6f, dim=%zu, x_bound=%.6f", 
                  i, numQubits[i], gGridSteps[i], registerDim, 
                  sqrt(2.0 * PI * registerDim));
    }
    
    // Compute following qubit counts (sum of qubits after each register)
    for (int i = 0; i < numReg; i++) {
        size_t followQubits = 0;
        for (int j = i + 1; j < numReg; j++) {
            followQubits += numQubits[j];
        }
        gFollowQubitCounts[i] = followQubits;
    }
    
    logInfo("Registers allocated successfully with managed memory");
}

void cvdvInitStateVector() {
    if (gNumReg == 0 || dRegisterArrays == nullptr) {
        logError("Must call cvdvAllocateRegisters before cvdvInitStateVector");
        return;
    }

    logInfo("Initializing state vector from register data");

    // Compute total state size using managed memory arrays
    gTotalSize = 1;
    for (int i = 0; i < gNumReg; i++) {
        gTotalSize *= gRegisterDims[i];
    }

    logDebug("Total state size: %zu", gTotalSize);

    // Allocate host memory for full state and download register arrays
    cuDoubleComplex* h_state = new cuDoubleComplex[gTotalSize];
    cuDoubleComplex** h_temp_regs = new cuDoubleComplex*[gNumReg];
    
    for (int i = 0; i < gNumReg; i++) {
        h_temp_regs[i] = new cuDoubleComplex[gRegisterDims[i]];
        checkCudaErrors(cudaMemcpy(h_temp_regs[i], dRegisterArrays[i], 
                                   gRegisterDims[i] * sizeof(cuDoubleComplex), 
                                   cudaMemcpyDeviceToHost));
    }

    // Compute tensor product
    // New layout: last register varies fastest
    // globalIdx = localIdx_R0 * (D1*...*D_{n-1}) + localIdx_R1 * (D2*...*D_{n-1}) + ... + localIdx_R_{n-1}
    for (size_t globalIdx = 0; globalIdx < gTotalSize; globalIdx++) {
        cuDoubleComplex product = make_cuDoubleComplex(1.0, 0.0);

        size_t idx = globalIdx;
        // Iterate from last register to first (last varies fastest)
        for (int reg = gNumReg - 1; reg >= 0; reg--) {
            size_t localIdx = idx % gRegisterDims[reg];
            idx /= gRegisterDims[reg];

            cuDoubleComplex reg_val = h_temp_regs[reg][localIdx];
            product = cuCmul(product, reg_val);
        }

        h_state[globalIdx] = product;
    }
    
    // Free temporary host arrays
    for (int i = 0; i < gNumReg; i++) {
        delete[] h_temp_regs[i];
    }
    delete[] h_temp_regs;

    // Free old device state if it exists
    if (dState != nullptr) {
        cudaFree(dState);
    }

    // Allocate and copy device state
    logDebug("Allocating device memory: %.3f GB", gTotalSize * sizeof(cuDoubleComplex) / (1024.0 * 1024.0 * 1024.0));
    checkCudaErrors(cudaMalloc(&dState, gTotalSize * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMemcpy(dState, h_state, gTotalSize * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));

    // Cleanup
    delete[] h_state;

    logInfo("State vector initialized: %d registers, total size: %zu", gNumReg, gTotalSize);
    printf("Initialized state with %d registers, total size: %zu\n", gNumReg, gTotalSize);
}

void cvdvFree() {
    // Free device memory
    if (dState != nullptr) {
        cudaFree(dState);
        dState = nullptr;
    }
    
    // Free managed memory for register metadata
    if (gQubitCounts != nullptr) {
        cudaFree(gQubitCounts);
        gQubitCounts = nullptr;
    }
    if (gFollowQubitCounts != nullptr) {
        cudaFree(gFollowQubitCounts);
        gFollowQubitCounts = nullptr;
    }
    if (gGridSteps != nullptr) {
        cudaFree(gGridSteps);
        gGridSteps = nullptr;
    }
    if (gRegisterDims != nullptr) {
        cudaFree(gRegisterDims);
        gRegisterDims = nullptr;
    }

    // Free device register arrays
    if (dRegisterArrays != nullptr) {
        for (int i = 0; i < gNumReg; i++) {
            if (dRegisterArrays[i] != nullptr) {
                checkCudaErrors(cudaFree(dRegisterArrays[i]));
            }
        }
        delete[] dRegisterArrays;
        dRegisterArrays = nullptr;
    }

    // Close log file
    if (gLogFile != nullptr) {
        time_t now = time(nullptr);
        fprintf(gLogFile, "=== CUDA Session Ended: %s", ctime(&now));
        fclose(gLogFile);
        gLogFile = nullptr;
    }

    gNumReg = 0;
    gTotalSize = 0;
}

#pragma endregion

#pragma region C API - State Initialization

void cvdvSetZero(int regIdx) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        logError("Invalid register index: %d", regIdx);
        return;
    }

    logInfo("Setting register %d to |0> state", regIdx);

    size_t dim = gRegisterDims[regIdx];

    // Set to zero first
    checkCudaErrors(cudaMemset(dRegisterArrays[regIdx], 0, dim * sizeof(cuDoubleComplex)));
    
    // Set first element to 1
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    checkCudaErrors(cudaMemcpy(dRegisterArrays[regIdx], &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    logDebug("Register %d set to |0> state", regIdx);
}

void cvdvSetCoherent(int regIdx, double alphaRe, double alphaIm) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        logError("Invalid register index: %d", regIdx);
        return;
    }

    logInfo("Setting register %d to coherent state |(%.3f, %.3f)>",
             regIdx, alphaRe, alphaIm);

    size_t cvDim = gRegisterDims[regIdx];
    double dx = gGridSteps[regIdx];

    if (fabs(dx) < 1e-15) {
        logError("Coherent state requires CV mode (dx > 0)");
        return;
    }

    // Compute coherent state directly in dRegisterArrays
    int block = 256;
    int grid = (cvDim + block - 1) / block;
    kernelSetCoherent<<<grid, block>>>(dRegisterArrays[regIdx], cvDim, dx, alphaRe, alphaIm);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    logDebug("Register %d set to coherent state", regIdx);
}

void cvdvSetFock(int regIdx, int n) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        logError("Invalid register index: %d", regIdx);
        return;
    }

    logInfo("Setting register %d to Fock state |%d>", regIdx, n);

    size_t cvDim = gRegisterDims[regIdx];
    double dx = gGridSteps[regIdx];

    if (fabs(dx) < 1e-15) {
        logError("Fock state requires CV mode (dx > 0)");
        return;
    }

    // Compute Fock state directly in dRegisterArrays
    int block = 256;
    int grid = (cvDim + block - 1) / block;
    kernelSetFock<<<grid, block>>>(dRegisterArrays[regIdx], cvDim, dx, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    logDebug("Register %d set to Fock state |%d>", regIdx, n);
}

void cvdvSetUniform(int regIdx) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        logError("Invalid register index: %d", regIdx);
        return;
    }

    size_t cvDim = gRegisterDims[regIdx];
    
    logInfo("Setting register %d to uniform superposition (dim=%zu)", regIdx, cvDim);

    // Create uniform state: all elements = 1/sqrt(N)
    double amplitude = 1.0 / sqrt(cvDim);
    
    cuDoubleComplex* h_temp = new cuDoubleComplex[cvDim];
    for (int i = 0; i < cvDim; i++) {
        h_temp[i] = make_cuDoubleComplex(amplitude, 0.0);
    }
    
    checkCudaErrors(cudaMemcpy(dRegisterArrays[regIdx], h_temp, 
                               cvDim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    delete[] h_temp;

    logDebug("Register %d set to uniform superposition", regIdx);
}

void cvdvSetFocks(int regIdx, double* coeffsRe, double* coeffsIm, int length) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        logError("Invalid register index: %d", regIdx);
        return;
    }

    logInfo("Setting register %d to Fock superposition with %d terms", regIdx, length);

    size_t cvDim = gRegisterDims[regIdx];
    double dx = gGridSteps[regIdx];

    if (fabs(dx) < 1e-15) {
        logError("Fock superposition requires CV mode (dx > 0)");
        return;
    }

    // Copy coefficients to device
    cuDoubleComplex* h_coeffs = new cuDoubleComplex[length];
    for (int i = 0; i < length; i++) {
        h_coeffs[i] = make_cuDoubleComplex(coeffsRe[i], coeffsIm[i]);
    }

    cuDoubleComplex* d_coeffs;
    checkCudaErrors(cudaMalloc(&d_coeffs, length * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMemcpy(d_coeffs, h_coeffs, length * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));

    // Compute Fock superposition directly in dRegisterArrays
    int block = 256;
    int grid = (cvDim + block - 1) / block;
    kernelSetFocks<<<grid, block>>>(dRegisterArrays[regIdx], cvDim, dx, d_coeffs, length);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(d_coeffs);
    delete[] h_coeffs;

    logDebug("Register %d set to Fock superposition", regIdx);
}

void cvdvSetCoeffs(int regIdx, double* coeffsRe, double* coeffsIm, int length) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        logError("Invalid register index: %d", regIdx);
        return;
    }

    logInfo("Setting register %d to custom coefficients with %d terms", regIdx, length);

    size_t cvDim = gRegisterDims[regIdx];

    if (length != cvDim) {
        logError("Coefficient array length (%d) must match register dimension (%zu)", length, cvDim);
        return;
    }

    // Copy coefficients to device (already assumed to be normalized)
    cuDoubleComplex* h_coeffs = new cuDoubleComplex[length];
    for (int i = 0; i < length; i++) {
        h_coeffs[i] = make_cuDoubleComplex(coeffsRe[i], coeffsIm[i]);
    }

    checkCudaErrors(cudaMemcpy(dRegisterArrays[regIdx], h_coeffs,
                               length * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    delete[] h_coeffs;

    logDebug("Register %d set to custom coefficients", regIdx);
}

#pragma endregion

#pragma region C API - Fourier Transforms

void cvdvFtQ2P(int regIdx) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    logDebug("ftQ2P called for register %d", regIdx);

    // Index-shifted QFT on specific register
    // Algorithm:
    // 1. Pre-phase correction: exp(i*pi*k*(N-1)/N)
    // 2. Standard FFT
    // 3. Post-phase correction: exp(i*pi*j*(N-1)/N)
    // 4. Normalization: 1/√N

    int block = 256;
    int grid = (gTotalSize + block - 1) / block;

    // Get register dimension and dx from managed memory (direct CPU access)
    size_t regDim = gRegisterDims[regIdx];
    double dx = gGridSteps[regIdx];

    // Step 1: Pre-phase correction: exp(i*π(N-1)/N * j)
    // In position representation: exp(i*π(N-1)/(N*dx) * x)
    double phaseCoeff = PI * (regDim - 1.0) / (regDim * dx);
    logDebug("Applying pre-phase correction: phaseCoeff=%.6f", phaseCoeff);
    kernelApplyOneModeQ<<<grid, block>>>(dState, gTotalSize,
                                          regIdx,
                                          gQubitCounts, gGridSteps, gFollowQubitCounts, phaseCoeff);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 2: Forward FFT using cuFFTPlanMany
    // New layout: last register is contiguous (stride 1), first register has largest stride
    // Stride for regIdx = product of dimensions after regIdx
    size_t regStride = 1;
    for (int i = regIdx + 1; i < gNumReg; i++) {
        regStride *= gRegisterDims[i];
    }
    
    // Number of complete FFTs to perform
    size_t numFFTs = gTotalSize / regDim;
    
    cufftHandle plan;
    int n = regDim;
    
    if (regStride == 1) {
        // Contiguous case: FFT dimension is contiguous, batch FFTs together
        int batch = numFFTs;
        logDebug("Contiguous FFT: n=%d, batch=%d", n, batch);
        cufftResult result = cufftPlan1d(&plan, n, CUFFT_Z2Z, batch);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        
        logDebug("Executing FFT...");
        result = cufftExecZ2Z(plan, dState, dState, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT forward execution failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        logDebug("FFT completed successfully");
    } else {
        // Strided case: FFT dimension is strided  
        // Layout: elements of one FFT are at stride regStride apart
        // Consecutive FFTs start at distance 1 (consecutive in inner dimension)
        int iStride = regStride, oStride = regStride;
        int iDist = 1, oDist = 1;
        int batch = regStride;  
        int inembed[1] = {n * (int)regStride};  // Logical size of input array
        int onembed[1] = {n * (int)regStride};  // Logical size of output array
        
        logDebug("FFT strided case: regIdx=%d, n=%d, stride=%zu, batch=%d, nembed=%d", 
                 regIdx, n, regStride, batch, inembed[0]);
        
        cufftResult result = cufftPlanMany(&plan, 1, &n, inembed, iStride, iDist,
                                            onembed, oStride, oDist, CUFFT_Z2Z, batch);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
        
        // Number of outer blocks to process
        size_t outerDim = gTotalSize / (regStride * regDim);
        logDebug("Processing %zu outer blocks", outerDim);
        
        // Process each outer block
        for (size_t o = 0; o < outerDim; o++) {
            // Start of this outer block
            cuDoubleComplex* blockStart = dState + o * (regStride * regDim);
            result = cufftExecZ2Z(plan, blockStart, blockStart, CUFFT_FORWARD);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT forward execution failed: %d at block %zu\n", result, o);
                exit(EXIT_FAILURE);
            }
        }
        logDebug("Strided FFT completed successfully");
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    cufftDestroy(plan);

    // Step 3: Post-phase correction: exp(i*π(N-1)/N * k)
    // In momentum representation: exp(i*π(N-1)/(N*dx) * p)
    logDebug("Applying post-phase correction: phaseCoeff=%.6f", phaseCoeff);
    kernelApplyOneModeQ<<<grid, block>>>(dState, gTotalSize,
                                          regIdx,
                                          gQubitCounts, gGridSteps, gFollowQubitCounts, phaseCoeff);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 4: Normalization (1/√N for unitary transform)
    double norm = 1.0 / sqrt((double)regDim);
    logDebug("Applying normalization: norm=%.6f", norm);
    kernelApplyScalarRegister<<<grid, block>>>(dState, gTotalSize,
                                                   regIdx,
                                                   gQubitCounts, norm);
    checkCudaErrors(cudaDeviceSynchronize());

    logDebug("ftQ2P completed for register %d", regIdx);
}

void cvdvFtP2Q(int regIdx) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Inverse index-shifted QFT on specific register
    // Algorithm:
    // 1. Pre-phase correction: exp(-i*pi*j*(N-1)/N)
    // 2. Standard IFFT
    // 3. Post-phase correction: exp(-i*pi*k*(N-1)/N)
    // 4. Normalization: 1/√N

    int block = 256;
    int grid = (gTotalSize + block - 1) / block;

    // Get register dimension and dx from managed memory (direct CPU access)
    size_t regDim = gRegisterDims[regIdx];
    double dx = gGridSteps[regIdx];

    // Step 1: Pre-phase correction (negative phase): exp(-i*π(N-1)/N * j)
    // In momentum representation: exp(-i*π(N-1)/(N*dx) * p)
    double phaseCoeff = -PI * (regDim - 1.0) / (regDim * dx);
    kernelApplyOneModeQ<<<grid, block>>>(dState, gTotalSize,
                                          regIdx,
                                          gQubitCounts, gGridSteps, gFollowQubitCounts, phaseCoeff);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 2: Inverse FFT using cuFFTPlanMany
    // New layout: last register is contiguous (stride 1), first register has largest stride
    // Stride for regIdx = product of dimensions after regIdx
    size_t regStride = 1;
    for (int i = regIdx + 1; i < gNumReg; i++) {
        regStride *= gRegisterDims[i];
    }
    
    // Number of complete FFTs to perform
    size_t numFFTs = gTotalSize / regDim;
    
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
        
        result = cufftExecZ2Z(plan, dState, dState, CUFFT_INVERSE);
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
        size_t outerDim = gTotalSize / (regStride * regDim);
        for (size_t o = 0; o < outerDim; o++) {
            cuDoubleComplex* blockStart = dState + o * (regStride * regDim);
            result = cufftExecZ2Z(plan, blockStart, blockStart, CUFFT_INVERSE);
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "cuFFT inverse execution failed: %d at block %zu\n", result, o);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    cufftDestroy(plan);

    // Step 3: Post-phase correction (negative phase): exp(-i*π(N-1)/N * k)
    // In position representation: exp(-i*π(N-1)/(N*dx) * x)
    kernelApplyOneModeQ<<<grid, block>>>(dState, gTotalSize,
                                          regIdx,
                                          gQubitCounts, gGridSteps, gFollowQubitCounts, phaseCoeff);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 4: Normalization (1/√N for unitary transform)
    double norm = 1.0 / sqrt((double)regDim);
    kernelApplyScalarRegister<<<grid, block>>>(dState, gTotalSize,
                                                   regIdx,
                                                   gQubitCounts, norm);
    checkCudaErrors(cudaDeviceSynchronize());
}

#pragma endregion

#pragma region C API - Gates

void cvdvDisplacement(int regIdx, double betaRe, double betaIm) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // D(α) = exp(-i*Im(α)*Re(α)) * D(i*Im(α)) * D(Re(α))
    // D(i*p0) = exp(i*sqrt(2)*p0*q) - phase in position space
    // D(q0) = exp(-i*sqrt(2)*q0*p) - phase in momentum space

    int block = 256;
    int grid = (gTotalSize + block - 1) / block;

    // Step 1: Apply D(i*Im(α)) = exp(i*sqrt(2)*Im(α)*q) in position space
    if (fabs(betaIm) > 1e-12) {
        kernelApplyOneModeQ<<<grid, block>>>(dState, gTotalSize,
                                                       regIdx,
                                                       gQubitCounts, gGridSteps, gFollowQubitCounts,
                                                       SQRT2 * betaIm);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Step 2: Apply D(Re(α)) = exp(-i*sqrt(2)*Re(α)*p) in momentum space
    if (fabs(betaRe) > 1e-12) {
        // Transform register to momentum space
        cvdvFtQ2P(regIdx);

        // Apply phase in momentum space
        kernelApplyOneModeQ<<<grid, block>>>(dState, gTotalSize,
                                                       regIdx,
                                                       gQubitCounts, gGridSteps, gFollowQubitCounts,
                                                       -SQRT2 * betaRe);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Transform back to position space
        cvdvFtP2Q(regIdx);
    }

    // Note: Global phase exp(-i*Im(α)*Re(α)) is ignored
}

void cvdvConditionalDisplacement(int targetReg, int ctrlReg, int ctrlQubit, double alphaRe, double alphaIm) {
    if (targetReg < 0 || targetReg >= gNumReg) {
        fprintf(stderr, "Invalid target register index: %d\n", targetReg);
        return;
    }
    if (ctrlReg < 0 || ctrlReg >= gNumReg) {
        fprintf(stderr, "Invalid control register index: %d\n", ctrlReg);
        return;
    }

    // Conditional displacement: CD(α) = CD(i*Im(α)) CD(Re(α))
    // CD(i*p0) = exp(i*sqrt(2)*p0*Z*q) - controlled phase in position space
    // CD(q0) = F^{-1} exp(-i*sqrt(2)*q0*Z*p) F - controlled phase in momentum space
    // where Z = |0⟩⟨0| - |1⟩⟨1|, so |0⟩ gets +α and |1⟩ gets -α

    int block = 256;
    int grid = (gTotalSize + block - 1) / block;

    // Step 1: Apply CD(i*Im(α)) = exp(i√2 Im(α) Z q) in position space
    if (fabs(alphaIm) > 1e-12) {
        kernelApplyControlledQ<<<grid, block>>>(dState, gTotalSize,
                                                targetReg, ctrlReg, ctrlQubit,
                                                gQubitCounts, gGridSteps, gFollowQubitCounts,
                                                SQRT2 * alphaIm);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Step 2: Apply CD(Re(α)) = F^{-1} exp(-i√2 Re(α) Z p) F
    if (fabs(alphaRe) > 1e-12) {
        // Transform target register to momentum space
        cvdvFtQ2P(targetReg);

        // Apply exp(-i√2 Re(α) Z p) in momentum space
        kernelApplyControlledQ<<<grid, block>>>(dState, gTotalSize,
                                                targetReg, ctrlReg, ctrlQubit,
                                                gQubitCounts, gGridSteps, gFollowQubitCounts,
                                                -SQRT2 * alphaRe);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Transform back to position space
        cvdvFtP2Q(targetReg);
    }
}

void cvdvPauliRotation(int regIdx, int qubitIdx, int axis, double theta) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int block = 256;
    int grid = (gTotalSize / 2 + block - 1) / block;

    kernelPauliRotation<<<grid, block>>>(dState, gTotalSize,
                                                      regIdx, qubitIdx,
                                                      gQubitCounts, gFollowQubitCounts,
                                                      axis, theta);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdvHadamard(int regIdx, int qubitIdx) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int block = 256;
    int grid = (gTotalSize / 2 + block - 1) / block;

    kernelHadamard<<<grid, block>>>(dState, gTotalSize, regIdx, qubitIdx, gQubitCounts, gFollowQubitCounts);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdvPhaseSquare(int regIdx, double t) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    int block = 256;
    int grid = (gTotalSize + block - 1) / block;

    kernelApplyOneModeQ2<<<grid, block>>>(dState, gTotalSize, regIdx, gQubitCounts, gGridSteps, gFollowQubitCounts, t);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdvRotation(int regIdx, double theta) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // R(θ) = exp(-i/2 tan(θ/2) q^2) exp(-i/2 sin(θ) p^2) exp(-i/2 tan(θ/2) q^2)
    double tanHalfTheta = tan(theta / 2.0);
    double sinTheta = sin(theta);

    // First: exp(-i/2 tan(θ/2) q^2) in position space
    cvdvPhaseSquare(regIdx, -0.5 * tanHalfTheta);

    // Second: exp(-i/2 sin(θ) p^2) in momentum space
    cvdvFtQ2P(regIdx);
    cvdvPhaseSquare(regIdx, -0.5 * sinTheta);
    cvdvFtP2Q(regIdx);

    // Third: exp(-i/2 tan(θ/2) q^2) in position space
    cvdvPhaseSquare(regIdx, -0.5 * tanHalfTheta);
}

void cvdvSqueezing(int regIdx, double r) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // S(r) = exp(i(e^{-r}-1)/(2te^r) p^2) exp(-i(te^r)/2 q^2) exp(i(1-e^{-r})/(2t) p^2) exp(i(t)/2 q^2)
    // where t = e^{-r/2} * sqrt(|1-e^{-r}|) minimizes sum of coefficients
    double expR = exp(r);
    double expMinusR = exp(-r);
    double t = exp(-r / 2.0) * sqrt(fabs(1.0 - expMinusR));

    // First: exp(i(e^{-r}-1)/(2te^r) p^2) in momentum space
    cvdvFtQ2P(regIdx);
    cvdvPhaseSquare(regIdx, (expMinusR - 1.0) / (2.0 * t * expR));
    cvdvFtP2Q(regIdx);

    // Second: exp(-i(te^r)/2 q^2) in position space
    cvdvPhaseSquare(regIdx, -0.5 * t * expR);

    // Third: exp(i(1-e^{-r})/(2t) p^2) in momentum space
    cvdvFtQ2P(regIdx);
    cvdvPhaseSquare(regIdx, (1.0 - expMinusR) / (2.0 * t));
    cvdvFtP2Q(regIdx);

    // Fourth: exp(i(t)/2 q^2) in position space
    cvdvPhaseSquare(regIdx, 0.5 * t);
}

void cvdvBeamSplitter(int reg1, int reg2, double theta) {
    if (reg1 < 0 || reg1 >= gNumReg || reg2 < 0 || reg2 >= gNumReg) {
        fprintf(stderr, "Invalid register indices: %d, %d\n", reg1, reg2);
        return;
    }
    if (reg1 == reg2) {
        fprintf(stderr, "Beam splitter requires two different registers\n");
        return;
    }

    // BS(theta) = exp(-i*tan(theta/4)*q1*q2/2) * exp(-i*sin(theta/2)*p1*p2/2) * exp(-i*tan(theta/4)*q1*q2/2)
    // where q and p are position and momentum operators
    
    double coeff_q = -tan(0.25 * theta);  // -tan(theta/4)/2
    double coeff_p = -sin(0.5  * theta);     // -sin(theta/2)/2
    
    int block = 256;
    int grid = (gTotalSize + block - 1) / block;
    
    // First: exp(-i*tan(theta/4)*q1*q2/2)
    kernelApplyTwoModeQQ<<<grid, block>>>(dState, gTotalSize,
                                        reg1, reg2,
                                        gQubitCounts, gGridSteps, gFollowQubitCounts,
                                        coeff_q);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Transform both registers to momentum basis
    cvdvFtQ2P(reg1);
    cvdvFtQ2P(reg2);
    
    // Second: exp(-i*sin(theta/2)*p1*p2/2) (applied in momentum basis, same kernel)
    kernelApplyTwoModeQQ<<<grid, block>>>(dState, gTotalSize,
                                        reg1, reg2,
                                        gQubitCounts, gGridSteps, gFollowQubitCounts,
                                        coeff_p);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Transform back to position basis
    cvdvFtP2Q(reg1);
    cvdvFtP2Q(reg2);
    
    // Third: exp(-i*tan(theta/4)*q1*q2/2)
    kernelApplyTwoModeQQ<<<grid, block>>>(dState, gTotalSize,
                                        reg1, reg2,
                                        gQubitCounts, gGridSteps, gFollowQubitCounts,
                                        coeff_q);
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdvQ1Q2Gate(int reg1, int reg2, double coeff) {
    if (reg1 < 0 || reg1 >= gNumReg || reg2 < 0 || reg2 >= gNumReg) {
        fprintf(stderr, "Invalid register indices: reg1=%d, reg2=%d\n", reg1, reg2);
        return;
    }
    
    if (reg1 == reg2) {
        fprintf(stderr, "Cannot apply Q1Q2 gate to the same register\n");
        return;
    }
    
    int block = 256;
    int grid = (gTotalSize + block - 1) / block;
    
    kernelApplyTwoModeQQ<<<grid, block>>>(dState, gTotalSize,
                                        reg1, reg2,
                                        gQubitCounts, gGridSteps, gFollowQubitCounts,
                                        coeff);
    checkCudaErrors(cudaDeviceSynchronize());
}

#pragma endregion

#pragma region C API - State Access

void cvdvGetWignerSingleSlice(int regIdx, int* sliceIndices, double* wignerOut,
                                   int wignerN, double wXMax, double wPMax) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Get register dimension and dx from managed memory (direct CPU access)
    size_t cvDim = gRegisterDims[regIdx];
    double dx = gGridSteps[regIdx];

    // Copy slice indices to device
    int* d_sliceIndices;
    checkCudaErrors(cudaMalloc(&d_sliceIndices, gNumReg * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_sliceIndices, sliceIndices, gNumReg * sizeof(int),
                               cudaMemcpyHostToDevice));

    double* d_wigner;
    checkCudaErrors(cudaMalloc(&d_wigner, wignerN * wignerN * sizeof(double)));

    int block = 256;
    int grid = (wignerN * wignerN + block - 1) / block;

    kernelComputeWignerSingleSlice<<<grid, block>>>(d_wigner, dState, regIdx, d_sliceIndices,
                                                         cvDim, dx, wignerN,
                                                         wXMax, wPMax, gQubitCounts, gFollowQubitCounts);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wignerOut, d_wigner, wignerN * wignerN * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(d_wigner);
    cudaFree(d_sliceIndices);
}

void cvdvGetWignerFullMode(int regIdx, double* wignerOut,
                                int wignerN, double wXMax, double wPMax) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Get register dimension and dx from managed memory (direct CPU access)
    size_t cvDim = gRegisterDims[regIdx];
    double dx = gGridSteps[regIdx];

    double* d_wigner;
    checkCudaErrors(cudaMalloc(&d_wigner, wignerN * wignerN * sizeof(double)));

    // Use larger blocks and limit grid size for better occupancy
    int block = 512;
    int totalPoints = wignerN * wignerN;
    int grid = min((totalPoints + block - 1) / block, 2048);

    kernelComputeWignerFullMode<<<grid, block>>>(d_wigner, dState, regIdx,
                                                      cvDim, dx, wignerN,
                                                      wXMax, wPMax, gNumReg, gQubitCounts, gFollowQubitCounts);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wignerOut, d_wigner, wignerN * wignerN * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(d_wigner);
}

void cvdvGetHusimiQFullMode(int regIdx, double* outHusimiQ, int qN, double qMax) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // Get register dimension and dx from managed memory (direct CPU access)
    double dx = gGridSteps[regIdx];

    double* d_husimiQ;
    checkCudaErrors(cudaMalloc(&d_husimiQ, qN * qN * sizeof(double)));

    // Use larger blocks and limit grid size for better occupancy
    int block = 512;
    int totalPoints = qN * qN;
    int grid = min((totalPoints + block - 1) / block, 2048);

    kernelComputeHusimiQFullMode<<<grid, block>>>(d_husimiQ, dState, regIdx, dx, qN,
                                                   qMax, gNumReg, gQubitCounts, gFollowQubitCounts);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(outHusimiQ, d_husimiQ, qN * qN * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(d_husimiQ);
}

void cvdvJointMeasure(int reg1Idx, int reg2Idx, double* jointProbsOut) {
    if (reg1Idx < 0 || reg1Idx >= gNumReg || reg2Idx < 0 || reg2Idx >= gNumReg) {
        fprintf(stderr, "Invalid register indices: %d, %d\n", reg1Idx, reg2Idx);
        return;
    }
    if (reg1Idx == reg2Idx) {
        fprintf(stderr, "Register indices must be different for joint measurement\n");
        return;
    }

    // Get register dimensions from managed memory (direct CPU access)
    size_t totalPairs = 1 << gQubitCounts[reg1Idx] << gQubitCounts[reg2Idx];

    // Allocate device memory for joint probabilities
    double* d_jointProb;
    checkCudaErrors(cudaMalloc(&d_jointProb, totalPairs * sizeof(double)));

    // Launch kernel
    int block = 256;
    int grid = min((int)((totalPairs + block - 1) / block), 256);

    kernelComputeJointMeasure<<<grid, block>>>(d_jointProb, dState,
                                                reg1Idx, reg2Idx,
                                                gTotalSize, gNumReg,
                                                gQubitCounts, gFollowQubitCounts);
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy result to CPU
    checkCudaErrors(cudaMemcpy(jointProbsOut, d_jointProb, totalPairs * sizeof(double),
                               cudaMemcpyDeviceToHost));

    // Clean up
    cudaFree(d_jointProb);
}

void cvdvGetState(double* realOut, double* imagOut) {
    if (gTotalSize == 0) {
        fprintf(stderr, "Error: State not initialized. Call cvdvInitStateVector first.\n");
        return;
    }

    cuDoubleComplex* h_state = new cuDoubleComplex[gTotalSize];
    checkCudaErrors(cudaMemcpy(h_state, dState, gTotalSize * sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < gTotalSize; i++) {
        realOut[i] = cuCreal(h_state[i]);
        imagOut[i] = cuCimag(h_state[i]);
    }

    delete[] h_state;
}

// CUDA kernel to compute marginal probabilities for a register
// Sums |amplitude|^2 over all other registers
__global__ void kernelComputeRegisterProbabilities(double* probabilities, const cuDoubleComplex* state,
                                                    int regIdx, int numReg,
                                                    const size_t* registerDims) {
    // Each thread computes probability for one basis state of the target register
    size_t regLocalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t regDim = registerDims[regIdx];
    
    if (regLocalIdx >= regDim) return;
    
    // Compute stride for this register
    size_t regStride = 1;
    for (int i = regIdx + 1; i < numReg; i++) {
        regStride *= registerDims[i];
    }
    
    // Compute total elements in all other registers
    size_t otherSize = 1;
    for (int i = 0; i < numReg; i++) {
        if (i != regIdx) otherSize *= registerDims[i];
    }
    
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
                localIdx = tempOtherIdx % registerDims[r];
                tempOtherIdx /= registerDims[r];
            }
            globalIdx += localIdx * currentStride;
            currentStride *= registerDims[r];
        }
        
        cuDoubleComplex amp = state[globalIdx];
        prob += cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
    }
    
    probabilities[regLocalIdx] = prob;
}

#pragma endregion

#pragma region C API - Getters

int cvdvGetNumRegisters() { return gNumReg; }

size_t cvdvGetTotalSize() { return gTotalSize; }

void cvdvGetRegisterInfo(int* qubitCountsOut, double* gridStepsOut) {
    if (gNumReg == 0) return;

    checkCudaErrors(cudaMemcpy(qubitCountsOut, gQubitCounts,
                               gNumReg * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(gridStepsOut, gGridSteps,
                               gNumReg * sizeof(double), cudaMemcpyDeviceToHost));
}

int cvdvGetRegisterDim(int regIdx) {
    if (regIdx < 0 || regIdx >= gNumReg) return -1;

    int qubit_count;
    checkCudaErrors(cudaMemcpy(&qubit_count, gQubitCounts + regIdx,
                               sizeof(int), cudaMemcpyDeviceToHost));
    return 1 << qubit_count;
}

double cvdvGetRegisterDx(int regIdx) {
    if (regIdx < 0 || regIdx >= gNumReg) return -1.0;

    double dx;
    checkCudaErrors(cudaMemcpy(&dx, gGridSteps + regIdx,
                               sizeof(double), cudaMemcpyDeviceToHost));
    return dx;
}

void cvdvMeasure(int regIdx, double* probabilitiesOut) {
    if (regIdx < 0 || regIdx >= gNumReg) {
        logError("Invalid register index: %d", regIdx);
        return;
    }
    if (dState == nullptr || gRegisterDims == nullptr) {
        logError("State not initialized");
        return;
    }

    logDebug("Computing marginal probabilities for register %d", regIdx);

    size_t regDim = gRegisterDims[regIdx];
    
    // Allocate device memory for probabilities
    double* d_probs;
    checkCudaErrors(cudaMalloc(&d_probs, regDim * sizeof(double)));
    
    // Launch kernel to compute probabilities
    int block = 256;
    int grid = (regDim + block - 1) / block;
    kernelComputeRegisterProbabilities<<<grid, block>>>(d_probs, dState, regIdx, gNumReg, gRegisterDims);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy results back to host
    checkCudaErrors(cudaMemcpy(probabilitiesOut, d_probs, regDim * sizeof(double), cudaMemcpyDeviceToHost));
}

void cvdvInnerProduct(double* realOut, double* imagOut) {
    if (dState == nullptr || dRegisterArrays == nullptr || gNumReg == 0) {
        logError("State or register arrays not initialized");
        *realOut = 0.0;
        *imagOut = 0.0;
        return;
    }

    logInfo("Computing inner product between state and register tensor product");

    // Prepare device array of register array pointers
    cuDoubleComplex** d_regArrayPtrs;
    checkCudaErrors(cudaMalloc(&d_regArrayPtrs, gNumReg * sizeof(cuDoubleComplex*)));
    checkCudaErrors(cudaMemcpy(d_regArrayPtrs, dRegisterArrays, 
                               gNumReg * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));

    // Allocate partial sums for reduction
    int threadsPerBlock = 256;
    int numBlocks = (gTotalSize + threadsPerBlock - 1) / threadsPerBlock;
    numBlocks = min(numBlocks, 1024);  // Cap at 1024 blocks
    
    cuDoubleComplex* d_partialSums;
    checkCudaErrors(cudaMalloc(&d_partialSums, numBlocks * sizeof(cuDoubleComplex)));

    // Launch kernel with shared memory for reduction
    size_t sharedMemSize = threadsPerBlock * sizeof(cuDoubleComplex);
    kernelComputeInnerProduct<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
        d_partialSums, dState, d_regArrayPtrs, gNumReg, gQubitCounts, gFollowQubitCounts, gTotalSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Download partial sums and reduce on host
    cuDoubleComplex* h_partialSums = new cuDoubleComplex[numBlocks];
    checkCudaErrors(cudaMemcpy(h_partialSums, d_partialSums, 
                               numBlocks * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < numBlocks; i++) {
        result = cuCadd(result, h_partialSums[i]);
    }

    *realOut = cuCreal(result);
    *imagOut = cuCimag(result);

    // Cleanup
    delete[] h_partialSums;
    checkCudaErrors(cudaFree(d_partialSums));
    checkCudaErrors(cudaFree(d_regArrayPtrs));

    logInfo("Inner product: (%.10f, %.10f)", *realOut, *imagOut);
}

#pragma endregion

}  // extern "C"
