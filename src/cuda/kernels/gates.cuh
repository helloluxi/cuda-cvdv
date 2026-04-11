#pragma once
#include "utils.cuh"

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

// Apply phase factor to a specific register: exp(i*phaseCoeff*x^3)
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

// Apply scalar multiplication to all state elements
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
