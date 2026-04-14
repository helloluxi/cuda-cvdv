// gates.cu - CV, DV, hybrid, and two-mode gate implementations

#include "types.h"
#include "kernels/gates.cuh"

// ── static helpers (not in extern "C") ──────────────────────────────────────

static void cvdvRotationSmall(CVDVContext* ctx, int regIdx, double theta);
static void cvdvControlledPhaseSquare(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit,
                                      double t);
static void cvdvConditionalRotationSmall(CVDVContext* ctx, int targetReg, int ctrlReg,
                                         int ctrlQubit, double theta);
static void cvdvBeamSplitterSmall(CVDVContext* ctx, int reg1, int reg2, double theta);
static void cvdvConditionalBeamSplitterSmall(CVDVContext* ctx, int reg1, int reg2, int ctrlReg,
                                             int ctrlQubit, double theta);

extern "C" {

void cvdvFtQ2P(CVDVContext* ctx, int regIdx) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    size_t totalSize = 1 << ctx->gTotalQbt;
    int grid = (totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    size_t regDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];

    // Step 1: Pre-phase correction: exp(i*π(N-1)/N * j)
    // In position representation: exp(i*π(N-1)/(N*dx) * x)
    double phaseCoeff = PI * (regDim - 1.0) / (regDim * dx);
    kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, totalSize, regIdx, ctx->gQbts,
                                            ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);

    // Step 2: Forward FFT — use cached plan
    size_t regStride = 1 << ctx->hFlwQbts[regIdx];
    cufftHandle plan = ctx->ftPlans[regIdx];

    if (regStride == 1) {
        cufftResult result = cufftExecZ2Z(plan, ctx->dState, ctx->dState, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT forward execution failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
    } else {
        // Strided case: loop over outer blocks (plan covers one block at a time)
        size_t regBlockSize = regStride << ctx->hQbts[regIdx];
        size_t outerDim = totalSize / regBlockSize;
        for (size_t o = 0; o < outerDim; o++) {
            cuDoubleComplex* blockStart =
                ctx->dState + (o << (ctx->hFlwQbts[regIdx] + ctx->hQbts[regIdx]));
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
    size_t regDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];

    // Step 1: Pre-phase correction (negative phase): exp(-i*π(N-1)/N * j)
    // In momentum representation: exp(-i*π(N-1)/(N*dx) * p)
    double phaseCoeff = -PI * (regDim - 1.0) / (regDim * dx);
    kernelPhaseX<<<grid, CUDA_BLOCK_SIZE>>>(ctx->dState, (1 << ctx->gTotalQbt), regIdx, ctx->gQbts,
                                            ctx->gGridSteps, ctx->gFlwQbts, phaseCoeff);

    // Step 2: Inverse FFT — use cached plan
    size_t regStride = 1 << ctx->hFlwQbts[regIdx];
    cufftHandle plan = ctx->ftPlans[regIdx];

    if (regStride == 1) {
        cufftResult result = cufftExecZ2Z(plan, ctx->dState, ctx->dState, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT inverse execution failed: %d\n", result);
            exit(EXIT_FAILURE);
        }
    } else {
        size_t regBlockSize = regStride << ctx->hQbts[regIdx];
        size_t outerDim = (1 << ctx->gTotalQbt) / regBlockSize;
        for (size_t o = 0; o < outerDim; o++) {
            cuDoubleComplex* blockStart =
                ctx->dState + (o << (ctx->hFlwQbts[regIdx] + ctx->hQbts[regIdx]));
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
    if (ctrlQubit < 0 || ctrlQubit >= ctx->hQbts[ctrlReg]) {
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
    if (ctx->hQbts[reg1] != ctx->hQbts[reg2]) {
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

void cvdvRotation(CVDVContext* ctx, int regIdx, double theta) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) {
        fprintf(stderr, "Invalid register index: %d\n", regIdx);
        return;
    }

    // For |θ| > π/4, decompose R(θ) = R(θ₀) R(θ-θ₀)
    // where θ₀ ∈ (π/2)Z is chosen so |θ-θ₀| ≤ π/4
    // R(π/2) = FT, R(π) = Parity, R(-π/2) = FT†

    double theta0 = rint(theta / (PI / 2.0)) * (PI / 2.0);
    double remainder = theta - theta0;

    int quarterTurns = (int)rint(theta0 / (PI / 2.0));
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

    cvdvRotationSmall(ctx, regIdx, remainder);
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
    if (ctrlQubit < 0 || ctrlQubit >= ctx->hQbts[ctrlReg]) {
        fprintf(stderr, "Invalid control qubit index: %d\n", ctrlQubit);
        return;
    }

    double theta0 = rint(theta / (PI / 2.0)) * (PI / 2.0);
    double remainder = theta - theta0;

    int quarterTurns = (int)rint(theta0 / (PI / 2.0));
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
    if (ctrlQubit < 0 || ctrlQubit >= ctx->hQbts[ctrlReg]) {
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

    int halfTurns = (int)round(theta0 / PI);
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

    cvdvBeamSplitterSmall(ctx, reg1, reg2, remainder);
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

}  // extern "C"

// ── static helper implementations ───────────────────────────────────────────

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

    // CR(θ) = exp(-i/2 Z tan(θ/2) q^2) exp(-i/2 Z sin(θ) p^2) exp(-i/2 Z tan(θ/2) q^2)
    double tanHalfTheta = tan(theta / 2.0);
    double sinTheta = sin(theta);

    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * tanHalfTheta);
    cvdvFtQ2P(ctx, targetReg);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * sinTheta);
    cvdvFtP2Q(ctx, targetReg);
    cvdvControlledPhaseSquare(ctx, targetReg, ctrlReg, ctrlQubit, -0.5 * tanHalfTheta);
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
