// context.cu - Context lifecycle: create, destroy, init, getters

#include "types.h"

extern "C" {

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
    ctx->hBatchPlanValid = false;
    ctx->hBatchCvDim = 0;
    ctx->hBatchChunkSize = 0;
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
    if (ctx->hBatchPlanValid) {
        cufftDestroy(ctx->hBatchFwdPlan);
        cufftDestroy(ctx->hBatchInvPlan);
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

    // Allocate device memory
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

// Copy a full flat state from an existing CUDA device pointer.
// d_src must be cuDoubleComplex* on the GPU with totalSize elements.
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

}  // extern "C"
