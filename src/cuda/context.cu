// context.cu - Context lifecycle: create, destroy, init, getters

#include "types.h"

extern "C" {

CVDVContext* cvdvCreate(int numReg, int* numQubits) {
    CVDVContext* ctx = new CVDVContext{};  // CPU-only struct, never accessed by GPU kernels
    ctx->gQbts = nullptr;
    ctx->gFlwQbts = nullptr;
    ctx->gGridSteps = nullptr;
    ctx->hQbts = nullptr;
    ctx->hFlwQbts = nullptr;
    ctx->hGridSteps = nullptr;
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
    ctx->dMeasureOut = nullptr;
    ctx->measurePlanCache = nullptr;
    ctx->ctHandleValid = false;
    ctx->dMeasureProbs = nullptr;
    ctx->measurePlanCacheCT = nullptr;

    // If no registers specified, return empty context
    if (numReg == 0 || numQubits == nullptr) {
        return ctx;
    }

    // Allocate registers
    ctx->gNumReg = numReg;

    // Allocate host + device copies of register metadata.
    // Host arrays (hQbts etc.) for CPU-side reads; device arrays (gQbts etc.)
    // for GPU kernel parameters.  No managed memory — eliminates UM runtime
    // overhead that was adding ~2ms per CUDA API call.
    size_t qbtsBytes  = numReg * sizeof(int);
    size_t stepsBytes = numReg * sizeof(double);
    ctx->hQbts      = new int[numReg];
    ctx->hFlwQbts   = new int[numReg];
    ctx->hGridSteps = new double[numReg];
    checkCudaErrors(cudaMalloc(&ctx->gQbts,      qbtsBytes));
    checkCudaErrors(cudaMalloc(&ctx->gFlwQbts,   qbtsBytes));
    checkCudaErrors(cudaMalloc(&ctx->gGridSteps,  stepsBytes));

    // Initialize metadata on host
    ctx->gTotalQbt = 0;
    for (int i = 0; i < numReg; i++) {
        ctx->hQbts[i] = numQubits[i];
        ctx->gTotalQbt += numQubits[i];
        size_t registerDim = 1 << numQubits[i];
        ctx->hGridSteps[i] = sqrt(2.0 * PI / registerDim);
    }
    for (int i = 0; i < numReg; i++) {
        int followQubits = 0;
        for (int j = i + 1; j < numReg; j++)
            followQubits += numQubits[j];
        ctx->hFlwQbts[i] = followQubits;
    }

    // Upload to device
    checkCudaErrors(cudaMemcpy(ctx->gQbts,      ctx->hQbts,      qbtsBytes,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ctx->gFlwQbts,   ctx->hFlwQbts,   qbtsBytes,  cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ctx->gGridSteps,  ctx->hGridSteps, stepsBytes, cudaMemcpyHostToDevice));

    // Build per-register cuFFT plans (parameters are fully determined now).
    ctx->ftPlans = (cufftHandle*)malloc(numReg * sizeof(cufftHandle));
    size_t totalSize = (size_t)1 << ctx->gTotalQbt;
    for (int i = 0; i < numReg; i++) {
        int n = 1 << ctx->hQbts[i];
        size_t regStride = (size_t)1 << ctx->hFlwQbts[i];
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
            if (mp.dSelQbts)    cudaFree(mp.dSelQbts);
            if (mp.dSelFlwQbts) cudaFree(mp.dSelFlwQbts);
            if (mp.dOutStrides) cudaFree(mp.dOutStrides);
        }
        delete cache;
        ctx->measurePlanCache = nullptr;
    }
    if (ctx->measurePlanCacheCT != nullptr) {
        auto* cache = static_cast<std::map<std::vector<int>, MeasurePlanCT>*>(ctx->measurePlanCacheCT);
        for (auto& [key, mp] : *cache) {
            if (mp.dWorkspace) cudaFree(mp.dWorkspace);
            cutensorDestroyPlan(mp.plan);
            cutensorDestroyPlanPreference(mp.planPref);
            cutensorDestroyOperationDescriptor(mp.opDesc);
            cutensorDestroyTensorDescriptor(mp.descIn);
            cutensorDestroyTensorDescriptor(mp.descOut);
        }
        delete cache;
        ctx->measurePlanCacheCT = nullptr;
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

    // Free device + host register metadata
    if (ctx->gQbts)      { cudaFree(ctx->gQbts);      ctx->gQbts = nullptr; }
    if (ctx->gFlwQbts)   { cudaFree(ctx->gFlwQbts);   ctx->gFlwQbts = nullptr; }
    if (ctx->gGridSteps) { cudaFree(ctx->gGridSteps);  ctx->gGridSteps = nullptr; }
    delete[] ctx->hQbts;      ctx->hQbts = nullptr;
    delete[] ctx->hFlwQbts;   ctx->hFlwQbts = nullptr;
    delete[] ctx->hGridSteps; ctx->hGridSteps = nullptr;

    delete ctx;
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
        size_t regDim = 1 << ctx->hQbts[i];
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
            size_t regDim = 1 << ctx->hQbts[reg];
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
    if (ctx->gQbts)      { cudaFree(ctx->gQbts);      ctx->gQbts = nullptr; }
    if (ctx->gFlwQbts)   { cudaFree(ctx->gFlwQbts);   ctx->gFlwQbts = nullptr; }
    if (ctx->gGridSteps) { cudaFree(ctx->gGridSteps);  ctx->gGridSteps = nullptr; }
    delete[] ctx->hQbts;      ctx->hQbts = nullptr;
    delete[] ctx->hFlwQbts;   ctx->hFlwQbts = nullptr;
    delete[] ctx->hGridSteps; ctx->hGridSteps = nullptr;
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
    if (!ctx || ctx->gNumReg == 0) return;
    memcpy(qubitCountsOut, ctx->hQbts, ctx->gNumReg * sizeof(int));
    memcpy(gridStepsOut, ctx->hGridSteps, ctx->gNumReg * sizeof(double));
}

int cvdvGetRegisterDim(CVDVContext* ctx, int regIdx) {
    if (!ctx || regIdx < 0 || regIdx >= ctx->gNumReg) return -1;
    return 1 << ctx->hQbts[regIdx];
}

double cvdvGetRegisterDx(CVDVContext* ctx, int regIdx) {
    if (!ctx || regIdx < 0 || regIdx >= ctx->gNumReg) return -1.0;
    return ctx->hGridSteps[regIdx];
}

}  // extern "C"
