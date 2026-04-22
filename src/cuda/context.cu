// context.cu - Context lifecycle: create, destroy, init, getters

#include "types.h"

extern "C" {

CVDVContext* cvdvCreate(int numReg, int* numQubits) {
    CVDVContext* ctx = new CVDVContext{};

    if (numReg == 0 || numQubits == nullptr) {
        return ctx;
    }

    ctx->gNumReg = numReg;

    // Allocate device memory via CudaVector
    ctx->dQbts.resize(numReg);
    ctx->dFlwQbts.resize(numReg);
    ctx->dGridSteps.resize(numReg);

    // Initialize metadata on host
    ctx->hQbts.resize(numReg);
    ctx->hFlwQbts.resize(numReg);
    ctx->hGridSteps.resize(numReg);

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
    checkCudaErrors(cudaMemcpy(ctx->gpQbts(), ctx->hQbts.data(),
                               numReg * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ctx->gpFlwQbts(), ctx->hFlwQbts.data(),
                               numReg * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ctx->gpGridSteps(), ctx->hGridSteps.data(),
                               numReg * sizeof(double), cudaMemcpyHostToDevice));

    // Build per-register cuFFT plans
    size_t totalSize = (size_t)1 << ctx->gTotalQbt;
    ctx->ftPlans.reserve(numReg);
    for (int i = 0; i < numReg; i++) {
        int n = 1 << ctx->hQbts[i];
        size_t regStride = (size_t)1 << ctx->hFlwQbts[i];
        if (regStride == 1) {
            int batch = (int)(totalSize / n);
            ctx->ftPlans.push_back(cudapp::CudaFftPlan::plan1d(n, CUFFT_Z2Z, batch));
        } else {
            int iStride = (int)regStride, oStride = (int)regStride;
            int iDist = 1, oDist = 1;
            int batch = (int)regStride;
            int nembed[1] = {n * (int)regStride};
            ctx->ftPlans.push_back(cudapp::CudaFftPlan::planMany(
                1, &n, nembed, iStride, iDist, nembed, oStride, oDist, CUFFT_Z2Z, batch));
        }
    }

    return ctx;
}

void cvdvDestroy(CVDVContext* ctx) {
    delete ctx;
}

// Build the full tensor-product state from per-register device pointers.
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
    std::vector<std::vector<cuDoubleComplex>> hTempRegs(numReg);
    for (int i = 0; i < numReg; i++) {
        size_t regDim = 1 << ctx->hQbts[i];
        hTempRegs[i].resize(regDim);
        checkCudaErrors(cudaMemcpy(hTempRegs[i].data(), reinterpret_cast<cuDoubleComplex*>(devicePtrs[i]),
                                   regDim * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // Compute tensor product on host (last register varies fastest)
    std::vector<cuDoubleComplex> hState(totalSize);
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

    // Allocate device memory and upload
    ctx->state.resize(totalSize);
    checkCudaErrors(cudaMemcpy(ctx->dState(), hState.data(), totalSize * sizeof(cuDoubleComplex),
                                    cudaMemcpyHostToDevice));
}

void cvdvFree(CVDVContext* ctx) {
    if (!ctx) return;
    ctx->state.clear();
    ctx->dQbts.clear();
    ctx->dFlwQbts.clear();
    ctx->dGridSteps.clear();
    ctx->hQbts.clear();
    ctx->hFlwQbts.clear();
    ctx->hGridSteps.clear();
    ctx->ftPlans.clear();
    ctx->gNumReg = 0;
    ctx->gTotalQbt = 0;
}

void cvdvSetStateFromDevicePtr(CVDVContext* ctx, void* d_src) {
    if (!ctx || !d_src) return;
    size_t totalSize = 1 << ctx->gTotalQbt;
    ctx->state.resize(totalSize);
    checkCudaErrors(cudaMemcpy(ctx->dState(), d_src, totalSize * sizeof(cuDoubleComplex),
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
    memcpy(qubitCountsOut, ctx->hQbts.data(), ctx->gNumReg * sizeof(int));
    memcpy(gridStepsOut, ctx->hGridSteps.data(), ctx->gNumReg * sizeof(double));
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
