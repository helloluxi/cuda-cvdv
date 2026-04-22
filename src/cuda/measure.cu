// measure.cu - Measurement, fidelity, and expectation values

#include "types.h"
#include "api.h"
#include "kernels/measure.cuh"

// ── static helpers ───────────────────────────────────────────────────────────

using DeviceComplexVec = cudapp::CudaVector<cuDoubleComplex, cudapp::CudaDeviceArena>;
using DeviceDoubleVec = cudapp::CudaVector<double, cudapp::CudaDeviceArena>;
using DeviceIntVec = cudapp::CudaVector<int, cudapp::CudaDeviceArena>;
using DevicePtrVec = cudapp::CudaVector<cuDoubleComplex*, cudapp::CudaDeviceArena>;
using DeviceLongLongVec = cudapp::CudaVector<int64_t, cudapp::CudaDeviceArena>;

static cuDoubleComplex runInnerProductKernel(CVDVContext* ctx, void** devicePtrs, int numReg) {
    DevicePtrVec dRegArrayPtrs(numReg);
    checkCudaErrors(cudaMemcpy(dRegArrayPtrs.data(), devicePtrs, numReg * sizeof(cuDoubleComplex*),
                                    cudaMemcpyHostToDevice));

    size_t totalSize = 1 << ctx->gTotalQbt;
    int numBlocks = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    numBlocks = min(numBlocks, 1024);

    DeviceComplexVec dPartialSums(numBlocks);

    size_t sharedMemSize = CUDA_BLOCK_SIZE * sizeof(cuDoubleComplex);
    kernelComputeInnerProduct<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemSize>>>(
        dPartialSums.data(), ctx->dState(), dRegArrayPtrs.data(), numReg, ctx->gpQbts(), ctx->gpFlwQbts(), totalSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<cuDoubleComplex> hPartialSums(numBlocks);
    checkCudaErrors(cudaMemcpy(hPartialSums.data(), dPartialSums.data(), numBlocks * sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost));
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < numBlocks; i++) result = cuCadd(result, hPartialSums[i]);

    return result;
}

static double computeExpectX2(CVDVContext* ctx, int regIdx) {
    size_t totalSize = 1ULL << ctx->gTotalQbt;
    int numBlocks = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    numBlocks = min(numBlocks, 1024);

    DeviceDoubleVec dPartialSums(numBlocks);

    kernelExpectX2<<<numBlocks, CUDA_BLOCK_SIZE>>>(dPartialSums.data(), ctx->dState(), totalSize, regIdx,
                                                   ctx->gpQbts(), ctx->gpGridSteps(), ctx->gpFlwQbts());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<double> hPartialSums(numBlocks);
    checkCudaErrors(cudaMemcpy(hPartialSums.data(), dPartialSums.data(), numBlocks * sizeof(double),
                                    cudaMemcpyDeviceToHost));
    double result = 0.0;
    for (int i = 0; i < numBlocks; i++) result += hPartialSums[i];
    return result;
}

// ── C API ────────────────────────────────────────────────────────────────────

extern "C" {

void cvdvMeasureMultiple(CVDVContext* ctx, const int* regIdxs, int numRegs, double* probsOut) {
    if (!ctx || !regIdxs || numRegs <= 0) return;
    for (int i = 0; i < numRegs; i++) {
        if (regIdxs[i] < 0 || regIdxs[i] >= ctx->gNumReg) {
            fprintf(stderr, "Invalid register index: %d\n", regIdxs[i]);
            return;
        }
    }

    size_t totalSize = (size_t)1 << ctx->gTotalQbt;

    DeviceDoubleVec dMeasureOut(totalSize);

    std::vector<int> qbts(ctx->gNumReg), flwQbts(ctx->gNumReg);
    memcpy(qbts.data(), ctx->hQbts.data(), ctx->gNumReg * sizeof(int));
    memcpy(flwQbts.data(), ctx->hFlwQbts.data(), ctx->gNumReg * sizeof(int));

    std::vector<int64_t> outExtents(numRegs);
    std::vector<int64_t> outStrides(numRegs);
    size_t outSize = 1;
    for (int i = 0; i < numRegs; i++) {
        outExtents[i] = 1LL << qbts[regIdxs[i]];
        outSize *= outExtents[i];
    }
    outStrides[0] = 1;
    for (int i = 1; i < numRegs; i++)
        outStrides[i] = outStrides[i - 1] * outExtents[i - 1];

    std::vector<int> selQbts(numRegs), selFlwQbts(numRegs);
    for (int i = 0; i < numRegs; i++) {
        selQbts[i]    = qbts[regIdxs[i]];
        selFlwQbts[i] = flwQbts[regIdxs[i]];
    }

    DeviceIntVec dSelQbts(numRegs);
    DeviceIntVec dSelFlwQbts(numRegs);
    DeviceLongLongVec dOutStrides(numRegs);
    checkCudaErrors(cudaMemcpy(dSelQbts.data(), selQbts.data(),
                               numRegs * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dSelFlwQbts.data(), selFlwQbts.data(),
                               numRegs * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dOutStrides.data(), outStrides.data(),
                               numRegs * sizeof(long long), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(dMeasureOut.data(), 0, outSize * sizeof(double)));

    int gridSize = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    gridSize = min(gridSize, 1024);
    size_t sharedMem = (outSize <= MEASURE_SHARED_HIST_MAX)
                           ? outSize * sizeof(double) : 0;
    kernelAbsSquareReduce<<<gridSize, CUDA_BLOCK_SIZE, sharedMem>>>(
        dMeasureOut.data(), ctx->dState(), totalSize,
        dSelQbts.data(), dSelFlwQbts.data(), numRegs,
        dOutStrides.data(), outSize);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(probsOut, dMeasureOut.data(), outSize * sizeof(double),
                               cudaMemcpyDeviceToHost));
}

void cvdvMeasureMultipleCT(CVDVContext* ctx, const int* regIdxs, int numRegs, double* probsOut) {
    if (!ctx || !regIdxs || numRegs <= 0) return;
    for (int i = 0; i < numRegs; i++) {
        if (regIdxs[i] < 0 || regIdxs[i] >= ctx->gNumReg) {
            fprintf(stderr, "Invalid register index: %d\n", regIdxs[i]);
            return;
        }
    }

    size_t totalSize = (size_t)1 << ctx->gTotalQbt;

    cutensorHandle_t ctHandle;
    cutensorStatus_t initStatus = cutensorCreate(&ctHandle);
    if (initStatus != CUTENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "cuTENSOR init failed: %d\n", initStatus);
        return;
    }

    DeviceDoubleVec dMeasureProbs(totalSize);
    DeviceDoubleVec dMeasureOut(totalSize);

    std::vector<int> qbts(ctx->gNumReg), flwQbts(ctx->gNumReg);
    memcpy(qbts.data(), ctx->hQbts.data(), ctx->gNumReg * sizeof(int));
    memcpy(flwQbts.data(), ctx->hFlwQbts.data(), ctx->gNumReg * sizeof(int));

    std::vector<long long> extents(ctx->gNumReg);
    std::vector<long long> strides(ctx->gNumReg);
    for (int r = 0; r < ctx->gNumReg; r++) {
        extents[r] = 1LL << qbts[r];
        strides[r] = 1LL << flwQbts[r];
    }
    std::vector<int64_t> outExtents(numRegs);
    std::vector<int64_t> outStridesVec(numRegs);
    size_t outSize = 1;
    for (int i = 0; i < numRegs; i++) {
        outExtents[i] = extents[regIdxs[i]];
        outSize *= outExtents[i];
    }
    outStridesVec[0] = 1;
    for (int i = 1; i < numRegs; i++)
        outStridesVec[i] = outStridesVec[i - 1] * outExtents[i - 1];

    std::vector<int32_t> modesIn(ctx->gNumReg), modesOut(numRegs);
    for (int i = 0; i < ctx->gNumReg; i++) modesIn[i] = i;
    for (int i = 0; i < numRegs; i++) modesOut[i] = regIdxs[i];

    MeasurePlanCT mp{};
    mp.outSize = outSize;
    cutensorHandle_t h = ctHandle;
    cutensorStatus_t status;
    uint32_t alignment = 256;
    cutensorComputeDescriptor_t computeDesc = CUTENSOR_COMPUTE_DESC_64F;

    status = cutensorCreateTensorDescriptor(h, &mp.descIn, (uint32_t)ctx->gNumReg,
                                            reinterpret_cast<const int64_t*>(extents.data()),
                                            reinterpret_cast<const int64_t*>(strides.data()),
                                            CUDA_R_64F, alignment);
    if (status != CUTENSOR_STATUS_SUCCESS) {
        fprintf(stderr, "cuTENSOR input desc failed: %d\n", status); return;
    }
    status = cutensorCreateTensorDescriptor(h, &mp.descOut, (uint32_t)numRegs,
                                            reinterpret_cast<const int64_t*>(outExtents.data()),
                                            reinterpret_cast<const int64_t*>(outStridesVec.data()),
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

    DeviceDoubleVec dWorkspace;
    if (mp.workspaceSize > 0) {
        dWorkspace.resize(mp.workspaceSize);
        mp.dWorkspace = dWorkspace.data();
    } else {
        mp.dWorkspace = nullptr;
    }

    // Step 1: Compute |ψ|² into scratch buffer
    int gridSize = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    kernelAbsSquareInPlace<<<gridSize, CUDA_BLOCK_SIZE>>>(dMeasureProbs.data(), ctx->dState(), totalSize);

    checkCudaErrors(cudaMemset(dMeasureOut.data(), 0, mp.outSize * sizeof(double)));

    // Step 2: cuTENSOR reduction
    const double one = 1.0, zero = 0.0;
    cutensorStatus_t reduceStatus = cutensorReduce(ctHandle, mp.plan,
                                                   &one, dMeasureProbs.data(),
                                                   &zero, dMeasureOut.data(), dMeasureOut.data(),
                                                   mp.dWorkspace, mp.workspaceSize, 0);
    if (reduceStatus != CUTENSOR_STATUS_SUCCESS)
        fprintf(stderr, "cuTENSOR reduction failed: %d\n", reduceStatus);

    checkCudaErrors(cudaStreamSynchronize(0));
    checkCudaErrors(cudaMemcpy(probsOut, dMeasureOut.data(), mp.outSize * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cutensorDestroyPlan(mp.plan);
    cutensorDestroyPlanPreference(mp.planPref);
    cutensorDestroyOperationDescriptor(mp.opDesc);
    cutensorDestroyTensorDescriptor(mp.descIn);
    cutensorDestroyTensorDescriptor(mp.descOut);
    cutensorDestroy(ctHandle);
}

void cvdvGetState(CVDVContext* ctx, double* realOut, double* imagOut) {
    if (!ctx) return;
    size_t totalSize = 1 << ctx->gTotalQbt;
    if (totalSize == 0) {
        fprintf(stderr, "Error: State not initialized. Call cvdvInitStateVector first.\n");
        return;
    }

    std::vector<cuDoubleComplex> hState(totalSize);
    checkCudaErrors(cudaMemcpy(hState.data(), ctx->dState(), totalSize * sizeof(cuDoubleComplex),
                                    cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < totalSize; i++) {
        realOut[i] = cuCreal(hState[i]);
        imagOut[i] = cuCimag(hState[i]);
    }
}

double cvdvGetNorm(CVDVContext* ctx) {
    if (!ctx) return 0.0;
    if (ctx->state.empty()) {
        fprintf(stderr, "State not initialized\n");
        return 0.0;
    }

    size_t totalSize = 1 << ctx->gTotalQbt;

    DeviceDoubleVec dNormOut(1);
    checkCudaErrors(cudaMemset(dNormOut.data(), 0, sizeof(double)));

    int gridSize = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    gridSize = min(gridSize, 1024);
    kernelAbsSquareReduce<<<gridSize, CUDA_BLOCK_SIZE, 0>>>(
        dNormOut.data(), ctx->dState(), totalSize,
        nullptr, nullptr, 0,
        nullptr, 1);
    checkCudaErrors(cudaGetLastError());

    double result = 0.0;
    checkCudaErrors(cudaMemcpy(&result, dNormOut.data(), sizeof(double),
                               cudaMemcpyDeviceToHost));
    return result;
}

void cvdvGetFidelity(CVDVContext* ctx, void** devicePtrs, int numReg, double* fidOut) {
    if (!ctx) return;
    if (ctx->state.empty() || ctx->gNumReg == 0) {
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

double cvdvGetPhotonNumber(CVDVContext* ctx, int regIdx) {
    if (!ctx || ctx->state.empty()) return -1.0;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return -1.0;

    double q2 = computeExpectX2(ctx, regIdx);

    cvdvFtQ2P(ctx, regIdx);
    double p2 = computeExpectX2(ctx, regIdx);
    cvdvFtP2Q(ctx, regIdx);

    return (q2 + p2 - 1.0) / 2.0;
}

void cvdvFidelityStatevectors(CVDVContext* ctx1, CVDVContext* ctx2, double* fidOut) {
    *fidOut = 0.0;
    if (!ctx1 || !ctx2) return;
    if (ctx1->state.empty() || ctx2->state.empty()) {
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

    DeviceComplexVec dPartialSums(numBlocks);

    size_t sharedMemSize = CUDA_BLOCK_SIZE * sizeof(cuDoubleComplex);
    kernelInnerProductStatevectors<<<numBlocks, CUDA_BLOCK_SIZE, sharedMemSize>>>(
        dPartialSums.data(), ctx1->dState(), ctx2->dState(), totalSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<cuDoubleComplex> hPartialSums(numBlocks);
    checkCudaErrors(cudaMemcpy(hPartialSums.data(), dPartialSums.data(),
                                     numBlocks * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < numBlocks; i++) result = cuCadd(result, hPartialSums[i]);

    double re = cuCreal(result), im = cuCimag(result);
    *fidOut = re * re + im * im;
}

}  // extern "C"
