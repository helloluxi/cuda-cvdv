// measure.cu - Measurement, fidelity, and expectation values

#include "types.h"
#include "api.h"
#include "kernels/measure.cuh"

// ── static helpers ───────────────────────────────────────────────────────────

// Helper: build device pointer-of-pointers from a host array of numReg device
// ptrs, run kernelComputeInnerProduct, reduce, free, and return the complex sum.
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

    // Per-call output scratch buffer (sized to totalSize — worst-case output)
    double* dMeasureOut = nullptr;
    checkCudaErrors(cudaMalloc(&dMeasureOut, totalSize * sizeof(double)));

    std::vector<int> qbts(ctx->gNumReg), flwQbts(ctx->gNumReg);
    memcpy(qbts.data(), ctx->hQbts, ctx->gNumReg * sizeof(int));
    memcpy(flwQbts.data(), ctx->hFlwQbts, ctx->gNumReg * sizeof(int));

    std::vector<long long> outExtents(numRegs);
    std::vector<long long> outStrides(numRegs);
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

    MeasurePlan mp{};
    mp.outSize = outSize;
    checkCudaErrors(cudaMalloc(&mp.dSelQbts,    numRegs * sizeof(int)));
    checkCudaErrors(cudaMalloc(&mp.dSelFlwQbts, numRegs * sizeof(int)));
    checkCudaErrors(cudaMalloc(&mp.dOutStrides,  numRegs * sizeof(long long)));
    checkCudaErrors(cudaMemcpy(mp.dSelQbts,    selQbts.data(),
                               numRegs * sizeof(int),    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mp.dSelFlwQbts, selFlwQbts.data(),
                               numRegs * sizeof(int),    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mp.dOutStrides,  outStrides.data(),
                               numRegs * sizeof(long long), cudaMemcpyHostToDevice));

    // Zero output before fused reduction (atomicAdd accumulates into it)
    checkCudaErrors(cudaMemset(dMeasureOut, 0, mp.outSize * sizeof(double)));

    // Fused abs-square + reduce: single kernel reads complex state, computes
    // |ψ|² on the fly, and atomically reduces into the output buffer.
    int gridSize = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    gridSize = min(gridSize, 1024);
    size_t sharedMem = (mp.outSize <= MEASURE_SHARED_HIST_MAX)
                           ? mp.outSize * sizeof(double) : 0;
    kernelAbsSquareReduce<<<gridSize, CUDA_BLOCK_SIZE, sharedMem>>>(
        dMeasureOut, ctx->dState, totalSize,
        mp.dSelQbts, mp.dSelFlwQbts, numRegs,
        mp.dOutStrides, mp.outSize);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(probsOut, dMeasureOut, mp.outSize * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(mp.dOutStrides);
    cudaFree(mp.dSelFlwQbts);
    cudaFree(mp.dSelQbts);
    cudaFree(dMeasureOut);
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

    double* dMeasureProbs = nullptr;
    checkCudaErrors(cudaMalloc(&dMeasureProbs, totalSize * sizeof(double)));
    double* dMeasureOut = nullptr;
    checkCudaErrors(cudaMalloc(&dMeasureOut, totalSize * sizeof(double)));

    std::vector<int> qbts(ctx->gNumReg), flwQbts(ctx->gNumReg);
    memcpy(qbts.data(), ctx->hQbts, ctx->gNumReg * sizeof(int));
    memcpy(flwQbts.data(), ctx->hFlwQbts, ctx->gNumReg * sizeof(int));

    std::vector<long long> extents(ctx->gNumReg);
    std::vector<long long> strides(ctx->gNumReg);
    for (int r = 0; r < ctx->gNumReg; r++) {
        extents[r] = 1LL << qbts[r];
        strides[r] = 1LL << flwQbts[r];
    }
    std::vector<int64_t> outExtents(numRegs);
    std::vector<int64_t> outStrides(numRegs);
    size_t outSize = 1;
    for (int i = 0; i < numRegs; i++) {
        outExtents[i] = extents[regIdxs[i]];
        outSize *= outExtents[i];
    }
    outStrides[0] = 1;
    for (int i = 1; i < numRegs; i++)
        outStrides[i] = outStrides[i - 1] * outExtents[i - 1];

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
                                            reinterpret_cast<const int64_t*>(outStrides.data()),
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

    // Step 1: Compute |ψ|² into scratch buffer
    int gridSize = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    kernelAbsSquareInPlace<<<gridSize, CUDA_BLOCK_SIZE>>>(dMeasureProbs, ctx->dState, totalSize);

    // Zero output before reduction
    checkCudaErrors(cudaMemset(dMeasureOut, 0, mp.outSize * sizeof(double)));

    // Step 2: cuTENSOR reduction
    const double one = 1.0, zero = 0.0;
    cutensorStatus_t reduceStatus = cutensorReduce(ctHandle, mp.plan,
                                                   &one, dMeasureProbs,
                                                   &zero, dMeasureOut, dMeasureOut,
                                                   mp.dWorkspace, mp.workspaceSize, 0);
    if (reduceStatus != CUTENSOR_STATUS_SUCCESS)
        fprintf(stderr, "cuTENSOR reduction failed: %d\n", reduceStatus);

    checkCudaErrors(cudaStreamSynchronize(0));
    checkCudaErrors(cudaMemcpy(probsOut, dMeasureOut, mp.outSize * sizeof(double),
                               cudaMemcpyDeviceToHost));

    if (mp.dWorkspace) cudaFree(mp.dWorkspace);
    cutensorDestroyPlan(mp.plan);
    cutensorDestroyPlanPreference(mp.planPref);
    cutensorDestroyOperationDescriptor(mp.opDesc);
    cutensorDestroyTensorDescriptor(mp.descIn);
    cutensorDestroyTensorDescriptor(mp.descOut);
    cudaFree(dMeasureProbs);
    cudaFree(dMeasureOut);
    cutensorDestroy(ctHandle);
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

double cvdvGetNorm(CVDVContext* ctx) {
    if (!ctx) return 0.0;
    if (ctx->dState == nullptr) {
        fprintf(stderr, "State not initialized\n");
        return 0.0;
    }

    size_t totalSize = 1 << ctx->gTotalQbt;

    // Single-element output buffer for norm accumulation.
    double* dNormOut = nullptr;
    checkCudaErrors(cudaMalloc(&dNormOut, sizeof(double)));
    checkCudaErrors(cudaMemset(dNormOut, 0, sizeof(double)));

    int gridSize = (int)((totalSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    gridSize = min(gridSize, 1024);
    // outSize=1 path uses blockReduceSum (no histogram) — selQbts/selFlwQbts unused.
    kernelAbsSquareReduce<<<gridSize, CUDA_BLOCK_SIZE, 0>>>(
        dNormOut, ctx->dState, totalSize,
        nullptr, nullptr, 0,
        nullptr, 1);
    checkCudaErrors(cudaGetLastError());

    double result = 0.0;
    checkCudaErrors(cudaMemcpy(&result, dNormOut, sizeof(double),
                               cudaMemcpyDeviceToHost));
    cudaFree(dNormOut);
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

// Compute mean photon number <n> = (<q²> + <p²> - 1) / 2 for register regIdx.
// Temporarily applies ftQ2P / ftP2Q to measure <p²> in momentum basis (dp = dx).
double cvdvGetPhotonNumber(CVDVContext* ctx, int regIdx) {
    if (!ctx || ctx->dState == nullptr) return -1.0;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return -1.0;

    double q2 = computeExpectX2(ctx, regIdx);

    cvdvFtQ2P(ctx, regIdx);
    double p2 = computeExpectX2(ctx, regIdx);
    cvdvFtP2Q(ctx, regIdx);

    return (q2 + p2 - 1.0) / 2.0;
}

// Compute fidelity |⟨psi1|psi2⟩|² between two CUDA statevectors of the same size.
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
