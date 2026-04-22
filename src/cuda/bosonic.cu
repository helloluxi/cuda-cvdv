// bosonic.cu - Wigner and Husimi Q phase-space functions

#include "types.h"
#include "kernels/bosonic.cuh"

extern "C" {

static void husimiFromWignerFft2(CVDVContext* ctx, int regIdx, double* husimiOut);

static void husimiFromOverlap(CVDVContext* ctx, int regIdx, double* husimiOut) {
    int cvDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];
    int sliceCount = 1 << (ctx->gTotalQbt - ctx->hQbts[regIdx]);
    size_t N2 = (size_t)cvDim * cvDim;
    size_t fixedBytes =
        (size_t)cvDim * sizeof(cuDoubleComplex) + 2 * N2 * sizeof(double);
    size_t dynamicBytes = (size_t)sliceCount * cvDim * sizeof(cuDoubleComplex) +
                          (size_t)sliceCount * N2 * sizeof(cuDoubleComplex);
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaError_t memInfo = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (memInfo != cudaSuccess || fixedBytes + dynamicBytes > (size_t)(freeBytes * 0.8)) {
        husimiFromWignerFft2(ctx, regIdx, husimiOut);
        return;
    }

    using cudapp::CudaVector;
    using cudapp::CudaDeviceArena;

    CudaVector<cuDoubleComplex, CudaDeviceArena> dHusimiG(cvDim);
    int gGrid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelComputeHusimiG<<<gGrid, CUDA_BLOCK_SIZE>>>(dHusimiG.data(), cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    auto hBatchFwdPlan = cudapp::CudaFftPlan::plan1d(cvDim, CUFFT_Z2Z, sliceCount);
    auto hBatchInvPlan = cudapp::CudaFftPlan::plan1d(cvDim, CUFFT_Z2Z, sliceCount * cvDim);

    CudaVector<cuDoubleComplex, CudaDeviceArena> dChunkPsi(sliceCount * cvDim);
    CudaVector<cuDoubleComplex, CudaDeviceArena> dChunkBuf(sliceCount * N2);
    CudaVector<double, CudaDeviceArena> dAccum(N2);
    checkCudaErrors(cudaMemset(dAccum.data(), 0, N2 * sizeof(double)));

    int gridChunkN = ((size_t)sliceCount * cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridChunkN2 = ((size_t)sliceCount * N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridN2 = (N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelExtractChunkPsi<<<gridChunkN, CUDA_BLOCK_SIZE>>>(
        dChunkPsi.data(), ctx->dState(), regIdx, cvDim, 0, sliceCount, ctx->gpQbts(), ctx->gpFlwQbts());
    cufftExecZ2Z(hBatchFwdPlan, dChunkPsi.data(), dChunkPsi.data(), CUFFT_FORWARD);
    kernelFillHusimiABatched<<<gridChunkN2, CUDA_BLOCK_SIZE>>>(
        dChunkBuf.data(), dChunkPsi.data(), dHusimiG.data(), cvDim, sliceCount);
    cufftExecZ2Z(hBatchInvPlan, dChunkBuf.data(), dChunkBuf.data(), CUFFT_INVERSE);
    kernelAccumHusimiPowerChunked<<<gridN2, CUDA_BLOCK_SIZE>>>(dAccum.data(), dChunkBuf.data(), N2, sliceCount);
    checkCudaErrors(cudaDeviceSynchronize());

    CudaVector<double, CudaDeviceArena> dHusimiQ(N2);
    kernelFinalizeHusimi<<<gridN2, CUDA_BLOCK_SIZE>>>(dHusimiQ.data(), dAccum.data(), cvDim);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(husimiOut, dHusimiQ.data(), N2 * sizeof(double), cudaMemcpyDeviceToHost));
}

static void husimiFromWignerFft2(CVDVContext* ctx, int regIdx, double* husimiOut) {
    int cvDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];
    size_t N2 = (size_t)cvDim * cvDim;

    using cudapp::CudaVector;
    using cudapp::CudaDeviceArena;

    CudaVector<cuDoubleComplex, CudaDeviceArena> dBuf(N2);
    int gridN2 = (N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelBuildWignerRow<<<gridN2, CUDA_BLOCK_SIZE>>>(dBuf.data(), ctx->dState(), regIdx, cvDim, dx,
                                                       ctx->gNumReg, ctx->gpQbts(), ctx->gpFlwQbts());
    checkCudaErrors(cudaDeviceSynchronize());

    {
        auto rowIfftPlan = cudapp::CudaFftPlan::plan1d(cvDim, CUFFT_Z2Z, cvDim);
        cufftExecZ2Z(rowIfftPlan, dBuf.data(), dBuf.data(), CUFFT_INVERSE);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // Fused: finalize Wigner + convert to complex in one pass (reuse dBuf)
    kernelFinalizeWignerComplex<<<gridN2, CUDA_BLOCK_SIZE>>>(dBuf.data(), dBuf.data(), cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    {
        auto plan2d = cudapp::CudaFftPlan::plan2d(cvDim, cvDim, CUFFT_Z2Z);
        cufftExecZ2Z(plan2d, dBuf.data(), dBuf.data(), CUFFT_FORWARD);
        kernelApplyHusimiGaussian2DFreq<<<gridN2, CUDA_BLOCK_SIZE>>>(dBuf.data(), cvDim, dx);
        cufftExecZ2Z(plan2d, dBuf.data(), dBuf.data(), CUFFT_INVERSE);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    CudaVector<double, CudaDeviceArena> dOut(N2);
    kernelComplexToRealNormalizedClamp<<<gridN2, CUDA_BLOCK_SIZE>>>(
        dBuf.data(), dOut.data(), (int)N2, (double)N2);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(husimiOut, dOut.data(), N2 * sizeof(double), cudaMemcpyDeviceToHost));
}

void cvdvGetWigner(CVDVContext* ctx, int regIdx, double* wignerOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    size_t cvDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];
    size_t N2 = cvDim * cvDim;

    using cudapp::CudaVector;
    using cudapp::CudaDeviceArena;

    CudaVector<cuDoubleComplex, CudaDeviceArena> dBuf(N2);

    int grid1 = (N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelBuildWignerRow<<<grid1, CUDA_BLOCK_SIZE>>>(dBuf.data(), ctx->dState(), regIdx, cvDim, dx,
                                                     ctx->gNumReg, ctx->gpQbts(), ctx->gpFlwQbts());
    checkCudaErrors(cudaDeviceSynchronize());

    {
        auto wPlan = cudapp::CudaFftPlan::plan1d(cvDim, CUFFT_Z2Z, cvDim);
        cufftExecZ2Z(wPlan, dBuf.data(), dBuf.data(), CUFFT_INVERSE);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    CudaVector<double, CudaDeviceArena> dWigner(N2);
    kernelFinalizeWigner<<<grid1, CUDA_BLOCK_SIZE>>>(dWigner.data(), dBuf.data(), cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wignerOut, dWigner.data(), N2 * sizeof(double),
                                    cudaMemcpyDeviceToHost));
}

void cvdvGetHusimiQ(CVDVContext* ctx, int regIdx, double* husimiOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    husimiFromOverlap(ctx, regIdx, husimiOut);
}

void cvdvGetHusimiQOverlap(CVDVContext* ctx, int regIdx, double* husimiOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    husimiFromOverlap(ctx, regIdx, husimiOut);
}

void cvdvGetHusimiQWigner(CVDVContext* ctx, int regIdx, double* husimiOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    husimiFromWignerFft2(ctx, regIdx, husimiOut);
}

}  // extern "C"
