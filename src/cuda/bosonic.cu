// cv.cu - Wigner and Husimi Q phase-space functions

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

    cuDoubleComplex* dHusimiG = nullptr;
    checkCudaErrors(cudaMalloc(&dHusimiG, cvDim * sizeof(cuDoubleComplex)));
    int gGrid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelComputeHusimiG<<<gGrid, CUDA_BLOCK_SIZE>>>(dHusimiG, cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    cufftHandle hBatchFwdPlan;
    cufftHandle hBatchInvPlan;
    cufftResult rf = cufftPlan1d(&hBatchFwdPlan, cvDim, CUFFT_Z2Z, sliceCount);
    cufftResult ri = cufftPlan1d(&hBatchInvPlan, cvDim, CUFFT_Z2Z, sliceCount * cvDim);
    if (rf != CUFFT_SUCCESS || ri != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT Husimi batch plan creation failed: %d %d\n", rf, ri);
        if (rf == CUFFT_SUCCESS) cufftDestroy(hBatchFwdPlan);
        if (ri == CUFFT_SUCCESS) cufftDestroy(hBatchInvPlan);
        cudaFree(dHusimiG);
        return;
    }

    cuDoubleComplex* dChunkPsi;
    cuDoubleComplex* dChunkBuf;
    double* dAccum;
    checkCudaErrors(cudaMalloc(&dChunkPsi, (size_t)sliceCount * cvDim * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMalloc(&dChunkBuf, (size_t)sliceCount * N2 * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMalloc(&dAccum, N2 * sizeof(double)));
    checkCudaErrors(cudaMemset(dAccum, 0, N2 * sizeof(double)));

    int gridChunkN = ((size_t)sliceCount * cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridChunkN2 = ((size_t)sliceCount * N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridN2 = (N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    kernelExtractChunkPsi<<<gridChunkN, CUDA_BLOCK_SIZE>>>(
        dChunkPsi, ctx->dState, regIdx, cvDim, 0, sliceCount, ctx->gQbts, ctx->gFlwQbts);
    cufftExecZ2Z(hBatchFwdPlan, dChunkPsi, dChunkPsi, CUFFT_FORWARD);
    kernelFillHusimiABatched<<<gridChunkN2, CUDA_BLOCK_SIZE>>>(
        dChunkBuf, dChunkPsi, dHusimiG, cvDim, sliceCount);
    cufftExecZ2Z(hBatchInvPlan, dChunkBuf, dChunkBuf, CUFFT_INVERSE);
    kernelAccumHusimiPowerChunked<<<gridN2, CUDA_BLOCK_SIZE>>>(dAccum, dChunkBuf, N2, sliceCount);
    checkCudaErrors(cudaDeviceSynchronize());

    double* dHusimiQ;
    checkCudaErrors(cudaMalloc(&dHusimiQ, N2 * sizeof(double)));
    kernelFinalizeHusimi<<<gridN2, CUDA_BLOCK_SIZE>>>(dHusimiQ, dAccum, cvDim);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(husimiOut, dHusimiQ, N2 * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dHusimiQ);
    cudaFree(dAccum);
    cudaFree(dChunkBuf);
    cudaFree(dChunkPsi);
    cudaFree(dHusimiG);
    cufftDestroy(hBatchInvPlan);
    cufftDestroy(hBatchFwdPlan);
}

static void husimiFromWignerFft2(CVDVContext* ctx, int regIdx, double* husimiOut) {
    int cvDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];
    size_t N2 = (size_t)cvDim * cvDim;

    cuDoubleComplex* dBuf;
    checkCudaErrors(cudaMalloc(&dBuf, N2 * sizeof(cuDoubleComplex)));
    int gridN2 = (N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelBuildWignerRow<<<gridN2, CUDA_BLOCK_SIZE>>>(dBuf, ctx->dState, regIdx, cvDim, dx,
                                                       ctx->gNumReg, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(cudaDeviceSynchronize());

    cufftHandle rowIfftPlan;
    if (cufftPlan1d(&rowIfftPlan, cvDim, CUFFT_Z2Z, cvDim) != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT Husimi row IFFT plan creation failed\n");
        cudaFree(dBuf);
        return;
    }
    cufftExecZ2Z(rowIfftPlan, dBuf, dBuf, CUFFT_INVERSE);
    cufftDestroy(rowIfftPlan);
    checkCudaErrors(cudaDeviceSynchronize());

    // Fused: finalize Wigner + convert to complex in one pass (reuse dBuf)
    kernelFinalizeWignerComplex<<<gridN2, CUDA_BLOCK_SIZE>>>(dBuf, dBuf, cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    cufftHandle plan2d;
    if (cufftPlan2d(&plan2d, cvDim, cvDim, CUFFT_Z2Z) != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT Husimi FFT2 plan creation failed\n");
        cudaFree(dBuf);
        return;
    }
    cufftExecZ2Z(plan2d, dBuf, dBuf, CUFFT_FORWARD);
    kernelApplyHusimiGaussian2DFreq<<<gridN2, CUDA_BLOCK_SIZE>>>(dBuf, cvDim, dx);
    cufftExecZ2Z(plan2d, dBuf, dBuf, CUFFT_INVERSE);
    cufftDestroy(plan2d);
    checkCudaErrors(cudaDeviceSynchronize());

    double* dOut;
    checkCudaErrors(cudaMalloc(&dOut, N2 * sizeof(double)));
    kernelComplexToRealNormalizedClamp<<<gridN2, CUDA_BLOCK_SIZE>>>(
        dBuf, dOut, (int)N2, (double)N2);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(husimiOut, dOut, N2 * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(dOut);
    cudaFree(dBuf);
}

void cvdvGetWigner(CVDVContext* ctx, int regIdx, double* wignerOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    size_t cvDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];

    cuDoubleComplex* dBuf;
    checkCudaErrors(cudaMalloc(&dBuf, cvDim * cvDim * sizeof(cuDoubleComplex)));

    int grid1 = (cvDim * cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    kernelBuildWignerRow<<<grid1, CUDA_BLOCK_SIZE>>>(dBuf, ctx->dState, regIdx, cvDim, dx,
                                                     ctx->gNumReg, ctx->gQbts, ctx->gFlwQbts);
    checkCudaErrors(cudaDeviceSynchronize());

    cufftHandle wPlan;
    cufftResult planRes = cufftPlan1d(&wPlan, cvDim, CUFFT_Z2Z, cvDim);
    if (planRes != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT wPlan creation failed: %d\n", planRes);
        cudaFree(dBuf);
        return;
    }
    cufftExecZ2Z(wPlan, dBuf, dBuf, CUFFT_INVERSE);
    checkCudaErrors(cudaDeviceSynchronize());

    double* dWigner;
    checkCudaErrors(cudaMalloc(&dWigner, cvDim * cvDim * sizeof(double)));
    kernelFinalizeWigner<<<grid1, CUDA_BLOCK_SIZE>>>(dWigner, dBuf, cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wignerOut, dWigner, cvDim * cvDim * sizeof(double),
                                    cudaMemcpyDeviceToHost));
    cudaFree(dWigner);
    cudaFree(dBuf);
    cufftDestroy(wPlan);
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
