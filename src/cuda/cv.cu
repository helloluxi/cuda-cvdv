// cv.cu - Wigner and Husimi Q phase-space functions

#include "types.h"
#include "kernels/cv.cuh"

extern "C" {

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

    // Fixed plan: batch = cvDim rows (only recreated when cvDim changes)
    if (!ctx->wPlanValid || ctx->wPlanCvDim != (int)cvDim) {
        if (ctx->wPlanValid) cufftDestroy(ctx->wPlan);
        cufftResult planRes = cufftPlan1d(&ctx->wPlan, cvDim, CUFFT_Z2Z, cvDim);
        if (planRes != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT wPlan creation failed: %d\n", planRes);
            cudaFree(dBuf);
            ctx->wPlanValid = false;
            return;
        }
        ctx->wPlanCvDim = (int)cvDim;
        ctx->wPlanValid = true;
    }
    cufftExecZ2Z(ctx->wPlan, dBuf, dBuf, CUFFT_INVERSE);
    checkCudaErrors(cudaDeviceSynchronize());

    double* dWigner;
    checkCudaErrors(cudaMalloc(&dWigner, cvDim * cvDim * sizeof(double)));
    kernelFinalizeWigner<<<grid1, CUDA_BLOCK_SIZE>>>(dWigner, dBuf, cvDim, dx);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wignerOut, dWigner, cvDim * cvDim * sizeof(double),
                                    cudaMemcpyDeviceToHost));
    cudaFree(dWigner);
    cudaFree(dBuf);
}

void cvdvGetHusimiQ(CVDVContext* ctx, int regIdx, double* husimiOut) {
    if (!ctx) return;
    if (regIdx < 0 || regIdx >= ctx->gNumReg) return;
    int cvDim = 1 << ctx->hQbts[regIdx];
    double dx = ctx->hGridSteps[regIdx];
    int sliceCount = 1 << (ctx->gTotalQbt - ctx->hQbts[regIdx]);

    // --- Compute chunk size: largest power-of-2 that keeps dBuf ≤ 16 MB ---
    // 16 MB ≈ half the L2 cache on high-end GPUs, so fill→IFFT→accum stay L2-resident.
    // For N=1024 (bytesPerSlice=16MB) this gives chunk=1 (same as old serial loop).
    // Smaller N (e.g. 256) gets chunk=16 and 16× fewer kernel launches.
    const size_t CHUNK_BUF_BUDGET = (size_t)16 * 1024 * 1024;
    size_t bytesPerSlice = (size_t)cvDim * cvDim * sizeof(cuDoubleComplex);
    int chunkSize = 1;
    while (chunkSize * 2 <= sliceCount &&
           (size_t)(chunkSize * 2) * bytesPerSlice <= CHUNK_BUF_BUDGET)
        chunkSize *= 2;

    // --- Lazy G[k]: analytic Gaussian kernel in frequency domain ---
    if (!ctx->hGValid || ctx->hGCvDim != cvDim || ctx->hGDx != dx) {
        if (ctx->dHusimiG != nullptr) cudaFree(ctx->dHusimiG);
        checkCudaErrors(cudaMalloc(&ctx->dHusimiG, cvDim * sizeof(cuDoubleComplex)));
        int gGrid = (cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
        kernelComputeHusimiG<<<gGrid, CUDA_BLOCK_SIZE>>>(ctx->dHusimiG, cvDim, dx);
        checkCudaErrors(cudaDeviceSynchronize());
        ctx->hGCvDim = cvDim;
        ctx->hGDx    = dx;
        ctx->hGValid = true;
    }

    // --- Lazy batched plans: forward (batch=chunkSize) and inverse (batch=chunkSize*N) ---
    if (!ctx->hBatchPlanValid || ctx->hBatchCvDim != cvDim || ctx->hBatchChunkSize != chunkSize) {
        if (ctx->hBatchPlanValid) {
            cufftDestroy(ctx->hBatchFwdPlan);
            cufftDestroy(ctx->hBatchInvPlan);
        }
        cufftResult rf = cufftPlan1d(&ctx->hBatchFwdPlan, cvDim, CUFFT_Z2Z, chunkSize);
        cufftResult ri = cufftPlan1d(&ctx->hBatchInvPlan, cvDim, CUFFT_Z2Z, chunkSize * cvDim);
        if (rf != CUFFT_SUCCESS || ri != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT Husimi batch plan creation failed: %d %d\n", rf, ri);
            return;
        }
        ctx->hBatchCvDim    = cvDim;
        ctx->hBatchChunkSize = chunkSize;
        ctx->hBatchPlanValid = true;
    }

    // --- Per-call buffers ---
    size_t N2 = (size_t)cvDim * cvDim;
    cuDoubleComplex* dChunkPsi;  // [chunkSize × N] extraction buffer
    cuDoubleComplex* dChunkBuf;  // [chunkSize × N × N] circulant / IFFT work buffer
    double* dAccum;              // [N × N] accumulated |H|²
    checkCudaErrors(cudaMalloc(&dChunkPsi, (size_t)chunkSize * cvDim * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMalloc(&dChunkBuf, (size_t)chunkSize * N2 * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMalloc(&dAccum,    N2 * sizeof(double)));
    checkCudaErrors(cudaMemset(dAccum, 0,  N2 * sizeof(double)));

    int gridChunkN  = ((size_t)chunkSize * cvDim + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridChunkN2 = ((size_t)chunkSize * N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    int gridN2      = (N2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;

    // Process slices in chunks — sliceCount and chunkSize are both powers-of-2
    for (int sliceBase = 0; sliceBase < sliceCount; sliceBase += chunkSize) {
        // Step 1: extract chunkSize slices at once → dChunkPsi [chunkSize × N]
        kernelExtractChunkPsi<<<gridChunkN, CUDA_BLOCK_SIZE>>>(
            dChunkPsi, ctx->dState, regIdx, cvDim, sliceBase, chunkSize,
            ctx->gQbts, ctx->gFlwQbts);

        // Step 2: batched forward FFT of all chunkSize slices at once
        cufftExecZ2Z(ctx->hBatchFwdPlan, dChunkPsi, dChunkPsi, CUFFT_FORWARD);

        // Step 3: fill circulant A[s, m, k] = Ψ_s[(m+k)%N] * G[k] for all (s, m, k)
        kernelFillHusimiABatched<<<gridChunkN2, CUDA_BLOCK_SIZE>>>(
            dChunkBuf, dChunkPsi, ctx->dHusimiG, cvDim, chunkSize);

        // Step 4: batched IFFT over k for each (s, m) → H_s[m, j]; batch = chunkSize * N
        cufftExecZ2Z(ctx->hBatchInvPlan, dChunkBuf, dChunkBuf, CUFFT_INVERSE);

        // Step 5: reduce |H_s[m,j]|² over s in chunk and accumulate into dAccum[m*N+j]
        kernelAccumHusimiPowerChunked<<<gridN2, CUDA_BLOCK_SIZE>>>(dAccum, dChunkBuf, N2, chunkSize);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    double* dHusimiQ;
    checkCudaErrors(cudaMalloc(&dHusimiQ, N2 * sizeof(double)));
    kernelFinalizeHusimi<<<gridN2, CUDA_BLOCK_SIZE>>>(dHusimiQ, dAccum, cvDim);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(husimiOut, dHusimiQ, N2 * sizeof(double),
                               cudaMemcpyDeviceToHost));
    cudaFree(dHusimiQ);
    cudaFree(dAccum);
    cudaFree(dChunkBuf);
    cudaFree(dChunkPsi);
}

}  // extern "C"
