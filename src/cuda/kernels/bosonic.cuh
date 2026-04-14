#pragma once
#include "utils.cuh"

// ============ Native Grid Wigner Function ============
// Build integrand g_i[k] for each x-row on native grid, tracing out other
// registers
__global__ void kernelBuildWignerRow(cuDoubleComplex* dBuf, const cuDoubleComplex* state,
                                     int regIdx, int cvDim, double dx, int numReg, const int* qbts,
                                     const int* flwQbts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cvDim * cvDim) return;
    int i = tid / cvDim;
    int k = tid % cvDim;
    int kDisp = k - (cvDim - 1) / 2;
    int iPy = i + kDisp;
    int iMy = i - kDisp;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    if (iPy >= 0 && iPy < cvDim && iMy >= 0 && iMy < cvDim) {
        size_t regStride = 1 << flwQbts[regIdx];
        int totalQubits = qbts[0] + flwQbts[0];
        size_t otherSize = 1 << (totalQubits - qbts[regIdx]);
        for (size_t otherIdx = 0; otherIdx < otherSize; otherIdx++) {
            size_t baseIdx = 0, remainingIdx = otherIdx;
            for (int r = numReg - 1; r >= 0; r--) {
                if (r == regIdx) continue;
                size_t rDim = 1 << qbts[r];
                size_t rStride = 1 << flwQbts[r];
                baseIdx += (remainingIdx % rDim) * rStride;
                remainingIdx /= rDim;
            }
            sum = cuCadd(
                sum, conjMul(state[baseIdx + iPy * regStride], state[baseIdx + iMy * regStride]));
        }
    }
    dBuf[i * cvDim + k] = sum;
}

// Apply phase correction + fftshift, write N×N real output
__global__ void kernelFinalizeWigner(double* wigner, const cuDoubleComplex* dFFTOut, int cvDim,
                                     double dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cvDim * cvDim) return;
    int jc = tid / cvDim;
    int i = tid % cvDim;
    int k_fft = (jc + cvDim / 2) % cvDim;
    cuDoubleComplex G = dFFTOut[i * cvDim + k_fft];
    double pj = (jc - cvDim / 2.0) * PI / ((double)cvDim * dx);
    double phase = -pj * (cvDim - 1) * dx;
    double s, c;
    sincos(phase, &s, &c);
    wigner[tid] = (cuCreal(G) * c - cuCimag(G) * s) * dx / PI;
}

// Fused: phase correction + fftshift → complex output (for Husimi FFT2 route)
// Avoids separate kernelRealToComplex pass
__global__ void kernelFinalizeWignerComplex(cuDoubleComplex* out, const cuDoubleComplex* dFFTOut,
                                            int cvDim, double dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cvDim * cvDim) return;
    int jc = tid / cvDim;
    int i = tid % cvDim;
    int k_fft = (jc + cvDim / 2) % cvDim;
    cuDoubleComplex G = dFFTOut[i * cvDim + k_fft];
    double pj = (jc - cvDim / 2.0) * PI / ((double)cvDim * dx);
    double phase = -pj * (cvDim - 1) * dx;
    double s, c;
    sincos(phase, &s, &c);
    double w = (cuCreal(G) * c - cuCimag(G) * s) * dx / PI;
    out[tid] = make_cuDoubleComplex(w, 0.0);
}

// ============ Native Grid Husimi Q Function ============

// Compute G[k] = DFT{g_0}[k] analytically.
// g_0 is the Gaussian kernel with peak at index 0 (not centered at (N-1)/2),
// so that the circular convolution in kernelFillHusimiA gives the correct
// windowed FFT.  G[k] is real and uses the "fftfreq" signed-frequency so that
// negative-frequency bins (k > N/2) are handled correctly.
//   g_0[m] = π^{-1/4} sqrt(dx) exp(-½ (m·dx)²)
//   G[k]   = sqrt(2) π^{1/4} / sqrt(dx) · exp(-½ p_eff²)  (real, no phase)
// where p_eff = 2π·k_signed / (N·dx),  k_signed = k for k≤N/2, k-N otherwise.
__global__ void kernelComputeHusimiG(cuDoubleComplex* dG, int N, double dx) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    int k_signed = (k <= N / 2) ? k : k - N;  // fftfreq: centre the frequency axis
    double p_eff = 2.0 * M_PI * k_signed / ((double)N * dx);
    double mag   = pow(M_PI, 0.25) * sqrt(2.0 / dx) * exp(-0.5 * p_eff * p_eff);
    dG[k] = make_cuDoubleComplex(mag, 0.0);
}

// Extract chunkSize slices at once into a contiguous [chunkSize × cvDim] buffer.
// tid = sliceOffset * cvDim + xIdx
__global__ void kernelExtractChunkPsi(cuDoubleComplex* dAllPsi, const cuDoubleComplex* state,
                                      int regIdx, int cvDim, size_t sliceBase, int chunkSize,
                                      const int* qbts, const int* flwQbts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunkSize * cvDim) return;
    int s    = tid / cvDim;
    int xIdx = tid % cvDim;
    size_t sliceIdx = sliceBase + s;
    int qbtsAfterCV = flwQbts[regIdx];
    size_t qbtsAfterCVMask = ((size_t)1 << qbtsAfterCV) - 1;
    dAllPsi[tid] = state[(sliceIdx & qbtsAfterCVMask) |
                         ((size_t)xIdx << flwQbts[regIdx]) |
                         ((sliceIdx & ~qbtsAfterCVMask) << qbts[regIdx])];
}

// Fill circulant A[s, m, k] = Psi_s[(m+k) % N] * G[k] for all slices in chunk.
// dAllPsi layout: [chunkSize × N], dBuf layout: [chunkSize × N × N].
// After batch IFFT over k (for each (s, m)), dBuf[s, m, j] = H_s[m, j].
// Budget ensures chunkSize*N*N ≤ 1M, so 32-bit tid is safe (avoids expensive 64-bit div).
__global__ void kernelFillHusimiABatched(cuDoubleComplex* dBuf, const cuDoubleComplex* dAllPsi,
                                          const cuDoubleComplex* dG, int N, int chunkSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= chunkSize * N * N) return;
    int s   = tid / (N * N);
    int rem = tid % (N * N);
    int m   = rem / N;
    int k   = rem % N;
    dBuf[tid] = cuCmul(dAllPsi[s * N + ((m + k) & (N - 1))], dG[k]);
}

// fftshift + normalize accumulated power.
// dAccum layout: [m * cvDim + j] where m = FFT p-bin, j = q-index.
// Output layout: [jc * cvDim + qIdx] where jc = centered p-index.
// Divides by PI·N² to correct for cuFFT's unnormalized IFFT (factor N per IFFT
// call, squared because we accumulate |H|²).
__global__ void kernelFinalizeHusimi(double* outQ, const double* dAccum, int cvDim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cvDim * cvDim) return;
    int jc   = tid / cvDim;  // centered p-index (row of output)
    int qIdx = tid % cvDim;  // q-index (col of output)
    int m    = (jc + cvDim / 2) % cvDim;  // FFT bin (un-shifted p-index)
    double N2 = (double)cvDim * (double)cvDim;
    outQ[tid] = dAccum[m * cvDim + qIdx] / (PI * N2);
}

// Reduce |H_s[m,j]|² over all slices s in the chunk and accumulate into dAccum.
// Each thread handles one (m, j) position, loops over chunkSize slices.
// dBuf layout: [chunkSize × N × N], dAccum layout: [N × N].
// No atomics needed since one thread owns each output position.
__global__ void kernelAccumHusimiPowerChunked(
    double* dAccum, const cuDoubleComplex* dBuf, int N2, int chunkSize) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= N2) return;
    double sum = 0.0;
    for (int s = 0; s < chunkSize; s++)
        sum += absSquare(dBuf[(size_t)s * N2 + pos]);
    dAccum[pos] += sum;
}

__global__ void kernelRealToComplex(const double* in, cuDoubleComplex* out, int n2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n2) return;
    out[tid] = make_cuDoubleComplex(in[tid], 0.0);
}

__global__ void kernelApplyHusimiGaussian2DFreq(cuDoubleComplex* spec, int N, double dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = N * N;
    if (tid >= n2) return;

    int pIdx = tid / N;
    int qIdx = tid % N;
    int kq = (qIdx <= N / 2) ? qIdx : (qIdx - N);
    int kp = (pIdx <= N / 2) ? pIdx : (pIdx - N);

    const double PI_LOCAL = 3.14159265358979323846;
    double Lq = (double)N * dx;
    double dp = PI_LOCAL / ((double)N * dx);
    double Lp = (double)N * dp;
    double wq = 2.0 * PI_LOCAL * (double)kq / Lq;
    double wp = 2.0 * PI_LOCAL * (double)kp / Lp;
    double g = exp(-0.25 * (wq * wq + wp * wp));

    cuDoubleComplex v = spec[tid];
    spec[tid] = make_cuDoubleComplex(cuCreal(v) * g, cuCimag(v) * g);
}

__global__ void kernelComplexToRealNormalizedClamp(const cuDoubleComplex* in, double* out, int n2,
                                                   double normFactor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n2) return;
    double v = cuCreal(in[tid]) / normFactor;
    out[tid] = (v < 0.0) ? 0.0 : v;
}

