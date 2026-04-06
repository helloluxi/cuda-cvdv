/**
 * benchmarks/kernel_profiling/workload.cu
 * Minimal CVDV workload: Wigner/Husimi readout kernels only.
 * Compiled and run by benchmarks/kernel_profiling/run.sh against build/libcvdv.so.
 *
 * Register layout: reg 0 = 1-qubit DV | reg 1 = 10-qubit CV
 */
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

// ── CVDVContext — mirrors src/cvdv.cu (no public header available) ────────
typedef struct {
    cuDoubleComplex* dState;
    int*    gQbts;
    int*    gFlwQbts;
    double* gGridSteps;
    int     gNumReg;
    int     gTotalQbt;
} CVDVContext;

// ── C API forward declarations ────────────────────────────────────────────
extern "C" {
    CVDVContext* cvdvCreate(int numReg, int* numQubits);
    void cvdvDestroy(CVDVContext* ctx);
    void cvdvInitFromSeparable(CVDVContext* ctx, void** devicePtrs, int numReg);

    // readout
    void cvdvGetWigner(CVDVContext*, int reg, double* out);
    void cvdvGetHusimiQ(CVDVContext*, int reg, double* out);
}

// ── state initialisation helpers ──────────────────────────────────────────
static const double TWO_PI = 6.28318530717958647692;

// Upload host array to a fresh device allocation; caller owns the device ptr.
static void* upload(const cuDoubleComplex* h, size_t n) {
    void* d;
    cudaMalloc(&d, n * sizeof(cuDoubleComplex));
    cudaMemcpy(d, h, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    return d;
}

// Uniform superposition: ψ[k] = 1/√dim for all k.
static void* makeUniform(int nQubits) {
    size_t dim = (size_t)1 << nQubits;
    cuDoubleComplex* h = new cuDoubleComplex[dim];
    double amp = 1.0 / sqrt((double)dim);
    for (size_t k = 0; k < dim; k++)
        h[k] = make_cuDoubleComplex(amp, 0.0);
    void* d = upload(h, dim);
    delete[] h;
    return d;
}

// Coherent state |α⟩ in discrete position basis (Gaussian, then normalise).
// Grid: dx = sqrt(2π/dim), x[k] = (k - (dim-1)/2) * dx
// Peak: q₀ = √2·Re(α), p₀ = √2·Im(α)
static void* makeCoherent(int nQubits, double alphaRe, double alphaIm) {
    size_t dim = (size_t)1 << nQubits;
    double dx  = sqrt(TWO_PI / dim);
    double q0  = sqrt(2.0) * alphaRe;
    double p0  = sqrt(2.0) * alphaIm;

    cuDoubleComplex* h = new cuDoubleComplex[dim];
    double norm2 = 0.0;
    for (size_t k = 0; k < dim; k++) {
        double x   = (k - (dim - 1) * 0.5) * dx;
        double env = exp(-0.5 * (x - q0) * (x - q0));
        double re  = env * cos(p0 * x);
        double im  = env * sin(p0 * x);
        h[k]  = make_cuDoubleComplex(re, im);
        norm2 += re * re + im * im;
    }
    double inv = 1.0 / sqrt(norm2);
    for (size_t k = 0; k < dim; k++)
        h[k] = make_cuDoubleComplex(h[k].x * inv, h[k].y * inv);

    void* d = upload(h, dim);
    delete[] h;
    return d;
}

// ── main ──────────────────────────────────────────────────────────────────
int main() {
    int qubits[] = {1, 10};
    CVDVContext* ctx = cvdvCreate(2, qubits);

    // Initialise separable state
    void* ptrs[2] = {
        makeUniform(1),              // reg 0: |+⟩
        makeCoherent(10, 2.0, 1.0), // reg 1: |α = 2+1j⟩
    };
    cvdvInitFromSeparable(ctx, ptrs, 2);
    cudaFree(ptrs[0]); cudaFree(ptrs[1]);

    // Readout paths of interest.
    const int cvDim = 1 << 10;
    double* wig = new double[cvDim * cvDim];
    cvdvGetWigner(ctx, 1, wig);                            // kernelBuildWignerRow + FFT + kernelFinalizeWigner
    delete[] wig;

    double* hus = new double[cvDim * cvDim];
    cvdvGetHusimiQ(ctx, 1, hus);                           // kernelBuildHusimiRows + FFT + kernelFinalizeHusimi
    delete[] hus;

    cvdvDestroy(ctx);
    return 0;
}
