/**
 * profiling/workload.cu
 * Minimal CVDV workload: each kernel family called exactly once.
 * Compiled and run by profiling/run.sh against build/libcvdv.so.
 *
 * Register layout: reg 0 = 1-qubit DV | reg 1 = 10-qubit CV | reg 2 = 10-qubit CV
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

    // single-mode CV
    void cvdvDisplacement(CVDVContext*, int reg, double re, double im);
    void cvdvRotation(CVDVContext*, int reg, double theta);
    void cvdvSqueeze(CVDVContext*, int reg, double r);
    void cvdvPhaseSquare(CVDVContext*, int reg, double t);   // shear: exp(i*t*q²)
    void cvdvPhaseCubic(CVDVContext*, int reg, double t);    // cubic: exp(i*t*q³)
    void cvdvFtQ2P(CVDVContext*, int reg);

    // two-mode CV
    void cvdvQ1Q2Gate(CVDVContext*, int r1, int r2, double coeff);
    void cvdvBeamSplitter(CVDVContext*, int r1, int r2, double theta);
    void cvdvSwapRegisters(CVDVContext*, int r1, int r2);

    // qubit (DV)
    void cvdvHadamard(CVDVContext*, int reg, int qubit);
    void cvdvPauliRotation(CVDVContext*, int reg, int qubit, int axis, double theta);
    void cvdvParity(CVDVContext*, int reg);

    // hybrid CV-DV
    void cvdvConditionalDisplacement(CVDVContext*, int tReg, int cReg, int cQubit, double re, double im);
    void cvdvConditionalRotation(CVDVContext*, int tReg, int cReg, int cQubit, double theta);
    void cvdvConditionalSqueeze(CVDVContext*, int tReg, int cReg, int cQubit, double r);
    void cvdvConditionalParity(CVDVContext*, int tReg, int cReg, int cQubit);
    void cvdvConditionalBeamSplitter(CVDVContext*, int r1, int r2, int cReg, int cQubit, double theta);

    // readout
    double cvdvGetNorm(CVDVContext*);
    void cvdvGetWignerFullMode(CVDVContext*, int reg, double* out, int N, double xMax, double pMax);
    void cvdvGetHusimiQFullMode(CVDVContext*, int reg, double* out, int N, double qMax, double pMax);

    // measurement
    void cvdvMeasure(CVDVContext*, int reg, double* probs);
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
    int qubits[] = {1, 10, 10};
    CVDVContext* ctx = cvdvCreate(3, qubits);

    // Initialise separable state
    void* ptrs[3] = {
        makeUniform(1),              // reg 0: |+⟩
        makeCoherent(10, 2.0, 1.0), // reg 1: |α = 2+1j⟩
        makeCoherent(10, 1.0, 0.5), // reg 2: |α = 1+0.5j⟩
    };
    cvdvInitFromSeparable(ctx, ptrs, 3);
    cudaFree(ptrs[0]); cudaFree(ptrs[1]); cudaFree(ptrs[2]);

    // ── single-mode CV (one call each) ────────────────────────────────────
    cvdvDisplacement(ctx, 1, 1.0, 0.5);      // kernelPhaseX + FT
    cvdvRotation(ctx, 1, 0.3);               // kernelPhaseX + FT
    cvdvSqueeze(ctx, 1, 0.5);               // kernelPhaseX2 + FT
    cvdvPhaseSquare(ctx, 1, 0.1);           // kernelPhaseX  (shear = exp(i·t·q²))
    cvdvPhaseCubic(ctx, 1, 0.05);           // kernelCPhaseX (cubic phase)
    cvdvFtQ2P(ctx, 1);                      // vector_fft / regular_fft

    // ── two-mode CV ───────────────────────────────────────────────────────
    cvdvQ1Q2Gate(ctx, 1, 2, 0.3);           // kernelPhaseXX
    cvdvBeamSplitter(ctx, 1, 2, 0.5);       // kernelPhaseXX + FT
    cvdvSwapRegisters(ctx, 1, 2);           // kernelSwapRegisters

    // ── qubit (DV) ────────────────────────────────────────────────────────
    cvdvHadamard(ctx, 0, 0);                // kernelHadamard
    cvdvPauliRotation(ctx, 0, 0, 0, 0.5);  // kernelPauliRotation (axis 0 = X)

    // ── hybrid CV-DV ──────────────────────────────────────────────────────
    cvdvConditionalDisplacement(ctx, 1, 0, 0, 1.0, 0.5);  // kernelCPhaseX
    cvdvConditionalRotation(ctx, 1, 0, 0, 0.3);            // kernelCPhaseX
    cvdvConditionalSqueeze(ctx, 1, 0, 0, 0.5);             // kernelCPhaseX2
    cvdvParity(ctx, 1);                                     // kernelParity
    cvdvConditionalParity(ctx, 1, 0, 0);                    // kernelConditionalParity
    cvdvConditionalBeamSplitter(ctx, 1, 2, 0, 0, 0.5);     // kernelCPhaseXX

    // ── readout ───────────────────────────────────────────────────────────
    cvdvGetNorm(ctx);                       // kernelComputeNorm

    double* wig = new double[51 * 51];
    cvdvGetWignerFullMode(ctx, 1, wig, 51, 5.0, 5.0);      // kernelBuildWignerIntegrand + FT + kernelExtractWigner
    delete[] wig;

    double* hus = new double[51 * 51];
    cvdvGetHusimiQFullMode(ctx, 1, hus, 51, 5.0, 5.0);     // kernelBuildHusimiWindowed + kernelAccumHusimiPower + kernelExtractHusimi
    delete[] hus;

    // ── measurement ───────────────────────────────────────────────────────
    double* probs = new double[1024];
    cvdvMeasure(ctx, 1, probs);             // kernelMeasureShared
    delete[] probs;

    cvdvDestroy(ctx);
    return 0;
}
