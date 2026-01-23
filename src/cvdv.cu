// cvdv.cu - CUDA library for CV-DV hybrid quantum simulation
// State vector: n qubits tensor product 1 bosonic mode (position representation)

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>

// ============================================================================
// ERROR CHECKING AND CONSTANTS
// ============================================================================

#define checkCudaErrors(val) do { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880;
constexpr double PI_POW_NEG_QUARTER = 0.75112554446494248286;  // PI^(-0.25)

// ============================================================================
// GLOBAL STATE
// ============================================================================

static cuDoubleComplex* d_state = nullptr;
static int g_dv_level = 0;   // qubit count
static int g_dv_dim = 1;     // 2^g_dv_level
static int g_cv_level = 0;   // grid level
static int g_cv_dim = 1;     // 2^g_cv_level
static double g_dx = 0.0;    // sqrt(2*pi / g_cv_dim)

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

// Position value at grid index: x_i = (i - cv_dim/2) * dx
__device__ __host__ inline double grid_x(int idx, int cv_dim, double dx) {
    return (idx - cv_dim / 2) * dx;
}

// Convert phase to complex exponential: e^{i*phase}
__device__ __host__ inline cuDoubleComplex phaseToZ(double phase) {
    return make_cuDoubleComplex(cos(phase), sin(phase));
}

// Multiply complex number by phase factor: z * e^{i*phase}
__device__ __host__ inline cuDoubleComplex cmulPhase(cuDoubleComplex z, double phase) {
    return cuCmul(phaseToZ(phase), z);
}

// Conjugate multiply: conj(a) * b
__device__ __host__ inline cuDoubleComplex conjMul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(cuConj(a), b);
}

// ============================================================================
// STATE INITIALIZATION KERNELS
// ============================================================================

__global__ void kernel_set_coherent(cuDoubleComplex* state, int cv_dim, double dx,
                                     double alpha_re, double alpha_im) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cv_dim) return;

    double x = grid_x(idx, cv_dim, dx);
    // Coherent state: |alpha> in position representation
    // q = sqrt(2) * Re(alpha), p = sqrt(2) * Im(alpha)
    double q = SQRT2 * alpha_re;
    double p = SQRT2 * alpha_im;

    double norm = PI_POW_NEG_QUARTER;
    double gauss = exp(-(x - q) * (x - q) / 2.0);
    double phase = p * x - p * q / 2.0;

    cuDoubleComplex phase_factor = phaseToZ(phase);
    double amplitude = norm * gauss;

    state[idx] = make_cuDoubleComplex(amplitude * cuCreal(phase_factor), 
                                      amplitude * cuCimag(phase_factor));
}

__global__ void kernel_set_fock(cuDoubleComplex* state, int cv_dim, double dx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cv_dim) return;

    double x = grid_x(idx, cv_dim, dx);

    // Fock state |n>: psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) * H_n(x) * exp(-x^2/2)
    // Hermite polynomial via recurrence: H_0 = 1, H_1 = 2x, H_{n+1} = 2x H_n - 2n H_{n-1}
    double H_prev = 1.0;
    double H_curr = 2.0 * x;

    if (n == 0) {
        H_curr = 1.0;
    } else if (n == 1) {
        H_curr = 2.0 * x;
    } else {
        for (int k = 1; k < n; k++) {
            double H_next = 2.0 * x * H_curr - 2.0 * k * H_prev;
            H_prev = H_curr;
            H_curr = H_next;
        }
    }

    // Normalization: 1/sqrt(2^n * n! * sqrt(pi))
    double norm = PI_POW_NEG_QUARTER;
    double factorial = 1.0;
    for (int k = 1; k <= n; k++) factorial *= k;
    norm /= sqrt(pow(2.0, n) * factorial);

    double val = norm * H_curr * exp(-x * x / 2.0);
    state[idx] = make_cuDoubleComplex(val, 0.0);
}

__global__ void kernel_set_focks(cuDoubleComplex* state, int cv_dim, double dx,
                                  const cuDoubleComplex* coeffs, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cv_dim) return;

    double x = grid_x(idx, cv_dim, dx);
    
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);

    // Compute linear combination of Fock states
    for (int n = 0; n < length; n++) {
        // Compute Hermite polynomial H_n(x)
        double H_prev = 1.0;
        double H_curr = 2.0 * x;

        if (n == 0) {
            H_curr = 1.0;
        } else if (n == 1) {
            H_curr = 2.0 * x;
        } else {
            for (int k = 1; k < n; k++) {
                double H_next = 2.0 * x * H_curr - 2.0 * k * H_prev;
                H_prev = H_curr;
                H_curr = H_next;
            }
        }

        // Normalization: 1/sqrt(2^n * n! * sqrt(pi))
        double norm = PI_POW_NEG_QUARTER;
        double factorial = 1.0;
        for (int k = 1; k <= n; k++) factorial *= k;
        norm /= sqrt(pow(2.0, n) * factorial);

        double psi_n = norm * H_curr * exp(-x * x / 2.0);
        
        // Add coefficient * |n>
        cuDoubleComplex term = cuCmul(coeffs[n], make_cuDoubleComplex(psi_n, 0.0));
        result = cuCadd(result, term);
    }

    state[idx] = result;
}

// ============================================================================
// CONTINUOUS FOURIER TRANSFORM KERNELS
// ============================================================================

// Continuous Fourier Transform: position to momentum representation
// ψ̃(p) = (1/√(2π)) ∫ ψ(x) exp(-i*p*x) dx
__global__ void kernel_ft_x_to_p(const cuDoubleComplex* src, cuDoubleComplex* dst,
                                  int total_size, int cv_dim, double dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int qubit_idx = idx / cv_dim;
    int p_idx = idx % cv_dim;

    double p = grid_x(p_idx, cv_dim, dx);  // momentum uses same grid
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    
    for (int x_idx = 0; x_idx < cv_dim; x_idx++) {
        double x = grid_x(x_idx, cv_dim, dx);
        double phase = -p * x;
        
        cuDoubleComplex phase_factor = phaseToZ(phase);
        cuDoubleComplex psi_x = src[qubit_idx * cv_dim + x_idx];
        
        sum = cuCadd(sum, cuCmul(phase_factor, psi_x));
    }
    
    // Normalization factor: dx/sqrt(2π)
    double norm = dx / sqrt(2.0 * PI);
    dst[idx] = make_cuDoubleComplex(norm * cuCreal(sum), norm * cuCimag(sum));
}

// Inverse Continuous Fourier Transform: momentum to position representation
// ψ(x) = (1/√(2π)) ∫ ψ̃(p) exp(i*p*x) dp
__global__ void kernel_ft_p_to_x(const cuDoubleComplex* src, cuDoubleComplex* dst,
                                  int total_size, int cv_dim, double dx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int qubit_idx = idx / cv_dim;
    int x_idx = idx % cv_dim;

    double x = grid_x(x_idx, cv_dim, dx);
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    
    for (int p_idx = 0; p_idx < cv_dim; p_idx++) {
        double p = grid_x(p_idx, cv_dim, dx);
        double phase = p * x;
        
        cuDoubleComplex phase_factor = phaseToZ(phase);
        cuDoubleComplex psi_p = src[qubit_idx * cv_dim + p_idx];
        
        sum = cuCadd(sum, cuCmul(phase_factor, psi_p));
    }
    
    // Normalization factor: dx/sqrt(2π)
    double norm = dx / sqrt(2.0 * PI);
    dst[idx] = make_cuDoubleComplex(norm * cuCreal(sum), norm * cuCimag(sum));
}

// ============================================================================
// GATE KERNELS
// ============================================================================

// Apply phase factor: exp(i*phase_coeff*x)
__global__ void kernel_apply_phase(cuDoubleComplex* state, int total_size,
                                   int cv_dim, double dx, double phase_coeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int x_idx = idx % cv_dim;
    double x = grid_x(x_idx, cv_dim, dx);
    
    state[idx] = cmulPhase(state[idx], phase_coeff * x);
}

// Compute norm squared for a DV slice
__global__ void kernel_compute_dv_norm(const cuDoubleComplex* state, double* norm_out,
                                        int slice_idx, int cv_dim) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_idx >= cv_dim) return;
    
    int idx = slice_idx * cv_dim + x_idx;
    cuDoubleComplex val = state[idx];
    double prob = cuCreal(val) * cuCreal(val) + cuCimag(val) * cuCimag(val);
    
    atomicAdd(norm_out, prob);
}

// Apply controlled phase: exp(i*phase_coeff*Z*x) where Z = |0⟩⟨0| - |1⟩⟨1|
__global__ void kernel_apply_controlled_phase(cuDoubleComplex* state, int total_size,
                                                int cv_dim, double dx, double phase_coeff,
                                                int ctrl_qubit, int dv_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int qubit_idx = idx / cv_dim;
    int x_idx = idx % cv_dim;
    
    int ctrl_mask = 1 << (dv_level - 1 - ctrl_qubit);
    double x = grid_x(x_idx, cv_dim, dx);
    double phase = phase_coeff * x;
    
    // Z operator: Z = |0⟩⟨0| - |1⟩⟨1|, so |0⟩ gets +α and |1⟩ gets -α
    if (qubit_idx & ctrl_mask) {
        // Control qubit is |1⟩: apply -phase (gets -α)
        state[idx] = cmulPhase(state[idx], -phase);
    } else {
        // Control qubit is |0⟩: apply +phase (gets +α)
        state[idx] = cmulPhase(state[idx], phase);
    }
}

// Conditional displacement: applies old method for controlled operations
__global__ void kernel_cond_displacement(const cuDoubleComplex* src, cuDoubleComplex* dst,
                                          int total_size, int cv_dim,
                                          double dx, double beta_re, double beta_im,
                                          int ctrl_qubit, int dv_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int qubit_idx = idx / cv_dim;
    int x_idx = idx % cv_dim;

    // Check if control qubit is |1>
    int ctrl_mask = 1 << (dv_level - 1 - ctrl_qubit);
    if (!(qubit_idx & ctrl_mask)) {
        // Control is |0>, just copy
        dst[idx] = src[idx];
        return;
    }

    // Apply displacement (using old direct method for conditional case)
    double q_beta = SQRT2 * beta_re;
    double p_beta = SQRT2 * beta_im;

    double x = grid_x(x_idx, cv_dim, dx);
    double x_shifted = x - q_beta;

    int x_shifted_idx = (int)round(x_shifted / dx) + cv_dim / 2;

    cuDoubleComplex val;
    if (x_shifted_idx < 0 || x_shifted_idx >= cv_dim) {
        val = make_cuDoubleComplex(0.0, 0.0);
    } else {
        val = src[qubit_idx * cv_dim + x_shifted_idx];
    }

    double phase = p_beta * (x - q_beta / 2.0);

    dst[idx] = cmulPhase(val, phase);
}

__global__ void kernel_pauli_rotation(cuDoubleComplex* state, int total_size, int cv_dim,
                                       int target_qubit, int dv_level,
                                       int axis, double theta) {
    // axis: 0=X, 1=Y, 2=Z
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size / 2) return;

    int target_mask = 1 << (dv_level - 1 - target_qubit);

    int x_idx = idx % cv_dim;
    int q_idx = idx / cv_dim;

    // Find qubit index pairs that differ only in target qubit
    int lower_bits = q_idx % target_mask;
    int upper_bits = (q_idx / target_mask) * target_mask * 2;
    int q0 = upper_bits + lower_bits;
    int q1 = q0 + target_mask;

    int full_idx0 = q0 * cv_dim + x_idx;
    int full_idx1 = q1 * cv_dim + x_idx;

    cuDoubleComplex a = state[full_idx0];
    cuDoubleComplex b = state[full_idx1];

    double c = cos(theta / 2.0);
    double s = sin(theta / 2.0);

    cuDoubleComplex new_a, new_b;

    if (axis == 0) {  // X
        new_a = make_cuDoubleComplex(c * cuCreal(a) + s * cuCimag(b),
                                      c * cuCimag(a) - s * cuCreal(b));
        new_b = make_cuDoubleComplex(s * cuCimag(a) + c * cuCreal(b),
                                      -s * cuCreal(a) + c * cuCimag(b));
    } else if (axis == 1) {  // Y
        new_a = make_cuDoubleComplex(c * cuCreal(a) - s * cuCreal(b),
                                      c * cuCimag(a) - s * cuCimag(b));
        new_b = make_cuDoubleComplex(s * cuCreal(a) + c * cuCreal(b),
                                      s * cuCimag(a) + c * cuCimag(b));
    } else {  // Z
        cuDoubleComplex phase0 = make_cuDoubleComplex(c, -s);
        cuDoubleComplex phase1 = make_cuDoubleComplex(c, s);
        new_a = cuCmul(phase0, a);
        new_b = cuCmul(phase1, b);
    }

    state[full_idx0] = new_a;
    state[full_idx1] = new_b;
}

__global__ void kernel_hadamard(cuDoubleComplex* state, int total_size, int cv_dim,
                                 int target_qubit, int dv_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = total_size / (2 * cv_dim);
    if (idx >= n_pairs * cv_dim) return;

    int x_idx = idx % cv_dim;
    int pair_idx = idx / cv_dim;

    int target_mask = 1 << (dv_level - 1 - target_qubit);

    int lower_bits = pair_idx % target_mask;
    int upper_bits = (pair_idx / target_mask) * target_mask * 2;
    int q0 = upper_bits + lower_bits;
    int q1 = q0 + target_mask;

    int full_idx0 = q0 * cv_dim + x_idx;
    int full_idx1 = q1 * cv_dim + x_idx;

    cuDoubleComplex a = state[full_idx0];
    cuDoubleComplex b = state[full_idx1];

    double inv_sqrt2 = 1.0 / SQRT2;

    cuDoubleComplex new_a = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(a) + cuCreal(b)),
        inv_sqrt2 * (cuCimag(a) + cuCimag(b))
    );
    cuDoubleComplex new_b = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(a) - cuCreal(b)),
        inv_sqrt2 * (cuCimag(a) - cuCimag(b))
    );

    state[full_idx0] = new_a;
    state[full_idx1] = new_b;
}

// ============================================================================
// CONTINUOUS FOURIER TRANSFORM KERNELS
// ============================================================================

// Apply scalar multiplication to complex array
__global__ void kernel_apply_scalar(cuDoubleComplex* data, int n, double scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    data[idx] = make_cuDoubleComplex(cuCreal(data[idx]) * scalar,
                                      cuCimag(data[idx]) * scalar);
}

// Apply index-dependent phase correction: data[idx] *= exp(i * phase_per_index * (idx % cv_dim))
__global__ void kernel_apply_index_phase(cuDoubleComplex* data, int total_size,
                                          int cv_dim, double phase_per_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int cv_idx = idx % cv_dim;
    double phase = phase_per_index * cv_idx;
    data[idx] = cmulPhase(data[idx], phase);
}

// ============================================================================
// UTILITY KERNELS
// ============================================================================

__global__ void kernel_compute_wigner(double* wigner, const cuDoubleComplex* state,
                                       int dv_dim, int cv_dim, double dx,
                                       int wigner_n, double w_x_max, double w_p_max) {
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= wigner_n * wigner_n) return;

    int wx_idx = w_idx % wigner_n;
    int wp_idx = w_idx / wigner_n;

    double w_dx = 2.0 * w_x_max / (wigner_n - 1);
    double w_dp = 2.0 * w_p_max / (wigner_n - 1);

    double wx = -w_x_max + wx_idx * w_dx;
    double wp = -w_p_max + wp_idx * w_dp;

    // W(x,p) = (1/pi) * integral psi*(x+y) psi(x-y) exp(2ipy) dy
    double wigner_val = 0.0;

    for (int q = 0; q < dv_dim; q++) {
        double real_sum = 0.0;
        double imag_sum = 0.0;

        for (int y_idx = 0; y_idx < cv_dim; y_idx++) {
            double y = grid_x(y_idx, cv_dim, dx);

            // x+y and x-y positions
            int xpy_idx = (int)round((wx + y) / dx) + cv_dim / 2;
            int xmy_idx = (int)round((wx - y) / dx) + cv_dim / 2;

            if (xpy_idx >= 0 && xpy_idx < cv_dim && xmy_idx >= 0 && xmy_idx < cv_dim) {
                cuDoubleComplex psi_xpy = state[q * cv_dim + xpy_idx];
                cuDoubleComplex psi_xmy = state[q * cv_dim + xmy_idx];

                cuDoubleComplex prod = conjMul(psi_xpy, psi_xmy);

                double phase = 2.0 * wp * y;
                cuDoubleComplex phase_factor = phaseToZ(phase);

                real_sum += cuCreal(prod) * cuCreal(phase_factor) - cuCimag(prod) * cuCimag(phase_factor);
                imag_sum += cuCreal(prod) * cuCimag(phase_factor) + cuCimag(prod) * cuCreal(phase_factor);
            }
        }

        wigner_val += real_sum * dx;
    }

    wigner[w_idx] = wigner_val / PI;
}

// Compute Wigner function for a specific qubit basis state slice
__global__ void kernel_compute_wigner_single_slice(double* wigner, const cuDoubleComplex* state,
                                                    int slice_idx, int cv_dim, double dx,
                                                    int wigner_n, double w_x_max, double w_p_max) {
    int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (w_idx >= wigner_n * wigner_n) return;

    int wx_idx = w_idx % wigner_n;
    int wp_idx = w_idx / wigner_n;

    double w_dx = 2.0 * w_x_max / (wigner_n - 1);
    double w_dp = 2.0 * w_p_max / (wigner_n - 1);

    double wx = -w_x_max + wx_idx * w_dx;
    double wp = -w_p_max + wp_idx * w_dp;

    // W(x,p) = (1/pi) * integral psi*(x+y) psi(x-y) exp(2ipy) dy
    double real_sum = 0.0;

    for (int y_idx = 0; y_idx < cv_dim; y_idx++) {
        double y = grid_x(y_idx, cv_dim, dx);

        // x+y and x-y positions
        int xpy_idx = (int)round((wx + y) / dx) + cv_dim / 2;
        int xmy_idx = (int)round((wx - y) / dx) + cv_dim / 2;

        if (xpy_idx >= 0 && xpy_idx < cv_dim && xmy_idx >= 0 && xmy_idx < cv_dim) {
            cuDoubleComplex psi_xpy = state[slice_idx * cv_dim + xpy_idx];
            cuDoubleComplex psi_xmy = state[slice_idx * cv_dim + xmy_idx];

            cuDoubleComplex prod = conjMul(psi_xpy, psi_xmy);

            double phase = 2.0 * wp * y;
            cuDoubleComplex phase_factor = phaseToZ(phase);

            real_sum += cuCreal(prod) * cuCreal(phase_factor) - cuCimag(prod) * cuCimag(phase_factor);
        }
    }

    wigner[w_idx] = (real_sum * dx) / PI;
}

// ============================================================================
// C API - INITIALIZATION AND CLEANUP
// ============================================================================

extern "C" {

int cvdv_init(int dv_level, int cv_level) {
    if (d_state != nullptr) {
        cudaFree(d_state);
    }

    g_dv_level = dv_level;
    g_dv_dim = 1 << dv_level;
    g_cv_level = cv_level;
    g_cv_dim = 1 << cv_level;
    g_dx = sqrt(2.0 * PI / g_cv_dim);

    size_t total_size = (size_t)g_dv_dim * g_cv_dim;

    checkCudaErrors(cudaMalloc(&d_state, total_size * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMemset(d_state, 0, total_size * sizeof(cuDoubleComplex)));

    return 0;
}

void cvdv_free() {
    if (d_state != nullptr) {
        cudaFree(d_state);
        d_state = nullptr;
    }
}

// ============================================================================
// C API - STATE INITIALIZATION
// ============================================================================

void cvdv_set_coherent(double alpha_re, double alpha_im) {
    int block = 256;
    int grid = (g_cv_dim + block - 1) / block;

    checkCudaErrors(cudaMemset(d_state, 0, g_dv_dim * g_cv_dim * sizeof(cuDoubleComplex)));

    kernel_set_coherent<<<grid, block>>>(d_state, g_cv_dim, g_dx, alpha_re, alpha_im);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdv_set_fock(int n) {
    int block = 256;
    int grid = (g_cv_dim + block - 1) / block;

    checkCudaErrors(cudaMemset(d_state, 0, g_dv_dim * g_cv_dim * sizeof(cuDoubleComplex)));

    kernel_set_fock<<<grid, block>>>(d_state, g_cv_dim, g_dx, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdv_set_focks(double* coeffs_re, double* coeffs_im, int length) {
    // Copy coefficients to device
    cuDoubleComplex* h_coeffs = new cuDoubleComplex[length];
    for (int i = 0; i < length; i++) {
        h_coeffs[i] = make_cuDoubleComplex(coeffs_re[i], coeffs_im[i]);
    }

    cuDoubleComplex* d_coeffs;
    checkCudaErrors(cudaMalloc(&d_coeffs, length * sizeof(cuDoubleComplex)));
    checkCudaErrors(cudaMemcpy(d_coeffs, h_coeffs, length * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (g_cv_dim + block - 1) / block;

    checkCudaErrors(cudaMemset(d_state, 0, g_dv_dim * g_cv_dim * sizeof(cuDoubleComplex)));

    kernel_set_focks<<<grid, block>>>(d_state, g_cv_dim, g_dx, d_coeffs, length);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(d_coeffs);
    delete[] h_coeffs;
}

// ============================================================================
// C API - FOURIER TRANSFORMS
// ============================================================================

void cvdv_ft_x_to_p() {
    // Index-shifted QFT: exp(-i*lambda^2*j_tilde*k_tilde)
    // where j_tilde = j - (N-1)/2, k_tilde = k - (N-1)/2
    // lambda^2 = 2*pi/N, so this becomes a standard FFT with phase corrections
    //
    // Decomposition: exp(-i*2pi*jk/N) * exp(i*pi*k*(N-1)/N) * exp(i*pi*j*(N-1)/N)
    // Algorithm:
    // 1. Pre-phase correction: exp(i*pi*k*(N-1)/N)
    // 2. Standard FFT
    // 3. Post-phase correction: exp(i*pi*j*(N-1)/N)
    // 4. Normalization: 1/√N

    int total_size = g_dv_dim * g_cv_dim;
    int block = 256;
    int grid = (total_size + block - 1) / block;

    // Step 1: Pre-phase correction
    double phase_per_index = PI * (g_cv_dim - 1.0) / g_cv_dim;
    kernel_apply_index_phase<<<grid, block>>>(d_state, total_size, g_cv_dim, phase_per_index);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 2: Forward FFT
    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, g_cv_dim, CUFFT_Z2Z, g_dv_dim);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
        exit(EXIT_FAILURE);
    }

    result = cufftExecZ2Z(plan, d_state, d_state, CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT forward execution failed: %d\n", result);
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    cufftDestroy(plan);

    // Step 3: Post-phase correction
    kernel_apply_index_phase<<<grid, block>>>(d_state, total_size, g_cv_dim, phase_per_index);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 4: Normalization (1/√N for unitary transform)
    double norm = 1.0 / sqrt((double)g_cv_dim);
    kernel_apply_scalar<<<grid, block>>>(d_state, total_size, norm);
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdv_ft_p_to_x() {
    // Inverse index-shifted QFT: exp(+i*lambda^2*j_tilde*k_tilde)
    // Decomposition: exp(+i*2pi*jk/N) * exp(-i*pi*j*(N-1)/N) * exp(-i*pi*k*(N-1)/N)
    // Algorithm:
    // 1. Pre-phase correction: exp(-i*pi*j*(N-1)/N)
    // 2. Standard IFFT
    // 3. Post-phase correction: exp(-i*pi*k*(N-1)/N)
    // 4. Normalization: √N (cuFFT IFFT gives 1/N, we want 1/√N)

    int total_size = g_dv_dim * g_cv_dim;
    int block = 256;
    int grid = (total_size + block - 1) / block;

    // Step 1: Pre-phase correction (negative phase)
    double phase_per_index = -PI * (g_cv_dim - 1.0) / g_cv_dim;
    kernel_apply_index_phase<<<grid, block>>>(d_state, total_size, g_cv_dim, phase_per_index);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 2: Inverse FFT
    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, g_cv_dim, CUFFT_Z2Z, g_dv_dim);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT plan creation failed: %d\n", result);
        exit(EXIT_FAILURE);
    }

    result = cufftExecZ2Z(plan, d_state, d_state, CUFFT_INVERSE);
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT inverse execution failed: %d\n", result);
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    cufftDestroy(plan);

    // Step 3: Post-phase correction (negative phase)
    kernel_apply_index_phase<<<grid, block>>>(d_state, total_size, g_cv_dim, phase_per_index);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 4: Normalization (1/√N for unitary transform)
    // Note: cuFFT INVERSE is un-normalized, just like FORWARD
    double norm = 1.0 / sqrt((double)g_cv_dim);
    kernel_apply_scalar<<<grid, block>>>(d_state, total_size, norm);
    checkCudaErrors(cudaDeviceSynchronize());
}

// ============================================================================
// C API - GATES
// ============================================================================

void cvdv_displacement(double beta_re, double beta_im) {
    // D(α) = exp(-i*Im(α)*Re(α)) * D(i*Im(α)) * D(Re(α))
    // D(i*p0) = exp(i*sqrt(2)*p0*q) - phase in position space
    // D(q0) = exp(-i*sqrt(2)*q0*p) - phase in momentum space
    
    int total_size = g_dv_dim * g_cv_dim;
    int block = 256;
    int grid = (total_size + block - 1) / block;

    // Step 1: Apply D(i*Im(α)) = exp(i*sqrt(2)*Im(α)*q) in position space
    if (fabs(beta_im) > 1e-12) {
        kernel_apply_phase<<<grid, block>>>(d_state, total_size,
                                             g_cv_dim, g_dx, SQRT2 * beta_im);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Step 2: Apply D(Re(α)) = exp(-i*sqrt(2)*Re(α)*p) in momentum space
    if (fabs(beta_re) > 1e-12) {
        // Transform to momentum space
        cvdv_ft_x_to_p();
        
        // Apply phase in momentum space
        kernel_apply_phase<<<grid, block>>>(d_state, total_size,
                                             g_cv_dim, g_dx, -SQRT2 * beta_re);
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Transform back to position space
        cvdv_ft_p_to_x();
    }
    
    // Note: Global phase exp(-i*Im(α)*Re(α)) is ignored
}

void cvdv_cd(double alpha_re, double alpha_im, int ctrl_qubit) {
    // Conditional displacement: CD(α) = CD(i*Im(α)) CD(Re(α))
    // CD(i*p0) = exp(i*sqrt(2)*p0*Z*q) - controlled phase in position space
    // CD(q0) = F^{-1} exp(-i*sqrt(2)*q0*Z*p) F - controlled phase in momentum space
    // where Z = |1⟩⟨1| - |0⟩⟨0|, so |1⟩ gets +α and |0⟩ gets -α
    
    int total_size = g_dv_dim * g_cv_dim;
    int block = 256;
    int grid = (total_size + block - 1) / block;

    // Step 1: Apply CD(i*Im(α)) = exp(i√2 Im(α) Z q) in position space
    if (fabs(alpha_im) > 1e-12) {
        kernel_apply_controlled_phase<<<grid, block>>>(d_state, total_size, g_cv_dim, g_dx,
                                                        SQRT2 * alpha_im, ctrl_qubit, g_dv_level);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // Step 2: Apply CD(Re(α)) = F^{-1} exp(-i√2 Re(α) Z p) F
    if (fabs(alpha_re) > 1e-12) {
        // Transform to momentum space
        cvdv_ft_x_to_p();
        
        // Apply exp(-i√2 Re(α) Z p) in momentum space
        kernel_apply_controlled_phase<<<grid, block>>>(d_state, total_size, g_cv_dim, g_dx,
                                                        -SQRT2 * alpha_re, ctrl_qubit, g_dv_level);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        
        // Transform back to position space
        cvdv_ft_p_to_x();
    }
}

void cvdv_pauli_rotation(int target_qubit, int axis, double theta) {
    int total_size = g_dv_dim * g_cv_dim;

    int block = 256;
    int grid = (total_size / 2 + block - 1) / block;

    kernel_pauli_rotation<<<grid, block>>>(d_state, total_size, g_cv_dim,
                                            target_qubit, g_dv_level, axis, theta);
    checkCudaErrors(cudaDeviceSynchronize());
}

void cvdv_hadamard(int target_qubit) {
    int total_size = g_dv_dim * g_cv_dim;
    int n_pairs = g_dv_dim / 2;

    int block = 256;
    int grid = (n_pairs * g_cv_dim + block - 1) / block;

    kernel_hadamard<<<grid, block>>>(d_state, total_size, g_cv_dim, target_qubit, g_dv_level);
    checkCudaErrors(cudaDeviceSynchronize());
}

// ============================================================================
// C API - STATE ACCESS
// ============================================================================



void cvdv_get_wigner(double* wigner_out, int wigner_n, double w_x_max, double w_p_max) {
    double* d_wigner;
    checkCudaErrors(cudaMalloc(&d_wigner, wigner_n * wigner_n * sizeof(double)));

    int block = 256;
    int grid = (wigner_n * wigner_n + block - 1) / block;

    kernel_compute_wigner<<<grid, block>>>(d_wigner, d_state, g_dv_dim, g_cv_dim,
                                            g_dx, wigner_n, w_x_max, w_p_max);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wigner_out, d_wigner, wigner_n * wigner_n * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(d_wigner);
}

void cvdv_get_wigner_single_slice(double* wigner_out, int slice_idx, 
                                   int wigner_n, double w_x_max, double w_p_max) {
    double* d_wigner;
    checkCudaErrors(cudaMalloc(&d_wigner, wigner_n * wigner_n * sizeof(double)));

    int block = 256;
    int grid = (wigner_n * wigner_n + block - 1) / block;

    kernel_compute_wigner_single_slice<<<grid, block>>>(d_wigner, d_state, slice_idx,
                                                         g_cv_dim, g_dx, wigner_n, 
                                                         w_x_max, w_p_max);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(wigner_out, d_wigner, wigner_n * wigner_n * sizeof(double),
                               cudaMemcpyDeviceToHost));

    cudaFree(d_wigner);
}

void cvdv_get_state(double* real_out, double* imag_out) {
    int total_size = g_dv_dim * g_cv_dim;

    cuDoubleComplex* h_state = new cuDoubleComplex[total_size];
    checkCudaErrors(cudaMemcpy(h_state, d_state, total_size * sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToHost));

    for (int i = 0; i < total_size; i++) {
        real_out[i] = cuCreal(h_state[i]);
        imag_out[i] = cuCimag(h_state[i]);
    }

    delete[] h_state;
}

// ============================================================================
// C API - GETTERS
// ============================================================================

int cvdv_get_dv_level() { return g_dv_level; }
int cvdv_get_dv_dim() { return g_dv_dim; }
int cvdv_get_cv_level() { return g_cv_level; }
int cvdv_get_cv_dim() { return g_cv_dim; }
double cvdv_get_dx() { return g_dx; }

void cvdv_get_dv_probs(double* probs_out) {
    // Compute probability (norm squared) for each DV slice
    double* d_norm;
    checkCudaErrors(cudaMalloc(&d_norm, sizeof(double)));
    
    for (int slice = 0; slice < g_dv_dim; slice++) {
        checkCudaErrors(cudaMemset(d_norm, 0, sizeof(double)));
        
        int block = 256;
        int grid = (g_cv_dim + block - 1) / block;
        kernel_compute_dv_norm<<<grid, block>>>(d_state, d_norm, slice, g_cv_dim);
        checkCudaErrors(cudaDeviceSynchronize());
        
        double norm;
        checkCudaErrors(cudaMemcpy(&norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
        probs_out[slice] = norm * g_dx;  // Multiply by dx for proper integration
    }
    
    cudaFree(d_norm);
}

}  // extern "C"
