"""
CVDV Library - Python wrapper for CUDA quantum simulator
"""

import numpy as np
import ctypes
from ctypes import c_int, c_double, c_size_t, POINTER
import subprocess
import os
from numpy import pi, sqrt

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
plt.rcParams.update({'font.size': 18, 'text.usetex': True})

# Get project paths
project_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(project_dir, 'build')

def compile_library():
    """Compile the CUDA library."""
    result = subprocess.run(
        ['bash', os.path.join(project_dir, 'run.sh')],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('Build failed')
    print("Library compiled successfully!")

def load_library():
    """Load the compiled CUDA library and set up function signatures."""
    lib_path = os.path.join(build_dir, 'libcvdv.so')
    lib = ctypes.CDLL(lib_path)
    
    # Define C function signatures
    
    # CVDVContext* cvdvCreate(int numReg, int* numQubits)
    lib.cvdvCreate.argtypes = [c_int, POINTER(c_int)]
    lib.cvdvCreate.restype = ctypes.c_void_p
    
    # void cvdvDestroy(CVDVContext* ctx)
    lib.cvdvDestroy.argtypes = [ctypes.c_void_p]
    lib.cvdvDestroy.restype = None
    
    # void cvdvInitStateVector(CVDVContext* ctx)
    lib.cvdvInitStateVector.argtypes = [ctypes.c_void_p]
    lib.cvdvInitStateVector.restype = None
    
    # void cvdvSetUniform(CVDVContext* ctx, int regIdx)
    lib.cvdvSetUniform.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvSetUniform.restype = None
    
    # void cvdvFree(CVDVContext* ctx)
    lib.cvdvFree.argtypes = [ctypes.c_void_p]
    lib.cvdvFree.restype = None
    
    # void cvdvSetZero(CVDVContext* ctx, int regIdx)
    lib.cvdvSetZero.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvSetZero.restype = None
    
    # void cvdvSetCoherent(CVDVContext* ctx, int regIdx, double alphaRe, double alphaIm)
    lib.cvdvSetCoherent.argtypes = [ctypes.c_void_p, c_int, c_double, c_double]
    lib.cvdvSetCoherent.restype = None
    
    # void cvdvSetFock(CVDVContext* ctx, int regIdx, int n)
    lib.cvdvSetFock.argtypes = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvSetFock.restype = None
    
    # void cvdvSetFocks(CVDVContext* ctx, int regIdx, double* coeffs, int length)
    lib.cvdvSetFocks.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double), c_int]
    lib.cvdvSetFocks.restype = None
    
    # void cvdvSetCoeffs(CVDVContext* ctx, int regIdx, double* coeffs, int length)
    lib.cvdvSetCoeffs.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double), c_int]
    lib.cvdvSetCoeffs.restype = None

    # void cvdvSetCat(CVDVContext* ctx, int regIdx, double* data, int length)
    lib.cvdvSetCat.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double), c_int]
    lib.cvdvSetCat.restype = None

    # void cvdvDisplacement(CVDVContext* ctx, int regIdx, double betaRe, double betaIm)
    lib.cvdvDisplacement.argtypes = [ctypes.c_void_p, c_int, c_double, c_double]
    lib.cvdvDisplacement.restype = None
    
    # void cvdvConditionalDisplacement(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double alphaRe, double alphaIm)
    lib.cvdvConditionalDisplacement.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double, c_double]
    lib.cvdvConditionalDisplacement.restype = None
    
    # void cvdvPauliRotation(CVDVContext* ctx, int regIdx, int targetQubit, int axis, double theta)
    lib.cvdvPauliRotation.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvPauliRotation.restype = None
    
    # void cvdvHadamard(CVDVContext* ctx, int regIdx, int targetQubit)
    lib.cvdvHadamard.argtypes = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvHadamard.restype = None

    # void cvdvParity(CVDVContext* ctx, int regIdx)
    lib.cvdvParity.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvParity.restype = None

    # void cvdvConditionalParity(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit)
    lib.cvdvConditionalParity.argtypes = [ctypes.c_void_p, c_int, c_int, c_int]
    lib.cvdvConditionalParity.restype = None

    # void cvdvSwapRegisters(CVDVContext* ctx, int reg1, int reg2)
    lib.cvdvSwapRegisters.argtypes = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvSwapRegisters.restype = None

    # void cvdvPhaseSquare(CVDVContext* ctx, int regIdx, double t)
    lib.cvdvPhaseSquare.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvPhaseSquare.restype = None

    # void cvdvPhaseCubic(CVDVContext* ctx, int regIdx, double t)
    lib.cvdvPhaseCubic.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvPhaseCubic.restype = None

    # void cvdvRotation(CVDVContext* ctx, int regIdx, double theta)
    lib.cvdvRotation.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvRotation.restype = None

    # void cvdvConditionalRotation(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double theta)
    lib.cvdvConditionalRotation.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalRotation.restype = None

    # void cvdvSqueeze(CVDVContext* ctx, int regIdx, double r)
    lib.cvdvSqueeze.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvSqueeze.restype = None

    # void cvdvConditionalSqueeze(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit, double r)
    lib.cvdvConditionalSqueeze.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalSqueeze.restype = None

    # void cvdvBeamSplitter(CVDVContext* ctx, int reg1, int reg2, double theta)
    lib.cvdvBeamSplitter.argtypes = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvBeamSplitter.restype = None

    # void cvdvConditionalBeamSplitter(CVDVContext* ctx, int reg1, int reg2, int ctrlReg, int ctrlQubit, double theta)
    lib.cvdvConditionalBeamSplitter.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalBeamSplitter.restype = None

    # void cvdvQ1Q2Gate(CVDVContext* ctx, int reg1, int reg2, double coeff)
    lib.cvdvQ1Q2Gate.argtypes = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvQ1Q2Gate.restype = None

    # void cvdvFtQ2P(CVDVContext* ctx, int regIdx)
    lib.cvdvFtQ2P.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvFtQ2P.restype = None
    
    # void cvdvFtP2Q(CVDVContext* ctx, int regIdx)
    lib.cvdvFtP2Q.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvFtP2Q.restype = None
    
    # void cvdvGetWignerSingleSlice(CVDVContext* ctx, int regIdx, int* sliceIndices, double* wignerOut, int wignerN, double wXMax, double wPMax)
    lib.cvdvGetWignerSingleSlice.argtypes = [ctypes.c_void_p, c_int, POINTER(c_int), POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetWignerSingleSlice.restype = None
    
    # void cvdvGetWignerFullMode(CVDVContext* ctx, int regIdx, double* wignerOut, int wignerN, double wXMax, double wPMax)
    lib.cvdvGetWignerFullMode.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetWignerFullMode.restype = None
    
    # void cvdvGetHusimiQFullMode(CVDVContext* ctx, int regIdx, double* husimiQOut, int qN, double qMax, double pMax)
    lib.cvdvGetHusimiQFullMode.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetHusimiQFullMode.restype = None
    
    # void cvdvJointMeasure(CVDVContext* ctx, int reg1Idx, int reg2Idx, double* jointProbsOut)
    lib.cvdvJointMeasure.argtypes = [ctypes.c_void_p, c_int, c_int, POINTER(c_double)]
    lib.cvdvJointMeasure.restype = None
    
    # void cvdvGetState(CVDVContext* ctx, double* realOut, double* imagOut)
    lib.cvdvGetState.argtypes = [ctypes.c_void_p, POINTER(c_double), POINTER(c_double)]
    lib.cvdvGetState.restype = None
    
    # int cvdvGetNumRegisters(CVDVContext* ctx)
    lib.cvdvGetNumRegisters.argtypes = [ctypes.c_void_p]
    lib.cvdvGetNumRegisters.restype = c_int
    
    # size_t cvdvGetTotalSize(CVDVContext* ctx)
    lib.cvdvGetTotalSize.argtypes = [ctypes.c_void_p]
    lib.cvdvGetTotalSize.restype = c_size_t
    
    # void cvdvGetRegisterInfo(CVDVContext* ctx, int* qubitCountsOut, double* gridStepsOut)
    lib.cvdvGetRegisterInfo.argtypes = [ctypes.c_void_p, POINTER(c_int), POINTER(c_double)]
    lib.cvdvGetRegisterInfo.restype = None
    
    # int cvdvGetRegisterDim(CVDVContext* ctx, int regIdx)
    lib.cvdvGetRegisterDim.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvGetRegisterDim.restype = c_int
    
    # double cvdvGetRegisterDx(CVDVContext* ctx, int regIdx)
    lib.cvdvGetRegisterDx.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvGetRegisterDx.restype = c_double
    
    # void cvdvMeasure(CVDVContext* ctx, int regIdx, double* probabilitiesOut)
    lib.cvdvMeasure.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvMeasure.restype = None
    
    # void cvdvInnerProduct(CVDVContext* ctx, double* realOut, double* imagOut)
    lib.cvdvInnerProduct.argtypes = [ctypes.c_void_p, POINTER(c_double), POINTER(c_double)]
    lib.cvdvInnerProduct.restype = None

    # double cvdvGetNorm(CVDVContext* ctx)
    lib.cvdvGetNorm.argtypes = [ctypes.c_void_p]
    lib.cvdvGetNorm.restype = c_double

    print(f"Library loaded successfully!")
    print(f"Debug logs are written to: {os.path.join(project_dir, 'cuda.log')}")
    print("NOTE: Log file is cleared each time CVDV() is instantiated")
    
    return lib

# Compile and load library
compile_library()
lib = load_library()


class CVDV:
    """Python wrapper for CUDA quantum simulator (register-based API).
    
    All registers are treated uniformly as discrete quantum systems with dimensions 2^numQubits.
    Grid steps (dx) are automatically calculated for position-space operations.
    
    INITIALIZATION PATTERN:
    1. Create CVDV instance (allocates registers)
    2. Call setXXX functions to initialize each register
    3. Call initStateVector() to build the tensor product state
    """
    
    def __init__(self, numQubits_list):
        """Initialize simulator with multiple registers.
        
        Grid steps (dx) are calculated automatically inside CUDA using:
            reg_dim = 2^numQubits
            dx = sqrt(2 * pi / reg_dim)
            x_bound = sqrt(2 * pi * reg_dim)
        
        Args:
            numQubits_list: List of qubit counts for each register
                           Register dimension will be 2^numQubits
        
        Example (Single register with 4096 grid points):
            CVDV([12])  # 2^12 = 4096 points
        
        Example (Two registers: small + large):
            CVDV([1, 10])  # Register 0: 2 qubits, Register 1: 1024 points
        """
        # Convert to C arrays
        numQubits_c = (c_int * len(numQubits_list))(*numQubits_list)
        
        # Create context with registers (merged create+allocate)
        self.ctx = lib.cvdvCreate(len(numQubits_list), numQubits_c)
        
        self.num_registers = len(numQubits_list)
        
        # Note: User must call setXXX functions followed by initStateVector()
        # before performing any operations
        
        # Retrieve register info from host (device state not yet created)
        self.qubit_counts = np.zeros(self.num_registers, dtype=np.int32)
        self.grid_steps = np.zeros(self.num_registers, dtype=np.float64)
        
        # Compute dimensions locally (device not yet initialized)
        self.register_dims = [1 << qubits for qubits in numQubits_list]
        self.total_size = 1
        for dim in self.register_dims:
            self.total_size *= dim
        
        # Compute grid steps locally
        for i in range(self.num_registers):
            self.qubit_counts[i] = numQubits_list[i]
            dim = self.register_dims[i]
            self.grid_steps[i] = sqrt(2 * pi / dim)
    
    def __del__(self):
        """Free CUDA resources."""
        try:
            if hasattr(self, 'ctx') and self.ctx:
                lib.cvdvDestroy(self.ctx)
                self.ctx = None
        except:
            pass
    
    def initStateVector(self):
        """Build tensor product state from register arrays and upload to device.
        
        Must be called after all setXXX functions and before any operations.
        """
        lib.cvdvInitStateVector(self.ctx)
    
    def setZero(self, regIdx):
        """Set register to |0⟩ state and upload to device."""
        lib.cvdvSetZero(self.ctx, regIdx)
    
    def setCoherent(self, regIdx, alpha):
        """Set register to coherent state |α⟩ and upload to device."""
        if isinstance(alpha, (int, float)):
            alpha = complex(alpha, 0.0)
        lib.cvdvSetCoherent(self.ctx, regIdx, c_double(alpha.real), c_double(alpha.imag))
    
    def setFock(self, regIdx, n):
        """Set register to Fock state |n⟩ and upload to device."""
        lib.cvdvSetFock(self.ctx, regIdx, n)
    
    def setUniform(self, regIdx):
        """Set register to uniform superposition: all basis states with amplitude 1/sqrt(N)."""
        lib.cvdvSetUniform(self.ctx, regIdx)
    
    def setFocks(self, regIdx, coeffs):
        """Set register to superposition of Fock states and upload to device.
        
        Args:
            regIdx: Register index
            coeffs: List/array of complex coefficients [c0, c1, c2, ...]
                   State will be: c0|0⟩ + c1|1⟩ + c2|2⟩ + ...
        """
        coeffs = np.array(coeffs, dtype=complex)
        # Interleave real and imaginary parts: [re0, im0, re1, im1, ...]
        coeffs_interleaved = np.empty(2 * len(coeffs), dtype=np.float64)
        coeffs_interleaved[0::2] = coeffs.real
        coeffs_interleaved[1::2] = coeffs.imag
        lib.cvdvSetFocks(self.ctx, regIdx,
                          coeffs_interleaved.ctypes.data_as(POINTER(c_double)),
                          len(coeffs))
    
    def setCoeffs(self, regIdx, coeffs):
        """Set register to arbitrary coefficient array directly.

        Args:
            regIdx: Register index
            coeffs: Array of complex coefficients (must match register dimension)
                   Coefficients should be pre-normalized.
        """
        coeffs = np.array(coeffs, dtype=complex)
        # Interleave real and imaginary parts: [re0, im0, re1, im1, ...]
        coeffs_interleaved = np.empty(2 * len(coeffs), dtype=np.float64)
        coeffs_interleaved[0::2] = coeffs.real
        coeffs_interleaved[1::2] = coeffs.imag
        lib.cvdvSetCoeffs(self.ctx, regIdx,
                          coeffs_interleaved.ctypes.data_as(POINTER(c_double)),
                          len(coeffs))

    def setCat(self, regIdx, cat_states):
        """Set register to cat state (superposition of coherent states) and normalize.

        Args:
            regIdx: Register index
            cat_states: List of tuples [(alpha0, coeff0), (alpha1, coeff1), ...]
                       where alpha_i are complex coherent state amplitudes
                       and coeff_i are complex coefficients.
                       State will be normalized: (c0|α0⟩ + c1|α1⟩ + ...) / norm

        Example:
            # Create cat state: (|α⟩ + |-α⟩) / √2
            sim.setCat(0, [(2.0, 1.0), (-2.0, 1.0)])
        """
        cat_states = [(complex(alpha), complex(coeff)) for alpha, coeff in cat_states]

        # Interleave: [alphaRe, alphaIm, coeffRe, coeffIm, ...]
        data = np.empty(4 * len(cat_states), dtype=np.float64)
        for i, (alpha, coeff) in enumerate(cat_states):
            data[4*i] = alpha.real
            data[4*i+1] = alpha.imag
            data[4*i+2] = coeff.real
            data[4*i+3] = coeff.imag

        lib.cvdvSetCat(self.ctx, regIdx,
                       data.ctypes.data_as(POINTER(c_double)),
                       len(cat_states))

    def d(self, regIdx, beta):
        """Apply displacement operator D(β) to register."""
        if isinstance(beta, (int, float)):
            beta = complex(beta, 0.0)
        lib.cvdvDisplacement(self.ctx, regIdx, c_double(beta.real), c_double(beta.imag))
    
    def cd(self, targetReg, ctrlReg, ctrlQubit, alpha):
        """Apply conditional displacement CD(α) controlled by qubit.
        
        Args:
            targetReg: Target register index (receives displacement)
            ctrlReg: Control register index
            ctrlQubit: Qubit index within control register
            alpha: Complex displacement parameter
        """
        if isinstance(alpha, (int, float)):
            alpha = complex(alpha, 0.0)
        lib.cvdvConditionalDisplacement(self.ctx, targetReg, ctrlReg, ctrlQubit, 
                   c_double(alpha.real), c_double(alpha.imag))
    
    def cr(self, targetReg, ctrlReg, ctrlQubit, theta):
        """Apply conditional rotation CR(θ) controlled by qubit.

        Implements: CR(θ) = exp(-i/2 Z tan(θ/2) Q²) exp(-i/2 Z sin(θ) P²) exp(-i/2 Z tan(θ/2) Q²)
        where Z acts on ctrlQubit and Q,P act on targetReg.
        |0⟩ gets R(θ), |1⟩ gets R(-θ).

        Args:
            targetReg: Target register index (receives rotation)
            ctrlReg: Control register index
            ctrlQubit: Qubit index within control register
            theta: Rotation angle in radians
        """
        lib.cvdvConditionalRotation(self.ctx, targetReg, ctrlReg, ctrlQubit, c_double(theta))

    def x(self, regIdx, targetQubit):
        lib.cvdvPauliRotation(self.ctx, regIdx, targetQubit, 0, pi)

    def y(self, regIdx, targetQubit):
        lib.cvdvPauliRotation(self.ctx, regIdx, targetQubit, 1, pi)

    def z(self, regIdx, targetQubit):
        lib.cvdvPauliRotation(self.ctx, regIdx, targetQubit, 2, pi)

    def rx(self, regIdx, targetQubit, theta):
        lib.cvdvPauliRotation(self.ctx, regIdx, targetQubit, 0, theta)

    def ry(self, regIdx, targetQubit, theta):
        lib.cvdvPauliRotation(self.ctx, regIdx, targetQubit, 1, theta)

    def rz(self, regIdx, targetQubit, theta):
        lib.cvdvPauliRotation(self.ctx, regIdx, targetQubit, 2, theta)
    
    def h(self, regIdx, targetQubit):
        """Apply Hadamard gate to qubit in register."""
        lib.cvdvHadamard(self.ctx, regIdx, targetQubit)

    def p(self, regIdx):
        """Apply parity gate: flips all qubits of a register (|j⟩ → |N-1-j⟩)."""
        lib.cvdvParity(self.ctx, regIdx)

    def cp(self, targetReg, ctrlReg, ctrlQubit):
        """Apply conditional parity: identity on |0⟩, parity on |1⟩ control branch."""
        lib.cvdvConditionalParity(self.ctx, targetReg, ctrlReg, ctrlQubit)

    def swap(self, reg1, reg2):
        """Swap the contents of two registers (must have same number of qubits)."""
        lib.cvdvSwapRegisters(self.ctx, reg1, reg2)

    def sheer(self, regIdx, t):
        """Apply phase square gate: exp(i*t*q^2) in position space.

        Args:
            regIdx: Register index
            t: Phase coefficient
        """
        lib.cvdvPhaseSquare(self.ctx, regIdx, t)

    def phaseCubic(self, regIdx, t):
        """Apply cubic phase gate: exp(i*t*q^3) in position space.

        Args:
            regIdx: Register index
            t: Phase coefficient
        """
        lib.cvdvPhaseCubic(self.ctx, regIdx, t)

    def r(self, regIdx, theta):
        """Apply rotation gate R(θ) in phase space.

        Implements: R(θ) = exp(-i/2 tan(θ/2) q^2) exp(-i/2 sin(θ) p^2) exp(-i/2 tan(θ/2) q^2)
        This is a symplectic rotation in phase space by angle θ.

        Args:
            regIdx: Register index
            theta: Rotation angle in radians
        """
        lib.cvdvRotation(self.ctx, regIdx, theta)

    def s(self, regIdx, r):
        """Apply squeezing gate S(r).

        Implements the squeezing operator decomposed into q^2 and p^2 phases.
        Positive r squeezes position and expands momentum.

        Args:
            regIdx: Register index
            r: Squeezing parameter
        """
        lib.cvdvSqueeze(self.ctx, regIdx, r)

    def cs(self, targetReg, ctrlReg, ctrlQubit, r):
        """Apply conditional squeezing gate CS(r) controlled by qubit.

        |0⟩ gets S(r), |1⟩ gets S(-r).

        Args:
            targetReg: Target register index (receives squeezing)
            ctrlReg: Control register index
            ctrlQubit: Qubit index within control register
            r: Squeezing parameter
        """
        lib.cvdvConditionalSqueeze(self.ctx, targetReg, ctrlReg, ctrlQubit, c_double(r))

    def bs(self, reg1, reg2, theta):
        """Apply beam splitter gate BS(θ) between two registers.

        Implements: BS(θ) = exp(-i*tan(θ/4)*q1*q2/2) * exp(-i*sin(θ/2)*p1*p2/2) * exp(-i*tan(θ/4)*q1*q2/2)
        where q and p are position and momentum operators.

        Args:
            reg1: First register index
            reg2: Second register index
            theta: Beam splitter angle in radians
        """
        lib.cvdvBeamSplitter(self.ctx, reg1, reg2, theta)

    def cbs(self, reg1, reg2, ctrlReg, ctrlQubit, theta):
        """Apply conditional beam splitter CBS(θ) controlled by qubit.

        |0⟩ gets BS(θ), |1⟩ gets BS(-θ).

        Args:
            reg1: First register index
            reg2: Second register index
            ctrlReg: Control register index
            ctrlQubit: Qubit index within control register
            theta: Beam splitter angle in radians
        """
        lib.cvdvConditionalBeamSplitter(self.ctx, reg1, reg2, ctrlReg, ctrlQubit, c_double(theta))

    def q1q2(self, reg1, reg2, coeff):
        """Apply Q1Q2 interaction gate between two registers.

        Implements: exp(i*coeff*q1*q2) where q1 and q2 are position operators.

        Args:
            reg1: First register index
            reg2: Second register index
            coeff: Interaction coefficient
        """
        lib.cvdvQ1Q2Gate(self.ctx, reg1, reg2, coeff)

    def ftQ2P(self, regIdx):
        """Apply Fourier transform: position to momentum representation."""
        lib.cvdvFtQ2P(self.ctx, regIdx)
    
    def ftP2Q(self, regIdx):
        """Apply inverse Fourier transform: momentum to position representation."""
        lib.cvdvFtP2Q(self.ctx, regIdx)
    
    def getWignerSingleSlice(self, regIdx, slice_indices, wignerN=101, 
                             wXMax=5.0, wPMax=5.0):
        """Compute Wigner function for register at specific slice.
        
        Args:
            regIdx: Register index to compute Wigner function for
            slice_indices: List/array of basis states for each register.
                          slice_indices[regIdx] is ignored (can be any value).
                          For other registers, specifies which basis state to condition on.
        
        Example:
            For 2 registers, to get Wigner of register 1 when register 0 is in |0⟩:
            getWignerSingleSlice(1, [0, -1])  # -1 is placeholder for register 1
        """
        if len(slice_indices) != self.num_registers:
            raise ValueError(f"slice_indices must have length {self.num_registers}")
        
        slice_indices_arr = np.array(slice_indices, dtype=np.int32)
        wigner = np.zeros(wignerN * wignerN, dtype=np.float64)
        lib.cvdvGetWignerSingleSlice(self.ctx, regIdx,
            slice_indices_arr.ctypes.data_as(POINTER(c_int)),
            wigner.ctypes.data_as(POINTER(c_double)),
            wignerN, wXMax, wPMax
        )
        return wigner.reshape((wignerN, wignerN))
    
    def getWignerFullMode(self, regIdx, wignerN=101, wXMax=5.0, wPMax=5.0):
        """Compute reduced Wigner function for register by tracing out all other registers.
        
        This efficiently sums over all possible states of other registers to compute
        the reduced density matrix's Wigner function.
        
        Args:
            regIdx: Register index to compute Wigner function for
            wignerN: Grid size for Wigner function (default: 101)
            wXMax: Maximum position value (default: 5.0)
            wPMax: Maximum momentum value (default: 5.0)
        
        Returns:
            2D numpy array of shape (wignerN, wignerN) containing Wigner function W(q,p)
        """
        wigner = np.zeros(wignerN * wignerN, dtype=np.float64)
        lib.cvdvGetWignerFullMode(self.ctx, regIdx,
            wigner.ctypes.data_as(POINTER(c_double)),
            wignerN, wXMax, wPMax
        )
        return wigner.reshape((wignerN, wignerN))
    
    def getHusimiQFullMode(self, regIdx, qN=101, qMax=5.0, pMax=5.0):
        """Compute Husimi Q function for register by tracing out all other registers.
        
        The Husimi Q function is defined as Q(q,p) = (1/π) ⟨α|ρ|α⟩ where α = (q + ip)/√2.
        It represents the probability density for the state to be in coherent state |α⟩.
        
        Args:
            regIdx: Register index to compute Q function for
            qN: Grid size for Q function (default: 101)
            qMax: Maximum position value (default: 5.0)
            pMax: Maximum momentum value (default: 5.0)
        
        Returns:
            2D numpy array of shape (qN, qN) containing Husimi Q function Q(q,p)
        """
        husimiQ = np.zeros(qN * qN, dtype=np.float64)
        lib.cvdvGetHusimiQFullMode(self.ctx, regIdx,
            husimiQ.ctypes.data_as(POINTER(c_double)),
            qN, qMax, pMax
        )
        return husimiQ.reshape((qN, qN))
    
    def jointMeasure(self, reg1Idx, reg2Idx):
        """Compute joint measurement probabilities for two DV registers.
        
        This computes the joint probability distribution P(i,j) for measuring
        state |i⟩ in reg1 and state |j⟩ in reg2, tracing out all other registers.
        
        Args:
            reg1Idx: First register index
            reg2Idx: Second register index
        
        Returns:
            2D numpy array of shape (dim1, dim2) where dim1 and dim2 are the
            dimensions of reg1 and reg2 respectively. Element [i,j] is P(i,j).
        """
        dim1 = self.register_dims[reg1Idx]
        dim2 = self.register_dims[reg2Idx]
        jointProbs = np.zeros(dim1 * dim2, dtype=np.float64)
        lib.cvdvJointMeasure(self.ctx, reg1Idx, reg2Idx,
            jointProbs.ctypes.data_as(POINTER(c_double))
        )
        return jointProbs.reshape((dim1, dim2))
    
    def getState(self):
        """Get full state vector as complex array."""
        real_arr = np.zeros(self.total_size, dtype=np.float64)
        imag_arr = np.zeros(self.total_size, dtype=np.float64)
        lib.cvdvGetState(self.ctx,
            real_arr.ctypes.data_as(POINTER(c_double)),
            imag_arr.ctypes.data_as(POINTER(c_double))
        )
        return real_arr + 1j * imag_arr
    
    def getXGrid(self, regIdx):
        """Get position grid points for register."""
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        return (np.arange(dim) - dim // 2) * dx
    
    def m(self, regIdx):
        """Compute measurement probabilities for all basis states of a register.
        
        Args:
            regIdx: Register index
        
        Returns:
            numpy array of probabilities for all basis states (dimension 2^numQubits)
        """
        dim = self.register_dims[regIdx]
        probs = np.zeros(dim, dtype=np.float64)
        lib.cvdvMeasure(self.ctx, regIdx, probs.ctypes.data_as(POINTER(c_double)))
        return probs
    
    def innerProduct(self):
        """Compute inner product between current state and register tensor product.

        Computes <current_state | register_tensor_product> where register_tensor_product
        is the tensor product of all register arrays (the state that would be created by
        calling initStateVector with the current register contents).

        Useful for:
        - Verifying state initialization (should return 1.0 right after initStateVector)
        - Computing overlap between evolved state and initial state
        - Debugging and validation

        Returns:
            complex: Inner product value
        """
        real_out = c_double(0.0)
        imag_out = c_double(0.0)
        lib.cvdvInnerProduct(self.ctx, ctypes.byref(real_out), ctypes.byref(imag_out))
        return complex(real_out.value, imag_out.value)

    def getNorm(self):
        """Compute norm of the state vector (sum of |state[i]|^2).

        Returns:
            float: Norm value (should be 1.0 for normalized states)
        """
        return lib.cvdvGetNorm(self.ctx)

    def info(self):
        """Print system information."""
        # Calculate VRAM usage (complex double = 16 bytes per element)
        vram_gb = (self.total_size * 16) / (1024 * 1024 * 1024)
        print(f"Number of registers: {self.num_registers}")
        print(f"Total state size: {self.total_size} elements ({vram_gb:.3f} GB in VRAM)")
        for i in range(self.num_registers):
            dim = self.register_dims[i]
            dx = self.grid_steps[i]
            x_bound = sqrt(2 * pi * dim)
            print(f"  Register {i}: dim={dim}, "
                  f"qubits={self.qubit_counts[i]}, dx={dx:.6f}, x_bound={x_bound:.6f}")
    
    def plot_wigner(self, regIdx, slice_indices=None, wignerN=201, wignerMax=5.0, 
                    cmap='RdBu', figsize=(7, 6), show=True):
        """Plot Wigner function for a register."""
        # Get Wigner function
        if slice_indices is not None:
            wigner = self.getWignerSingleSlice(regIdx, slice_indices, 
                                              wignerN=wignerN, wXMax=wignerMax, wPMax=wignerMax)
        else:
            wigner = self.getWignerFullMode(regIdx, wignerN=wignerN, 
                                           wXMax=wignerMax, wPMax=wignerMax)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        vmax = np.max(np.abs(wigner))
        im = ax.imshow(wigner, extent=[-wignerMax, wignerMax, -wignerMax, wignerMax],
                      origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_xlabel(r'$q$')
        ax.set_ylabel(r'$p$')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, ax
