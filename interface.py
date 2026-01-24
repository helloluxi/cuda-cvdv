"""
CVDV Library - Python wrapper for CUDA quantum simulator
"""

import numpy as np
import ctypes
from ctypes import c_int, c_double, c_size_t, POINTER
import subprocess
import os
from numpy import pi, sqrt

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
    
    # void cvdvAllocateRegisters(int numReg, int* numQubits)
    lib.cvdvAllocateRegisters.argtypes = [c_int, POINTER(c_int)]
    lib.cvdvAllocateRegisters.restype = None
    
    # void cvdvInitStateVector()
    lib.cvdvInitStateVector.argtypes = []
    lib.cvdvInitStateVector.restype = None
    
    # void cvdvSetUniform(int regIdx)
    lib.cvdvSetUniform.argtypes = [c_int]
    lib.cvdvSetUniform.restype = None
    
    # void cvdvFree()
    lib.cvdvFree.argtypes = []
    lib.cvdvFree.restype = None
    
    # void cvdvSetZero(int regIdx)
    lib.cvdvSetZero.argtypes = [c_int]
    lib.cvdvSetZero.restype = None
    
    # void cvdvSetCoherent(int regIdx, double alphaRe, double alphaIm)
    lib.cvdvSetCoherent.argtypes = [c_int, c_double, c_double]
    lib.cvdvSetCoherent.restype = None
    
    # void cvdvSetFock(int regIdx, int n)
    lib.cvdvSetFock.argtypes = [c_int, c_int]
    lib.cvdvSetFock.restype = None
    
    # void cvdvSetFocks(int regIdx, double* coeffsRe, double* coeffsIm, int length)
    lib.cvdvSetFocks.argtypes = [c_int, POINTER(c_double), POINTER(c_double), c_int]
    lib.cvdvSetFocks.restype = None
    
    # void cvdvDisplacement(int regIdx, double betaRe, double betaIm)
    lib.cvdvDisplacement.argtypes = [c_int, c_double, c_double]
    lib.cvdvDisplacement.restype = None
    
    # void cvdvConditionalDisplacement(int targetReg, int ctrlReg, int ctrlQubit, double alphaRe, double alphaIm)
    lib.cvdvConditionalDisplacement.argtypes = [c_int, c_int, c_int, c_double, c_double]
    lib.cvdvConditionalDisplacement.restype = None
    
    # void cvdvPauliRotation(int regIdx, int targetQubit, int axis, double theta)
    lib.cvdvPauliRotation.argtypes = [c_int, c_int, c_int, c_double]
    lib.cvdvPauliRotation.restype = None
    
    # void cvdvHadamard(int regIdx, int targetQubit)
    lib.cvdvHadamard.argtypes = [c_int, c_int]
    lib.cvdvHadamard.restype = None

    # void cvdvPhaseSquare(int regIdx, double t)
    lib.cvdvPhaseSquare.argtypes = [c_int, c_double]
    lib.cvdvPhaseSquare.restype = None

    # void cvdvRotation(int regIdx, double theta)
    lib.cvdvRotation.argtypes = [c_int, c_double]
    lib.cvdvRotation.restype = None

    # void cvdvSqueezing(int regIdx, double r)
    lib.cvdvSqueezing.argtypes = [c_int, c_double]
    lib.cvdvSqueezing.restype = None

    # void cvdvBeamSplitter(int reg1, int reg2, double theta)
    lib.cvdvBeamSplitter.argtypes = [c_int, c_int, c_double]
    lib.cvdvBeamSplitter.restype = None

    # void cvdvFtQ2P(int regIdx)
    lib.cvdvFtQ2P.argtypes = [c_int]
    lib.cvdvFtQ2P.restype = None
    
    # void cvdvFtP2Q(int regIdx)
    lib.cvdvFtP2Q.argtypes = [c_int]
    lib.cvdvFtP2Q.restype = None
    
    # void cvdvGetWignerSingleSlice(int regIdx, int* sliceIndices, double* wignerOut, int wignerN, double wXMax, double wPMax)
    lib.cvdvGetWignerSingleSlice.argtypes = [c_int, POINTER(c_int), POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetWignerSingleSlice.restype = None
    
    # void cvdvGetWignerFullMode(int regIdx, double* wignerOut, int wignerN, double wXMax, double wPMax)
    lib.cvdvGetWignerFullMode.argtypes = [c_int, POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetWignerFullMode.restype = None
    
    # void cvdvGetState(double* realOut, double* imagOut)
    lib.cvdvGetState.argtypes = [POINTER(c_double), POINTER(c_double)]
    lib.cvdvGetState.restype = None
    
    # int cvdvGetNumRegisters()
    lib.cvdvGetNumRegisters.argtypes = []
    lib.cvdvGetNumRegisters.restype = c_int
    
    # size_t cvdvGetTotalSize()
    lib.cvdvGetTotalSize.argtypes = []
    lib.cvdvGetTotalSize.restype = c_size_t
    
    # void cvdvGetRegisterInfo(int* qubitCountsOut, double* gridStepsOut)
    lib.cvdvGetRegisterInfo.argtypes = [POINTER(c_int), POINTER(c_double)]
    lib.cvdvGetRegisterInfo.restype = None
    
    # int cvdvGetRegisterDim(int regIdx)
    lib.cvdvGetRegisterDim.argtypes = [c_int]
    lib.cvdvGetRegisterDim.restype = c_int
    
    # double cvdvGetRegisterDx(int regIdx)
    lib.cvdvGetRegisterDx.argtypes = [c_int]
    lib.cvdvGetRegisterDx.restype = c_double
    
    # void cvdvMeasure(int regIdx, double* probabilitiesOut)
    lib.cvdvMeasure.argtypes = [c_int, POINTER(c_double)]
    lib.cvdvMeasure.restype = None
    
    # void cvdvInnerProduct(double* realOut, double* imagOut)
    lib.cvdvInnerProduct.argtypes = [POINTER(c_double), POINTER(c_double)]
    lib.cvdvInnerProduct.restype = None
    
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
        self.num_registers = len(numQubits_list)
        
        # Convert to C arrays
        numQubits_c = (c_int * self.num_registers)(*numQubits_list)
        
        # Allocate registers (this clears cuda.log and creates new session)
        lib.cvdvAllocateRegisters(self.num_registers, numQubits_c)
        
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
            lib.cvdvFree()
        except:
            pass
    
    def initStateVector(self):
        """Build tensor product state from register arrays and upload to device.
        
        Must be called after all setXXX functions and before any operations.
        """
        lib.cvdvInitStateVector()
    
    def setZero(self, regIdx):
        """Set register to |0⟩ state and upload to device."""
        lib.cvdvSetZero(regIdx)
    
    def setCoherent(self, regIdx, alpha):
        """Set register to coherent state |α⟩ and upload to device."""
        if isinstance(alpha, (int, float)):
            alpha = complex(alpha, 0.0)
        lib.cvdvSetCoherent(regIdx, c_double(alpha.real), c_double(alpha.imag))
    
    def setFock(self, regIdx, n):
        """Set register to Fock state |n⟩ and upload to device."""
        lib.cvdvSetFock(regIdx, n)
    
    def setUniform(self, regIdx):
        """Set register to uniform superposition: all basis states with amplitude 1/sqrt(N)."""
        lib.cvdvSetUniform(regIdx)
    
    def setFocks(self, regIdx, coeffs):
        """Set register to superposition of Fock states and upload to device.
        
        Args:
            regIdx: Register index
            coeffs: List/array of complex coefficients [c0, c1, c2, ...]
                   State will be: c0|0⟩ + c1|1⟩ + c2|2⟩ + ...
        """
        coeffs = np.array(coeffs, dtype=complex)
        coeffs_re = coeffs.real.astype(np.float64)
        coeffs_im = coeffs.imag.astype(np.float64)
        lib.cvdvSetFocks(regIdx,
                          coeffs_re.ctypes.data_as(POINTER(c_double)),
                          coeffs_im.ctypes.data_as(POINTER(c_double)),
                          len(coeffs))
    
    def displacement(self, regIdx, beta):
        """Apply displacement operator D(β) to register."""
        if isinstance(beta, (int, float)):
            beta = complex(beta, 0.0)
        lib.cvdvDisplacement(regIdx, c_double(beta.real), c_double(beta.imag))
    
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
        lib.cvdvConditionalDisplacement(targetReg, ctrlReg, ctrlQubit, 
                   c_double(alpha.real), c_double(alpha.imag))
    
    def pauliRotation(self, regIdx, targetQubit, axis, theta):
        """Apply Pauli rotation to qubit in register.
        
        Args:
            regIdx: Register index
            targetQubit: Qubit index within register
            axis: 0=X, 1=Y, 2=Z
            theta: Rotation angle
        """
        lib.cvdvPauliRotation(regIdx, targetQubit, axis, theta)
    
    def hadamard(self, regIdx, targetQubit):
        """Apply Hadamard gate to qubit in register."""
        lib.cvdvHadamard(regIdx, targetQubit)

    def phaseSquare(self, regIdx, t):
        """Apply phase square gate: exp(i*t*q^2) in position space.

        Args:
            regIdx: Register index
            t: Phase coefficient
        """
        lib.cvdvPhaseSquare(regIdx, t)

    def rotation(self, regIdx, theta):
        """Apply rotation gate R(θ) in phase space.

        Implements: R(θ) = exp(-i/2 tan(θ/2) q^2) exp(-i/2 sin(θ) p^2) exp(-i/2 tan(θ/2) q^2)
        This is a symplectic rotation in phase space by angle θ.

        Args:
            regIdx: Register index
            theta: Rotation angle in radians
        """
        lib.cvdvRotation(regIdx, theta)

    def squeezing(self, regIdx, r):
        """Apply squeezing gate S(r).

        Implements the squeezing operator decomposed into q^2 and p^2 phases.
        Positive r squeezes position and expands momentum.

        Args:
            regIdx: Register index
            r: Squeezing parameter
        """
        lib.cvdvSqueezing(regIdx, r)

    def beamSplitter(self, reg1, reg2, theta):
        """Apply beam splitter gate BS(θ) between two registers.

        Implements: BS(θ) = exp(-i*tan(θ/4)*q1*q2/2) * exp(-i*sin(θ/2)*p1*p2/2) * exp(-i*tan(θ/4)*q1*q2/2)
        where q and p are position and momentum operators.

        Args:
            reg1: First register index
            reg2: Second register index
            theta: Beam splitter angle in radians
        """
        lib.cvdvBeamSplitter(reg1, reg2, theta)

    def ftQ2P(self, regIdx):
        """Apply Fourier transform: position to momentum representation."""
        lib.cvdvFtQ2P(regIdx)
    
    def ftP2Q(self, regIdx):
        """Apply inverse Fourier transform: momentum to position representation."""
        lib.cvdvFtP2Q(regIdx)
    
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
        lib.cvdvGetWignerSingleSlice(regIdx,
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
        lib.cvdvGetWignerFullMode(regIdx,
            wigner.ctypes.data_as(POINTER(c_double)),
            wignerN, wXMax, wPMax
        )
        return wigner.reshape((wignerN, wignerN))
    
    def getState(self):
        """Get full state vector as complex array."""
        real_arr = np.zeros(self.total_size, dtype=np.float64)
        imag_arr = np.zeros(self.total_size, dtype=np.float64)
        lib.cvdvGetState(
            real_arr.ctypes.data_as(POINTER(c_double)),
            imag_arr.ctypes.data_as(POINTER(c_double))
        )
        return real_arr + 1j * imag_arr
    
    def getXGrid(self, regIdx):
        """Get position grid points for register."""
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        return (np.arange(dim) - dim // 2) * dx
    
    def measure(self, regIdx):
        """Compute measurement probabilities for all basis states of a register.
        
        Args:
            regIdx: Register index
        
        Returns:
            numpy array of probabilities for all basis states (dimension 2^numQubits)
        """
        dim = self.register_dims[regIdx]
        probs = np.zeros(dim, dtype=np.float64)
        lib.cvdvMeasure(regIdx, probs.ctypes.data_as(POINTER(c_double)))
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
        lib.cvdvInnerProduct(ctypes.byref(real_out), ctypes.byref(imag_out))
        return complex(real_out.value, imag_out.value)
    
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
