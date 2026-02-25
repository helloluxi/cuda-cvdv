"""
CVDV Library - Python wrapper for CUDA quantum simulator
"""

from typing import List, Tuple, Optional, Union, Sequence, Any
import numpy as np
import numpy.typing as npt
import ctypes
from ctypes import c_int, c_double, c_size_t, POINTER
import subprocess
import os
from numpy import pi, sqrt

import torch
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
plt.rcParams.update({'font.size': 18, 'text.usetex': True})

from .separable import SeparableState

# Get project paths (adjusted for src/ directory)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(project_dir, 'build')

# Lazy-loaded CUDA library (only compiled/loaded when backend='cuda' is first used)
_lib: Optional[ctypes.CDLL] = None

def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _compile_and_load()
    return _lib

def _compile_and_load() -> ctypes.CDLL:
    result = subprocess.run(
        ['make', '-C', project_dir, 'build'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError('Build failed')
    print("Library compiled successfully!")

    lib_path = os.path.join(build_dir, 'libcvdv.so')
    lib = ctypes.CDLL(lib_path)

    lib.cvdvCreate.argtypes = [c_int, POINTER(c_int)]
    lib.cvdvCreate.restype = ctypes.c_void_p
    lib.cvdvDestroy.argtypes = [ctypes.c_void_p]
    lib.cvdvDestroy.restype = None
    lib.cvdvInitFromSeparable.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), c_int]
    lib.cvdvInitFromSeparable.restype = None
    lib.cvdvFree.argtypes = [ctypes.c_void_p]
    lib.cvdvFree.restype = None
    lib.cvdvDisplacement.argtypes = [ctypes.c_void_p, c_int, c_double, c_double]
    lib.cvdvDisplacement.restype = None
    lib.cvdvConditionalDisplacement.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double, c_double]
    lib.cvdvConditionalDisplacement.restype = None
    lib.cvdvPauliRotation.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvPauliRotation.restype = None
    lib.cvdvHadamard.argtypes = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvHadamard.restype = None
    lib.cvdvParity.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvParity.restype = None
    lib.cvdvConditionalParity.argtypes = [ctypes.c_void_p, c_int, c_int, c_int]
    lib.cvdvConditionalParity.restype = None
    lib.cvdvSwapRegisters.argtypes = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvSwapRegisters.restype = None
    lib.cvdvPhaseSquare.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvPhaseSquare.restype = None
    lib.cvdvPhaseCubic.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvPhaseCubic.restype = None
    lib.cvdvRotation.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvRotation.restype = None
    lib.cvdvConditionalRotation.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalRotation.restype = None
    lib.cvdvSqueeze.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvSqueeze.restype = None
    lib.cvdvConditionalSqueeze.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalSqueeze.restype = None
    lib.cvdvBeamSplitter.argtypes = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvBeamSplitter.restype = None
    lib.cvdvConditionalBeamSplitter.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalBeamSplitter.restype = None
    lib.cvdvQ1Q2Gate.argtypes = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvQ1Q2Gate.restype = None
    lib.cvdvFtQ2P.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvFtQ2P.restype = None
    lib.cvdvFtP2Q.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvFtP2Q.restype = None
    lib.cvdvGetWignerSingleSlice.argtypes = [ctypes.c_void_p, c_int, POINTER(c_int), POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetWignerSingleSlice.restype = None
    lib.cvdvGetWignerFullMode.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetWignerFullMode.restype = None
    lib.cvdvGetHusimiQFullMode.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double), c_int, c_double, c_double]
    lib.cvdvGetHusimiQFullMode.restype = None
    lib.cvdvJointMeasure.argtypes = [ctypes.c_void_p, c_int, c_int, POINTER(c_double)]
    lib.cvdvJointMeasure.restype = None
    lib.cvdvGetState.argtypes = [ctypes.c_void_p, POINTER(c_double), POINTER(c_double)]
    lib.cvdvGetState.restype = None
    lib.cvdvGetNumRegisters.argtypes = [ctypes.c_void_p]
    lib.cvdvGetNumRegisters.restype = c_int
    lib.cvdvGetTotalSize.argtypes = [ctypes.c_void_p]
    lib.cvdvGetTotalSize.restype = c_size_t
    lib.cvdvGetRegisterInfo.argtypes = [ctypes.c_void_p, POINTER(c_int), POINTER(c_double)]
    lib.cvdvGetRegisterInfo.restype = None
    lib.cvdvGetRegisterDim.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvGetRegisterDim.restype = c_int
    lib.cvdvGetRegisterDx.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvGetRegisterDx.restype = c_double
    lib.cvdvMeasure.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvMeasure.restype = None
    lib.cvdvGetFidelity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), c_int, POINTER(c_double)]
    lib.cvdvGetFidelity.restype = None
    lib.cvdvGetNorm.argtypes = [ctypes.c_void_p]
    lib.cvdvGetNorm.restype = c_double
    lib.cvdvSetStateFromDevicePtr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cvdvSetStateFromDevicePtr.restype = None

    print(f"Library loaded successfully!")
    print(f"Debug logs are written to: {os.path.join(project_dir, 'cuda.log')}")
    print("NOTE: Log file is cleared each time CVDV() is instantiated")
    return lib


class CVDV:
    """Quantum simulator supporting cuda, torch-cuda, and torch-cpu backends.

    INITIALIZATION PATTERN:
        sim = CVDV([8, 1], backend='cuda')   # or 'torch-cuda' / 'torch-cpu'
        sep = SeparableState([8, 1])
        sep.setCoherent(0, 2.0)
        sep.setZero(1)
        sim.initStateVector(sep)
    """

    def __init__(self, numQubits_list: List[int],
                 backend: str = 'cuda') -> None:
        assert backend in ('cuda', 'torch-cuda', 'torch-cpu'), \
            f"backend must be 'cuda', 'torch-cuda', or 'torch-cpu', got {backend!r}"
        self.backend = backend
        self.num_registers = len(numQubits_list)
        self.register_dims = [1 << q for q in numQubits_list]
        self.total_size = 1
        for d in self.register_dims:
            self.total_size *= d
        self.qubit_counts = np.array(numQubits_list, dtype=np.int32)
        self.grid_steps = np.array(
            [sqrt(2 * pi / d) for d in self.register_dims], dtype=np.float64
        )

        if backend == 'cuda':
            lib = _get_lib()
            numQubits_c = (c_int * len(numQubits_list))(*numQubits_list)
            self.ctx = lib.cvdvCreate(len(numQubits_list), numQubits_c)
        else:
            
            dev = 'cuda' if backend == 'torch-cuda' else 'cpu'
            self.device = torch.device(dev)
            self.state: Any = None  # torch.Tensor, set by initStateVector

    def __del__(self):
        try:
            if self.backend == 'cuda' and hasattr(self, 'ctx') and self.ctx:
                _get_lib().cvdvDestroy(self.ctx)
                self.ctx = None
        except:
            pass

    # ==================== State Initialization ====================

    def initStateVector(self, sep: 'SeparableState') -> None:
        """Build tensor-product state from a SeparableState (all backends)."""
        self._initFromSeparable(sep)

    def _initFromSeparable(self, sep: 'SeparableState') -> None:
        sep.validate()
        if sep.num_registers != self.num_registers:
            raise ValueError(
                f"SeparableState has {sep.num_registers} registers, CVDV expects {self.num_registers}"
            )
        for i, (sq, rq) in enumerate(zip(sep.qubit_counts, self.qubit_counts)):
            if sq != rq:
                raise ValueError(
                    f"Register {i}: SeparableState has {sq} qubits, CVDV has {rq}"
                )

        if self.backend == 'cuda':
            lib = _get_lib()
            ptr_arr = (ctypes.c_void_p * self.num_registers)(
                *[ctypes.c_void_p(arr.data_ptr()) for arr in sep.register_arrays]  # type: ignore[union-attr]
            )
            lib.cvdvInitFromSeparable(self.ctx, ptr_arr, c_int(self.num_registers))

        else:
            
            arrays = [arr.to(self.device) for arr in sep.register_arrays]  # type: ignore[union-attr]
            if self.num_registers == 1:
                state = arrays[0].clone()
            else:
                state = torch.outer(arrays[0], arrays[1])
                for i in range(2, self.num_registers):
                    state = state.unsqueeze(-1) * arrays[i].reshape(*([1] * i), -1)
            norm = torch.sqrt(torch.sum(torch.abs(state) ** 2))
            self.state = state / norm

    # ==================== Gate Operations ====================
    # All gates dispatch on self.backend.

    def d(self, regIdx: int, beta: Union[complex, float, int]) -> None:
        """Apply displacement operator D(β) to register."""
        if isinstance(beta, (int, float)):
            beta = complex(beta, 0.0)
        if self.backend == 'cuda':
            _get_lib().cvdvDisplacement(self.ctx, regIdx, c_double(beta.real), c_double(beta.imag))
        else:
            if abs(beta.imag) > 1e-12:
                
                x = self._tPositionGrid(regIdx)
                phase = torch.exp(1j * sqrt(2) * beta.imag * x).to(torch.cdouble)
                self._tApplyPhase(regIdx, phase)
            if abs(beta.real) > 1e-12:
                self.ftQ2P(regIdx)
                
                p = self._tPositionGrid(regIdx)
                phase = torch.exp(-1j * sqrt(2) * beta.real * p).to(torch.cdouble)
                self._tApplyPhase(regIdx, phase)
                self.ftP2Q(regIdx)

    def cd(self, targetReg: int, ctrlReg: int, ctrlQubit: int, alpha: Union[complex, float, int]) -> None:
        """Apply conditional displacement CD(α) controlled by qubit."""
        if isinstance(alpha, (int, float)):
            alpha = complex(alpha)
        if self.backend == 'cuda':
            _get_lib().cvdvConditionalDisplacement(self.ctx, targetReg, ctrlReg, ctrlQubit,
                       c_double(alpha.real), c_double(alpha.imag))
        else:
            if abs(alpha.imag) > 1e-12:
                self._tApplyCondPhaseQ(targetReg, ctrlReg, ctrlQubit, sqrt(2) * alpha.imag)
            if abs(alpha.real) > 1e-12:
                self.ftQ2P(targetReg)
                self._tApplyCondPhaseQ(targetReg, ctrlReg, ctrlQubit, -sqrt(2) * alpha.real)
                self.ftP2Q(targetReg)

    def cr(self, targetReg: int, ctrlReg: int, ctrlQubit: int, theta: float) -> None:
        """Apply conditional rotation CR(θ) controlled by qubit."""
        if self.backend == 'cuda':
            _get_lib().cvdvConditionalRotation(self.ctx, targetReg, ctrlReg, ctrlQubit, c_double(theta))
        else:
            ratio = theta / (pi / 2)
            theta0 = int(np.floor(ratio + 0.5)) * (pi / 2)
            remainder = theta - theta0
            quarter_turns = (int(np.floor(ratio + 0.5)) % 4 + 4) % 4
            if quarter_turns == 1:
                self.ftQ2P(targetReg)
                self.cp(targetReg, ctrlReg, ctrlQubit)
                self.rz(ctrlReg, ctrlQubit, pi / 2)
            elif quarter_turns == 2:
                self.p(targetReg)
                self.rz(ctrlReg, ctrlQubit, pi)
            elif quarter_turns == 3:
                self.ftP2Q(targetReg)
                self.cp(targetReg, ctrlReg, ctrlQubit)
                self.rz(ctrlReg, ctrlQubit, -pi / 2)
            if abs(remainder) > 1e-15:
                tan_half = np.tan(remainder / 2)
                sin_theta = np.sin(remainder)
                self._tApplyCondPhaseQ2(targetReg, ctrlReg, ctrlQubit, -0.5 * tan_half)
                self.ftQ2P(targetReg)
                self._tApplyCondPhaseQ2(targetReg, ctrlReg, ctrlQubit, -0.5 * sin_theta)
                self.ftP2Q(targetReg)
                self._tApplyCondPhaseQ2(targetReg, ctrlReg, ctrlQubit, -0.5 * tan_half)

    def x(self, regIdx: int, targetQubit: int) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 0, pi)
        else:
            self.rx(regIdx, targetQubit, pi)

    def y(self, regIdx: int, targetQubit: int) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 1, pi)
        else:
            self.ry(regIdx, targetQubit, pi)

    def z(self, regIdx: int, targetQubit: int) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 2, pi)
        else:
            self.rz(regIdx, targetQubit, pi)

    def rx(self, regIdx: int, targetQubit: int, theta: float) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 0, theta)
        else:
            
            c = np.cos(theta / 2); s = np.sin(theta / 2)
            mat = torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=torch.cdouble, device=self.device)
            self._tApplyQubitGate(regIdx, targetQubit, mat)

    def ry(self, regIdx: int, targetQubit: int, theta: float) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 1, theta)
        else:
            
            c = np.cos(theta / 2); s = np.sin(theta / 2)
            mat = torch.tensor([[c, -s], [s, c]], dtype=torch.cdouble, device=self.device)
            self._tApplyQubitGate(regIdx, targetQubit, mat)

    def rz(self, regIdx: int, targetQubit: int, theta: float) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 2, theta)
        else:
            
            mat = torch.tensor([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]],
                                dtype=torch.cdouble, device=self.device)
            self._tApplyQubitGate(regIdx, targetQubit, mat)

    def h(self, regIdx: int, targetQubit: int) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvHadamard(self.ctx, regIdx, targetQubit)
        else:
            
            mat = torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble, device=self.device) / sqrt(2)
            self._tApplyQubitGate(regIdx, targetQubit, mat)

    def p(self, regIdx: int) -> None:
        """Apply parity gate: flips all qubits of a register (|j⟩ → |N-1-j⟩)."""
        if self.backend == 'cuda':
            _get_lib().cvdvParity(self.ctx, regIdx)
        else:
            
            self.state = torch.flip(self.state, dims=[regIdx])

    def cp(self, targetReg: int, ctrlReg: int, ctrlQubit: int) -> None:
        """Apply conditional parity."""
        if self.backend == 'cuda':
            _get_lib().cvdvConditionalParity(self.ctx, targetReg, ctrlReg, ctrlQubit)
        else:
            
            ctrl_dim = self.register_dims[ctrlReg]
            n_ctrl = self.qubit_counts[ctrlReg]
            perm = list(range(self.num_registers))
            perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
            actual_target = targetReg if targetReg != 0 else ctrlReg
            perm[1], perm[actual_target] = perm[actual_target], perm[1]
            state = self.state.permute(*perm)
            new_shape = [2] * n_ctrl + list(state.shape[1:])
            state = state.reshape(new_shape)
            qperm = list(range(n_ctrl))
            qperm[0], qperm[ctrlQubit] = qperm[ctrlQubit], qperm[0]
            state = state.permute(*qperm + list(range(n_ctrl, len(new_shape))))
            state[1] = torch.flip(state[1], dims=[n_ctrl - 1])
            state = state.permute(*[qperm.index(i) for i in range(n_ctrl)] + list(range(n_ctrl, len(state.shape))))
            state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl:]))
            state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
            self.state = state

    def swap(self, reg1: int, reg2: int) -> None:
        """Swap the contents of two registers (must have same number of qubits)."""
        if self.backend == 'cuda':
            _get_lib().cvdvSwapRegisters(self.ctx, reg1, reg2)
        else:
            if self.qubit_counts[reg1] != self.qubit_counts[reg2]:
                raise ValueError(f"SWAP requires registers with same number of qubits")
            perm = list(range(self.num_registers))
            perm[reg1], perm[reg2] = perm[reg2], perm[reg1]
            self.state = self.state.permute(*perm)

    def sheer(self, regIdx: int, t: float) -> None:
        """Apply phase square gate: exp(i*t*q^2) in position space."""
        if self.backend == 'cuda':
            _get_lib().cvdvPhaseSquare(self.ctx, regIdx, t)
        else:
            
            x = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * t * x ** 2).to(torch.cdouble))

    def phaseCubic(self, regIdx: int, t: float) -> None:
        """Apply cubic phase gate: exp(i*t*q^3) in position space."""
        if self.backend == 'cuda':
            _get_lib().cvdvPhaseCubic(self.ctx, regIdx, t)
        else:
            
            x = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * t * x ** 3).to(torch.cdouble))

    def r(self, regIdx: int, theta: float) -> None:
        """Apply rotation gate R(θ) in phase space."""
        if self.backend == 'cuda':
            _get_lib().cvdvRotation(self.ctx, regIdx, theta)
        else:
            ratio = theta / (pi / 2)
            theta0 = int(np.floor(ratio + 0.5)) * (pi / 2)
            remainder = theta - theta0
            quarter_turns = (int(np.floor(ratio + 0.5)) % 4 + 4) % 4
            if quarter_turns == 1:
                self.ftQ2P(regIdx)
            elif quarter_turns == 2:
                self.p(regIdx)
            elif quarter_turns == 3:
                self.ftP2Q(regIdx)
            if abs(remainder) > 1e-15:
                self.sheer(regIdx, -0.5 * np.tan(remainder / 2))
                self.ftQ2P(regIdx)
                self.sheer(regIdx, -0.5 * np.sin(remainder))
                self.ftP2Q(regIdx)
                self.sheer(regIdx, -0.5 * np.tan(remainder / 2))

    def s(self, regIdx: int, r: float) -> None:
        """Apply squeezing gate S(r)."""
        if self.backend == 'cuda':
            _get_lib().cvdvSqueeze(self.ctx, regIdx, r)
        else:
            exp_r = np.exp(r)
            exp_mr = np.exp(-r)
            t = np.exp(-r / 2.0) * np.sqrt(abs(1.0 - exp_mr))
            self.sheer(regIdx, 0.5 * t)
            self.ftQ2P(regIdx)
            self.sheer(regIdx, (1.0 - exp_mr) / (2.0 * t))
            self.ftP2Q(regIdx)
            self.sheer(regIdx, -0.5 * t * exp_r)
            self.ftQ2P(regIdx)
            self.sheer(regIdx, (exp_mr - 1.0) / (2.0 * t * exp_r))
            self.ftP2Q(regIdx)

    def cs(self, targetReg: int, ctrlReg: int, ctrlQubit: int, r: float) -> None:
        """Apply conditional squeezing gate CS(r) controlled by qubit."""
        if self.backend == 'cuda':
            _get_lib().cvdvConditionalSqueeze(self.ctx, targetReg, ctrlReg, ctrlQubit, c_double(r))
        else:
            ch_r = np.cosh(r); sh_r = np.sinh(r)
            sv = np.sqrt(2.0 * abs(np.sinh(0.5 * r)))
            self.sheer(targetReg, 0.5 * sv * ch_r)
            self._tApplyCondPhaseQ2(targetReg, ctrlReg, ctrlQubit, -0.5 * sv * sh_r)
            self.ftQ2P(targetReg)
            self.sheer(targetReg, 0.5 * (ch_r - 1) / sv)
            self._tApplyCondPhaseQ2(targetReg, ctrlReg, ctrlQubit, 0.5 * sh_r / sv)
            self.ftP2Q(targetReg)
            self.sheer(targetReg, -0.5 * sv)
            self.ftQ2P(targetReg)
            self.sheer(targetReg, 0.5 * (ch_r - 1) / sv)
            self._tApplyCondPhaseQ2(targetReg, ctrlReg, ctrlQubit, -0.5 * sh_r / sv)
            self.ftP2Q(targetReg)

    def bs(self, reg1: int, reg2: int, theta: float) -> None:
        """Apply beam splitter gate BS(θ) between two registers."""
        if self.backend == 'cuda':
            _get_lib().cvdvBeamSplitter(self.ctx, reg1, reg2, theta)
        else:
            ratio = theta / pi
            theta0 = int(np.floor(ratio + 0.5)) * pi
            remainder = theta - theta0
            half_turns = (int(np.floor(ratio + 0.5)) % 4 + 4) % 4
            if half_turns == 1:
                self.ftQ2P(reg1); self.ftQ2P(reg2); self.swap(reg1, reg2)
            elif half_turns == 2:
                self.p(reg1); self.p(reg2)
            elif half_turns == 3:
                self.ftP2Q(reg1); self.ftP2Q(reg2); self.swap(reg1, reg2)
            if abs(remainder) > 1e-15:
                tq = np.tan(remainder / 4); sh = np.sin(remainder / 2)
                self.q1q2(reg1, reg2, -tq)
                self.ftQ2P(reg1); self.ftQ2P(reg2)
                self.q1q2(reg1, reg2, -sh)
                self.ftP2Q(reg1); self.ftP2Q(reg2)
                self.q1q2(reg1, reg2, -tq)

    def cbs(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, theta: float) -> None:
        """Apply conditional beam splitter CBS(θ) controlled by qubit."""
        if self.backend == 'cuda':
            _get_lib().cvdvConditionalBeamSplitter(self.ctx, reg1, reg2, ctrlReg, ctrlQubit, c_double(theta))
        else:
            ratio = theta / pi
            theta0 = int(np.floor(ratio + 0.5)) * pi
            remainder = theta - theta0
            half_turns = (int(np.floor(ratio + 0.5)) % 4 + 4) % 4
            if half_turns == 1:
                self.ftQ2P(reg1); self.ftQ2P(reg2)
                self.cp(reg1, ctrlReg, ctrlQubit); self.cp(reg2, ctrlReg, ctrlQubit)
                self.swap(reg1, reg2)
            elif half_turns == 2:
                self.p(reg1); self.p(reg2)
            elif half_turns == 3:
                self.ftP2Q(reg1); self.ftP2Q(reg2)
                self.cp(reg1, ctrlReg, ctrlQubit); self.cp(reg2, ctrlReg, ctrlQubit)
                self.swap(reg1, reg2)
            if abs(remainder) > 1e-15:
                tq = np.tan(remainder / 4); sh = np.sin(remainder / 2)
                self._tApplyCondQ1Q2(reg1, reg2, ctrlReg, ctrlQubit, -tq)
                self.ftQ2P(reg1); self.ftQ2P(reg2)
                self._tApplyCondQ1Q2(reg1, reg2, ctrlReg, ctrlQubit, -sh)
                self.ftP2Q(reg1); self.ftP2Q(reg2)
                self._tApplyCondQ1Q2(reg1, reg2, ctrlReg, ctrlQubit, -tq)

    def q1q2(self, reg1: int, reg2: int, coeff: float) -> None:
        """Apply Q1Q2 interaction gate: exp(i*coeff*q1*q2)."""
        if self.backend == 'cuda':
            _get_lib().cvdvQ1Q2Gate(self.ctx, reg1, reg2, coeff)
        else:
            
            q1 = self._tPositionGrid(reg1)
            q2 = self._tPositionGrid(reg2)
            phase_matrix = torch.exp(1j * coeff * q1[:, None] * q2[None, :]).to(torch.cdouble)
            shape = [1] * self.num_registers
            shape[reg1] = self.register_dims[reg1]
            shape[reg2] = self.register_dims[reg2]
            self.state = self.state * phase_matrix.reshape(shape)

    def ftQ2P(self, regIdx: int) -> None:
        """Apply Fourier transform: position to momentum representation."""
        if self.backend == 'cuda':
            _get_lib().cvdvFtQ2P(self.ctx, regIdx)
        else:
            
            dx = self.grid_steps[regIdx]; dim = self.register_dims[regIdx]
            phaseCoeff = pi * (dim - 1.0) / (dim * dx)
            x = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * x).to(torch.cdouble))
            self.state = torch.fft.fft(self.state, dim=regIdx)
            p = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * p).to(torch.cdouble))
            self.state = self.state / sqrt(dim)

    def ftP2Q(self, regIdx: int) -> None:
        """Apply inverse Fourier transform: momentum to position representation."""
        if self.backend == 'cuda':
            _get_lib().cvdvFtP2Q(self.ctx, regIdx)
        else:
            
            dx = self.grid_steps[regIdx]; dim = self.register_dims[regIdx]
            phaseCoeff = -pi * (dim - 1.0) / (dim * dx)
            p = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * p).to(torch.cdouble))
            self.state = torch.fft.ifft(self.state, dim=regIdx, norm='forward')
            x = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * x).to(torch.cdouble))
            self.state = self.state / sqrt(dim)

    # ==================== Measurements & Observables ====================

    def getState(self) -> npt.NDArray[np.complex128]:
        """Get full state vector as complex array."""
        if self.backend == 'cuda':
            real_arr = np.zeros(self.total_size, dtype=np.float64)
            imag_arr = np.zeros(self.total_size, dtype=np.float64)
            _get_lib().cvdvGetState(self.ctx,
                real_arr.ctypes.data_as(POINTER(c_double)),
                imag_arr.ctypes.data_as(POINTER(c_double))
            )
            return real_arr + 1j * imag_arr
        else:
            return self.state.flatten().cpu().numpy()

    def getXGrid(self, regIdx: int) -> npt.NDArray[np.float64]:
        """Get position grid points for register."""
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        return (np.arange(dim) - (dim - 1) * 0.5) * dx

    def m(self, regIdx: int) -> npt.NDArray[np.float64]:
        """Compute measurement probabilities for all basis states of a register."""
        if self.backend == 'cuda':
            dim = self.register_dims[regIdx]
            probs = np.zeros(dim, dtype=np.float64)
            _get_lib().cvdvMeasure(self.ctx, regIdx, probs.ctypes.data_as(POINTER(c_double)))
            return probs
        else:
            
            probs = torch.abs(self.state) ** 2
            dims_to_sum = [i for i in range(self.num_registers) if i != regIdx]
            for d in sorted(dims_to_sum, reverse=True):
                probs = torch.sum(probs, dim=d)
            return probs.cpu().numpy()

    def jointMeasure(self, reg1Idx: int, reg2Idx: int) -> npt.NDArray[np.float64]:
        """Compute joint measurement probabilities for two registers."""
        if self.backend == 'cuda':
            dim1 = self.register_dims[reg1Idx]; dim2 = self.register_dims[reg2Idx]
            jointProbs = np.zeros(dim1 * dim2, dtype=np.float64)
            _get_lib().cvdvJointMeasure(self.ctx, reg1Idx, reg2Idx,
                jointProbs.ctypes.data_as(POINTER(c_double))
            )
            return jointProbs.reshape((dim1, dim2))
        else:
            
            probs = torch.abs(self.state) ** 2
            dims_to_sum = [i for i in range(self.num_registers) if i not in [reg1Idx, reg2Idx]]
            for d in sorted(dims_to_sum, reverse=True):
                probs = torch.sum(probs, dim=d)
            if reg1Idx > reg2Idx:
                probs = probs.transpose(0, 1)
            return probs.cpu().numpy()

    def getFidelity(self, sep: 'SeparableState') -> float:
        """Compute |⟨sep|ψ⟩|² between the current state and a SeparableState."""
        sep.validate()
        if self.backend == 'cuda':
            ptr_arr = (ctypes.c_void_p * self.num_registers)(
                *[ctypes.c_void_p(arr.data_ptr()) for arr in sep.register_arrays]  # type: ignore[union-attr]
            )
            fid_out = c_double(0.0)
            _get_lib().cvdvGetFidelity(self.ctx, ptr_arr, c_int(self.num_registers),
                                       ctypes.byref(fid_out))
            return float(fid_out.value)
        else:
            arrays = [arr.to(self.device) for arr in sep.register_arrays]  # type: ignore[union-attr]
            if self.num_registers == 1:
                ref = arrays[0]
            else:
                ref = torch.outer(arrays[0], arrays[1])
                for i in range(2, self.num_registers):
                    ref = ref.unsqueeze(-1) * arrays[i].reshape(*([1] * i), -1)
            ref_flat = ref.reshape(-1)
            inner = torch.sum(torch.conj(ref_flat) * self.state.reshape(-1))
            return float(torch.abs(inner).item() ** 2)

    def getNorm(self) -> float:
        """Compute norm of the state vector."""
        if self.backend == 'cuda':
            return _get_lib().cvdvGetNorm(self.ctx)
        else:
            
            return float(torch.sqrt(torch.sum(torch.abs(self.state) ** 2)).cpu().item())

    # ==================== Phase Space Functions ====================

    def getWignerSingleSlice(self, regIdx: int, slice_indices: Sequence[int], wignerN: int = 101,
                             wXMax: float = 5.0, wPMax: float = 5.0) -> npt.NDArray[np.float64]:
        """Compute Wigner function for register at specific slice."""
        if len(slice_indices) != self.num_registers:
            raise ValueError(f"slice_indices must have length {self.num_registers}")
        if self.backend == 'cuda':
            slice_indices_arr = np.array(slice_indices, dtype=np.int32)
            wigner = np.zeros(wignerN * wignerN, dtype=np.float64)
            _get_lib().cvdvGetWignerSingleSlice(self.ctx, regIdx,
                slice_indices_arr.ctypes.data_as(POINTER(c_int)),
                wigner.ctypes.data_as(POINTER(c_double)),
                wignerN, wXMax, wPMax
            )
            return wigner.reshape((wignerN, wignerN))
        else:
            
            perm = list(range(self.num_registers))
            perm[0], perm[regIdx] = perm[regIdx], perm[0]
            state = self.state.permute(*perm)
            for i in range(1, self.num_registers):
                actual_reg = perm[i]
                state = state.select(i, slice_indices[actual_reg])
            psi = state.cpu().numpy()
            dim = self.register_dims[regIdx]; dx = self.grid_steps[regIdx]
            x_grid = np.linspace(-wXMax, wXMax, wignerN)
            p_grid = np.linspace(-wPMax, wPMax, wignerN)
            fft_results = np.zeros((wignerN, dim), dtype=np.complex128)
            for i, x in enumerate(x_grid):
                integrand = np.zeros(dim, dtype=np.complex128)
                for j in range(dim):
                    y = (j - (dim - 1) / 2) * dx
                    ip = int(round((x + y) / dx)) + dim // 2
                    im = int(round((x - y) / dx)) + dim // 2
                    if 0 <= ip < dim and 0 <= im < dim:
                        integrand[j] = np.conj(psi[ip]) * psi[im]
                fft_results[i, :] = np.fft.ifft(integrand) * dim
            wigner = np.zeros((wignerN, wignerN), dtype=np.float64)
            dp = pi / (dim * dx)
            for k, p in enumerate(p_grid):
                k_s = int(round(p / dp + dim / 2))
                k_s = max(0, min(dim - 1, k_s))
                k_fft = (k_s + dim // 2) % dim
                p_act = (k_s - dim / 2) * dp
                phase_corr = np.exp(-1j * p_act * (dim - 1) * dx)
                for i in range(wignerN):
                    wigner[k, i] = np.real(phase_corr * fft_results[i, k_fft]) * dx / pi
            return wigner

    def getWignerFullMode(self, regIdx: int, wignerN: int = 101, wXMax: float = 5.0, wPMax: float = 5.0) -> npt.NDArray[np.float64]:
        """Compute reduced Wigner function by tracing out all other registers."""
        if self.backend == 'cuda':
            wigner = np.zeros(wignerN * wignerN, dtype=np.float64)
            _get_lib().cvdvGetWignerFullMode(self.ctx, regIdx,
                wigner.ctypes.data_as(POINTER(c_double)),
                wignerN, wXMax, wPMax
            )
            return wigner.reshape((wignerN, wignerN))
        else:
            
            perm = list(range(self.num_registers))
            perm[0], perm[regIdx] = perm[regIdx], perm[0]
            state = self.state.permute(*perm)
            dim = self.register_dims[regIdx]; other_size = self.total_size // dim
            psi = state.reshape(dim, other_size).cpu().numpy()
            dx = self.grid_steps[regIdx]
            x_grid = np.linspace(-wXMax, wXMax, wignerN)
            p_grid = np.linspace(-wPMax, wPMax, wignerN)
            fft_results = np.zeros((wignerN, dim), dtype=np.complex128)
            for i, x in enumerate(x_grid):
                integrand = np.zeros(dim, dtype=np.complex128)
                for j in range(dim):
                    y = (j - (dim - 1) / 2) * dx
                    ip = int(round((x + y) / dx)) + dim // 2
                    im = int(round((x - y) / dx)) + dim // 2
                    if 0 <= ip < dim and 0 <= im < dim:
                        integrand[j] = np.dot(np.conj(psi[ip, :]), psi[im, :])
                fft_results[i, :] = np.fft.ifft(integrand) * dim
            wigner = np.zeros((wignerN, wignerN), dtype=np.float64)
            dp = pi / (dim * dx)
            for k, p in enumerate(p_grid):
                k_s = int(round(p / dp + dim / 2))
                k_s = max(0, min(dim - 1, k_s))
                k_fft = (k_s + dim // 2) % dim
                p_act = (k_s - dim / 2) * dp
                phase_corr = np.exp(-1j * p_act * (dim - 1) * dx)
                for i in range(wignerN):
                    wigner[k, i] = np.real(phase_corr * fft_results[i, k_fft]) * dx / pi
            return wigner

    def getHusimiQFullMode(self, regIdx: int, qN: int = 101, qMax: float = 5.0, pMax: float = 5.0) -> npt.NDArray[np.float64]:
        """Compute Husimi Q function by tracing out all other registers."""
        if self.backend == 'cuda':
            husimiQ = np.zeros(qN * qN, dtype=np.float64)
            _get_lib().cvdvGetHusimiQFullMode(self.ctx, regIdx,
                husimiQ.ctypes.data_as(POINTER(c_double)),
                qN, qMax, pMax
            )
            return husimiQ.reshape((qN, qN))
        else:
            
            perm = list(range(self.num_registers))
            perm[0], perm[regIdx] = perm[regIdx], perm[0]
            state = self.state.permute(*perm)
            target_dim = self.register_dims[regIdx]; other_size = self.total_size // target_dim
            psi = state.reshape(target_dim, other_size).cpu().numpy()
            dx = self.grid_steps[regIdx]
            x_values = np.array([(j - (target_dim - 1) / 2) * dx for j in range(target_dim)])
            husimiQ = np.zeros((qN, qN), dtype=np.float64)
            q_grid = np.linspace(-qMax, qMax, qN)
            p_grid = np.linspace(-pMax, pMax, qN)
            PI_POW_NEG_QUARTER = pi ** (-0.25)
            windowed_signals = np.zeros((qN, target_dim), dtype=np.float64)
            for i, q in enumerate(q_grid):
                window = np.exp(-0.5 * (x_values - q) ** 2) * PI_POW_NEG_QUARTER * np.sqrt(dx)
                for slice_idx in range(other_size):
                    windowed = window * psi[:, slice_idx]
                    fft_result = np.fft.fft(windowed)
                    windowed_signals[i, :] += np.abs(fft_result) ** 2
            dp = 2.0 * pi / (target_dim * dx)
            for j, p_val in enumerate(p_grid):
                for i in range(qN):
                    k_s = int(round(p_val / dp + target_dim / 2))
                    k_s = max(0, min(target_dim - 1, k_s))
                    k_fft = (k_s + target_dim // 2) % target_dim
                    husimiQ[j, i] = windowed_signals[i, k_fft] / pi
            return husimiQ

    def getWigner(self, regIdx: int, bound: float) -> npt.NDArray[np.float64]:
        """Compute Wigner function on the native grids, cropped to [-bound,+bound]^2."""
        dx = self.grid_steps[regIdx]
        dp = np.pi / (self.register_dims[regIdx] * dx)
        N = int(round(2 * bound / dx)) + 1
        wXMax = (N - 1) / 2 * dx
        n_p_bins = int(round(bound / dp))
        wPMax = n_p_bins * dp
        return self.getWignerFullMode(regIdx, wignerN=N, wXMax=wXMax, wPMax=wPMax)

    def getHusimiQ(self, regIdx: int, bound: float) -> npt.NDArray[np.float64]:
        """Compute Husimi Q function on the native grids, cropped to [-bound,+bound]^2."""
        dx = self.grid_steps[regIdx]
        dp = 2 * np.pi / (self.register_dims[regIdx] * dx)
        N = int(round(2 * bound / dx)) + 1
        qMax = (N - 1) / 2 * dx
        n_p_bins = int(round(bound / dp))
        pMax = n_p_bins * dp
        return self.getHusimiQFullMode(regIdx, qN=N, qMax=qMax, pMax=pMax)

    # ==================== Info & Plotting ====================

    def info(self) -> None:
        """Print system information."""
        vram_gb = (self.total_size * 16) / (1024 * 1024 * 1024)
        print(f"Backend: {self.backend}")
        print(f"Number of registers: {self.num_registers}")
        print(f"Total state size: {self.total_size} elements ({vram_gb:.3f} GB)")
        for i in range(self.num_registers):
            dim = self.register_dims[i]; dx = self.grid_steps[i]
            x_bound = sqrt(2 * pi * dim)
            print(f"  Register {i}: dim={dim}, qubits={self.qubit_counts[i]}, dx={dx:.6f}, x_bound={x_bound:.6f}")

    def plotWigner(self, regIdx: int, slice_indices: Optional[Sequence[int]] = None, wignerN: int = 201, wignerMax: float = 5.0,
                    cmap: str = 'RdBu', figsize: Tuple[int, int] = (7, 6), show: bool = True) -> Tuple[Any, Any]:
        """Plot Wigner function for a register."""
        if slice_indices is not None:
            wigner = self.getWignerSingleSlice(regIdx, slice_indices,
                                              wignerN=wignerN, wXMax=wignerMax, wPMax=wignerMax)
        else:
            wigner = self.getWignerFullMode(regIdx, wignerN=wignerN,
                                           wXMax=wignerMax, wPMax=wignerMax)
        fig, ax = plt.subplots(figsize=figsize)
        vmax = np.max(np.abs(wigner))
        im = ax.imshow(wigner, extent=(-wignerMax, wignerMax, -wignerMax, wignerMax),
                      origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_xlabel(r'$q$'); ax.set_ylabel(r'$p$')
        plt.colorbar(im, ax=ax); plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    # ==================== Torch-backend helpers (private) ====================

    def _tPositionGrid(self, regIdx: int):
        
        dim = self.register_dims[regIdx]; dx = self.grid_steps[regIdx]
        idx = torch.arange(dim, device=self.device, dtype=torch.float64)
        return (idx - (dim - 1) * 0.5) * dx

    def _tApplyPhase(self, regIdx: int, phase) -> None:
        shape = [1] * self.num_registers
        shape[regIdx] = -1
        self.state = self.state * phase.reshape(shape)

    def _tApplyQubitGate(self, regIdx: int, targetQubit: int, gate_matrix) -> None:
        
        dim = self.register_dims[regIdx]; n_q = self.qubit_counts[regIdx]
        perm = list(range(self.num_registers))
        perm[0], perm[regIdx] = perm[regIdx], perm[0]
        state = self.state.permute(*perm)
        other_dims = list(state.shape[1:])
        state = state.reshape([2] * n_q + other_dims)
        qperm = list(range(n_q))
        qperm[0], qperm[targetQubit] = qperm[targetQubit], qperm[0]
        state = state.permute(*qperm + list(range(n_q, len(state.shape))))
        orig_shape = state.shape
        state = gate_matrix @ state.reshape(2, -1)
        state = state.reshape(orig_shape)
        state = state.permute(*[qperm.index(i) for i in range(n_q)] + list(range(n_q, len(state.shape))))
        state = state.reshape([dim] + other_dims)
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        self.state = state

    def _tApplyCondPhaseQ(self, targetReg: int, ctrlReg: int, ctrlQubit: int, coeff: float) -> None:
        """exp(i*coeff*Z*q) where Z on control qubit, q on target register."""
        
        q = self._tPositionGrid(targetReg)
        phase_p = torch.exp(1j * coeff * q).to(torch.cdouble)
        phase_m = torch.exp(-1j * coeff * q).to(torch.cdouble)
        ctrl_dim = self.register_dims[ctrlReg]; n_ctrl = self.qubit_counts[ctrlReg]
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_target = targetReg if targetReg != 0 else ctrlReg
        perm[1], perm[actual_target] = perm[actual_target], perm[1]
        state = self.state.permute(*perm)
        new_shape = [2] * n_ctrl + list(state.shape[1:])
        state = state.reshape(new_shape)
        qperm = list(range(n_ctrl))
        qperm[0], qperm[ctrlQubit] = qperm[ctrlQubit], qperm[0]
        state = state.permute(*qperm + list(range(n_ctrl, len(new_shape))))
        pshape = [1] * (len(state.shape) - len(state.shape[n_ctrl:]) + 1)
        pshape[n_ctrl] = -1
        state[0] = state[0] * phase_p.reshape(pshape)[0]
        state[1] = state[1] * phase_m.reshape(pshape)[0]
        state = state.permute(*[qperm.index(i) for i in range(n_ctrl)] + list(range(n_ctrl, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl:]))
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        self.state = state

    def _tApplyCondPhaseQ2(self, targetReg: int, ctrlReg: int, ctrlQubit: int, t: float) -> None:
        """exp(i*t*Z*q^2) where Z on control qubit, q on target register."""
        
        q = self._tPositionGrid(targetReg)
        phase_p = torch.exp(1j * t * q ** 2).to(torch.cdouble)
        phase_m = torch.exp(-1j * t * q ** 2).to(torch.cdouble)
        ctrl_dim = self.register_dims[ctrlReg]; n_ctrl = self.qubit_counts[ctrlReg]
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_target = targetReg if targetReg != 0 else ctrlReg
        perm[1], perm[actual_target] = perm[actual_target], perm[1]
        state = self.state.permute(*perm)
        new_shape = [2] * n_ctrl + list(state.shape[1:])
        state = state.reshape(new_shape)
        qperm = list(range(n_ctrl))
        qperm[0], qperm[ctrlQubit] = qperm[ctrlQubit], qperm[0]
        state = state.permute(*qperm + list(range(n_ctrl, len(new_shape))))
        pshape = [1] * (len(state.shape) - len(state.shape[n_ctrl:]) + 1)
        pshape[n_ctrl] = -1
        state[0] = state[0] * phase_p.reshape(pshape)[0]
        state[1] = state[1] * phase_m.reshape(pshape)[0]
        state = state.permute(*[qperm.index(i) for i in range(n_ctrl)] + list(range(n_ctrl, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl:]))
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        self.state = state

    def _tApplyCondQ1Q2(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, coeff: float) -> None:
        """exp(i*coeff*Z*q1*q2) where Z on control qubit."""
        
        q1 = self._tPositionGrid(reg1); q2 = self._tPositionGrid(reg2)
        pm_p = torch.exp(1j * coeff * q1[:, None] * q2[None, :]).to(torch.cdouble)
        pm_m = torch.exp(-1j * coeff * q1[:, None] * q2[None, :]).to(torch.cdouble)
        ctrl_dim = self.register_dims[ctrlReg]; n_ctrl = self.qubit_counts[ctrlReg]
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_reg1 = reg1 if reg1 != 0 else ctrlReg
        perm[1], perm[actual_reg1] = perm[actual_reg1], perm[1]
        actual_reg2_idx = perm.index(reg2)
        perm[2], perm[actual_reg2_idx] = perm[actual_reg2_idx], perm[2]
        state = self.state.permute(*perm)
        new_shape = [2] * n_ctrl + list(state.shape[1:])
        state = state.reshape(new_shape)
        qperm = list(range(n_ctrl))
        qperm[0], qperm[ctrlQubit] = qperm[ctrlQubit], qperm[0]
        state = state.permute(*qperm + list(range(n_ctrl, len(new_shape))))
        shape_p = [1] * len(state.shape)
        shape_p[n_ctrl] = self.register_dims[reg1]
        shape_p[n_ctrl + 1] = self.register_dims[reg2]
        state[0] = state[0] * pm_p.reshape(shape_p)[0]
        state[1] = state[1] * pm_m.reshape(shape_p)[0]
        state = state.permute(*[qperm.index(i) for i in range(n_ctrl)] + list(range(n_ctrl, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl:]))
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        self.state = state


# CVDVTorch kept as a backward-compatible alias
def CVDVTorch(numQubits_list: List[int], device: str = 'cuda') -> CVDV:
    """Backward-compatible alias: CVDVTorch([...], device='cuda') → CVDV([...], backend='torch-cuda')."""
    return CVDV(numQubits_list, backend=f'torch-{device}')


__all__ = ['CVDV', 'CVDVTorch', 'SeparableState']
