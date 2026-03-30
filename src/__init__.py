"""
CVDV Library - Python wrapper for CUDA quantum simulator
"""

from typing import List, Tuple, Optional, Sequence, Any
import numpy as np
import numpy.typing as npt
import ctypes
from ctypes import c_int, c_double, c_size_t, POINTER
import subprocess
import os
from numpy import pi, sqrt

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

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
    lib_path = os.path.join(build_dir, 'libcvdv.so')
    dev_mode = os.environ.get('CVDV_DEV', '0') not in ('0', '')
    if dev_mode:
        result = subprocess.run(
            ['make', '-C', project_dir, 'build'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f'Build failed:\n{result.stderr}')
    if not os.path.exists(lib_path):
        raise RuntimeError(f'Library not found at {lib_path}. Run `make build` first.')
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
    lib.cvdvGetWigner.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetWigner.restype = None
    lib.cvdvGetHusimiQ.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetHusimiQ.restype = None
    lib.cvdvJointMeasure.argtypes = [ctypes.c_void_p, c_int, c_int, POINTER(c_double)]
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
    lib.cvdvGetPhotonNumber.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvGetPhotonNumber.restype = c_double
    lib.cvdvSetStateFromDevicePtr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cvdvSetStateFromDevicePtr.restype = None
    lib.cvdvFidelityStatevectors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, POINTER(c_double)]
    lib.cvdvFidelityStatevectors.restype = None

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
        if backend in ('torch-cuda', 'torch-cpu') and not _torch_available:
            raise ImportError("PyTorch is required for torch backends: pip install torch")
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

    def d(self, regIdx: int, beta) -> None:
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

    def cd(self, targetReg: int, ctrlReg: int, ctrlQubit: int, alpha) -> None:
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

    def cr(self, targetReg: int, ctrlReg: int, ctrlQubit: int, theta) -> None:
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

    def rx(self, regIdx: int, targetQubit: int, theta) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 0, theta)
        else:
            
            c = np.cos(theta / 2); s = np.sin(theta / 2)
            mat = torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=torch.cdouble, device=self.device)
            self._tApplyQubitGate(regIdx, targetQubit, mat)

    def ry(self, regIdx: int, targetQubit: int, theta) -> None:
        if self.backend == 'cuda':
            _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 1, theta)
        else:
            
            c = np.cos(theta / 2); s = np.sin(theta / 2)
            mat = torch.tensor([[c, -s], [s, c]], dtype=torch.cdouble, device=self.device)
            self._tApplyQubitGate(regIdx, targetQubit, mat)

    def rz(self, regIdx: int, targetQubit: int, theta) -> None:
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

    def sheer(self, regIdx: int, t) -> None:
        """Apply phase square gate: exp(i*t*q^2) in position space."""
        if self.backend == 'cuda':
            _get_lib().cvdvPhaseSquare(self.ctx, regIdx, t)
        else:
            
            x = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * t * x ** 2).to(torch.cdouble))

    def phaseCubic(self, regIdx: int, t) -> None:
        """Apply cubic phase gate: exp(i*t*q^3) in position space."""
        if self.backend == 'cuda':
            _get_lib().cvdvPhaseCubic(self.ctx, regIdx, t)
        else:
            
            x = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * t * x ** 3).to(torch.cdouble))

    def r(self, regIdx: int, theta) -> None:
        """Apply rotation gate R(θ) in phase space."""
        if self.backend == 'cuda':
            _get_lib().cvdvRotation(self.ctx, regIdx, theta)
        else:
            ratio = theta / (pi / 2)
            theta0 = int(np.round(ratio)) * (pi / 2)
            remainder = theta - theta0
            quarter_turns = (int(np.round(ratio)) % 4 + 4) % 4
            if quarter_turns == 1:
                self.ftQ2P(regIdx)
                # self.state = self.state * torch.exp(torch.tensor(-0.25j * pi))
            elif quarter_turns == 2:
                self.p(regIdx)
                # self.state = self.state * torch.exp(torch.tensor(-0.50j * pi))
            elif quarter_turns == 3:
                self.ftP2Q(regIdx)
                # self.state = self.state * torch.exp(torch.tensor(-0.75j * pi))
            if abs(remainder) > 1e-15:
                self.sheer(regIdx, -0.5 * np.tan(remainder / 2))
                self.ftQ2P(regIdx)
                self.sheer(regIdx, -0.5 * np.sin(remainder))
                self.ftP2Q(regIdx)
                self.sheer(regIdx, -0.5 * np.tan(remainder / 2))

    def s(self, regIdx: int, r) -> None:
        """Apply squeezing gate S(r)."""
        if self.backend == 'cuda':
            _get_lib().cvdvSqueeze(self.ctx, regIdx, r)
        else:
            exp_half_r = np.exp(0.5 * r)
            exp_minus_half_r = np.exp(-0.5 * r)
            sqrt_exp_r_minus_1 = np.sqrt(abs(np.exp(r) - 1.0))
            sign = 1 if r >= 0 else -1
            
            # First
            self.ftQ2P(regIdx)
            self.sheer(regIdx, -0.5 * exp_half_r * sqrt_exp_r_minus_1)
            self.ftP2Q(regIdx)
            
            # Second
            self.sheer(regIdx, 0.5 * sign * exp_minus_half_r * sqrt_exp_r_minus_1)
            
            # Third
            self.ftQ2P(regIdx)
            self.sheer(regIdx, 0.5 * exp_minus_half_r * sqrt_exp_r_minus_1)
            self.ftP2Q(regIdx)
            
            # Fourth
            self.sheer(regIdx, -0.5 * sign * exp_half_r * sqrt_exp_r_minus_1)

    def cs(self, targetReg: int, ctrlReg: int, ctrlQubit: int, r) -> None:
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

    def bs(self, reg1: int, reg2: int, theta) -> None:
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

    def cbs(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, theta) -> None:
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

    def q1q2(self, reg1: int, reg2: int, coeff) -> None:
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
            # Global phase correction: exp(-i*π*(N-1)²/(2N)) to match dvsim-code convention
            global_phase = torch.exp(torch.tensor(1j * pi * (dim - 1)**2 / (2 * dim), dtype=torch.cdouble, device=self.device))
            self.state = self.state * global_phase

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
            # Global phase correction: conjugate of ftQ2P: exp(+i*π*(N-1)²/(2N))
            global_phase = torch.exp(torch.tensor(-1j * pi * (dim - 1)**2 / (2 * dim), dtype=torch.cdouble, device=self.device))
            self.state = self.state * global_phase

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

    def initFromArray(self, arr) -> None:
        """Initialize state vector from a flat complex array (torch CUDA tensor or numpy array)."""
        if self.backend != 'cuda':
            raise ValueError("initFromArray is only supported for the CUDA backend")
        if _torch_available and isinstance(arr, torch.Tensor):
            tensor = arr.reshape(-1).contiguous().to(dtype=torch.cdouble, device='cuda')
        else:
            arr_np = np.asarray(arr, dtype=np.complex128).reshape(-1)
            if not _torch_available:
                raise RuntimeError("PyTorch is required for initFromArray")
            tensor = torch.from_numpy(arr_np).to(device='cuda', dtype=torch.cdouble).contiguous()
        _get_lib().cvdvSetStateFromDevicePtr(self.ctx, ctypes.c_void_p(tensor.data_ptr()))

    def fidelityWith(self, other: 'CVDV') -> float:
        """Compute |⟨self|other⟩|² between two CUDA state vectors."""
        if self.backend != 'cuda' or other.backend != 'cuda':
            raise ValueError("fidelityWith requires both instances to use the CUDA backend")
        fid_out = c_double(0.0)
        _get_lib().cvdvFidelityStatevectors(self.ctx, other.ctx, ctypes.byref(fid_out))
        return float(fid_out.value)

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

    def getPhotonNumber(self, regIdx: int) -> float:
        """Compute mean photon number <n> = (<q²> + <p²> - 1) / 2 for a register.

        <q²> is computed in the position basis; the state is temporarily transformed
        to momentum via ftQ2P to compute <p²>, then restored with ftP2Q.
        Since dp = dx in this simulator, the same grid helper applies in both bases.
        """
        if self.backend == 'cuda':
            return float(_get_lib().cvdvGetPhotonNumber(self.ctx, c_int(regIdx)))
        else:
            x = self._tPositionGrid(regIdx)
            shape = [1] * self.num_registers
            shape[regIdx] = -1
            q2 = float(torch.sum(torch.abs(self.state) ** 2 * x.reshape(shape) ** 2).real.cpu().item())

            self.ftQ2P(regIdx)
            p = self._tPositionGrid(regIdx)  # dp = dx, same grid values
            p2 = float(torch.sum(torch.abs(self.state) ** 2 * p.reshape(shape) ** 2).real.cpu().item())
            self.ftP2Q(regIdx)

            return (q2 + p2 - 1.0) / 2.0

    # ==================== Phase Space Functions ====================

    def getWigner(self, regIdx: int) -> npt.NDArray[np.float64]:
        """Wigner function on native N×N grid. Row=p-index, col=x-index.
        Output coordinates: x_i = (i-(N-1)/2)*dx, p_j = (j-N/2)*π/(N*dx)."""
        dim = self.register_dims[regIdx]
        if self.backend == 'cuda':
            out = np.zeros(dim * dim, dtype=np.float64)
            _get_lib().cvdvGetWigner(self.ctx, regIdx, out.ctypes.data_as(POINTER(c_double)))
            return out.reshape((dim, dim))
        else:
            # CPU fallback: native grid (i, k) instead of arbitrary wXMax
            dx = self.grid_steps[regIdx]
            # Trace out other registers (reduce to single-register density matrix diagonal)
            psi = self._get_reduced_state_cpu(regIdx)  # shape (dim,), marginal wavefunction
            fft_results = np.zeros((dim, dim), dtype=np.complex128)
            for i in range(dim):
                integrand = np.zeros(dim, dtype=np.complex128)
                for k in range(dim):
                    kDisp = k - (dim - 1) // 2
                    iPy = i + kDisp
                    iMy = i - kDisp
                    if 0 <= iPy < dim and 0 <= iMy < dim:
                        integrand[k] = np.conj(psi[iPy]) * psi[iMy]
                fft_results[i, :] = np.fft.ifft(integrand) * dim
            dp = pi / (dim * dx)
            wigner = np.zeros((dim, dim), dtype=np.float64)
            for jc in range(dim):
                k_fft = (jc + dim // 2) % dim
                pj = (jc - dim / 2) * dp
                phase_corr = np.exp(-1j * pj * (dim - 1) * dx)
                for i in range(dim):
                    wigner[jc, i] = np.real(phase_corr * fft_results[i, k_fft]) * dx / pi
            return wigner

    def getHusimiQ(self, regIdx: int) -> npt.NDArray[np.float64]:
        """Husimi Q function on native N×N grid. Row=p-index, col=q-index.
        Output coordinates: q_i = (i-(N-1)/2)*dx, p_j = (j-N/2)*2π/(N*dx)."""
        dim = self.register_dims[regIdx]
        if self.backend == 'cuda':
            out = np.zeros(dim * dim, dtype=np.float64)
            _get_lib().cvdvGetHusimiQ(self.ctx, regIdx, out.ctypes.data_as(POINTER(c_double)))
            return out.reshape((dim, dim))
        else:
            # CPU fallback: native grid positions for q
            dx = self.grid_steps[regIdx]
            psi = self._get_reduced_state_cpu(regIdx)  # shape (dim,)
            x_grid = np.array([(k - (dim-1)/2) * dx for k in range(dim)])
            accum = np.zeros((dim, dim), dtype=np.float64)
            for qi in range(dim):
                sample_q = (qi - (dim-1)/2) * dx
                window = (pi**(-0.25)) * np.exp(-0.5*(x_grid - sample_q)**2) * np.sqrt(dx)
                h = psi * window
                H = np.fft.fft(h)
                accum[qi, :] += np.abs(H)**2
            dp = 2 * pi / (dim * dx)
            husimiQ = np.zeros((dim, dim), dtype=np.float64)
            for jc in range(dim):
                k_fft = (jc + dim // 2) % dim
                for qi in range(dim):
                    husimiQ[jc, qi] = accum[qi, k_fft] / pi
            return husimiQ


    def _get_reduced_state_cpu(self, regIdx: int) -> npt.NDArray[np.complex128]:
        """Get reduced single-register wavefunction for CPU fallback."""
        if self.num_registers == 1:
            return self.state.cpu().numpy()
        else:
            # For multi-register systems, trace out other registers
            # Move target register to first dimension
            perm = list(range(self.num_registers))
            perm[0], perm[regIdx] = perm[regIdx], perm[0]
            state = self.state.permute(*perm)
            
            # Sum over all other dimensions to get reduced density matrix diagonal
            # This gives us the marginal wavefunction for the target register
            other_dims = tuple(range(1, self.num_registers))
            rho_diag = torch.sum(torch.abs(state) ** 2, dim=other_dims)
            return torch.sqrt(rho_diag).cpu().numpy()

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

    def _tApplyCondPhaseQ(self, targetReg: int, ctrlReg: int, ctrlQubit: int, coeff) -> None:
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

    def _tApplyCondPhaseQ2(self, targetReg: int, ctrlReg: int, ctrlQubit: int, t) -> None:
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

    def _tApplyCondQ1Q2(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, coeff) -> None:
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


# ==================== Plotting Methods ====================

    def plotWigner(self, regIdx: int, bound: float = 5.0,
                   cmap: str = 'RdBu', figsize: Tuple[int, int] = (7, 6),
                   show: bool = True) -> Tuple[Any, Any]:
        """Plot Wigner function for a register, cropped to [-bound, +bound]²."""
        import matplotlib.pyplot as plt
        try:
            import scienceplots  # noqa: F401
            plt.style.use(['science'])
            plt.rcParams.update({'font.size': 24, 'text.usetex': True})
        except ImportError:
            pass
        wigner = self.getWigner(regIdx)
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        dp = np.pi / (dim * dx)
        x_vals = np.array([(k - (dim-1)/2) * dx for k in range(dim)])
        p_vals = np.array([(j - dim/2) * dp for j in range(dim)])
        x_mask = np.abs(x_vals) <= bound
        p_mask = np.abs(p_vals) <= bound
        x_plot = x_vals[x_mask]; p_plot = p_vals[p_mask]
        W_plot = wigner[np.ix_(p_mask, x_mask)]
        fig, ax = plt.subplots(figsize=figsize)
        vmax = np.max(np.abs(W_plot))
        im = ax.pcolormesh(x_plot, p_plot, W_plot, cmap=cmap, vmin=-vmax, vmax=vmax,
                           shading='auto')
        ax.set_xlabel(r'$q$'); ax.set_ylabel(r'$p$')
        fig.colorbar(im, ax=ax); plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plotHusimiQ(self, regIdx: int, bound: float = 5.0,
                    cmap: str = 'viridis', figsize: Tuple[int, int] = (7, 6),
                    show: bool = True) -> Tuple[Any, Any]:
        """Plot Husimi Q function for a register, cropped to [-bound, +bound]²."""
        import matplotlib.pyplot as plt
        try:
            import scienceplots  # noqa: F401
            plt.style.use(['science'])
            plt.rcParams.update({'font.size': 24, 'text.usetex': True})
        except ImportError:
            pass
        Q = self.getHusimiQ(regIdx)
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        dp = 2 * np.pi / (dim * dx)
        q_vals = np.array([(k - (dim-1)/2) * dx for k in range(dim)])
        p_vals = np.array([(j - dim/2) * dp for j in range(dim)])
        q_mask = np.abs(q_vals) <= bound
        p_mask = np.abs(p_vals) <= bound
        q_plot = q_vals[q_mask]; p_plot = p_vals[p_mask]
        Q_plot = Q[np.ix_(p_mask, q_mask)]
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(q_plot, p_plot, Q_plot, cmap=cmap, shading='auto')
        ax.set_xlabel(r'$q$'); ax.set_ylabel(r'$p$')
        fig.colorbar(im, ax=ax); plt.tight_layout()
        if show:
            plt.show()
        return fig, ax


# CVDVTorch kept as a backward-compatible alias
def CVDVTorch(numQubits_list: List[int], device: str = 'cuda') -> CVDV:
    """Backward-compatible alias: CVDVTorch([...], device='cuda') → CVDV([...], backend='torch-cuda')."""
    return CVDV(numQubits_list, backend=f'torch-{device}')


__all__ = ['CVDV', 'CVDVTorch', 'SeparableState']
