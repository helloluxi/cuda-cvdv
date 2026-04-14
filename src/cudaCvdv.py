"""
CUDA-only CVDV wrapper around libcvdv.so.
"""

from typing import List, Tuple, Optional, Any
import ctypes
from ctypes import c_int, c_double, c_size_t, POINTER
import os

import numpy as np
import numpy.typing as npt
from numpy import pi, sqrt
import torch

from .separable import SeparableState

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
build_dir = os.path.join(project_dir, "build")
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _compile_and_load()
    return _lib


def _compile_and_load() -> ctypes.CDLL:
    lib_path = os.path.join(build_dir, "libcvdv.so")
    if not os.path.exists(lib_path):
        raise RuntimeError(f"Library not found at {lib_path}. Run `make build` first.")

    lib = ctypes.CDLL(lib_path)
    lib.cvdvCreate.argtypes = [c_int, POINTER(c_int)]
    lib.cvdvCreate.restype = ctypes.c_void_p
    lib.cvdvDestroy.argtypes = [ctypes.c_void_p]
    lib.cvdvDestroy.restype = None
    lib.cvdvInitFromSeparable.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), c_int]
    lib.cvdvInitFromSeparable.restype = None
    lib.cvdvDisplacement.argtypes = [ctypes.c_void_p, c_int, c_double, c_double]
    lib.cvdvConditionalDisplacement.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double, c_double]
    lib.cvdvPauliRotation.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvHadamard.argtypes = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvParity.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvConditionalParity.argtypes = [ctypes.c_void_p, c_int, c_int, c_int]
    lib.cvdvSwapRegisters.argtypes = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvPhaseSquare.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvPhaseCubic.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvRotation.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvConditionalRotation.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvSqueeze.argtypes = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvConditionalSqueeze.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvBeamSplitter.argtypes = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvConditionalBeamSplitter.argtypes = [ctypes.c_void_p, c_int, c_int, c_int, c_int, c_double]
    lib.cvdvQ1Q2Gate.argtypes = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvFtQ2P.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvFtP2Q.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvGetWigner.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetHusimiQ.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetHusimiQOverlap.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetHusimiQWigner.argtypes = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvMeasureMultiple.argtypes = [ctypes.c_void_p, POINTER(c_int), c_int, POINTER(c_double)]
    lib.cvdvMeasureMultiple.restype = None
    lib.cvdvGetState.argtypes = [ctypes.c_void_p, POINTER(c_double), POINTER(c_double)]
    lib.cvdvGetState.restype = None
    lib.cvdvGetFidelity.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), c_int, POINTER(c_double)]
    lib.cvdvGetNorm.argtypes = [ctypes.c_void_p]
    lib.cvdvGetNorm.restype = c_double
    lib.cvdvGetPhotonNumber.argtypes = [ctypes.c_void_p, c_int]
    lib.cvdvGetPhotonNumber.restype = c_double
    lib.cvdvSetStateFromDevicePtr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cvdvFidelityStatevectors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, POINTER(c_double)]
    return lib


class CudaCvdv:
    """CUDA-only simulator. Device and dtype are fixed to CUDA complex128."""

    def __init__(self, numQubits_list: List[int]) -> None:
        self.num_registers = len(numQubits_list)
        self.register_dims = [1 << q for q in numQubits_list]
        self.total_size = int(np.prod(self.register_dims, dtype=np.int64))
        self.qubit_counts = np.array(numQubits_list, dtype=np.int32)
        self.grid_steps = np.array([sqrt(2 * pi / d) for d in self.register_dims], dtype=np.float64)
        numQubits_c = (c_int * len(numQubits_list))(*numQubits_list)
        self.ctx = _get_lib().cvdvCreate(len(numQubits_list), numQubits_c)

    def __del__(self):
        try:
            if hasattr(self, "ctx") and self.ctx:
                _get_lib().cvdvDestroy(self.ctx)
                self.ctx = None
        except Exception:
            pass

    def initStateVector(self, sep: "SeparableState") -> None:
        sep.validate()
        ptr_arr = (ctypes.c_void_p * self.num_registers)(*[ctypes.c_void_p(arr.data_ptr()) for arr in sep.register_arrays])  # type: ignore[union-attr]
        _get_lib().cvdvInitFromSeparable(self.ctx, ptr_arr, c_int(self.num_registers))

    def d(self, regIdx: int, beta) -> None:
        beta = complex(beta)
        _get_lib().cvdvDisplacement(self.ctx, regIdx, c_double(beta.real), c_double(beta.imag))

    def cd(self, targetReg: int, ctrlReg: int, ctrlQubit: int, alpha) -> None:
        alpha = complex(alpha)
        _get_lib().cvdvConditionalDisplacement(self.ctx, targetReg, ctrlReg, ctrlQubit, c_double(alpha.real), c_double(alpha.imag))

    def cr(self, targetReg: int, ctrlReg: int, ctrlQubit: int, theta) -> None:
        _get_lib().cvdvConditionalRotation(self.ctx, targetReg, ctrlReg, ctrlQubit, c_double(theta))

    def x(self, regIdx: int, targetQubit: int) -> None: _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 0, pi)
    def y(self, regIdx: int, targetQubit: int) -> None: _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 1, pi)
    def z(self, regIdx: int, targetQubit: int) -> None: _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 2, pi)
    def rx(self, regIdx: int, targetQubit: int, theta) -> None: _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 0, theta)
    def ry(self, regIdx: int, targetQubit: int, theta) -> None: _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 1, theta)
    def rz(self, regIdx: int, targetQubit: int, theta) -> None: _get_lib().cvdvPauliRotation(self.ctx, regIdx, targetQubit, 2, theta)
    def h(self, regIdx: int, targetQubit: int) -> None: _get_lib().cvdvHadamard(self.ctx, regIdx, targetQubit)
    def p(self, regIdx: int) -> None: _get_lib().cvdvParity(self.ctx, regIdx)
    def cp(self, targetReg: int, ctrlReg: int, ctrlQubit: int) -> None: _get_lib().cvdvConditionalParity(self.ctx, targetReg, ctrlReg, ctrlQubit)
    def swap(self, reg1: int, reg2: int) -> None: _get_lib().cvdvSwapRegisters(self.ctx, reg1, reg2)
    def sheer(self, regIdx: int, t) -> None: _get_lib().cvdvPhaseSquare(self.ctx, regIdx, t)
    def phaseCubic(self, regIdx: int, t) -> None: _get_lib().cvdvPhaseCubic(self.ctx, regIdx, t)
    def r(self, regIdx: int, theta) -> None: _get_lib().cvdvRotation(self.ctx, regIdx, theta)
    def s(self, regIdx: int, r) -> None: _get_lib().cvdvSqueeze(self.ctx, regIdx, r)
    def cs(self, targetReg: int, ctrlReg: int, ctrlQubit: int, r) -> None: _get_lib().cvdvConditionalSqueeze(self.ctx, targetReg, ctrlReg, ctrlQubit, c_double(r))
    def bs(self, reg1: int, reg2: int, theta) -> None: _get_lib().cvdvBeamSplitter(self.ctx, reg1, reg2, theta)
    def cbs(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, theta) -> None: _get_lib().cvdvConditionalBeamSplitter(self.ctx, reg1, reg2, ctrlReg, ctrlQubit, c_double(theta))
    def q1q2(self, reg1: int, reg2: int, coeff) -> None: _get_lib().cvdvQ1Q2Gate(self.ctx, reg1, reg2, coeff)
    def ftQ2P(self, regIdx: int) -> None: _get_lib().cvdvFtQ2P(self.ctx, regIdx)
    def ftP2Q(self, regIdx: int) -> None: _get_lib().cvdvFtP2Q(self.ctx, regIdx)

    def getState(self) -> npt.NDArray[np.complex128]:
        real_arr = np.zeros(self.total_size, dtype=np.float64)
        imag_arr = np.zeros(self.total_size, dtype=np.float64)
        _get_lib().cvdvGetState(self.ctx, real_arr.ctypes.data_as(POINTER(c_double)), imag_arr.ctypes.data_as(POINTER(c_double)))
        return real_arr + 1j * imag_arr

    def getXGrid(self, regIdx: int) -> npt.NDArray[np.float64]:
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        return (np.arange(dim) - (dim - 1) * 0.5) * dx

    def m(self, *regIdxs) -> "float | npt.NDArray[np.float64]":
        if len(regIdxs) == 1 and isinstance(regIdxs[0], (list, tuple)):
            regIdxs = tuple(regIdxs[0])
        if not regIdxs:
            return float(_get_lib().cvdvGetNorm(self.ctx))
        regs = list(regIdxs)
        out_size = int(np.prod([self.register_dims[r] for r in regs]))
        out = np.zeros(out_size, dtype=np.float64)
        regs_c = (c_int * len(regs))(*regs)
        _get_lib().cvdvMeasureMultiple(self.ctx, regs_c, c_int(len(regs)),
                                       out.ctypes.data_as(POINTER(c_double)))
        # cuTENSOR outputs column-major, transpose to row-major
        shape = tuple(self.register_dims[r] for r in regs)
        if len(shape) == 1:
            return out.reshape(shape)
        return out.reshape(shape[::-1]).T

    def initFromArray(self, arr) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for CUDA state transfer")
        if isinstance(arr, torch.Tensor):
            tensor = arr.reshape(-1).contiguous().to(dtype=torch.cdouble, device="cuda")
        else:
            tensor = torch.from_numpy(np.asarray(arr, dtype=np.complex128).reshape(-1)).to(device="cuda", dtype=torch.cdouble).contiguous()
        _get_lib().cvdvSetStateFromDevicePtr(self.ctx, ctypes.c_void_p(tensor.data_ptr()))

    def fidelityWith(self, other: "CudaCvdv") -> float:
        fid_out = c_double(0.0)
        _get_lib().cvdvFidelityStatevectors(self.ctx, other.ctx, ctypes.byref(fid_out))
        return float(fid_out.value)

    def getFidelity(self, sep: "SeparableState") -> float:
        sep.validate()
        ptr_arr = (ctypes.c_void_p * self.num_registers)(*[ctypes.c_void_p(arr.data_ptr()) for arr in sep.register_arrays])  # type: ignore[union-attr]
        fid_out = c_double(0.0)
        _get_lib().cvdvGetFidelity(self.ctx, ptr_arr, c_int(self.num_registers), ctypes.byref(fid_out))
        return float(fid_out.value)

    def getPhotonNumber(self, regIdx: int) -> float:
        return float(_get_lib().cvdvGetPhotonNumber(self.ctx, c_int(regIdx)))

    def getWigner(self, regIdx: int, bound: float | None = None) -> npt.NDArray[np.float64]:
        dim = self.register_dims[regIdx]
        out = np.zeros(dim * dim, dtype=np.float64)
        _get_lib().cvdvGetWigner(self.ctx, regIdx, out.ctypes.data_as(POINTER(c_double)))
        W = out.reshape((dim, dim))
        if bound is None:
            return W
        dx = self.grid_steps[regIdx]
        dp = np.pi / (dim * dx)
        x_vals = (np.arange(dim) - (dim - 1) / 2) * dx
        p_vals = (np.arange(dim) - dim / 2) * dp
        return W[np.ix_(np.abs(p_vals) <= bound, np.abs(x_vals) <= bound)]

    def getHusimiQ(self, regIdx: int, bound: float | None = None) -> npt.NDArray[np.float64]:
        dim = self.register_dims[regIdx]
        out = np.zeros(dim * dim, dtype=np.float64)
        _get_lib().cvdvGetHusimiQ(self.ctx, regIdx, out.ctypes.data_as(POINTER(c_double)))
        Q = out.reshape((dim, dim))
        if bound is None:
            return Q
        dx = self.grid_steps[regIdx]
        dp = 2 * np.pi / (dim * dx)
        q_vals = (np.arange(dim) - (dim - 1) / 2) * dx
        p_vals = (np.arange(dim) - dim / 2) * dp
        return Q[np.ix_(np.abs(p_vals) <= bound, np.abs(q_vals) <= bound)]

    def plotWigner(self, regIdx: int, bound: float = 5.0, cmap: str = "RdBu", figsize: Tuple[int, int] = (7, 6), show: bool = True) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt
        wigner = self.getWigner(regIdx)
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        dp = np.pi / (dim * dx)
        x_vals = np.array([(k - (dim - 1) / 2) * dx for k in range(dim)])
        p_vals = np.array([(j - dim / 2) * dp for j in range(dim)])
        x_mask = np.abs(x_vals) <= bound
        p_mask = np.abs(p_vals) <= bound
        fig, ax = plt.subplots(figsize=figsize)
        W_plot = wigner[np.ix_(p_mask, x_mask)]
        vmax = np.max(np.abs(W_plot))
        im = ax.pcolormesh(x_vals[x_mask], p_vals[p_mask], W_plot, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$p$")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plotHusimiQ(self, regIdx: int, bound: float = 5.0, cmap: str = "viridis", figsize: Tuple[int, int] = (7, 6), show: bool = True) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt
        Q = self.getHusimiQ(regIdx)
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        dp = 2 * np.pi / (dim * dx)
        q_vals = np.array([(k - (dim - 1) / 2) * dx for k in range(dim)])
        p_vals = np.array([(j - dim / 2) * dp for j in range(dim)])
        q_mask = np.abs(q_vals) <= bound
        p_mask = np.abs(p_vals) <= bound
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(q_vals[q_mask], p_vals[p_mask], Q[np.ix_(p_mask, q_mask)], cmap=cmap, shading="auto")
        ax.set_xlabel(r"$q$")
        ax.set_ylabel(r"$p$")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
