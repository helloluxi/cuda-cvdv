"""
Pure PyTorch implementation of CVDV.
"""

from typing import List, Tuple, Any
import numpy as np
import numpy.typing as npt
from numpy import pi, sqrt
import torch

from .separable import SeparableState


class TorchCvdv:
    """Torch-only simulator with configurable device and dtype."""

    def __init__(self, numQubits_list: List[int], device: str = "cuda", dtype: Any = torch.cdouble) -> None:
        self.num_registers = len(numQubits_list)
        self.register_dims = [1 << q for q in numQubits_list]
        self.total_size = int(np.prod(self.register_dims, dtype=np.int64))
        self.qubit_counts = np.array(numQubits_list, dtype=np.int32)
        self.grid_steps = np.array([sqrt(2 * pi / d) for d in self.register_dims], dtype=np.float64)
        self.device = torch.device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.state: Any = None
        # flwQbts[r] = total qubits in all registers AFTER r (last register = LSB in flat index)
        self.flwQbts = [int(sum(self.qubit_counts[r + 1:])) for r in range(self.num_registers)]
        self._pos_grids: dict = {}  # lazy cache: regIdx → position grid tensor

    @staticmethod
    def _resolve_dtype(dtype: Any):
        if dtype in (np.complex128, "complex128"):
            return torch.cdouble
        if dtype in (np.complex64, "complex64"):
            return torch.cfloat
        return dtype

    def initStateVector(self, sep: "SeparableState") -> None:
        sep.validate()
        arrays = [arr.to(device=self.device, dtype=self.dtype) for arr in sep.register_arrays]  # type: ignore[union-attr]
        if self.num_registers == 1:
            state = arrays[0].clone()
        else:
            state = torch.outer(arrays[0], arrays[1])
            for i in range(2, self.num_registers):
                state = state.unsqueeze(-1) * arrays[i].reshape(*([1] * i), -1)
        norm = torch.sqrt(torch.sum(torch.abs(state) ** 2))
        self.state = state / norm

    def d(self, regIdx: int, beta) -> None:
        beta = complex(beta)
        if abs(beta.imag) > 1e-12:
            x = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(1j * sqrt(2) * beta.imag * x).to(self.dtype))
        if abs(beta.real) > 1e-12:
            self.ftQ2P(regIdx)
            p = self._tPositionGrid(regIdx)
            self._tApplyPhase(regIdx, torch.exp(-1j * sqrt(2) * beta.real * p).to(self.dtype))
            self.ftP2Q(regIdx)

    def cd(self, targetReg: int, ctrlReg: int, ctrlQubit: int, alpha) -> None:
        alpha = complex(alpha)
        if abs(alpha.imag) > 1e-12:
            self._tApplyCondPhaseQ(targetReg, ctrlReg, ctrlQubit, sqrt(2) * alpha.imag)
        if abs(alpha.real) > 1e-12:
            self.ftQ2P(targetReg)
            self._tApplyCondPhaseQ(targetReg, ctrlReg, ctrlQubit, -sqrt(2) * alpha.real)
            self.ftP2Q(targetReg)

    def cr(self, targetReg: int, ctrlReg: int, ctrlQubit: int, theta) -> None:
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

    def x(self, regIdx: int, targetQubit: int) -> None: self.rx(regIdx, targetQubit, pi)
    def y(self, regIdx: int, targetQubit: int) -> None: self.ry(regIdx, targetQubit, pi)
    def z(self, regIdx: int, targetQubit: int) -> None: self.rz(regIdx, targetQubit, pi)

    def rx(self, regIdx: int, targetQubit: int, theta) -> None:
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        mat = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=self.dtype, device=self.device)
        self._tApplyQubitGate(regIdx, targetQubit, mat)

    def ry(self, regIdx: int, targetQubit: int, theta) -> None:
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        mat = torch.tensor([[c, -s], [s, c]], dtype=self.dtype, device=self.device)
        self._tApplyQubitGate(regIdx, targetQubit, mat)

    def rz(self, regIdx: int, targetQubit: int, theta) -> None:
        mat = torch.tensor([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=self.dtype, device=self.device)
        self._tApplyQubitGate(regIdx, targetQubit, mat)

    def h(self, regIdx: int, targetQubit: int) -> None:
        mat = torch.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=self.device) / sqrt(2)
        self._tApplyQubitGate(regIdx, targetQubit, mat)

    def p(self, regIdx: int) -> None:
        self.state = torch.flip(self.state, dims=[regIdx])

    def cp(self, targetReg: int, ctrlReg: int, ctrlQubit: int) -> None:
        # Mirror CUDA kernelConditionalParity: flat-index swap, no reshape.
        target_dim = self.register_dims[targetReg]
        all_idx = torch.arange(self.total_size, device=self.device, dtype=torch.long)
        ctrl_bit_pos = self.flwQbts[ctrlReg] + self.qubit_counts[ctrlReg] - 1 - ctrlQubit
        ctrl_bit = (all_idx >> ctrl_bit_pos) & 1
        target_local = (all_idx >> self.flwQbts[targetReg]) & (target_dim - 1)
        flipped = (target_dim - 1) - target_local
        # Process each pair only once (local < flipped) to avoid double-swap
        active = (ctrl_bit == 1) & (target_local < flipped)
        src = all_idx[active]
        dst = src + (flipped[active] - target_local[active]) * (1 << self.flwQbts[targetReg])
        flat = self.state.contiguous().reshape(-1).clone()
        flat[src] = self.state.reshape(-1)[dst]
        flat[dst] = self.state.reshape(-1)[src]
        self.state = flat.reshape(self.state.shape)

    def swap(self, reg1: int, reg2: int) -> None:
        if self.qubit_counts[reg1] != self.qubit_counts[reg2]:
            raise ValueError("SWAP requires registers with same number of qubits")
        perm = list(range(self.num_registers))
        perm[reg1], perm[reg2] = perm[reg2], perm[reg1]
        self.state = self.state.permute(*perm)

    def sheer(self, regIdx: int, t) -> None:
        x = self._tPositionGrid(regIdx)
        self._tApplyPhase(regIdx, torch.exp(1j * t * x ** 2).to(self.dtype))

    def phaseCubic(self, regIdx: int, t) -> None:
        x = self._tPositionGrid(regIdx)
        self._tApplyPhase(regIdx, torch.exp(1j * t * x ** 3).to(self.dtype))

    def r(self, regIdx: int, theta) -> None:
        ratio = theta / (pi / 2)
        theta0 = int(np.round(ratio)) * (pi / 2)
        remainder = theta - theta0
        quarter_turns = (int(np.round(ratio)) % 4 + 4) % 4
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

    def s(self, regIdx: int, r) -> None:
        exp_half_r = np.exp(0.5 * r)
        exp_minus_half_r = np.exp(-0.5 * r)
        sqrt_exp_r_minus_1 = np.sqrt(abs(np.exp(r) - 1.0))
        sign = 1 if r >= 0 else -1
        self.ftQ2P(regIdx)
        self.sheer(regIdx, -0.5 * exp_half_r * sqrt_exp_r_minus_1)
        self.ftP2Q(regIdx)
        self.sheer(regIdx, 0.5 * sign * exp_minus_half_r * sqrt_exp_r_minus_1)
        self.ftQ2P(regIdx)
        self.sheer(regIdx, 0.5 * exp_minus_half_r * sqrt_exp_r_minus_1)
        self.ftP2Q(regIdx)
        self.sheer(regIdx, -0.5 * sign * exp_half_r * sqrt_exp_r_minus_1)

    def cs(self, targetReg: int, ctrlReg: int, ctrlQubit: int, r) -> None:
        ch_r = np.cosh(r)
        sh_r = np.sinh(r)
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
        ratio = theta / pi
        theta0 = int(np.floor(ratio + 0.5)) * pi
        remainder = theta - theta0
        half_turns = (int(np.floor(ratio + 0.5)) % 4 + 4) % 4
        if half_turns == 1:
            self.ftQ2P(reg1)
            self.ftQ2P(reg2)
            self.swap(reg1, reg2)
        elif half_turns == 2:
            self.p(reg1)
            self.p(reg2)
        elif half_turns == 3:
            self.ftP2Q(reg1)
            self.ftP2Q(reg2)
            self.swap(reg1, reg2)
        if abs(remainder) > 1e-15:
            tq = np.tan(remainder / 4)
            sh = np.sin(remainder / 2)
            self.q1q2(reg1, reg2, -tq)
            self.ftQ2P(reg1)
            self.ftQ2P(reg2)
            self.q1q2(reg1, reg2, -sh)
            self.ftP2Q(reg1)
            self.ftP2Q(reg2)
            self.q1q2(reg1, reg2, -tq)

    def cbs(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, theta) -> None:
        ratio = theta / pi
        theta0 = int(np.floor(ratio + 0.5)) * pi
        remainder = theta - theta0
        half_turns = (int(np.floor(ratio + 0.5)) % 4 + 4) % 4
        if half_turns == 1:
            self.ftQ2P(reg1)
            self.ftQ2P(reg2)
            self.cp(reg1, ctrlReg, ctrlQubit)
            self.cp(reg2, ctrlReg, ctrlQubit)
            self.swap(reg1, reg2)
        elif half_turns == 2:
            self.p(reg1)
            self.p(reg2)
        elif half_turns == 3:
            self.ftP2Q(reg1)
            self.ftP2Q(reg2)
            self.cp(reg1, ctrlReg, ctrlQubit)
            self.cp(reg2, ctrlReg, ctrlQubit)
            self.swap(reg1, reg2)
        if abs(remainder) > 1e-15:
            tq = np.tan(remainder / 4)
            sh = np.sin(remainder / 2)
            self._tApplyCondQ1Q2(reg1, reg2, ctrlReg, ctrlQubit, -tq)
            self.ftQ2P(reg1)
            self.ftQ2P(reg2)
            self._tApplyCondQ1Q2(reg1, reg2, ctrlReg, ctrlQubit, -sh)
            self.ftP2Q(reg1)
            self.ftP2Q(reg2)
            self._tApplyCondQ1Q2(reg1, reg2, ctrlReg, ctrlQubit, -tq)

    def q1q2(self, reg1: int, reg2: int, coeff) -> None:
        q1 = self._tPositionGrid(reg1)
        q2 = self._tPositionGrid(reg2)
        phase_matrix = torch.exp(1j * coeff * q1[:, None] * q2[None, :]).to(self.dtype)
        shape = [1] * self.num_registers
        shape[reg1] = self.register_dims[reg1]
        shape[reg2] = self.register_dims[reg2]
        self.state = self.state * phase_matrix.reshape(shape)

    def ftQ2P(self, regIdx: int) -> None:
        dx = self.grid_steps[regIdx]
        dim = self.register_dims[regIdx]
        phaseCoeff = pi * (dim - 1.0) / (dim * dx)
        x = self._tPositionGrid(regIdx)
        self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * x).to(self.dtype))
        self.state = torch.fft.fft(self.state, dim=regIdx)
        p = self._tPositionGrid(regIdx)
        self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * p).to(self.dtype))
        self.state = self.state / sqrt(dim)
        global_phase = torch.exp(torch.tensor(1j * pi * (dim - 1) ** 2 / (2 * dim), dtype=self.dtype, device=self.device))
        self.state = self.state * global_phase

    def ftP2Q(self, regIdx: int) -> None:
        dx = self.grid_steps[regIdx]
        dim = self.register_dims[regIdx]
        phaseCoeff = -pi * (dim - 1.0) / (dim * dx)
        p = self._tPositionGrid(regIdx)
        self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * p).to(self.dtype))
        self.state = torch.fft.ifft(self.state, dim=regIdx, norm="forward")
        x = self._tPositionGrid(regIdx)
        self._tApplyPhase(regIdx, torch.exp(1j * phaseCoeff * x).to(self.dtype))
        self.state = self.state / sqrt(dim)
        global_phase = torch.exp(torch.tensor(-1j * pi * (dim - 1) ** 2 / (2 * dim), dtype=self.dtype, device=self.device))
        self.state = self.state * global_phase

    def getState(self) -> npt.NDArray[np.complex128]:
        return self.state.flatten().detach().cpu().numpy()

    def getXGrid(self, regIdx: int) -> npt.NDArray[np.float64]:
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        return (np.arange(dim) - (dim - 1) * 0.5) * dx

    def m(self, *regIdxs) -> "float | npt.NDArray[np.float64]":
        if len(regIdxs) == 1 and isinstance(regIdxs[0], (list, tuple)):
            regIdxs = tuple(regIdxs[0])
        if not regIdxs:
            return float(torch.sqrt(torch.sum(torch.abs(self.state) ** 2)).cpu().item())
        regs = list(regIdxs)
        probs = torch.abs(self.state) ** 2
        dims_to_sum = [i for i in range(self.num_registers) if i not in regs]
        for d in sorted(dims_to_sum, reverse=True):
            probs = torch.sum(probs, dim=d)
        # After summing, remaining dims are in ascending sorted order; permute to match regs order
        sorted_regs = sorted(regs)
        perm = [sorted_regs.index(r) for r in regs]
        if perm != list(range(len(regs))):
            probs = probs.permute(*perm)
        return probs.detach().cpu().numpy()

    def getFidelity(self, sep: "SeparableState") -> float:
        sep.validate()
        arrays = [arr.to(device=self.device, dtype=self.dtype) for arr in sep.register_arrays]  # type: ignore[union-attr]
        if self.num_registers == 1:
            ref = arrays[0]
        else:
            ref = torch.outer(arrays[0], arrays[1])
            for i in range(2, self.num_registers):
                ref = ref.unsqueeze(-1) * arrays[i].reshape(*([1] * i), -1)
        ref_flat = ref.reshape(-1)
        inner = torch.sum(torch.conj(ref_flat) * self.state.reshape(-1))
        return float(torch.abs(inner).item() ** 2)

    def getPhotonNumber(self, regIdx: int) -> float:
        x = self._tPositionGrid(regIdx)
        shape = [1] * self.num_registers
        shape[regIdx] = -1
        q2 = float(torch.sum(torch.abs(self.state) ** 2 * x.reshape(shape) ** 2).real.cpu().item())
        self.ftQ2P(regIdx)
        p = self._tPositionGrid(regIdx)
        p2 = float(torch.sum(torch.abs(self.state) ** 2 * p.reshape(shape) ** 2).real.cpu().item())
        self.ftP2Q(regIdx)
        return (q2 + p2 - 1.0) / 2.0

    def getWigner(self, regIdx: int) -> npt.NDArray[np.float64]:
        # Vectorized on GPU — mirrors CUDA kernelBuildWignerRow + kernelFinalizeWigner.
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        slices = self._get_target_slices_gpu(regIdx)
        state_xs = slices.transpose(0, 1)
        rho = state_xs @ torch.conj(state_xs).transpose(0, 1)
        i_idx = torch.arange(dim, device=self.device, dtype=torch.long)
        k_idx = torch.arange(dim, device=self.device, dtype=torch.long)
        kDisp = k_idx - (dim - 1) // 2                        # (dim,)
        raw_iPy = i_idx[:, None] + kDisp[None, :]             # (dim, dim)
        raw_iMy = i_idx[:, None] - kDisp[None, :]
        valid = (raw_iPy >= 0) & (raw_iPy < dim) & (raw_iMy >= 0) & (raw_iMy < dim)
        iPy = raw_iPy.clamp(0, dim - 1)
        iMy = raw_iMy.clamp(0, dim - 1)
        integrand = torch.where(valid, rho[iMy, iPy],
                                torch.zeros((dim, dim), dtype=self.dtype, device=self.device))
        fft_results = torch.fft.ifft(integrand, dim=1) * dim  # (dim, dim)
        dp = pi / (dim * dx)
        jc = torch.arange(dim, device=self.device, dtype=torch.long)
        k_fft = (jc + dim // 2) % dim
        pj = (jc.double() - dim / 2) * dp
        phase_corr = torch.exp(torch.tensor(-1j * (dim - 1) * dx, dtype=self.dtype, device=self.device) * pj.to(self.dtype))
        # fft_results.T[k_fft] → shape (dim_jc, dim_i): result[jc,i] = fft_results[i, k_fft[jc]]
        wigner = (phase_corr[:, None] * fft_results.T[k_fft]).real * (dx / pi)
        return wigner.cpu().numpy()

    def getHusimiQ(self, regIdx: int) -> npt.NDArray[np.float64]:
        # Vectorized on GPU — mirrors CUDA kernelBuildHusimiRows + kernelFinalizeHusimi.
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        slices = self._get_target_slices_gpu(regIdx)
        x_grid = self._tPositionGrid(regIdx)
        # Gaussian window for all q-samples at once: shape (dim_q, dim_x)
        window = (pi ** -0.25) * torch.exp(
            -0.5 * (x_grid[None, :] - x_grid[:, None]) ** 2
        ) * sqrt(dx)                                           # real, broadcast
        accum = torch.zeros((dim, dim), dtype=torch.float64, device=self.device)
        complex_bytes = torch.empty((), dtype=self.dtype).element_size()
        chunk_budget = 16 * 1024 * 1024
        bytes_per_chunk = max(1, 2 * dim * dim * complex_bytes)
        chunk_size = max(1, min(slices.shape[0], chunk_budget // bytes_per_chunk))
        for start in range(0, slices.shape[0], chunk_size):
            chunk = slices[start:start + chunk_size]
            h = chunk[:, None, :] * window[None, :, :].to(self.dtype)
            H = torch.fft.fft(h, dim=2)
            accum = accum + torch.sum(torch.abs(H) ** 2, dim=0)
        k_fft = (torch.arange(dim, device=self.device) + dim // 2) % dim
        husimiQ = accum[:, k_fft].T / pi                       # (dim, dim)
        return husimiQ.cpu().numpy()

    def _get_target_slices_gpu(self, regIdx: int):
        """Return all target-register slices as a 2-D GPU tensor with shape (num_slices, dim)."""
        if self.num_registers == 1:
            return self.state.reshape(1, -1)
        perm = list(range(self.num_registers))
        perm[0], perm[regIdx] = perm[regIdx], perm[0]
        state = self.state.permute(*perm).contiguous()
        return state.reshape(self.register_dims[regIdx], -1).transpose(0, 1)

    def _get_target_slices_cpu(self, regIdx: int) -> npt.NDArray[np.complex128]:
        return self._get_target_slices_gpu(regIdx).detach().cpu().numpy()

    def _tPositionGrid(self, regIdx: int):
        if regIdx not in self._pos_grids:
            dim = self.register_dims[regIdx]
            dx = self.grid_steps[regIdx]
            idx = torch.arange(dim, device=self.device, dtype=torch.float64)
            self._pos_grids[regIdx] = (idx - (dim - 1) * 0.5) * dx
        return self._pos_grids[regIdx]

    def _tApplyPhase(self, regIdx: int, phase) -> None:
        shape = [1] * self.num_registers
        shape[regIdx] = -1
        self.state = self.state * phase.reshape(shape)

    def _tApplyQubitGate(self, regIdx: int, targetQubit: int, gate_matrix) -> None:
        # Mirror CUDA qubitPairIndices: work on the flat index, no reshaping.
        # globalBit = flwQbts[r] + qbts[r] - 1 - qubitIdx  (qubit 0 = MSB of register)
        global_bit = self.flwQbts[regIdx] + self.qubit_counts[regIdx] - 1 - targetQubit
        mask = 1 << global_bit
        pair_idx = torch.arange(self.total_size // 2, device=self.device, dtype=torch.long)
        # Insert a 0-bit at global_bit to get idx0; set that bit to get idx1
        idx0 = ((pair_idx >> global_bit) << (global_bit + 1)) | (pair_idx & (mask - 1))
        idx1 = idx0 | mask
        flat = self.state.contiguous().reshape(-1)
        s0, s1 = flat[idx0], flat[idx1]   # gather — creates independent tensors
        flat = flat.clone()
        flat[idx0] = gate_matrix[0, 0] * s0 + gate_matrix[0, 1] * s1
        flat[idx1] = gate_matrix[1, 0] * s0 + gate_matrix[1, 1] * s1
        self.state = flat.reshape(self.state.shape)

    def _tApplyCondPhaseQ(self, targetReg: int, ctrlReg: int, ctrlQubit: int, coeff) -> None:
        q = self._tPositionGrid(targetReg)
        phase_p = torch.exp(1j * coeff * q).to(self.dtype)
        phase_m = torch.exp(-1j * coeff * q).to(self.dtype)
        self._apply_conditional_phase_vectors(targetReg, ctrlReg, ctrlQubit, phase_p, phase_m)

    def _tApplyCondPhaseQ2(self, targetReg: int, ctrlReg: int, ctrlQubit: int, t) -> None:
        q = self._tPositionGrid(targetReg)
        phase_p = torch.exp(1j * t * q ** 2).to(self.dtype)
        phase_m = torch.exp(-1j * t * q ** 2).to(self.dtype)
        self._apply_conditional_phase_vectors(targetReg, ctrlReg, ctrlQubit, phase_p, phase_m)

    def _apply_conditional_phase_vectors(self, targetReg: int, ctrlReg: int, ctrlQubit: int, phase_p, phase_m) -> None:
        # Mirror CUDA kernelCPhaseX / kernelCPhaseX2: single pass on flat index, no reshape.
        all_idx = torch.arange(self.total_size, device=self.device, dtype=torch.long)
        # Local index within target register selects which phase value to apply
        target_local = (all_idx >> self.flwQbts[targetReg]) & (self.register_dims[targetReg] - 1)
        # Ctrl qubit bit in flat index
        ctrl_bit_pos = self.flwQbts[ctrlReg] + self.qubit_counts[ctrlReg] - 1 - ctrlQubit
        ctrl_bit = (all_idx >> ctrl_bit_pos) & 1
        # |0⟩ ctrl → phase_p,  |1⟩ ctrl → phase_m  (Z operator convention)
        phase = torch.where(ctrl_bit == 0, phase_p[target_local], phase_m[target_local])
        self.state = (self.state.reshape(-1) * phase).reshape(self.state.shape)

    def _tApplyCondQ1Q2(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, coeff) -> None:
        # Mirror CUDA kernelCPhaseXX: single pass on flat index, no reshape.
        q1 = self._tPositionGrid(reg1)
        q2 = self._tPositionGrid(reg2)
        all_idx = torch.arange(self.total_size, device=self.device, dtype=torch.long)
        local1 = (all_idx >> self.flwQbts[reg1]) & (self.register_dims[reg1] - 1)
        local2 = (all_idx >> self.flwQbts[reg2]) & (self.register_dims[reg2] - 1)
        ctrl_bit_pos = self.flwQbts[ctrlReg] + self.qubit_counts[ctrlReg] - 1 - ctrlQubit
        ctrl_bit = (all_idx >> ctrl_bit_pos) & 1
        phase_val = (coeff * q1[local1] * q2[local2]).to(self.dtype)
        phase = torch.where(ctrl_bit == 0,
                            torch.exp(1j * phase_val),
                            torch.exp(-1j * phase_val))
        self.state = (self.state.reshape(-1) * phase).reshape(self.state.shape)

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


