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
        self.state = state.permute(*[perm.index(i) for i in range(self.num_registers)])

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

    def m(self, regIdx: int) -> npt.NDArray[np.float64]:
        probs = torch.abs(self.state) ** 2
        dims_to_sum = [i for i in range(self.num_registers) if i != regIdx]
        for d in sorted(dims_to_sum, reverse=True):
            probs = torch.sum(probs, dim=d)
        return probs.detach().cpu().numpy()

    def jointMeasure(self, reg1Idx: int, reg2Idx: int) -> npt.NDArray[np.float64]:
        probs = torch.abs(self.state) ** 2
        dims_to_sum = [i for i in range(self.num_registers) if i not in [reg1Idx, reg2Idx]]
        for d in sorted(dims_to_sum, reverse=True):
            probs = torch.sum(probs, dim=d)
        if reg1Idx > reg2Idx:
            probs = probs.transpose(0, 1)
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

    def getNorm(self) -> float:
        return float(torch.sqrt(torch.sum(torch.abs(self.state) ** 2)).cpu().item())

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
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        psi = self._get_reduced_state_cpu(regIdx)
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
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        psi = self._get_reduced_state_cpu(regIdx)
        x_grid = np.array([(k - (dim - 1) / 2) * dx for k in range(dim)])
        accum = np.zeros((dim, dim), dtype=np.float64)
        for qi in range(dim):
            sample_q = (qi - (dim - 1) / 2) * dx
            window = (pi ** (-0.25)) * np.exp(-0.5 * (x_grid - sample_q) ** 2) * np.sqrt(dx)
            h = psi * window
            H = np.fft.fft(h)
            accum[qi, :] += np.abs(H) ** 2
        husimiQ = np.zeros((dim, dim), dtype=np.float64)
        for jc in range(dim):
            k_fft = (jc + dim // 2) % dim
            for qi in range(dim):
                husimiQ[jc, qi] = accum[qi, k_fft] / pi
        return husimiQ

    def _get_reduced_state_cpu(self, regIdx: int) -> npt.NDArray[np.complex128]:
        if self.num_registers == 1:
            return self.state.detach().cpu().numpy()
        perm = list(range(self.num_registers))
        perm[0], perm[regIdx] = perm[regIdx], perm[0]
        state = self.state.permute(*perm)
        rho_diag = torch.sum(torch.abs(state) ** 2, dim=tuple(range(1, self.num_registers)))
        return torch.sqrt(rho_diag).detach().cpu().numpy()

    def _tPositionGrid(self, regIdx: int):
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        idx = torch.arange(dim, device=self.device, dtype=torch.float64)
        return (idx - (dim - 1) * 0.5) * dx

    def _tApplyPhase(self, regIdx: int, phase) -> None:
        shape = [1] * self.num_registers
        shape[regIdx] = -1
        self.state = self.state * phase.reshape(shape)

    def _tApplyQubitGate(self, regIdx: int, targetQubit: int, gate_matrix) -> None:
        dim = self.register_dims[regIdx]
        n_q = self.qubit_counts[regIdx]
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
        self.state = state.permute(*[perm.index(i) for i in range(self.num_registers)])

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
        ctrl_dim = self.register_dims[ctrlReg]
        n_ctrl = self.qubit_counts[ctrlReg]
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_target = targetReg if targetReg != 0 else ctrlReg
        perm[1], perm[actual_target] = perm[actual_target], perm[1]
        state = self.state.permute(*perm)
        state = state.reshape([2] * n_ctrl + list(state.shape[1:]))
        qperm = list(range(n_ctrl))
        qperm[0], qperm[ctrlQubit] = qperm[ctrlQubit], qperm[0]
        state = state.permute(*qperm + list(range(n_ctrl, len(state.shape))))
        pshape = [1] * len(state[0].shape)
        pshape[n_ctrl - 1] = -1
        state[0] = state[0] * phase_p.reshape(pshape)
        state[1] = state[1] * phase_m.reshape(pshape)
        state = state.permute(*[qperm.index(i) for i in range(n_ctrl)] + list(range(n_ctrl, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl:]))
        self.state = state.permute(*[perm.index(i) for i in range(self.num_registers)])

    def _tApplyCondQ1Q2(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, coeff) -> None:
        q1 = self._tPositionGrid(reg1)
        q2 = self._tPositionGrid(reg2)
        pm_p = torch.exp(1j * coeff * q1[:, None] * q2[None, :]).to(self.dtype)
        pm_m = torch.exp(-1j * coeff * q1[:, None] * q2[None, :]).to(self.dtype)
        ctrl_dim = self.register_dims[ctrlReg]
        n_ctrl = self.qubit_counts[ctrlReg]
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_reg1 = reg1 if reg1 != 0 else ctrlReg
        perm[1], perm[actual_reg1] = perm[actual_reg1], perm[1]
        actual_reg2_idx = perm.index(reg2)
        perm[2], perm[actual_reg2_idx] = perm[actual_reg2_idx], perm[2]
        state = self.state.permute(*perm)
        state = state.reshape([2] * n_ctrl + list(state.shape[1:]))
        qperm = list(range(n_ctrl))
        qperm[0], qperm[ctrlQubit] = qperm[ctrlQubit], qperm[0]
        state = state.permute(*qperm + list(range(n_ctrl, len(state.shape))))
        shape_p = [1] * len(state[0].shape)
        shape_p[n_ctrl - 1] = self.register_dims[reg1]
        shape_p[n_ctrl] = self.register_dims[reg2]
        state[0] = state[0] * pm_p.reshape(shape_p)
        state[1] = state[1] * pm_m.reshape(shape_p)
        state = state.permute(*[qperm.index(i) for i in range(n_ctrl)] + list(range(n_ctrl, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl:]))
        self.state = state.permute(*[perm.index(i) for i in range(self.num_registers)])

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


