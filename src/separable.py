"""
SeparableState — per-register state initialization, backend-agnostic.

Stores each register's 1-D wavefunction as a torch.cdouble CUDA tensor so that
the CUDA backend can consume the raw device pointer without a CPU round-trip.
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch


class SeparableState:
    """Per-register state container (separable product state).

    Usage::

        sep = SeparableState([8, 1])   # two registers
        sep.setCoherent(0, 2.0 + 1j)
        sep.setZero(1)
        # Then pass to CVDV:
        sim = CVDV([8, 1])
        sim.initStateVector(sep)
    """

    def __init__(self, numQubits_list: List[int], device: str = 'cuda') -> None:
        self.device = torch.device(device)
        self.num_registers = len(numQubits_list)
        self.qubit_counts = list(numQubits_list)
        self.register_dims = [1 << n for n in numQubits_list]
        self.grid_steps = [(2 * torch.pi / d) ** 0.5 for d in self.register_dims]
        self.register_arrays: List[Optional[torch.Tensor]] = [None] * self.num_registers

    # ------------------------------------------------------------------ helpers

    def _x(self, regIdx: int) -> torch.Tensor:
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        idx = torch.arange(dim, device=self.device, dtype=torch.float64)
        return (idx - (dim - 1) * 0.5) * dx

    # ------------------------------------------------------------------ setters

    def setZero(self, regIdx: int) -> None:
        """Set register to |0⟩ (vacuum / ground state in position basis)."""
        arr = torch.zeros(self.register_dims[regIdx], dtype=torch.cdouble, device=self.device)
        arr[0] = 1.0
        self.register_arrays[regIdx] = arr

    def setCoherent(self, regIdx: int, alpha: Union[complex, float, int]) -> None:
        """Set register to coherent state |α⟩."""
        if isinstance(alpha, (int, float)):
            alpha = complex(alpha, 0.0)
        dx = self.grid_steps[regIdx]
        x = self._x(regIdx)
        q = 2 ** 0.5 * alpha.real
        p = 2 ** 0.5 * alpha.imag
        gauss = torch.exp(-0.5 * (x - q) ** 2)
        phase = torch.exp(1j * (p * x - p * q / 2.0)).to(torch.cdouble)
        psi = (torch.pi ** (-0.25)) * (dx ** 0.5) * gauss * phase
        self.register_arrays[regIdx] = psi

    def setFock(self, regIdx: int, n: int) -> None:
        """Set register to Fock state |n⟩."""
        dx = self.grid_steps[regIdx]
        x = self._x(regIdx)
        psi_prev = torch.zeros_like(x)
        psi_curr = torch.exp(-0.5 * x * x) * (torch.pi ** (-0.25)) * (dx ** 0.5)
        for k in range(1, n + 1):
            psi_next = (2.0 / k) ** 0.5 * x * psi_curr - ((k - 1.0) / k) ** 0.5 * psi_prev
            psi_prev = psi_curr
            psi_curr = psi_next
        self.register_arrays[regIdx] = psi_curr.to(torch.cdouble)

    def setUniform(self, regIdx: int) -> None:
        """Set register to uniform superposition 1/√N Σ|k⟩."""
        dim = self.register_dims[regIdx]
        arr = torch.ones(dim, dtype=torch.cdouble, device=self.device) / (dim ** 0.5)
        self.register_arrays[regIdx] = arr

    def setFocks(self, regIdx: int, coeffs: Union[Sequence[complex], torch.Tensor]) -> None:
        """Set register to superposition of Fock states Σ cₙ|n⟩ (normalized)."""
        if not isinstance(coeffs, torch.Tensor):
            coeffs_t = torch.tensor(list(coeffs), dtype=torch.cdouble, device=self.device)
        else:
            coeffs_t = coeffs.to(dtype=torch.cdouble, device=self.device)
        if not torch.any(torch.abs(coeffs_t) > 1e-12):
            self.setZero(regIdx)
            return
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        x = self._x(regIdx)
        max_n = len(coeffs_t)
        fock_states = torch.zeros((max_n, dim), dtype=torch.float64, device=self.device)
        fock_states[0] = torch.exp(-0.5 * x * x) * (torch.pi ** (-0.25)) * (dx ** 0.5)
        if max_n > 1:
            fock_states[1] = (2.0 ** 0.5) * x * fock_states[0]
            for k in range(2, max_n):
                fock_states[k] = ((2.0 / k) ** 0.5 * x * fock_states[k - 1]
                                  - ((k - 1.0) / k) ** 0.5 * fock_states[k - 2])
        state = torch.sum(coeffs_t.unsqueeze(1) * fock_states.to(torch.cdouble), dim=0)
        self.register_arrays[regIdx] = state

    def setCoeffs(self, regIdx: int, coeffs: Union[Sequence[complex], torch.Tensor]) -> None:
        """Set register to an arbitrary pre-normalized coefficient array."""
        if not isinstance(coeffs, torch.Tensor):
            self.register_arrays[regIdx] = torch.tensor(
                list(coeffs), dtype=torch.cdouble, device=self.device
            )
        else:
            self.register_arrays[regIdx] = coeffs.to(dtype=torch.cdouble, device=self.device)

    def setCat(self, regIdx: int,
               cat_states: Sequence[Tuple[Union[complex, float], Union[complex, float]]]) -> None:
        """Set register to normalized cat state Σ cₖ|αₖ⟩."""
        dx = self.grid_steps[regIdx]
        x = self._x(regIdx)
        alphas = [complex(a) for a, _ in cat_states]
        coeffs = [complex(c) for _, c in cat_states]
        q_vals = torch.tensor(
            [2 ** 0.5 * a.real for a in alphas], device=self.device, dtype=torch.float64)
        p_vals = torch.tensor(
            [2 ** 0.5 * a.imag for a in alphas], device=self.device, dtype=torch.float64)
        coeffs_t = torch.tensor(coeffs, dtype=torch.cdouble, device=self.device)
        gauss = torch.exp(-0.5 * (x[None, :] - q_vals[:, None]) ** 2)
        phase = p_vals[:, None] * x[None, :] - (p_vals * q_vals)[:, None] / 2.0
        coh = (torch.pi ** (-0.25)) * (dx ** 0.5) * gauss * torch.exp(1j * phase).to(torch.cdouble)
        state = torch.sum(coeffs_t[:, None] * coh, dim=0)
        norm = torch.sqrt(torch.sum(torch.abs(state) ** 2))
        self.register_arrays[regIdx] = state / norm

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        """Raise if any register has not been initialized."""
        for i, arr in enumerate(self.register_arrays):
            if arr is None:
                raise RuntimeError(f"Register {i} not initialized in SeparableState")
