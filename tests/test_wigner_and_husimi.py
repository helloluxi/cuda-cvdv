import os
import sys

import numpy as np
import torch
from numpy import pi, sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CudaCvdv, SeparableState, TorchCvdv


class TestTorchPhaseSpaceClosedForm:
    NQUBITS = 8
    ALPHA = 1.0 + 0.5j
    WIGNER_ATOL = 1e-2
    HUSIMI_ATOL = 5e-3

    @staticmethod
    def _wigner_exact(x_grid, p_grid, alpha, dx):
        q0 = sqrt(2) * alpha.real
        p0 = sqrt(2) * alpha.imag
        x_mesh, p_mesh = np.meshgrid(x_grid, p_grid)
        return (dx / pi) * np.exp(-(x_mesh - q0) ** 2 - (p_mesh - p0) ** 2)

    @staticmethod
    def _husimi_exact(q_grid, p_grid, alpha):
        q0 = sqrt(2) * alpha.real
        p0 = sqrt(2) * alpha.imag
        q_mesh, p_mesh = np.meshgrid(q_grid, p_grid)
        return (1 / pi) * np.exp(-0.5 * (q_mesh - q0) ** 2 - 0.5 * (p_mesh - p0) ** 2)

    def _make_sim(self):
        sim = TorchCvdv([self.NQUBITS], device='cuda', dtype=np.complex128)
        sep = SeparableState([self.NQUBITS])
        sep.setCoherent(0, self.ALPHA)
        sim.initStateVector(sep)
        return sim

    @staticmethod
    def _wigner_grids(sim, reg_idx):
        dx = sim.grid_steps[reg_idx]
        dim = sim.register_dims[reg_idx]
        dp = pi / (dim * dx)
        x_grid = np.array([(k - (dim - 1) / 2) * dx for k in range(dim)])
        p_grid = np.array([(j - dim / 2) * dp for j in range(dim)])
        return x_grid, p_grid

    @staticmethod
    def _husimi_grids(sim, reg_idx):
        dx = sim.grid_steps[reg_idx]
        dim = sim.register_dims[reg_idx]
        dp = 2 * pi / (dim * dx)
        q_grid = np.array([(k - (dim - 1) / 2) * dx for k in range(dim)])
        p_grid = np.array([(j - dim / 2) * dp for j in range(dim)])
        return q_grid, p_grid

    def test_wigner_coherent_closed_form(self):
        sim = self._make_sim()
        wigner = sim.getWigner(0)
        x_grid, p_grid = self._wigner_grids(sim, 0)
        wigner_exact = self._wigner_exact(x_grid, p_grid, self.ALPHA, sim.grid_steps[0])
        np.testing.assert_allclose(wigner, wigner_exact, atol=self.WIGNER_ATOL)
        assert wigner.shape == (sim.register_dims[0], sim.register_dims[0])

    def test_husimi_coherent_closed_form(self):
        sim = self._make_sim()
        husimi_q = sim.getHusimiQ(0)
        q_grid, p_grid = self._husimi_grids(sim, 0)
        husimi_exact = self._husimi_exact(q_grid, p_grid, self.ALPHA)
        np.testing.assert_allclose(husimi_q, husimi_exact, atol=self.HUSIMI_ATOL)
        assert husimi_q.shape == (sim.register_dims[0], sim.register_dims[0])
        assert np.all(husimi_q >= -1e-10)

    def test_wigner_multiregister_coherent_closed_form(self):
        sim = TorchCvdv([self.NQUBITS, 1], device='cuda', dtype=np.complex128)
        sep = SeparableState([self.NQUBITS, 1])
        sep.setCoherent(0, self.ALPHA)
        sep.setZero(1)
        sim.initStateVector(sep)
        wigner = sim.getWigner(0)
        x_grid, p_grid = self._wigner_grids(sim, 0)
        wigner_exact = self._wigner_exact(x_grid, p_grid, self.ALPHA, sim.grid_steps[0])
        np.testing.assert_allclose(wigner, wigner_exact, atol=self.WIGNER_ATOL)
        assert wigner.shape == (sim.register_dims[0], sim.register_dims[0])

    def test_husimi_multiregister_coherent_closed_form(self):
        sim = TorchCvdv([self.NQUBITS, 1], device='cuda', dtype=np.complex128)
        sep = SeparableState([self.NQUBITS, 1])
        sep.setCoherent(0, self.ALPHA)
        sep.setZero(1)
        sim.initStateVector(sep)
        husimi_q = sim.getHusimiQ(0)
        q_grid, p_grid = self._husimi_grids(sim, 0)
        husimi_exact = self._husimi_exact(q_grid, p_grid, self.ALPHA)
        np.testing.assert_allclose(husimi_q, husimi_exact, atol=self.HUSIMI_ATOL)
        assert husimi_q.shape == (sim.register_dims[0], sim.register_dims[0])
        assert np.all(husimi_q >= -1e-10)


class TestTorchPhaseSpaceMultiRegisterConsistency:
    NQUBITS = 7
    ALPHA = 1.0 + 0.75j
    ATOL = 1e-2
    RTOL = 1e-2

    def _make_entangled_pair(self):
        dims = [self.NQUBITS, 1]
        sep_plus = SeparableState(dims)
        sep_plus.setCoherent(0, self.ALPHA)
        sep_plus.setZero(1)
        psi_plus = sep_plus.register_arrays[0]
        sep_minus = SeparableState(dims)
        sep_minus.setCoherent(0, -self.ALPHA)
        sep_minus.setZero(1)
        psi_minus = sep_minus.register_arrays[0]
        state = torch.stack((psi_plus, psi_minus), dim=1) / np.sqrt(2)
        state = state.to(device='cuda', dtype=torch.cdouble).contiguous()

        cuda_sim = CudaCvdv(dims)
        cuda_sim.initFromArray(state)

        torch_sim = TorchCvdv(dims, device='cuda', dtype=np.complex128)
        torch_sim.state = state

        return cuda_sim, torch_sim

    def test_wigner_entangled_state_matches_cuda(self):
        cuda_sim, torch_sim = self._make_entangled_pair()
        cuda_wigner = cuda_sim.getWigner(0)
        torch_wigner = torch_sim.getWigner(0)
        np.testing.assert_allclose(torch_wigner, cuda_wigner, atol=self.ATOL, rtol=self.RTOL)

    def test_husimi_entangled_state_matches_cuda(self):
        cuda_sim, torch_sim = self._make_entangled_pair()
        cuda_husimi = cuda_sim.getHusimiQ(0)
        torch_husimi = torch_sim.getHusimiQ(0)
        np.testing.assert_allclose(torch_husimi, cuda_husimi, atol=self.ATOL, rtol=self.RTOL)
        assert np.all(torch_husimi >= -1e-10)
