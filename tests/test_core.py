
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from numpy import pi, sqrt
from src import CVDV, SeparableState

class TestCoreOperations:
    """Test core quantum operations using inner product validation."""

    def test_vacuum_state_initialization(self):
        """Test that vacuum state |0⟩ = coherent state |α=0⟩."""
        sim = CVDV([10])
        sep = SeparableState([10])
        sep.setFock(0, 0)
        sim.initStateVector(sep)
        # |0⟩ Fock == coherent |α=0⟩
        sep2 = SeparableState([10])
        sep2.setCoherent(0, 0)
        fid = sim.getFidelity(sep2)
        assert np.abs(fid - 1.0) < 1e-10, f"Vacuum state fidelity: {fid}"

    def test_displacement_on_vacuum(self):
        """Test displacement operator: D(α)|0⟩ = |α⟩."""
        alpha = (np.random.random() - 0.5) * 4 + 1j * (np.random.random() - 0.5) * 4

        sim = CVDV([10])
        sep = SeparableState([10])
        sep.setFock(0, 0)
        sim.initStateVector(sep)
        sim.d(0, alpha)

        sep2 = SeparableState([10])
        sep2.setCoherent(0, alpha)
        fid = sim.getFidelity(sep2)
        assert np.abs(fid - 1.0) < 1e-10, f"D({alpha})|0⟩ should equal |{alpha}⟩, fidelity: {fid}"

    def test_displacement_composition(self):
        """Test that D(α)D(β)|0⟩ = D(α+β)|0⟩."""
        alpha = (np.random.random() - 0.5) * 4 + 1j * (np.random.random() - 0.5) * 4
        beta = (np.random.random() - 0.5) * 4 + 1j * (np.random.random() - 0.5) * 4

        sim = CVDV([10])
        sep = SeparableState([10])
        sep.setFock(0, 0)
        sim.initStateVector(sep)
        sim.d(0, beta)
        sim.d(0, alpha)

        sep2 = SeparableState([10])
        sep2.setCoherent(0, alpha + beta)
        fid = sim.getFidelity(sep2)
        assert np.abs(fid - 1.0) < 1e-10, f"D({alpha})D({beta})|0⟩ should equal D({alpha+beta})|0⟩, fidelity: {fid}"

    def test_fourier_transform(self):
        """Test that FT_Q2P followed by FT_P2Q returns to original state."""
        nFock = np.random.randint(0, 20)

        sim = CVDV([10])
        sep = SeparableState([10])
        sep.setFock(0, nFock)
        sim.initStateVector(sep)

        # FT maps |nFock⟩ → i^n |nFock⟩, so fidelity stays 1
        sim.ftQ2P(0)
        fid = sim.getFidelity(sep)
        assert np.abs(fid - 1.0) < 1e-10, f"FT_Q2P fidelity for |{nFock}⟩: {fid}"

        sim.ftP2Q(0)
        fid = sim.getFidelity(sep)
        assert np.abs(fid - 1.0) < 1e-10, f"FT_P2Q fidelity for |{nFock}⟩: {fid}"

    def test_qubit_gates(self):
        """Test all available qubit gates: Hadamard and Pauli rotations (Rx, Ry, Rz)."""
        # Hadamard: H|0⟩ = |+⟩
        sim = CVDV([1])
        sep = SeparableState([1])
        sep.setZero(0)
        sim.initStateVector(sep)
        sim.h(0, 0)
        sep_plus = SeparableState([1])
        sep_plus.setUniform(0)
        fid = sim.getFidelity(sep_plus)
        assert np.abs(fid - 1.0) < 1e-10, f"H|0⟩ should equal |+⟩, fidelity: {fid}"

        # Rx(π)|+⟩ = |+⟩
        sim.rx(0, 0, pi)
        fid = sim.getFidelity(sep_plus)
        assert np.abs(fid - 1.0) < 1e-10, f"Rx(π)|+⟩ should equal |+⟩, fidelity: {fid}"

        # Rz(π)|+⟩ = |-⟩
        sim.rz(0, 0, pi)
        sep_minus = SeparableState([1])
        sep_minus.setCoeffs(0, [1/sqrt(2), -1/sqrt(2)])
        fid = sim.getFidelity(sep_minus)
        assert np.abs(fid - 1.0) < 1e-10, f"Rz(π)|+⟩ should equal |-⟩, fidelity: {fid}"

        # Ry(π)|-⟩ = i|+⟩, fidelity with |+⟩ = 1
        sim.ry(0, 0, pi)
        fid = sim.getFidelity(sep_plus)
        assert np.abs(fid - 1.0) < 1e-10, f"Ry(π)|-⟩ should equal i|+⟩, fidelity: {fid}"

        # Measure H Rz(theta)|+⟩ with random theta
        theta = (np.random.random() - 0.5) * 2 * pi
        sim.rz(0, 0, theta)
        sim.h(0, 0)
        probs = sim.m(0)
        assert np.abs(probs[0] - np.cos(theta/2)**2) < 1e-10, f"Measurement probability mismatch for theta={theta}: {probs[0]}"
        assert np.abs(probs[1] - np.sin(theta/2)**2) < 1e-10, f"Measurement probability mismatch for theta={theta}: {probs[1]}"

    def test_conditional_displacement_commutator(self):
        """Test CD commutator identity: Rz(4xy) D(-iy) CD(-x) D(iy) CD(x) = I"""
        x = (np.random.random() - 0.5) * 2
        y = (np.random.random() - 0.5) * 2

        sim = CVDV([1, 10])
        sep = SeparableState([1, 10])
        sep.setUniform(0)
        sep.setFock(1, 0)
        sim.initStateVector(sep)

        sim.cd(targetReg=1, ctrlReg=0, ctrlQubit=0, alpha=x)
        sim.d(1, 1j * y)
        sim.cd(targetReg=1, ctrlReg=0, ctrlQubit=0, alpha=-x)
        sim.d(1, -1j * y)
        sim.rz(0, 0, 4 * x * y)

        fid = sim.getFidelity(sep)
        assert np.abs(fid - 1.0) < 1e-10, f"CD commutator should be identity for x={x}, y={y}, fidelity: {fid}"

    def test_rotation(self):
        """Test phase space rotation on coherent state."""
        alpha = 3.0

        sim = CVDV([10])
        sep = SeparableState([10])
        sep.setCoherent(0, alpha)
        sim.initStateVector(sep)

        theta = (np.random.random() - 0.5) * 10
        sim.r(0, theta)
        sep2 = SeparableState([10])
        sep2.setCoherent(0, alpha * np.exp(-1j * theta))
        fid = sim.getFidelity(sep2)
        assert np.abs(fid - 1.0) < 1e-10, f"Rotation({theta}) fidelity: {fid}"

    def test_squeeze(self):
        """Test squeezing operation conjugated displacement."""
        alpha = (np.random.random() - 0.5) * 6
        r = (np.random.random() - 0.5) * 2

        sim = CVDV([10])
        sep = SeparableState([10])
        sep.setCoherent(0, 3j)
        sim.initStateVector(sep)
        sim.s(0, -r)
        sim.d(0, alpha)
        sim.s(0, r)
        sim.d(0, -alpha*np.exp(-r))
        fid = sim.getFidelity(sep)
        assert np.abs(fid - 1.0) < 1e-10, f"Squeezing conjugated displacement fidelity for alpha={alpha}, r={r}: {fid}"

    def test_beam_splitter(self):
        """Test beam splitter operation on Fock state + vacuum state pair."""
        nFock = np.random.randint(3, 10)
        sim = CVDV([10, 10])
        sep = SeparableState([10, 10])
        sep.setFock(0, nFock)
        sep.setFock(1, 0)
        sim.initStateVector(sep)
        theta = (np.random.random() - 0.5) * 10
        sim.bs(0, 1, theta)
        p1 = np.cos(theta/2)**2
        p2 = np.sin(theta/2)**2
        ips = np.zeros(nFock + 1)
        theoretical = p2**nFock
        for j in range(nFock+1):
            if j > 0:
                theoretical *= (nFock - j + 1) / j * p1 / p2
            sep2 = SeparableState([10, 10])
            sep2.setFock(0, j)
            sep2.setFock(1, nFock - j)
            ips[j] = theoretical - sim.getFidelity(sep2)

        assert np.allclose(ips, 0, atol=1e-10), f"BS({theta}) on |{nFock}⟩|0⟩ fidelity deviations: {ips}"

    def test_conditional_rotation(self):
        nFock = np.random.randint(0, 32)
        theta = (np.random.random() - 0.5) * 10

        sim = CVDV([1, 10])
        sep = SeparableState([1, 10])
        sep.setUniform(0)
        sep.setFock(1, nFock)
        sim.initStateVector(sep)

        sim.cr(targetReg=1, ctrlReg=0, ctrlQubit=0, theta=theta)
        sim.rz(0, 0, theta * -(2*nFock + 1))

        fid = sim.getFidelity(sep)
        assert np.abs(fid - 1.0) < 1e-10, f"CR({theta}) on |+⟩|{nFock}⟩ fidelity: {fid}"

    def test_conditional_squeeze(self):
        r = (np.random.random() - 0.5) * 2
        alpha = (np.random.random() - 0.5) * 2

        sim = CVDV([1, 10])
        sep = SeparableState([1, 10])
        sep.setUniform(0)
        sep.setFock(1, 0)
        sim.initStateVector(sep)

        sim.cs(targetReg=1, ctrlReg=0, ctrlQubit=0, r=r)
        sim.d(1, alpha)
        sim.cs(targetReg=1, ctrlReg=0, ctrlQubit=0, r=-r)
        sim.d(1, -alpha * np.cosh(r))
        sim.cd(1, 0, 0, -alpha * np.sinh(r))

        fid = sim.getFidelity(sep)
        assert np.abs(fid - 1.0) < 1e-10, f"CS({r}) on |+⟩|vac⟩ fidelity: {fid}"

    def test_conditional_beam_splitter(self):
        """Test beam splitter operation on Fock state + vacuum state pair."""
        nFock = np.random.randint(3, 10)
        sim = CVDV([1, 10, 10])
        sep = SeparableState([1, 10, 10])
        sep.setUniform(0)
        sep.setFock(1, nFock)
        sep.setFock(2, 0)
        sim.initStateVector(sep)
        theta = (np.random.random() - 0.5) * 10
        sim.cbs(1, 2, 0, 0, theta)
        sim.cp(2, 0, 0)
        sim.bs(1, 2, -theta)
        fid = sim.getFidelity(sep)
        assert np.abs(fid - 1.0) < 1e-10, f"CBS({theta}) on |+⟩|{nFock}⟩|0⟩ fidelity: {fid}"

class TestPhaseSpaceClosedForm:
    """Compare computed Wigner and Husimi Q against exact closed-form solutions for coherent states.

    Conventions (matching the simulator's discrete representation):
      The state is stored with norm sum_k |psi[k]|^2 = 1 (discrete), so
      psi_cont(x) = psi_stored[k] / sqrt(dx).  This introduces a factor of dx
      into the Wigner function relative to the continuous formula.

      Coherent state |β⟩: q₀ = √2 Re β, p₀ = √2 Im β  (quadrature units).

      Simulator Wigner (discrete-normalised, peak = dx/π):
        W_β(x, p) = (dx/π) exp(-(x - q₀)² - (p - p₀)²)

      Continuous-physics Wigner (peak = 2/π):
        W_β(x, p) = (2/π) exp(-2|α - β|²),  α = (x + ip)/√2
                  = (2/π) exp(-(x - q₀)² - (p - p₀)²)

      The simulator output equals (dx/2) × the continuous Wigner.

      Simulator Husimi Q (discrete-normalised):
        Q_β(x, p) = (dx / (2π)) exp(-½(x - q₀)² - ½(p - p₀)²)

      Continuous Husimi Q (peak = 1/π):
        Q_β(α) = (1/π) exp(-|α - β|²)   (α = (x + ip)/√2)
    """

    # 8 qubits → dim=256, dx≈0.157.  Use bound=3.0 so the coherent state peak
    # (at q₀=√2≈1.41, p₀=√2·0.5≈0.71) is well away from the boundary.
    NQUBITS = 8
    ALPHA = 1.0 + 0.5j
    BOUND = 3.0
    WIGNER_ATOL = 1e-2   # Wigner can go negative; tails have O(dx²) discretization noise
    HUSIMI_ATOL = 5e-3   # Husimi is non-negative and smoother

    @staticmethod
    def _wigner_exact(x_grid, p_grid, alpha, dx):
        """Simulator Wigner: (dx/π) exp(-(x-q₀)²-(p-p₀)²), peak = dx/π."""
        q0 = sqrt(2) * alpha.real
        p0 = sqrt(2) * alpha.imag
        X, P = np.meshgrid(x_grid, p_grid)   # shape (N_p, N_x)
        return (dx / pi) * np.exp(-(X - q0) ** 2 - (P - p0) ** 2)

    @staticmethod
    def _husimi_exact(x_grid, p_grid, alpha):
        """Continuous Husimi Q: (1/π) exp(-½(x-q₀)²-½(p-p₀)²), peak = 1/π."""
        q0 = sqrt(2) * alpha.real
        p0 = sqrt(2) * alpha.imag
        X, P = np.meshgrid(x_grid, p_grid)   # shape (N_p, N_q)
        return (1 / pi) * np.exp(-0.5 * (X - q0) ** 2 - 0.5 * (P - p0) ** 2)

    def _make_sim(self):
        sim = CVDV([self.NQUBITS])
        sep = SeparableState([self.NQUBITS])
        sep.setCoherent(0, self.ALPHA)
        sim.initStateVector(sep)
        return sim

    @staticmethod
    def _wigner_grids(sim, regIdx):
        """Return (x_grid, p_grid) for native N×N grid."""
        dx = sim.grid_steps[regIdx]
        dim = sim.register_dims[regIdx]
        dp = pi / (dim * dx)
        x_grid = np.array([(k - (dim-1)/2) * dx for k in range(dim)])
        p_grid = np.array([(j - dim/2) * dp for j in range(dim)])
        return x_grid, p_grid

    @staticmethod
    def _husimi_grids(sim, regIdx):
        """Return (q_grid, p_grid) for native N×N grid."""
        dx = sim.grid_steps[regIdx]
        dim = sim.register_dims[regIdx]
        dp = 2 * pi / (dim * dx)
        q_grid = np.array([(k - (dim-1)/2) * dx for k in range(dim)])
        p_grid = np.array([(j - dim/2) * dp for j in range(dim)])
        return q_grid, p_grid

    def test_wigner_coherent_closed_form(self):
        """W for coherent state matches (dx/π) exp(-(x-q₀)²-(p-p₀)²) to WIGNER_ATOL."""
        sim = self._make_sim()
        W = sim.getWigner(0)                         # no bound
        x_grid, p_grid = self._wigner_grids(sim, 0)  # no bound
        W_exact = self._wigner_exact(x_grid, p_grid, self.ALPHA, sim.grid_steps[0])
        np.testing.assert_allclose(W, W_exact, atol=self.WIGNER_ATOL,
                                   err_msg="Wigner vs closed form failed")
        assert W.shape == (sim.register_dims[0], sim.register_dims[0])
        del sim

    def test_husimi_coherent_closed_form(self):
        """Q for coherent state matches (1/π) exp(-½(x-q₀)²-½(p-p₀)²) to HUSIMI_ATOL."""
        sim = self._make_sim()
        Q = sim.getHusimiQ(0)                        # no bound
        q_grid, p_grid = self._husimi_grids(sim, 0)  # no bound
        Q_exact = self._husimi_exact(q_grid, p_grid, self.ALPHA)
        np.testing.assert_allclose(Q, Q_exact, atol=self.HUSIMI_ATOL,
                                   err_msg="Husimi Q vs closed form failed")
        assert Q.shape == (sim.register_dims[0], sim.register_dims[0])
        assert np.all(Q >= -1e-10)   # Husimi is non-negative
        del sim


class TestPhotonNumber:
    """Test getPhotonNumber against exact analytic values.

    Coherent state |α⟩: <n> = |α|²
    Fock state |n⟩:     <n> = n
    """

    NQUBITS = 8
    ATOL = 1e-2  # discretization noise comparable to Wigner tests

    def _make_sim(self, nqubits=None):
        nq = nqubits if nqubits is not None else self.NQUBITS
        sim = CVDV([nq])
        return sim

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5 + 1.0j, -0.5 + 2.0j, 2.0])
    def test_coherent_state(self, alpha):
        """<n> = |α|² for coherent state |α⟩."""
        sim = self._make_sim()
        sep = SeparableState([self.NQUBITS])
        sep.setCoherent(0, alpha)
        sim.initStateVector(sep)
        n_meas = sim.getPhotonNumber(0)
        n_exact = abs(alpha) ** 2
        assert abs(n_meas - n_exact) < self.ATOL, \
            f"Coherent |α={alpha}⟩: <n>={n_meas:.6f}, expected {n_exact:.6f}"
        del sim

    @pytest.mark.parametrize("n_fock", [0, 1, 2, 3, 5])
    def test_fock_state(self, n_fock):
        """<n> = n for Fock state |n⟩."""
        sim = self._make_sim()
        sep = SeparableState([self.NQUBITS])
        sep.setFock(0, n_fock)
        sim.initStateVector(sep)
        n_meas = sim.getPhotonNumber(0)
        assert abs(n_meas - n_fock) < self.ATOL, \
            f"Fock |{n_fock}⟩: <n>={n_meas:.6f}, expected {n_fock}"
        del sim

    def test_state_restored_after_call(self):
        """getPhotonNumber must not modify the state (ftQ2P/ftP2Q are undone)."""
        sim = self._make_sim()
        sep = SeparableState([self.NQUBITS])
        sep.setCoherent(0, 1.0 + 0.5j)
        sim.initStateVector(sep)
        state_before = sim.getState().copy()
        sim.getPhotonNumber(0)
        state_after = sim.getState()
        np.testing.assert_allclose(state_before, state_after, atol=1e-6,
                                   err_msg="State changed after getPhotonNumber")
        del sim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
