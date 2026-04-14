"""
Ground-truth tests for position-basis and momentum-basis measurements
of coherent Gaussian states.

The m(regIdx) call returns the marginal probability distribution
  P(k) = |ψ_k|²  for position-basis indices k = 0..N-1,
with physical position x_k = (k - (N-1)/2) * dx,  dx = sqrt(2π/N).

Coherent-state analytical results (all exact in the continuum limit):

  Position axis:   P(x_k) ∝ exp(-(x_k - q₀)²),   q₀ = sqrt(2) Re(α)
  Momentum axis:   P(k)   = |FT[ψ](k)|²            (numerically exact)

For multi-register product states the joint distribution factorises as
  P(i, j) = P₀(i) · P₁(j).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from numpy import pi, sqrt

from src import CudaCvdv, SeparableState


# ── helpers ──────────────────────────────────────────────────────────────────

def _position_grid(nq: int):
    N = 1 << nq
    dx = sqrt(2 * pi / N)
    x = (np.arange(N) - (N - 1) / 2) * dx
    return x, dx


def _coherent_wavefunction(nq: int, alpha: complex) -> np.ndarray:
    """Discrete coherent-state wavefunction ψ(x_k)."""
    x, _ = _position_grid(nq)
    q0 = sqrt(2) * alpha.real
    p0 = sqrt(2) * alpha.imag
    psi = np.exp(-0.5 * (x - q0) ** 2 + 1j * p0 * x)
    psi /= np.linalg.norm(psi)
    return psi


def _apply_ftQ2P_numpy(psi: np.ndarray, nq: int) -> np.ndarray:
    """Replicate gates.cu::cvdvFtQ2P in numpy.

    Steps: pre-phase → CUFFT_FORWARD (numpy fft) / √N → post-phase.
    The global phase is omitted; it cancels in |·|².
    """
    N = 1 << nq
    dx = sqrt(2 * pi / N)
    x = (np.arange(N) - (N - 1) / 2) * dx
    c = pi * (N - 1) / (N * dx)        # phase coefficient  π(N-1)/(N·dx)
    psi = psi * np.exp(1j * c * x)     # pre-phase
    psi = np.fft.fft(psi) / sqrt(N)    # forward DFT + normalise
    psi = psi * np.exp(1j * c * x)     # post-phase (same formula on output indices)
    return psi


def _gaussian_probs(x: np.ndarray, center: float) -> np.ndarray:
    """Normalised Gaussian P(k) ∝ exp(-(x_k - center)²)."""
    unnorm = np.exp(-(x - center) ** 2)
    return unnorm / unnorm.sum()


def _make_coherent(nq: int, alpha: complex) -> CudaCvdv:
    sim = CudaCvdv([nq])
    sep = SeparableState([nq])
    sep.setCoherent(0, alpha)
    sim.initStateVector(sep)
    return sim


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCoherentMeasurementGroundTruth:
    """m(regIdx) ground-truth checks against analytic Gaussian distributions."""

    NQUBITS  = 8      # N = 256; good Gaussian resolution with small boundary artefacts
    ATOL     = 3e-3   # per-bin absolute tolerance (discretisation ~ O(dx²))
    ATOL_MOM = 5e-3   # momentum axis: FT doubles the discretisation error

    # ── position axis ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("alpha", [
        0.0,            # vacuum: Gaussian at origin
        0.7 + 0.0j,     # real displacement  → peak shifts in x
        0.0 + 0.9j,     # imaginary α        → no shift in x (phase only)
        0.6 + 0.8j,
        -0.5 + 1.1j,
    ])
    def test_coherent_position_probs(self, alpha):
        """P(x_k) ∝ exp(-(x_k - q₀)²),  q₀ = √2 Re(α)."""
        alpha = complex(alpha)
        sim = _make_coherent(self.NQUBITS, alpha)
        probs = sim.m(0)
        x, _ = _position_grid(self.NQUBITS)
        q0 = sqrt(2) * alpha.real
        P_expected = _gaussian_probs(x, q0)
        assert abs(probs.sum() - 1.0) < 1e-10, "probs must sum to 1"
        np.testing.assert_allclose(
            probs, P_expected, atol=self.ATOL,
            err_msg=f"|α={alpha}⟩ position distribution mismatch")
        del sim

    def test_imaginary_alpha_no_x_shift(self):
        """Pure imaginary α shifts p, not x: position peak stays at origin."""
        alpha = 1.5j
        sim = _make_coherent(self.NQUBITS, alpha)
        probs = sim.m(0)
        x, dx = _position_grid(self.NQUBITS)
        # Peak bin should be within one grid step of x=0
        x_peak = x[np.argmax(probs)]
        assert abs(x_peak) <= dx + 1e-10, \
            f"Pure imaginary α should not shift x peak; got x_peak={x_peak:.4f}"
        del sim

    # ── momentum axis (after ftQ2P) ───────────────────────────────────────────

    @pytest.mark.parametrize("alpha", [
        1j,             # pure imaginary → peaked in p
        0.5 + 1.5j,
        -0.4 + 1.0j,
        0.0 + 2.0j,     # larger p displacement
    ])
    def test_coherent_momentum_probs(self, alpha):
        """After ftQ2P, m(0) matches |FT[ψ_analytic]|² exactly."""
        alpha = complex(alpha)
        sim = _make_coherent(self.NQUBITS, alpha)
        sim.ftQ2P(0)
        probs = sim.m(0)

        psi = _coherent_wavefunction(self.NQUBITS, alpha)
        psi_ft = _apply_ftQ2P_numpy(psi, self.NQUBITS)
        P_expected = np.abs(psi_ft) ** 2
        P_expected /= P_expected.sum()

        assert abs(probs.sum() - 1.0) < 1e-10
        np.testing.assert_allclose(
            probs, P_expected, atol=self.ATOL_MOM,
            err_msg=f"|α={alpha}⟩ momentum distribution mismatch")
        del sim

    # ── multi-register: marginals and joint distributions ────────────────────

    def test_marginal_independence(self):
        """m(0) is independent of the state of register 1 for product states."""
        nq0, nq1 = 6, 6
        alpha0 = 0.8 + 0.3j
        results = []
        for alpha1 in [0.0, 1.5 - 0.8j, -1.0 + 0.5j]:
            sim = CudaCvdv([nq0, nq1])
            sep = SeparableState([nq0, nq1])
            sep.setCoherent(0, alpha0)
            sep.setCoherent(1, complex(alpha1))
            sim.initStateVector(sep)
            results.append(sim.m(0).copy())
            del sim
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0], results[i], atol=1e-10,
                err_msg=f"m(0) must not depend on register 1's state (case {i})")

    def test_marginal_matches_single_register(self):
        """m(0) in a two-register system equals m(0) of the same state alone."""
        nq0, nq1 = 7, 5
        alpha0, alpha1 = 0.6 + 0.4j, -0.3 + 0.9j

        sim1 = CudaCvdv([nq0])
        sep1 = SeparableState([nq0])
        sep1.setCoherent(0, alpha0)
        sim1.initStateVector(sep1)
        p_single = sim1.m(0).copy()
        del sim1

        sim2 = CudaCvdv([nq0, nq1])
        sep2 = SeparableState([nq0, nq1])
        sep2.setCoherent(0, alpha0)
        sep2.setCoherent(1, alpha1)
        sim2.initStateVector(sep2)
        p_marginal = sim2.m(0).copy()
        del sim2

        np.testing.assert_allclose(
            p_marginal, p_single, atol=1e-10,
            err_msg="Marginal of reg 0 must equal single-register m(0)")

    def test_joint_prob_factorises(self):
        """For |α₀⟩ ⊗ |α₁⟩, m(0,1)[i,j] = m(0)[i] · m(1)[j]."""
        nq0, nq1 = 5, 6
        alpha0, alpha1 = 0.7 + 0.2j, -0.4 + 1.0j

        sim = CudaCvdv([nq0, nq1])
        sep = SeparableState([nq0, nq1])
        sep.setCoherent(0, alpha0)
        sep.setCoherent(1, alpha1)
        sim.initStateVector(sep)

        p0  = sim.m(0)
        p1  = sim.m(1)
        p01 = sim.m(0, 1)
        del sim

        P_outer = np.outer(p0, p1)
        assert p01.shape == P_outer.shape
        assert abs(p01.sum() - 1.0) < 1e-10
        np.testing.assert_allclose(
            p01, P_outer, atol=1e-8,
            err_msg="Joint distribution must factorise for a product state")

    # ── normalization ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.0 + 1.0j, -0.5 + 1.5j])
    def test_prob_normalization(self, alpha):
        """m(regIdx) sums to 1 and is non-negative for any coherent state."""
        sim = _make_coherent(self.NQUBITS, complex(alpha))
        probs = sim.m(0)
        assert abs(probs.sum() - 1.0) < 1e-10, \
            f"|α={alpha}⟩: probs sum to {probs.sum():.12f}"
        assert np.all(probs >= -1e-15), "Probabilities must be non-negative"
        del sim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
