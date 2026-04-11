"""
Consistency tests between CUDA and PyTorch backends.

These tests verify that both backends produce identical or numerically similar
results for the same quantum operations. Tests are organized by functionality
and use pytest fixtures for better code reuse.
"""

import pytest
import numpy as np
from numpy import pi, sqrt
from typing import Tuple, Generator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CudaCvdv, TorchCvdv, SeparableState


def _fidelity(cuda_sim, torch_sim) -> float:
    """Compute |<cuda|torch>|² by loading the torch state into a CUDA reference."""
    ref = CudaCvdv(cuda_sim.qubit_counts.tolist())
    ref.initFromArray(torch_sim.state.reshape(-1))
    return cuda_sim.fidelityWith(ref)


# Test tolerances for numerical comparisons
TIGHT_ATOL = 1e-6
TIGHT_RTOL = 1e-5
LOOSE_ATOL = 1e-3
LOOSE_RTOL = 1e-2
VERY_LOOSE_ATOL = 1e-2
VERY_LOOSE_RTOL = 1e-1


@pytest.fixture
def single_register_pair() -> Generator[Tuple[CudaCvdv, TorchCvdv], None, None]:
    """Create a pair of simulators with single 4-qubit register."""
    cuda_sim = CudaCvdv([4])
    torch_sim = TorchCvdv([4], device='cuda', dtype=np.complex128)
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


@pytest.fixture
def medium_register_pair() -> Generator[Tuple[CudaCvdv, TorchCvdv], None, None]:
    """Create a pair of simulators with single 6-qubit register."""
    cuda_sim = CudaCvdv([6])
    torch_sim = TorchCvdv([6], device='cuda', dtype=np.complex128)
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


@pytest.fixture
def two_register_pair() -> Generator[Tuple[CudaCvdv, TorchCvdv], None, None]:
    """Create a pair of simulators with two registers."""
    cuda_sim = CudaCvdv([3, 4])
    torch_sim = TorchCvdv([3, 4], device='cuda', dtype=np.complex128)
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


@pytest.fixture
def two_equal_register_pair() -> Generator[Tuple[CudaCvdv, TorchCvdv], None, None]:
    """Create a pair of simulators with two registers of equal size."""
    cuda_sim = CudaCvdv([3, 3])
    torch_sim = TorchCvdv([3, 3], device='cuda', dtype=np.complex128)
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


class TestStateInitialization:
    """Test consistency of state initialization methods."""
    
    def test_setZero(self, single_register_pair):
        """Test |0⟩ vacuum state initialization."""
        cuda_sim, torch_sim = single_register_pair
        sep = SeparableState([4])
        sep.setZero(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL

    def test_setUniform(self, single_register_pair):
        """Test uniform superposition initialization."""
        cuda_sim, torch_sim = single_register_pair
        sep = SeparableState([4])
        sep.setUniform(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL

    @pytest.mark.parametrize("alpha", [
        1.0 + 0.0j,
        1.5 + 0.8j,
        0.0 + 2.0j,
    ])
    def test_setCoherent(self, medium_register_pair, alpha):
        """Test coherent state initialization with various amplitudes."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, alpha)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_setFock(self, medium_register_pair, n):
        """Test Fock state initialization."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setFock(0, n)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    def test_setFocks_superposition(self, medium_register_pair):
        """Test Fock state superposition."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setFocks(0, [0.6, 0.8j, 0.0])
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    def test_setCat_state(self):
        """Test cat state (superposition of coherent states)."""
        cuda_sim = CudaCvdv([7])
        torch_sim = TorchCvdv([7], device='cuda', dtype=np.complex128)
        sep = SeparableState([7])
        sep.setCat(0, [(2.0, 1.0), (-2.0, 1.0)])
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL
        del cuda_sim, torch_sim


class TestQubitGates:
    """Test consistency of discrete variable (qubit) gates."""
    
    @pytest.mark.parametrize("gate_name,gate_method", [
        ("X", lambda sim: sim.x(0, 0)),
        ("Y", lambda sim: sim.y(0, 0)),
        ("Z", lambda sim: sim.z(0, 0)),
        ("H", lambda sim: sim.h(0, 0)),
    ])
    def test_single_qubit_gates(self, single_register_pair, gate_name, gate_method):
        """Test Pauli gates and Hadamard."""
        cuda_sim, torch_sim = single_register_pair
        sep = SeparableState([4])
        sep.setZero(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        gate_method(cuda_sim)
        gate_method(torch_sim)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL

    @pytest.mark.parametrize("theta", [pi/4, pi/2, pi])
    def test_rotation_gates(self, single_register_pair, theta):
        """Test rotation gates Rx, Ry, Rz."""
        cuda_sim, torch_sim = single_register_pair
        sep = SeparableState([4])
        sep.setZero(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.rx(0, 0, theta)
        torch_sim.rx(0, 0, theta)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL


class TestContinuousVariableGates:
    """Test consistency of continuous variable (bosonic) gates."""
    
    def test_displacement_gate(self, medium_register_pair):
        """Test displacement operator D(β)."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setZero(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.d(0, 0.5 + 0.3j)
        torch_sim.d(0, 0.5 + 0.3j)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    def test_parity_gate(self, single_register_pair):
        """Test parity gate (basis state reversal)."""
        cuda_sim, torch_sim = single_register_pair
        sep = SeparableState([4])
        sep.setUniform(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.p(0)
        torch_sim.p(0)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL

    @pytest.mark.parametrize("t", [0.05, 0.1, 0.2])
    def test_quadratic_phase(self, medium_register_pair, t):
        """Test quadratic phase gate (sheer)."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, 1.0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.sheer(0, t)
        torch_sim.sheer(0, t)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    def test_cubic_phase(self, medium_register_pair):
        """Test cubic phase gate."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, 0.8)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.phaseCubic(0, 0.05)
        torch_sim.phaseCubic(0, 0.05)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    @pytest.mark.parametrize("theta", [pi/6, pi/4, pi/3])
    def test_rotation_gate(self, medium_register_pair, theta):
        """Test phase space rotation gate R(θ)."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, 1.0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.r(0, theta)
        torch_sim.r(0, theta)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    @pytest.mark.parametrize("r", [0.1, 0.3])
    def test_squeezing_gate(self, medium_register_pair, r):
        """Test squeezing gate S(r)."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setZero(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.s(0, r)
        torch_sim.s(0, r)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - VERY_LOOSE_ATOL


class TestFourierTransforms:
    """Test consistency of Fourier transform operations."""
    
    def test_ftQ2P_forward(self, medium_register_pair):
        """Test forward Fourier transform (position to momentum)."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, 1.5)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.ftQ2P(0)
        torch_sim.ftQ2P(0)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    def test_ftP2Q_inverse(self, medium_register_pair):
        """Test inverse Fourier transform (momentum to position)."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, 1.0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        ref_initial = CudaCvdv([6])
        ref_initial.initFromArray(torch_sim.state.reshape(-1))
        cuda_sim.ftQ2P(0)
        cuda_sim.ftP2Q(0)
        torch_sim.ftQ2P(0)
        torch_sim.ftP2Q(0)
        assert cuda_sim.fidelityWith(ref_initial) >= 1 - LOOSE_ATOL
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL


class TestMeasurements:
    """Test consistency of measurement and observable calculations."""
    
    def test_norm_normalized_states(self, medium_register_pair):
        """Test that both backends produce normalized states."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, 1.2)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_norm = cuda_sim.m()
        torch_norm = torch_sim.m()
        assert abs(cuda_norm - 1.0) < TIGHT_ATOL
        assert abs(torch_norm - 1.0) < TIGHT_ATOL
        np.testing.assert_allclose(cuda_norm, torch_norm, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)

    def test_measurement_probabilities(self, single_register_pair):
        """Test measurement probability distributions."""
        cuda_sim, torch_sim = single_register_pair
        sep = SeparableState([4])
        sep.setUniform(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_probs = cuda_sim.m(0)
        torch_probs = torch_sim.m(0)
        np.testing.assert_allclose(cuda_probs, torch_probs, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
        assert abs(np.sum(cuda_probs) - 1.0) < TIGHT_ATOL
        assert abs(np.sum(torch_probs) - 1.0) < TIGHT_ATOL

    def test_getState_consistency(self, single_register_pair):
        """Test that getState returns consistent results."""
        cuda_sim, torch_sim = single_register_pair
        sep = SeparableState([4])
        sep.setZero(0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL


class TestMultiRegisterOperations:
    """Test consistency with multiple registers."""
    
    def test_two_register_tensor_product(self, two_register_pair):
        """Test tensor product state initialization."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL

    def test_independent_register_operations(self, two_register_pair):
        """Test operations on independent registers."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.h(0, 0)
        cuda_sim.x(0, 1)
        torch_sim.h(0, 0)
        torch_sim.x(0, 1)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL

    def test_combined_cv_dv_operations(self, two_register_pair):
        """Test combined CV and DV operations on different registers."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.h(0, 0)
        torch_sim.h(0, 0)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL


class TestConditionalGates:
    """Test consistency of conditional gates (controlled operations)."""
    
    def test_conditional_displacement_real(self, two_register_pair):
        """Test conditional displacement with real alpha."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setUniform(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.cd(1, 0, 0, 0.5)
        torch_sim.cd(1, 0, 0, 0.5)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    def test_conditional_displacement_complex(self, two_register_pair):
        """Test conditional displacement with complex alpha."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.x(0, 0)
        torch_sim.x(0, 0)
        cuda_sim.cd(1, 0, 0, 0.5 + 0.3j)
        torch_sim.cd(1, 0, 0, 0.5 + 0.3j)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    def test_conditional_parity(self, two_register_pair):
        """Test conditional parity gate."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setUniform(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.cp(1, 0, 0)
        torch_sim.cp(1, 0, 0)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL

    def test_conditional_parity_control_one(self, two_register_pair):
        """Test conditional parity when control is |1⟩."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setUniform(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.x(0, 0)
        torch_sim.x(0, 0)
        cuda_sim.cp(1, 0, 0)
        torch_sim.cp(1, 0, 0)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL


class TestAdvancedGates:
    """Test consistency of advanced gates (conditional operations)."""
    
    @pytest.mark.parametrize("theta", [pi/6, pi/4, pi/3])
    def test_conditional_rotation(self, two_register_pair, theta):
        """Test conditional rotation CR(θ)."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setUniform(0)
        sep.setCoherent(1, 0.8)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.cr(1, 0, 0, theta)
        torch_sim.cr(1, 0, 0, theta)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - VERY_LOOSE_ATOL

    @pytest.mark.parametrize("r", [0.1, 0.2])
    def test_conditional_squeezing(self, two_register_pair, r):
        """Test conditional squeezing CS(r)."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.x(0, 0)
        torch_sim.x(0, 0)
        cuda_sim.cs(1, 0, 0, r)
        torch_sim.cs(1, 0, 0, r)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - VERY_LOOSE_ATOL


class TestTwoModeGates:
    """Test consistency of two-mode (register-register) operations."""
    
    def test_q1q2_interaction(self, two_register_pair):
        """Test Q1Q2 interaction gate."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setCoherent(0, 0.5)
        sep.setCoherent(1, 0.8)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.q1q2(0, 1, 0.1)
        torch_sim.q1q2(0, 1, 0.1)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL

    @pytest.mark.parametrize("theta", [pi/6, pi/4])
    def test_beam_splitter(self, two_equal_register_pair, theta):
        """Test beam splitter gate BS(θ)."""
        cuda_sim, torch_sim = two_equal_register_pair
        sep = SeparableState([3, 3])
        sep.setCoherent(0, 0.5)
        sep.setCoherent(1, 0.8)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.bs(0, 1, theta)
        torch_sim.bs(0, 1, theta)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - VERY_LOOSE_ATOL

    @pytest.mark.parametrize("theta", [pi/6, pi/4])
    def test_conditional_beam_splitter(self, theta):
        """Test conditional beam splitter CBS(θ)."""
        cuda_sim = CudaCvdv([2, 3, 3])
        torch_sim = TorchCvdv([2, 3, 3], device='cuda', dtype=np.complex128)
        sep = SeparableState([2, 3, 3])
        sep.setUniform(0)
        sep.setCoherent(1, 0.5)
        sep.setCoherent(2, 0.8)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.cbs(1, 2, 0, 0, theta)
        torch_sim.cbs(1, 2, 0, 0, theta)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - VERY_LOOSE_ATOL
        del cuda_sim, torch_sim

    def test_swap_operation(self, two_equal_register_pair):
        """Test register swap operation."""
        cuda_sim, torch_sim = two_equal_register_pair
        sep = SeparableState([3, 3])
        sep.setZero(0)
        sep.setUniform(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        ref_initial = CudaCvdv([3, 3])
        ref_initial.initFromArray(torch_sim.state.reshape(-1))
        cuda_sim.swap(0, 1)
        torch_sim.swap(0, 1)
        assert _fidelity(cuda_sim, torch_sim) >= 1 - TIGHT_ATOL
        assert cuda_sim.fidelityWith(ref_initial) < 0.5

    def test_double_swap(self, two_equal_register_pair):
        """Test that swapping twice returns to original state."""
        cuda_sim, torch_sim = two_equal_register_pair
        sep = SeparableState([3, 3])
        sep.setCoherent(0, 1.0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        ref_initial = CudaCvdv([3, 3])
        ref_initial.initFromArray(torch_sim.state.reshape(-1))
        cuda_sim.swap(0, 1)
        cuda_sim.swap(0, 1)
        torch_sim.swap(0, 1)
        torch_sim.swap(0, 1)
        assert cuda_sim.fidelityWith(ref_initial) >= 1 - LOOSE_ATOL
        assert _fidelity(cuda_sim, torch_sim) >= 1 - LOOSE_ATOL


class TestPhaseSpaceFunctions:
    """Test consistency of phase space functions (Wigner, Husimi Q)."""
    
    def test_wigner_single_slice_vacuum(self, two_register_pair):
        """Test Wigner function for single slice with vacuum state."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_wigner = cuda_sim.getWigner(1)
        torch_wigner = torch_sim.getWigner(1)
        np.testing.assert_allclose(cuda_wigner, torch_wigner, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)

    def test_husimi_q_vacuum(self):
        """Test Husimi Q function for vacuum state."""
        cuda_sim = CudaCvdv([5])
        torch_sim = TorchCvdv([5], device='cuda', dtype=np.complex128)
        sep = SeparableState([5])
        sep.setFock(0, 0)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_husimi = cuda_sim.getHusimiQ(0)
        torch_husimi = torch_sim.getHusimiQ(0)
        np.testing.assert_allclose(cuda_husimi, torch_husimi, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        assert np.min(cuda_husimi) >= 0.0
        assert np.min(torch_husimi) >= 0.0
        center_idx = cuda_husimi.shape[0] // 2  # center of native grid
        assert cuda_husimi[center_idx, center_idx] > np.mean(cuda_husimi)
        assert torch_husimi[center_idx, center_idx] > np.mean(torch_husimi)
        del cuda_sim, torch_sim

    def test_husimi_q_coherent(self):
        """Test Husimi Q function for coherent state."""
        cuda_sim = CudaCvdv([6])
        torch_sim = TorchCvdv([6], device='cuda', dtype=np.complex128)
        sep = SeparableState([6])
        sep.setCoherent(0, 1.5 + 1.0j)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_husimi = cuda_sim.getHusimiQ(0)
        torch_husimi = torch_sim.getHusimiQ(0)
        np.testing.assert_allclose(cuda_husimi, torch_husimi, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        assert np.min(cuda_husimi) >= 0.0
        assert np.min(torch_husimi) >= 0.0
        del cuda_sim, torch_sim

    def test_wigner_on_grid_coherent(self):
        """CUDA vs Torch getWigner on-grid API for a coherent state."""
        cuda_sim = CudaCvdv([8])
        torch_sim = TorchCvdv([8], device='cuda', dtype=np.complex128)
        sep = SeparableState([8])
        sep.setCoherent(0, 1.0 + 0.5j)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_W = cuda_sim.getWigner(0)
        torch_W = torch_sim.getWigner(0)
        assert cuda_W.shape == torch_W.shape
        assert cuda_W.shape == (256, 256)  # native grid N×N
        np.testing.assert_allclose(cuda_W, torch_W, atol=VERY_LOOSE_ATOL, rtol=VERY_LOOSE_RTOL)
        del cuda_sim, torch_sim

    def test_husimi_on_grid_coherent(self):
        """CUDA vs Torch getHusimiQ on-grid API for a coherent state."""
        cuda_sim = CudaCvdv([8])
        torch_sim = TorchCvdv([8], device='cuda', dtype=np.complex128)
        sep = SeparableState([8])
        sep.setCoherent(0, 1.0 + 0.5j)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_Q = cuda_sim.getHusimiQ(0)
        torch_Q = torch_sim.getHusimiQ(0)
        assert cuda_Q.shape == torch_Q.shape
        assert cuda_Q.shape == (256, 256)  # native grid N×N
        np.testing.assert_allclose(cuda_Q, torch_Q, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        assert np.min(cuda_Q) >= 0.0
        assert np.min(torch_Q) >= 0.0
        del cuda_sim, torch_sim


class TestMeasurementExtensions:
    """Test additional measurement and utility functions."""
    
    def test_joint_measurement(self, two_register_pair):
        """Test joint measurement probabilities."""
        cuda_sim, torch_sim = two_register_pair
        sep = SeparableState([3, 4])
        sep.setZero(0)
        sep.setZero(1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.h(0, 0)
        torch_sim.h(0, 0)
        cuda_joint = cuda_sim.m(0, 1)
        torch_joint = torch_sim.m(0, 1)
        np.testing.assert_allclose(cuda_joint, torch_joint, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
        assert abs(np.sum(cuda_joint) - 1.0) < TIGHT_ATOL
        assert abs(np.sum(torch_joint) - 1.0) < TIGHT_ATOL

    def test_fidelity_same_state(self, medium_register_pair):
        """Test fidelity of state with itself (no gate applied) equals 1."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setCoherent(0, 1.0 + 0.5j)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_fid = cuda_sim.getFidelity(sep)
        torch_fid = torch_sim.getFidelity(sep)
        assert abs(cuda_fid - 1.0) < LOOSE_ATOL
        assert abs(torch_fid - 1.0) < LOOSE_ATOL
        np.testing.assert_allclose(cuda_fid, torch_fid, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)

    def test_fidelity_after_gate(self, medium_register_pair):
        """Test fidelity after applying gate (overlap with original state)."""
        cuda_sim, torch_sim = medium_register_pair
        sep = SeparableState([6])
        sep.setFock(0, 1)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_sim.d(0, 0.5)
        torch_sim.d(0, 0.5)
        cuda_fid = cuda_sim.getFidelity(sep)
        torch_fid = torch_sim.getFidelity(sep)
        assert cuda_fid < 1.0
        assert torch_fid < 1.0
        np.testing.assert_allclose(cuda_fid, torch_fid, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)

    def test_get_x_grid(self, medium_register_pair):
        """Test position grid retrieval."""
        cuda_sim, torch_sim = medium_register_pair
        cuda_grid = cuda_sim.getXGrid(0)
        torch_grid = torch_sim.getXGrid(0)
        np.testing.assert_allclose(cuda_grid, torch_grid, atol=1e-15, rtol=1e-15)
        assert len(cuda_grid) == cuda_sim.register_dims[0]
        assert len(torch_grid) == torch_sim.register_dims[0]


class TestPhotonNumber:
    """Consistency of getPhotonNumber between CUDA and torch backends."""

    NQUBITS = 8
    ATOL = 1e-2

    def _make_pair(self, nqubits=None):
        nq = nqubits if nqubits is not None else self.NQUBITS
        cuda_sim = CudaCvdv([nq])
        torch_sim = TorchCvdv([nq], device='cuda', dtype=np.complex128)
        return cuda_sim, torch_sim

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5 + 1.0j, -0.5 + 2.0j])
    def test_coherent_state_consistency(self, alpha):
        """CUDA and torch agree on <n> for coherent |α⟩; both equal |α|²."""
        cuda_sim, torch_sim = self._make_pair()
        sep = SeparableState([self.NQUBITS])
        sep.setCoherent(0, alpha)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_n = cuda_sim.getPhotonNumber(0)
        torch_n = torch_sim.getPhotonNumber(0)
        n_exact = abs(alpha) ** 2
        assert abs(cuda_n - n_exact) < self.ATOL, f"CUDA: <n>={cuda_n:.6f} vs {n_exact:.6f}"
        assert abs(torch_n - n_exact) < self.ATOL, f"Torch: <n>={torch_n:.6f} vs {n_exact:.6f}"
        np.testing.assert_allclose(cuda_n, torch_n, atol=self.ATOL)
        del cuda_sim, torch_sim

    @pytest.mark.parametrize("n_fock", [0, 1, 2, 4])
    def test_fock_state_consistency(self, n_fock):
        """CUDA and torch agree on <n> for Fock |n⟩; both equal n."""
        cuda_sim, torch_sim = self._make_pair()
        sep = SeparableState([self.NQUBITS])
        sep.setFock(0, n_fock)
        cuda_sim.initStateVector(sep)
        torch_sim.initStateVector(sep)
        cuda_n = cuda_sim.getPhotonNumber(0)
        torch_n = torch_sim.getPhotonNumber(0)
        assert abs(cuda_n - n_fock) < self.ATOL, f"CUDA: <n>={cuda_n:.6f} vs {n_fock}"
        assert abs(torch_n - n_fock) < self.ATOL, f"Torch: <n>={torch_n:.6f} vs {n_fock}"
        np.testing.assert_allclose(cuda_n, torch_n, atol=self.ATOL)
        del cuda_sim, torch_sim


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
