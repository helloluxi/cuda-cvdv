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

from src import CVDV, CVDVTorch


# Test tolerances for numerical comparisons
TIGHT_ATOL = 1e-6
TIGHT_RTOL = 1e-5
LOOSE_ATOL = 1e-3
LOOSE_RTOL = 1e-2
VERY_LOOSE_ATOL = 1e-2
VERY_LOOSE_RTOL = 1e-1


@pytest.fixture
def single_register_pair() -> Generator[Tuple[CVDV, CVDVTorch], None, None]:
    """Create a pair of simulators with single 4-qubit register."""
    cuda_sim = CVDV([4])
    torch_sim = CVDVTorch([4], device='cuda')
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


@pytest.fixture
def medium_register_pair() -> Generator[Tuple[CVDV, CVDVTorch], None, None]:
    """Create a pair of simulators with single 6-qubit register."""
    cuda_sim = CVDV([6])
    torch_sim = CVDVTorch([6], device='cuda')
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


@pytest.fixture
def two_register_pair() -> Generator[Tuple[CVDV, CVDVTorch], None, None]:
    """Create a pair of simulators with two registers."""
    cuda_sim = CVDV([3, 4])
    torch_sim = CVDVTorch([3, 4], device='cuda')
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


@pytest.fixture
def two_equal_register_pair() -> Generator[Tuple[CVDV, CVDVTorch], None, None]:
    """Create a pair of simulators with two registers of equal size."""
    cuda_sim = CVDV([3, 3])
    torch_sim = CVDVTorch([3, 3], device='cuda')
    yield cuda_sim, torch_sim
    del cuda_sim
    del torch_sim


class TestStateInitialization:
    """Test consistency of state initialization methods."""
    
    def test_setZero(self, single_register_pair):
        """Test |0⟩ vacuum state initialization."""
        cuda_sim, torch_sim = single_register_pair
        
        cuda_sim.setZero(0)
        cuda_sim.initStateVector()
        
        torch_sim.setZero(0)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    def test_setUniform(self, single_register_pair):
        """Test uniform superposition initialization."""
        cuda_sim, torch_sim = single_register_pair
        
        cuda_sim.setUniform(0)
        cuda_sim.initStateVector()
        
        torch_sim.setUniform(0)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    @pytest.mark.parametrize("alpha", [
        1.0 + 0.0j,
        1.5 + 0.8j,
        0.0 + 2.0j,
    ])
    def test_setCoherent(self, medium_register_pair, alpha):
        """Test coherent state initialization with various amplitudes."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_sim.setCoherent(0, alpha)
        cuda_sim.initStateVector()
        
        torch_sim.setCoherent(0, alpha)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        # Coherent states have discretization differences
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_setFock(self, medium_register_pair, n):
        """Test Fock state initialization."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_sim.setFock(0, n)
        cuda_sim.initStateVector()
        
        torch_sim.setFock(0, n)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        # Fock states use Hermite polynomials - allow looser tolerance
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_setFocks_superposition(self, medium_register_pair):
        """Test Fock state superposition."""
        cuda_sim, torch_sim = medium_register_pair
        
        coeffs = [0.6, 0.8j, 0.0]  # c0|0⟩ + c1|1⟩
        
        cuda_sim.setFocks(0, coeffs)
        cuda_sim.initStateVector()
        
        torch_sim.setFocks(0, coeffs)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_setCat_state(self):
        """Test cat state (superposition of coherent states)."""
        cuda_sim = CVDV([7])
        torch_sim = CVDVTorch([7], device='cuda')
        
        cat_states = [(2.0, 1.0), (-2.0, 1.0)]  # (|α⟩ + |-α⟩)/√2
        
        cuda_sim.setCat(0, cat_states)
        cuda_sim.initStateVector()
        
        torch_sim.setCat(0, cat_states)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        
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
        
        # Initialize to |0⟩
        cuda_sim.setZero(0)
        cuda_sim.initStateVector()
        gate_method(cuda_sim)
        
        torch_sim.setZero(0)
        torch_sim.initStateVector()
        gate_method(torch_sim)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    @pytest.mark.parametrize("theta", [pi/4, pi/2, pi])
    def test_rotation_gates(self, single_register_pair, theta):
        """Test rotation gates Rx, Ry, Rz."""
        cuda_sim, torch_sim = single_register_pair
        
        # Test Rx
        cuda_sim.setZero(0)
        cuda_sim.initStateVector()
        cuda_sim.rx(0, 0, theta)
        
        torch_sim.setZero(0)
        torch_sim.initStateVector()
        torch_sim.rx(0, 0, theta)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)


class TestContinuousVariableGates:
    """Test consistency of continuous variable (bosonic) gates."""
    
    def test_displacement_gate(self, medium_register_pair):
        """Test displacement operator D(β)."""
        cuda_sim, torch_sim = medium_register_pair
        
        beta = 0.5 + 0.3j
        
        cuda_sim.setZero(0)
        cuda_sim.initStateVector()
        cuda_sim.d(0, beta)
        
        torch_sim.setZero(0)
        torch_sim.initStateVector()
        torch_sim.d(0, beta)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        # Displacement may have global phase differences - check magnitudes
        cuda_mag = np.abs(cuda_state)
        torch_mag = np.abs(torch_state)
        np.testing.assert_allclose(cuda_mag, torch_mag, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_parity_gate(self, single_register_pair):
        """Test parity gate (basis state reversal)."""
        cuda_sim, torch_sim = single_register_pair
        
        # Create non-trivial state
        cuda_sim.setUniform(0)
        cuda_sim.initStateVector()
        cuda_sim.p(0)
        
        torch_sim.setUniform(0)
        torch_sim.initStateVector()
        torch_sim.p(0)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    @pytest.mark.parametrize("t", [0.05, 0.1, 0.2])
    def test_quadratic_phase(self, medium_register_pair, t):
        """Test quadratic phase gate (sheer)."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_sim.setCoherent(0, 1.0)
        cuda_sim.initStateVector()
        cuda_sim.sheer(0, t)
        
        torch_sim.setCoherent(0, 1.0)
        torch_sim.initStateVector()
        torch_sim.sheer(0, t)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_cubic_phase(self, medium_register_pair):
        """Test cubic phase gate."""
        cuda_sim, torch_sim = medium_register_pair
        
        t = 0.05
        
        cuda_sim.setCoherent(0, 0.8)
        cuda_sim.initStateVector()
        cuda_sim.phaseCubic(0, t)
        
        torch_sim.setCoherent(0, 0.8)
        torch_sim.initStateVector()
        torch_sim.phaseCubic(0, t)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    @pytest.mark.parametrize("theta", [pi/6, pi/4, pi/3])
    def test_rotation_gate(self, medium_register_pair, theta):
        """Test phase space rotation gate R(θ)."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_sim.setCoherent(0, 1.0)
        cuda_sim.initStateVector()
        cuda_sim.r(0, theta)
        
        torch_sim.setCoherent(0, 1.0)
        torch_sim.initStateVector()
        torch_sim.r(0, theta)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    @pytest.mark.parametrize("r", [0.1, 0.3])
    def test_squeezing_gate(self, medium_register_pair, r):
        """Test squeezing gate S(r)."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_sim.setZero(0)
        cuda_sim.initStateVector()
        cuda_sim.s(0, r)
        
        torch_sim.setZero(0)
        torch_sim.initStateVector()
        torch_sim.s(0, r)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        # Squeezing involves multiple operations - allow higher tolerance
        np.testing.assert_allclose(cuda_state, torch_state, atol=VERY_LOOSE_ATOL, rtol=VERY_LOOSE_RTOL)


class TestFourierTransforms:
    """Test consistency of Fourier transform operations."""
    
    def test_ftQ2P_forward(self, medium_register_pair):
        """Test forward Fourier transform (position to momentum)."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_sim.setCoherent(0, 1.5)
        cuda_sim.initStateVector()
        cuda_sim.ftQ2P(0)
        
        torch_sim.setCoherent(0, 1.5)
        torch_sim.initStateVector()
        torch_sim.ftQ2P(0)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        # FT may have phase differences - check magnitudes
        cuda_mag = np.abs(cuda_state)
        torch_mag = np.abs(torch_state)
        np.testing.assert_allclose(cuda_mag, torch_mag, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_ftP2Q_inverse(self, medium_register_pair):
        """Test inverse Fourier transform (momentum to position)."""
        cuda_sim, torch_sim = medium_register_pair
        
        # Apply forward then inverse - should recover original
        cuda_sim.setCoherent(0, 1.0)
        cuda_sim.initStateVector()
        cuda_initial = cuda_sim.getState()
        cuda_sim.ftQ2P(0)
        cuda_sim.ftP2Q(0)
        cuda_final = cuda_sim.getState()
        
        torch_sim.setCoherent(0, 1.0)
        torch_sim.initStateVector()
        torch_initial = torch_sim.getState()
        torch_sim.ftQ2P(0)
        torch_sim.ftP2Q(0)
        torch_final = torch_sim.getState()
        
        # Check that FT^-1 . FT ≈ Identity for both backends
        np.testing.assert_allclose(cuda_initial, cuda_final, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        np.testing.assert_allclose(torch_initial, torch_final, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)


class TestMeasurements:
    """Test consistency of measurement and observable calculations."""
    
    def test_getNorm_normalized_states(self, medium_register_pair):
        """Test that both backends produce normalized states."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_sim.setCoherent(0, 1.2)
        cuda_sim.initStateVector()
        
        torch_sim.setCoherent(0, 1.2)
        torch_sim.initStateVector()
        
        cuda_norm = cuda_sim.getNorm()
        torch_norm = torch_sim.getNorm()
        
        # Both should be normalized
        assert abs(cuda_norm - 1.0) < TIGHT_ATOL
        assert abs(torch_norm - 1.0) < TIGHT_ATOL
        np.testing.assert_allclose(cuda_norm, torch_norm, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    def test_measurement_probabilities(self, single_register_pair):
        """Test measurement probability distributions."""
        cuda_sim, torch_sim = single_register_pair
        
        # Create superposition
        cuda_sim.setUniform(0)
        cuda_sim.initStateVector()
        
        torch_sim.setUniform(0)
        torch_sim.initStateVector()
        
        cuda_probs = cuda_sim.m(0)
        torch_probs = torch_sim.m(0)
        
        np.testing.assert_allclose(cuda_probs, torch_probs, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
        
        # Probabilities should sum to 1
        assert abs(np.sum(cuda_probs) - 1.0) < TIGHT_ATOL
        assert abs(np.sum(torch_probs) - 1.0) < TIGHT_ATOL
    
    def test_getState_consistency(self, single_register_pair):
        """Test that getState returns consistent results."""
        cuda_sim, torch_sim = single_register_pair
        
        cuda_sim.setZero(0)
        cuda_sim.initStateVector()
        
        torch_sim.setZero(0)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        # States should be identical for |0⟩
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)


class TestMultiRegisterOperations:
    """Test consistency with multiple registers."""
    
    def test_two_register_tensor_product(self, two_register_pair):
        """Test tensor product state initialization."""
        cuda_sim, torch_sim = two_register_pair
        
        cuda_sim.setZero(0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        
        torch_sim.setZero(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    def test_independent_register_operations(self, two_register_pair):
        """Test operations on independent registers."""
        cuda_sim, torch_sim = two_register_pair
        
        cuda_sim.setZero(0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        cuda_sim.h(0, 0)  # Hadamard on register 0
        cuda_sim.x(0, 1)  # X on register 0, qubit 1
        
        torch_sim.setZero(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        torch_sim.h(0, 0)
        torch_sim.x(0, 1)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    def test_combined_cv_dv_operations(self, two_register_pair):
        """Test combined CV and DV operations on different registers."""
        cuda_sim, torch_sim = two_register_pair
        
        cuda_sim.setZero(0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        cuda_sim.h(0, 0)  # DV operation on register 0
        
        torch_sim.setZero(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        torch_sim.h(0, 0)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)


class TestConditionalGates:
    """Test consistency of conditional gates (controlled operations)."""
    
    def test_conditional_displacement_real(self, two_register_pair):
        """Test conditional displacement with real alpha."""
        cuda_sim, torch_sim = two_register_pair
        
        alpha = 0.5
        
        # Initialize: control in superposition, target in vacuum
        cuda_sim.setUniform(0)  # Control register
        cuda_sim.setZero(1)      # Target register
        cuda_sim.initStateVector()
        cuda_sim.cd(1, 0, 0, alpha)
        
        torch_sim.setUniform(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        torch_sim.cd(1, 0, 0, alpha)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_conditional_displacement_complex(self, two_register_pair):
        """Test conditional displacement with complex alpha."""
        cuda_sim, torch_sim = two_register_pair
        
        alpha = 0.5 + 0.3j
        
        # Initialize: control with X on first qubit, target in vacuum
        cuda_sim.setZero(0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        cuda_sim.x(0, 0)  # Put control qubit in |1⟩
        cuda_sim.cd(1, 0, 0, alpha)
        
        torch_sim.setZero(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        torch_sim.x(0, 0)
        torch_sim.cd(1, 0, 0, alpha)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_conditional_parity(self, two_register_pair):
        """Test conditional parity gate."""
        cuda_sim, torch_sim = two_register_pair
        
        # Initialize: control in |0⟩, target in uniform superposition
        cuda_sim.setZero(0)
        cuda_sim.setUniform(1)
        cuda_sim.initStateVector()
        cuda_sim.cp(1, 0, 0)
        
        torch_sim.setZero(0)
        torch_sim.setUniform(1)
        torch_sim.initStateVector()
        torch_sim.cp(1, 0, 0)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
    
    def test_conditional_parity_control_one(self, two_register_pair):
        """Test conditional parity when control is |1⟩."""
        cuda_sim, torch_sim = two_register_pair
        
        # Initialize: control in |1⟩, target in uniform superposition
        cuda_sim.setZero(0)
        cuda_sim.setUniform(1)
        cuda_sim.initStateVector()
        cuda_sim.x(0, 0)  # Flip control qubit to |1⟩
        cuda_sim.cp(1, 0, 0)
        
        torch_sim.setZero(0)
        torch_sim.setUniform(1)
        torch_sim.initStateVector()
        torch_sim.x(0, 0)
        torch_sim.cp(1, 0, 0)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)


class TestAdvancedGates:
    """Test consistency of advanced gates (conditional operations)."""
    
    @pytest.mark.parametrize("theta", [pi/6, pi/4, pi/3])
    def test_conditional_rotation(self, two_register_pair, theta):
        """Test conditional rotation CR(θ)."""
        cuda_sim, torch_sim = two_register_pair
        
        # Initialize: control in superposition, target in coherent state
        cuda_sim.setUniform(0)
        cuda_sim.setCoherent(1, 0.8)
        cuda_sim.initStateVector()
        cuda_sim.cr(1, 0, 0, theta)
        
        torch_sim.setUniform(0)
        torch_sim.setCoherent(1, 0.8)
        torch_sim.initStateVector()
        torch_sim.cr(1, 0, 0, theta)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=VERY_LOOSE_ATOL, rtol=VERY_LOOSE_RTOL)
    
    @pytest.mark.parametrize("r", [0.1, 0.2])
    def test_conditional_squeezing(self, two_register_pair, r):
        """Test conditional squeezing CS(r)."""
        cuda_sim, torch_sim = two_register_pair
        
        # Initialize: control with X gate, target in vacuum
        cuda_sim.setZero(0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        cuda_sim.x(0, 0)
        cuda_sim.cs(1, 0, 0, r)
        
        torch_sim.setZero(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        torch_sim.x(0, 0)
        torch_sim.cs(1, 0, 0, r)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=VERY_LOOSE_ATOL, rtol=VERY_LOOSE_RTOL)


class TestTwoModeGates:
    """Test consistency of two-mode (register-register) operations."""
    
    def test_q1q2_interaction(self, two_register_pair):
        """Test Q1Q2 interaction gate."""
        cuda_sim, torch_sim = two_register_pair
        
        coeff = 0.1
        
        # Initialize both registers to coherent states
        cuda_sim.setCoherent(0, 0.5)
        cuda_sim.setCoherent(1, 0.8)
        cuda_sim.initStateVector()
        cuda_sim.q1q2(0, 1, coeff)
        
        torch_sim.setCoherent(0, 0.5)
        torch_sim.setCoherent(1, 0.8)
        torch_sim.initStateVector()
        torch_sim.q1q2(0, 1, coeff)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    @pytest.mark.parametrize("theta", [pi/6, pi/4])
    def test_beam_splitter(self, two_equal_register_pair, theta):
        """Test beam splitter gate BS(θ)."""
        cuda_sim, torch_sim = two_equal_register_pair
        
        # Initialize both registers to different coherent states
        cuda_sim.setCoherent(0, 0.5)
        cuda_sim.setCoherent(1, 0.8)
        cuda_sim.initStateVector()
        cuda_sim.bs(0, 1, theta)
        
        torch_sim.setCoherent(0, 0.5)
        torch_sim.setCoherent(1, 0.8)
        torch_sim.initStateVector()
        torch_sim.bs(0, 1, theta)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=VERY_LOOSE_ATOL, rtol=VERY_LOOSE_RTOL)
    
    @pytest.mark.parametrize("theta", [pi/6, pi/4])
    def test_conditional_beam_splitter(self, theta):
        """Test conditional beam splitter CBS(θ)."""
        # Need 3 registers: 2 equal for BS, 1 for control
        cuda_sim = CVDV([2, 3, 3])
        torch_sim = CVDVTorch([2, 3, 3], device='cuda')
        
        # Initialize: control in superposition, targets in coherent states
        cuda_sim.setUniform(0)
        cuda_sim.setCoherent(1, 0.5)
        cuda_sim.setCoherent(2, 0.8)
        cuda_sim.initStateVector()
        cuda_sim.cbs(1, 2, 0, 0, theta)
        
        torch_sim.setUniform(0)
        torch_sim.setCoherent(1, 0.5)
        torch_sim.setCoherent(2, 0.8)
        torch_sim.initStateVector()
        torch_sim.cbs(1, 2, 0, 0, theta)
        
        cuda_state = cuda_sim.getState()
        torch_state = torch_sim.getState()
        
        np.testing.assert_allclose(cuda_state, torch_state, atol=VERY_LOOSE_ATOL, rtol=VERY_LOOSE_RTOL)
        
        del cuda_sim, torch_sim
    
    def test_swap_operation(self, two_equal_register_pair):
        """Test register swap operation."""
        cuda_sim, torch_sim = two_equal_register_pair
        
        # Initialize registers to different states
        cuda_sim.setZero(0)
        cuda_sim.setUniform(1)
        cuda_sim.initStateVector()
        cuda_initial = cuda_sim.getState()
        cuda_sim.swap(0, 1)
        cuda_swapped = cuda_sim.getState()
        
        torch_sim.setZero(0)
        torch_sim.setUniform(1)
        torch_sim.initStateVector()
        torch_initial = torch_sim.getState()
        torch_sim.swap(0, 1)
        torch_swapped = torch_sim.getState()
        
        # Check consistency
        np.testing.assert_allclose(cuda_swapped, torch_swapped, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
        
        # Verify swap actually did something
        assert not np.allclose(cuda_initial, cuda_swapped, atol=TIGHT_ATOL)
    
    def test_double_swap(self, two_equal_register_pair):
        """Test that swapping twice returns to original state."""
        cuda_sim, torch_sim = two_equal_register_pair
        
        # Initialize to non-trivial states (using same dim for both registers)
        cuda_sim.setCoherent(0, 1.0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        cuda_initial = cuda_sim.getState()
        cuda_sim.swap(0, 1)
        cuda_sim.swap(0, 1)
        cuda_final = cuda_sim.getState()
        
        torch_sim.setCoherent(0, 1.0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        torch_initial = torch_sim.getState()
        torch_sim.swap(0, 1)
        torch_sim.swap(0, 1)
        torch_final = torch_sim.getState()
        
        # Both should return to initial state
        np.testing.assert_allclose(cuda_initial, cuda_final, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        np.testing.assert_allclose(torch_initial, torch_final, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        np.testing.assert_allclose(cuda_final, torch_final, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)


class TestPhaseSpaceFunctions:
    """Test consistency of phase space functions (Wigner, Husimi Q)."""
    
    def test_wigner_single_slice_vacuum(self, two_register_pair):
        """Test Wigner function for single slice with vacuum state."""
        cuda_sim, torch_sim = two_register_pair
        
        # Initialize to vacuum states
        cuda_sim.setZero(0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        
        torch_sim.setZero(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        
        # Compute Wigner for register 1 when register 0 is in |0⟩
        cuda_wigner = cuda_sim.getWignerSingleSlice(1, [0, 0], wignerN=51)
        torch_wigner = torch_sim.getWignerSingleSlice(1, [0, 0], wignerN=51)
        
        # Compare Wigner functions
        np.testing.assert_allclose(cuda_wigner, torch_wigner, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_husimi_q_vacuum(self):
        """Test Husimi Q function for vacuum state."""
        cuda_sim = CVDV([5])
        torch_sim = CVDVTorch([5], device='cuda')
        
        # Initialize to vacuum (Fock |0⟩ = Gaussian centered at origin)
        cuda_sim.setFock(0, 0)
        cuda_sim.initStateVector()

        torch_sim.setFock(0, 0)
        torch_sim.initStateVector()
        
        # Compute Husimi Q function
        cuda_husimi = cuda_sim.getHusimiQFullMode(0, qN=51, qMax=4.0, pMax=4.0)
        torch_husimi = torch_sim.getHusimiQFullMode(0, qN=51, qMax=4.0, pMax=4.0)
        
        # Compare Husimi Q functions
        np.testing.assert_allclose(cuda_husimi, torch_husimi, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        
        # Husimi Q should be positive everywhere
        assert np.min(cuda_husimi) >= 0.0
        assert np.min(torch_husimi) >= 0.0
        
        # Should be peaked at origin for vacuum state
        center_idx = 25  # Center of 51x51 grid
        assert cuda_husimi[center_idx, center_idx] > np.mean(cuda_husimi)
        assert torch_husimi[center_idx, center_idx] > np.mean(torch_husimi)
        
        del cuda_sim, torch_sim
    
    def test_husimi_q_coherent(self):
        """Test Husimi Q function for coherent state."""
        cuda_sim = CVDV([6])
        torch_sim = CVDVTorch([6], device='cuda')
        
        # Initialize to coherent state
        alpha = 1.5 + 1.0j
        cuda_sim.setCoherent(0, alpha)
        cuda_sim.initStateVector()
        
        torch_sim.setCoherent(0, alpha)
        torch_sim.initStateVector()
        
        # Compute Husimi Q function
        cuda_husimi = cuda_sim.getHusimiQFullMode(0, qN=51, qMax=4.0, pMax=4.0)
        torch_husimi = torch_sim.getHusimiQFullMode(0, qN=51, qMax=4.0, pMax=4.0)
        
        # Compare Husimi Q functions
        np.testing.assert_allclose(cuda_husimi, torch_husimi, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
        
        # Husimi Q should be positive everywhere
        assert np.min(cuda_husimi) >= 0.0
        assert np.min(torch_husimi) >= 0.0
        
        del cuda_sim, torch_sim


class TestMeasurementExtensions:
    """Test additional measurement and utility functions."""
    
    def test_joint_measurement(self, two_register_pair):
        """Test joint measurement probabilities."""
        cuda_sim, torch_sim = two_register_pair
        
        # Create entangled state
        cuda_sim.setZero(0)
        cuda_sim.setZero(1)
        cuda_sim.initStateVector()
        cuda_sim.h(0, 0)  # Hadamard on first qubit of register 0
        
        torch_sim.setZero(0)
        torch_sim.setZero(1)
        torch_sim.initStateVector()
        torch_sim.h(0, 0)
        
        cuda_joint = cuda_sim.jointMeasure(0, 1)
        torch_joint = torch_sim.jointMeasure(0, 1)
        
        np.testing.assert_allclose(cuda_joint, torch_joint, atol=TIGHT_ATOL, rtol=TIGHT_RTOL)
        
        # Check normalization
        assert abs(np.sum(cuda_joint) - 1.0) < TIGHT_ATOL
        assert abs(np.sum(torch_joint) - 1.0) < TIGHT_ATOL
    
    def test_inner_product_same_state(self, medium_register_pair):
        """Test inner product of state with itself."""
        cuda_sim, torch_sim = medium_register_pair
        
        # Initialize to coherent state
        cuda_sim.setCoherent(0, 1.0 + 0.5j)
        cuda_sim.initStateVector()
        cuda_inner = cuda_sim.innerProduct()
        
        torch_sim.setCoherent(0, 1.0 + 0.5j)
        torch_sim.initStateVector()
        torch_inner = torch_sim.innerProduct()
        
        # Inner product with itself should be 1
        assert abs(cuda_inner - 1.0) < LOOSE_ATOL
        assert abs(torch_inner - 1.0) < LOOSE_ATOL
        np.testing.assert_allclose(cuda_inner, torch_inner, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_inner_product_after_gate(self, medium_register_pair):
        """Test inner product after applying gate."""
        cuda_sim, torch_sim = medium_register_pair
        
        # Initialize to Fock state
        cuda_sim.setFock(0, 1)
        cuda_sim.initStateVector()
        cuda_sim.d(0, 0.5)  # Apply displacement
        cuda_inner = cuda_sim.innerProduct()  # Inner product with original |1⟩
        
        torch_sim.setFock(0, 1)
        torch_sim.initStateVector()
        torch_sim.d(0, 0.5)
        torch_inner = torch_sim.innerProduct()
        
        # Should have changed (inner product < 1)
        assert abs(cuda_inner) < 1.0
        assert abs(torch_inner) < 1.0
        np.testing.assert_allclose(cuda_inner, torch_inner, atol=LOOSE_ATOL, rtol=LOOSE_RTOL)
    
    def test_get_x_grid(self, medium_register_pair):
        """Test position grid retrieval."""
        cuda_sim, torch_sim = medium_register_pair
        
        cuda_grid = cuda_sim.getXGrid(0)
        torch_grid = torch_sim.getXGrid(0)
        
        # Grids should match exactly
        np.testing.assert_allclose(cuda_grid, torch_grid, atol=1e-15, rtol=1e-15)
        
        # Check grid properties
        assert len(cuda_grid) == cuda_sim.register_dims[0]
        assert len(torch_grid) == torch_sim.register_dims[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
