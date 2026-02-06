
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from numpy import pi, sqrt
from interface import CVDV

class TestCoreOperations:
    """Test core quantum operations using inner product validation."""
    
    def test_vacuum_state_initialization(self):
        """Test that vacuum state |0⟩ is correctly initialized."""
        sim = CVDV([10])  # 2^10 = 1024 grid points
        sim.setFock(0, 0)  # Set to vacuum |0⟩
        sim.initStateVector()
        
        # After initialization, state should match register state
        sim.setCoherent(0, 0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Vacuum state overlap: {overlap}"
    
    def test_displacement_on_vacuum(self):
        """Test displacement operator: D(α)|0⟩ = |α⟩."""
        alpha = 1.5 - 0.5j
        
        # Path 1: Apply displacement to vacuum
        sim = CVDV([10])
        sim.setFock(0, 0)
        sim.initStateVector()
        sim.displacement(0, alpha)
        
        # Test overlap with |alpha>
        sim.setCoherent(0, alpha)
        overlap = np.abs(sim.innerProduct())
        
        assert np.abs(overlap - 1.0) < 1e-10, f"D({alpha})|0⟩ should equal |{alpha}⟩, overlap: {overlap}"
    
    def test_displacement_composition(self):
        """Test that D(α)D(β)|0⟩ = D(α+β)|0⟩."""
        alpha = 1.0 + 0.5j
        beta = 0.5 - 1.0j
        
        # First path: D(α)D(β)|0⟩
        sim = CVDV([10])
        sim.setFock(0, 0)
        sim.initStateVector()
        sim.displacement(0, beta)
        sim.displacement(0, alpha)
        
        # Test overlap with |alpha + beta>
        sim.setCoherent(0, alpha + beta)
        overlap = np.abs(sim.innerProduct())
        
        assert np.abs(overlap - 1.0) < 1e-10, f"D({alpha})D({beta})|0⟩ should equal D({alpha+beta})|0⟩, overlap: {overlap}"
    
    def test_fourier_transform(self):
        """Test that FT_Q2P followed by FT_P2Q returns to original state."""
        sim = CVDV([10])
        sim.setFock(0, 4)
        sim.initStateVector()
        
        # Apply FT Q->P and expect the same |4> state
        sim.ftQ2P(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"FT_Q2P overlap: {overlap}"
        
        # Apply FT P->Q and expect the same |4> state
        sim.ftP2Q(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"FT_P2Q overlap: {overlap}"
    
    def test_qubit_gates(self):
        """Test all available qubit gates: Hadamard and Pauli rotations (Rx, Ry, Rz)."""
        # Hadamard: H|0⟩ = |+⟩
        sim = CVDV([1])
        sim.setZero(0)
        sim.initStateVector()
        sim.hadamard(0, 0)
        sim.setUniform(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"H|0⟩ should equal |+⟩, overlap: {overlap}"

        # Rx(π)|+⟩ = |+⟩
        sim.pauliRotation(0, 0, 0, pi)  # Rx(π)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Rx(π)|+⟩ should equal |+⟩, overlap: {overlap}"

        # Rz(π/2)|+⟩ = |-⟩
        sim.pauliRotation(0, 0, 2, pi)  # Rz(π)
        sim.setCoeffs(0, [1/sqrt(2), -1/sqrt(2)])
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Rz(π)|+⟩ should equal |-⟩, overlap: {overlap}"

        # Ry(π/2)|-⟩ = i|+⟩
        sim.pauliRotation(0, 0, 1, pi)  # Ry(π)
        sim.setUniform(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Ry(π)|-⟩ should equal i|+⟩, overlap: {overlap}"

        # Measure H Rz(theta)|+⟩
        theta = 0.5
        sim.pauliRotation(0, 0, 2, theta)
        sim.hadamard(0, 0)
        probs = sim.measure(0)
        assert np.abs(probs[0] - np.cos(theta/2)**2) < 1e-10, f"Measurement probability mismatch: {probs[0]}"
        assert np.abs(probs[1] - np.sin(theta/2)**2) < 1e-10, f"Measurement probability mismatch: {probs[1]}"

    def test_conditional_displacement_commutator(self):
        """Test CD commutator identity: Rz(4xy) D(-iy) CD(-x) D(iy) CD(x) = I"""
        x = 0.6
        y = 0.8

        sim = CVDV([1, 10])  # 1 qubit + 1024-point CV mode
        sim.setUniform(0)    # |+⟩ qubit
        sim.setFock(1, 0)    # vacuum mode
        sim.initStateVector()

        # Apply CD(x) -> D(iy) -> CD(-x) -> D(-iy) -> Rz(-2xy)
        sim.cd(targetReg=1, ctrlReg=0, ctrlQubit=0, alpha=x)
        sim.displacement(1, 1j * y)
        sim.cd(targetReg=1, ctrlReg=0, ctrlQubit=0, alpha=-x)
        sim.displacement(1, -1j * y)
        sim.pauliRotation(0, 0, 2, 4 * x * y)

        # Should return to initial state |+⟩ ⊗ |vac⟩
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"CD commutator should be identity, overlap: {overlap}"
    
    def test_rotation(self):
        """Test phase space rotation on coherent state."""
        alpha = 3.0
        
        sim = CVDV([10])
        sim.setCoherent(0, alpha)
        sim.initStateVector()
        
        # Apply -pi/4 rotation and test overlap with |alpha*exp(i*pi/4)>
        sim.rotation(0, -pi/4)
        sim.setCoherent(0, alpha * np.exp(1j * pi/4))
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Rotation(-π/4) overlap: {overlap}"
        
        # Apply 3*pi/2 rotation and test overlap with |-i*alpha>
        sim.rotation(0, 3 * pi/4)
        sim.setCoherent(0, -1j * alpha)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Rotation(3π/2) overlap: {overlap}"

    def test_squeeze(self):
        """Test squeezing operation conjugated displacement."""
        alpha = 3.0
        r = 0.5
        
        sim = CVDV([10])
        sim.setCoherent(0, 3j)
        sim.initStateVector()
        sim.squeeze(0, -r)
        sim.displacement(0, alpha)
        sim.squeeze(0, r)
        sim.displacement(0, -alpha*np.exp(r))
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Squeezing conjugated displacement overlap: {overlap}"

    def test_beam_splitter(self):
        """Test squeezing operation on Fock state + vacuum state pair."""
        nFock = 5

        sim = CVDV([10, 10])
        sim.setFock(0, nFock)
        sim.setFock(1, 0)
        sim.initStateVector()
        sim.beamSplitter(0, 1, pi / 2)

        from scipy.special import comb
        ips = np.array([comb(nFock, j, exact=True) / 2**nFock for j in range(nFock+1)])
        for j in range(nFock+1):
            sim.setFock(0, j)
            sim.setFock(1, nFock - j)
            ips[j] -= np.abs(sim.innerProduct()) ** 2
        
        assert np.allclose(ips, 0, atol=1e-10), "Beam splitter output should match binomial distribution"

        sim.beamSplitter(0, 1, -1.5 * pi)
        sim.setFock(0, 0)
        sim.setFock(1, nFock)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Beam splitter inverse overlap: {overlap}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
