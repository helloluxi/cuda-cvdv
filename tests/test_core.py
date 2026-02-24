
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from numpy import pi, sqrt
from src import CVDV

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
        alpha = (np.random.random() - 0.5) * 4 + 1j * (np.random.random() - 0.5) * 4
        
        # Path 1: Apply displacement to vacuum
        sim = CVDV([10])
        sim.setFock(0, 0)
        sim.initStateVector()
        sim.d(0, alpha)
        
        # Test overlap with |alpha>
        sim.setCoherent(0, alpha)
        overlap = np.abs(sim.innerProduct())
        
        assert np.abs(overlap - 1.0) < 1e-10, f"D({alpha})|0⟩ should equal |{alpha}⟩, overlap: {overlap}"
    
    def test_displacement_composition(self):
        """Test that D(α)D(β)|0⟩ = D(α+β)|0⟩."""
        alpha = (np.random.random() - 0.5) * 4 + 1j * (np.random.random() - 0.5) * 4
        beta = (np.random.random() - 0.5) * 4 + 1j * (np.random.random() - 0.5) * 4
        
        # First path: D(α)D(β)|0⟩
        sim = CVDV([10])
        sim.setFock(0, 0)
        sim.initStateVector()
        sim.d(0, beta)
        sim.d(0, alpha)
        
        # Test overlap with |alpha + beta>
        sim.setCoherent(0, alpha + beta)
        overlap = np.abs(sim.innerProduct())
        
        assert np.abs(overlap - 1.0) < 1e-10, f"D({alpha})D({beta})|0⟩ should equal D({alpha+beta})|0⟩, overlap: {overlap}"
    
    def test_fourier_transform(self):
        """Test that FT_Q2P followed by FT_P2Q returns to original state."""
        nFock = np.random.randint(0, 20)  # Random Fock state from 0 to 19
        
        sim = CVDV([10])
        sim.setFock(0, nFock)
        sim.initStateVector()
        
        # Apply FT Q->P and expect the same |nFock> state
        sim.ftQ2P(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"FT_Q2P overlap for |{nFock}⟩: {overlap}"
        
        # Apply FT P->Q and expect the same |nFock> state
        sim.ftP2Q(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"FT_P2Q overlap for |{nFock}⟩: {overlap}"
    
    def test_qubit_gates(self):
        """Test all available qubit gates: Hadamard and Pauli rotations (Rx, Ry, Rz)."""
        # Hadamard: H|0⟩ = |+⟩
        sim = CVDV([1])
        sim.setZero(0)
        sim.initStateVector()
        sim.h(0, 0)
        sim.setUniform(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"H|0⟩ should equal |+⟩, overlap: {overlap}"

        # Rx(π)|+⟩ = |+⟩
        sim.rx(0, 0, pi)  # Rx(π)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Rx(π)|+⟩ should equal |+⟩, overlap: {overlap}"

        # Rz(π)|+⟩ = |-⟩
        sim.rz(0, 0, pi)  # Rz(π)
        sim.setCoeffs(0, [1/sqrt(2), -1/sqrt(2)])
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Rz(π)|+⟩ should equal |-⟩, overlap: {overlap}"

        # Ry(π)|-⟩ = i|+⟩
        sim.ry(0, 0, pi)  # Ry(π)
        sim.setUniform(0)
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Ry(π)|-⟩ should equal i|+⟩, overlap: {overlap}"

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

        sim = CVDV([1, 10])  # 1 qubit + 1024-point CV mode
        sim.setUniform(0)    # |+⟩ qubit
        sim.setFock(1, 0)    # vacuum mode
        sim.initStateVector()

        # Apply CD(x) -> D(iy) -> CD(-x) -> D(-iy) -> Rz(-2xy)
        sim.cd(targetReg=1, ctrlReg=0, ctrlQubit=0, alpha=x)
        sim.d(1, 1j * y)
        sim.cd(targetReg=1, ctrlReg=0, ctrlQubit=0, alpha=-x)
        sim.d(1, -1j * y)
        sim.rz(0, 0, 4 * x * y)

        # Should return to initial state |+⟩ ⊗ |vac⟩
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"CD commutator should be identity for x={x}, y={y}, overlap: {overlap}"
    
    def test_rotation(self):
        """Test phase space rotation on coherent state."""
        alpha = 3.0
        
        sim = CVDV([10])
        sim.setCoherent(0, alpha)
        sim.initStateVector()
        
        # Apply random rotation and test overlap
        theta = (np.random.random() - 0.5) * 10
        sim.r(0, theta)
        sim.setCoherent(0, alpha * np.exp(-1j * theta))
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Rotation(${theta}) overlap: {overlap}"

    def test_squeeze(self):
        """Test squeezing operation conjugated displacement."""
        alpha = (np.random.random() - 0.5) * 6
        r = (np.random.random() - 0.5) * 2
        
        sim = CVDV([10])
        sim.setCoherent(0, 3j)
        sim.initStateVector()
        sim.s(0, -r)
        sim.d(0, alpha)
        sim.s(0, r)
        sim.d(0, -alpha*np.exp(-r))
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"Squeezing conjugated displacement overlap for alpha={alpha}, r={r}: {overlap}"

    def test_beam_splitter(self):
        """Test beam splitter operation on Fock state + vacuum state pair."""
        nFock = np.random.randint(3, 10)  # Random Fock state from 3 to 9
        sim = CVDV([10, 10])
        sim.setFock(0, nFock)
        sim.setFock(1, 0)
        sim.initStateVector()
        # Apply beam splitter with random angle
        theta = (np.random.random() - 0.5) * 10
        sim.bs(0, 1, theta)
        # Calculate overlap deviations using binomial coefficient recursion
        p1 = np.cos(theta/2)**2
        p2 = np.sin(theta/2)**2
        ips = np.zeros(nFock + 1)
        # Initialize for j=0: C(nFock, 0) = 1
        theoretical = p2**nFock
        for j in range(nFock+1):
            if j > 0:
                # Use recursion: C(n,k) = C(n,k-1) * (n-k+1) / k
                theoretical *= (nFock - j + 1) / j * p1 / p2
            sim.setFock(0, j)
            sim.setFock(1, nFock - j)
            ips[j] = theoretical - np.abs(sim.innerProduct()) ** 2
        
        assert np.allclose(ips, 0, atol=1e-10), f"BS({theta}$) on |{nFock}⟩|0⟩ overlap deviations: {ips}"

    def test_conditional_rotation(self):
        nFock = np.random.randint(0, 32)
        theta = (np.random.random() - 0.5) * 10
        
        sim = CVDV([1, 10])  # 1 qubit + 1024-point CV mode
        sim.setUniform(0)    # |+⟩ qubit
        sim.setFock(1, nFock)    # Fock state
        sim.initStateVector()

        sim.cr(targetReg=1, ctrlReg=0, ctrlQubit=0, theta=theta)
        sim.rz(0, 0, theta * -(2*nFock + 1))

        # Test overlap with initial state
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"CR({theta}) on |+⟩|{nFock}⟩ overlap: {overlap}"

    def test_conditional_squeeze(self):
        r = (np.random.random() - 0.5) * 2
        alpha = (np.random.random() - 0.5) * 2
        
        sim = CVDV([1, 10])  # 1 qubit + 1024-point CV mode
        sim.setUniform(0)    # |+⟩ qubit
        sim.setFock(1, 0)    # vacuum mode
        sim.initStateVector()

        sim.cs(targetReg=1, ctrlReg=0, ctrlQubit=0, r=r)
        sim.d(1, alpha)
        sim.cs(targetReg=1, ctrlReg=0, ctrlQubit=0, r=-r)
        sim.d(1, -alpha * np.cosh(r))
        sim.cd(1, 0, 0, -alpha * np.sinh(r))

        # Test overlap with initial state
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"CS({r}) on |+⟩|vac⟩ overlap: {overlap}"

    def test_conditional_beam_splitter(self):
        """Test beam splitter operation on Fock state + vacuum state pair."""
        nFock = np.random.randint(3, 10)  # Random Fock state from 3 to 9
        sim = CVDV([1, 10, 10])
        sim.setUniform(0)    # |+⟩ qubit
        sim.setFock(1, nFock)
        sim.setFock(2, 0)
        sim.initStateVector()
        # Apply beam splitter with random angle
        theta = (np.random.random() - 0.5) * 10
        sim.cbs(1, 2, 0, 0, theta)
        sim.cp(2, 0, 0)
        sim.bs(1, 2, -theta)
        # Test overlap with initial state
        overlap = np.abs(sim.innerProduct())
        assert np.abs(overlap - 1.0) < 1e-10, f"CBS({theta}$) on |+⟩|{nFock}⟩|0⟩ overlap: {overlap}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
