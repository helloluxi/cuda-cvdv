"""
PyTorch backend for CVDV quantum simulator (Refactored with multi-dim tensor storage)
State is stored as multi-dimensional tensor with shape (dim0, dim1, ..., dimN)
where each dimension corresponds to a register.
All operations use torch.cdouble on CUDA device with efficient vectorization.
"""
from typing import List, Tuple, Optional, Union, Callable, Sequence
import torch
import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])
plt.rcParams.update({'font.size': 18, 'text.usetex': True})


class CVDVTorch:
    """PyTorch-based quantum simulator (register-based API) with multi-dim tensor storage.
    
    All registers are treated uniformly as discrete quantum systems with dimensions 2^numQubits.
    State vector is stored as a multi-dimensional torch tensor with shape (dim0, dim1, ..., dimN)
    on CUDA device, eliminating reshape overhead.
    
    INITIALIZATION PATTERN:
    1. Create CVDVTorch instance (allocates registers)
    2. Call setXXX functions to initialize each register
    3. Call initStateVector() to build the tensor product state
    """
    
    def __init__(self, numQubits_list: List[int], device: str = 'cuda') -> None:
        """Initialize simulator with multiple registers.
        
        Args:
            numQubits_list: List of qubit counts for each register
                           Register dimension will be 2^numQubits
            device: PyTorch device ('cuda' or 'cpu', default: 'cuda')
        """
        self.device = torch.device(device)
        self.num_registers = len(numQubits_list)
        self.qubit_counts = np.array(numQubits_list, dtype=np.int32)
        
        # Compute dimensions
        self.register_dims = [1 << qubits for qubits in numQubits_list]
        self.total_size = int(np.prod(self.register_dims))
        
        # Compute grid steps
        self.grid_steps = np.zeros(self.num_registers, dtype=np.float64)
        for i in range(self.num_registers):
            dim = self.register_dims[i]
            self.grid_steps[i] = sqrt(2 * pi / dim)
        
        # Initialize register arrays (will be used to build state vector)
        self.register_arrays: List[Optional[torch.Tensor]] = [None] * self.num_registers
        
        # State vector as multi-dimensional tensor: shape = (dim0, dim1, ..., dimN)
        self.state: Optional[torch.Tensor] = None
    
    def __del__(self) -> None:
        """Cleanup resources."""
        if self.state is not None:
            del self.state
        for arr in self.register_arrays:
            if arr is not None:
                del arr
    
    # ==================== State Initialization ====================
    
    def initStateVector(self) -> None:
        """Build tensor product state from register arrays as multi-dim tensor."""
        
        # Verify all registers are initialized
        for i in range(self.num_registers):
            assert self.register_arrays[i] is not None, f"Register {i} not initialized"
        
        # Compute tensor product using meshgrid-style outer products
        if self.num_registers == 1:
            state = self.register_arrays[0].clone()
        else:
            # Build multi-dimensional tensor via outer products
            # Start with first two registers
            state = torch.outer(self.register_arrays[0], self.register_arrays[1])
            
            # Continue with remaining registers
            for i in range(2, self.num_registers):
                # state has shape (*prev_dims,), next_reg has shape (dim_i,)
                # Reshape state to (..., 1) and next_reg to (1, ..., 1, dim_i)
                state = state.unsqueeze(-1) * self.register_arrays[i].reshape(
                    *([1] * i), -1
                )
        
        # Renormalize (to handle discretization properly)
        norm = torch.sqrt(torch.sum(torch.abs(state) ** 2))
        state = state / norm
        
        # Store as multi-dimensional tensor
        self.state = state
    
    def setZero(self, regIdx: int) -> None:
        """Set register to |0⟩ state."""
        dim = self.register_dims[regIdx]
        arr = torch.zeros(dim, dtype=torch.cdouble, device=self.device)
        arr[0] = 1.0
        self.register_arrays[regIdx] = arr
    
    def setCoherent(self, regIdx: int, alpha: Union[complex, float, int]) -> None:
        """Set register to coherent state |α⟩."""
        if isinstance(alpha, (int, float)):
            alpha = complex(alpha, 0.0)
        
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        
        # Position grid - matching CUDA's gridX: x_i = (i - (dim-1)/2) * dx
        indices = torch.arange(dim, device=self.device, dtype=torch.float64)
        x = (indices - (dim - 1) * 0.5) * dx
        
        # Coherent state parameters
        q = sqrt(2) * alpha.real  # position expectation
        p = sqrt(2) * alpha.imag  # momentum expectation
        
        # Gaussian envelope
        norm = pi ** (-0.25)
        gauss = torch.exp(-0.5 * (x - q) ** 2)
        
        # Phase factor including global phase term
        phase = p * x - p * q / 2.0
        phase_factor = torch.exp(1j * phase).to(torch.cdouble)
        
        # Amplitude with normalization matching CUDA
        amplitude = norm * gauss * sqrt(dx)
        psi = amplitude * phase_factor
        
        self.register_arrays[regIdx] = psi
    
    def setFock(self, regIdx: int, n: int) -> None:
        """Set register to Fock state |n⟩."""
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        
        # Position grid - matching CUDA's gridX: x_i = (i - (dim-1)/2) * dx
        indices = torch.arange(dim, device=self.device, dtype=torch.float64)
        x = (indices - (dim - 1) * 0.5) * dx
        
        # Fock state using normalized Hermite function recurrence
        # psi_0(x) = exp(-x^2/2) / pi^(1/4) * sqrt(dx)
        # psi_n(x) = sqrt(2/n) * x * psi_{n-1}(x) - sqrt((n-1)/n) * psi_{n-2}(x)
        psi_prev = torch.zeros_like(x, dtype=torch.float64)
        psi_curr = torch.exp(-0.5 * x * x) * (pi ** (-0.25)) * sqrt(dx)
        
        for k in range(1, n + 1):
            psi_next = sqrt(2.0 / k) * x * psi_curr - sqrt((k - 1.0) / k) * psi_prev
            psi_prev = psi_curr
            psi_curr = psi_next
        
        self.register_arrays[regIdx] = psi_curr.to(torch.cdouble)
    
    def setUniform(self, regIdx: int) -> None:
        """Set register to uniform superposition."""
        dim = self.register_dims[regIdx]
        arr = torch.ones(dim, dtype=torch.cdouble, device=self.device) / sqrt(dim)
        self.register_arrays[regIdx] = arr
    
    def setFocks(self, regIdx: int, coeffs: Union[List[complex], np.ndarray]) -> None:
        """Set register to superposition of Fock states (vectorized)."""
        coeffs = np.array(coeffs, dtype=complex)
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        
        # Position grid
        indices = torch.arange(dim, device=self.device, dtype=torch.float64)
        x = (indices - (dim - 1) * 0.5) * dx
        
        max_n = len(coeffs)
        mask = np.abs(coeffs) > 1e-12
        if not np.any(mask):
            self.setZero(regIdx)
            return
        
        # Vectorized: compute all Fock states up to max_n simultaneously
        fock_states = torch.zeros((max_n, dim), dtype=torch.float64, device=self.device)
        fock_states[0] = torch.exp(-0.5 * x * x) * (pi ** (-0.25)) * sqrt(dx)
        
        if max_n > 1:
            fock_states[1] = sqrt(2.0) * x * fock_states[0]
            for k in range(2, max_n):
                fock_states[k] = sqrt(2.0 / k) * x * fock_states[k-1] - sqrt((k - 1.0) / k) * fock_states[k-2]
        
        # Vectorized sum
        coeffs_torch = torch.tensor(coeffs, dtype=torch.cdouble, device=self.device)
        state = torch.sum(coeffs_torch.unsqueeze(1) * fock_states.to(torch.cdouble), dim=0)
        
        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(state) ** 2) * torch.tensor(dx, device=self.device))
        state /= norm
        
        self.register_arrays[regIdx] = state
    
    def setCoeffs(self, regIdx: int, coeffs: Union[List[complex], np.ndarray]) -> None:
        """Set register to arbitrary coefficient array directly."""
        coeffs = np.array(coeffs, dtype=complex)
        arr = torch.tensor(coeffs, dtype=torch.cdouble, device=self.device)
        self.register_arrays[regIdx] = arr
    
    def setCat(self, regIdx: int, cat_states: Sequence[Tuple[Union[complex, float], Union[complex, float]]]) -> None:
        """Set register to cat state (superposition of coherent states) - vectorized."""
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        
        # Position grid
        indices = torch.arange(dim, device=self.device, dtype=torch.float64)
        x = (indices - (dim - 1) * 0.5) * dx
        
        # Extract alphas and coefficients
        alphas = []
        coeffs = []
        for alpha, coeff in cat_states:
            if isinstance(alpha, (int, float)):
                alpha = complex(alpha)
            alphas.append(alpha)
            coeffs.append(complex(coeff))
        
        alphas_np = np.array(alphas, dtype=complex)
        coeffs_np = np.array(coeffs, dtype=complex)
        
        # Vectorized construction: shape (n_states, dim)
        n_states = len(alphas)
        q_vals = sqrt(2) * alphas_np.real
        p_vals = sqrt(2) * alphas_np.imag
        
        q_vals_torch = torch.tensor(q_vals, device=self.device, dtype=torch.float64)
        p_vals_torch = torch.tensor(p_vals, device=self.device, dtype=torch.float64)
        
        # Gaussian envelopes (vectorized)
        gauss = torch.exp(-0.5 * (x[None, :] - q_vals_torch[:, None]) ** 2)
        
        # Phase factors (vectorized)
        phase = p_vals_torch[:, None] * x[None, :] - (p_vals_torch * q_vals_torch)[:, None] / 2.0
        phase_factor = torch.exp(1j * phase).to(torch.cdouble)
        
        # Combine
        norm_factor = pi ** (-0.25) * sqrt(dx)
        coherent_states = norm_factor * gauss * phase_factor
        
        # Sum with coefficients
        coeffs_torch = torch.tensor(coeffs_np, dtype=torch.cdouble, device=self.device)
        state = torch.sum(coeffs_torch[:, None] * coherent_states, dim=0)
        
        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(state) ** 2) * torch.tensor(dx, device=self.device))
        state /= norm
        
        self.register_arrays[regIdx] = state
    
    # ==================== Helper Functions ====================
    
    def _get_position_grid(self, regIdx: int) -> torch.Tensor:
        """Get position grid for register."""
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        indices = torch.arange(dim, device=self.device, dtype=torch.float64)
        return (indices - (dim - 1) * 0.5) * dx
    
    def _apply_phase_to_register(self, regIdx: int, phase: torch.Tensor) -> None:
        """Apply phase factor to a single register dimension.
        
        Args:
            regIdx: Register index
            phase: 1D tensor of shape (dim,) with phase factors
        """
        assert self.state is not None, "State not initialized"
        # Reshape phase to broadcast correctly: (1, ..., 1, dim, 1, ..., 1)
        shape = [1] * self.num_registers
        shape[regIdx] = -1
        phase_broadcast = phase.reshape(shape)
        self.state = self.state * phase_broadcast
    
    # ==================== Displacement Gates ====================
    
    def d(self, regIdx: int, beta: Union[complex, float, int]) -> None:
        """Apply displacement operator D(β) to register."""
        if isinstance(beta, (int, float)):
            beta = complex(beta, 0.0)
        
        # Step 1: Apply D(i*Im(β)) = exp(i*√2*Im(β)*q) in position space
        if abs(beta.imag) > 1e-12:
            x = self._get_position_grid(regIdx)
            phase = torch.exp(1j * sqrt(2) * beta.imag * x).to(torch.cdouble)
            self._apply_phase_to_register(regIdx, phase)
        
        # Step 2: Apply D(Re(β)) = exp(-i*√2*Re(β)*p) in momentum space
        if abs(beta.real) > 1e-12:
            self.ftQ2P(regIdx)
            p = self._get_position_grid(regIdx)  # Same grid structure
            phase = torch.exp(-1j * sqrt(2) * beta.real * p).to(torch.cdouble)
            self._apply_phase_to_register(regIdx, phase)
            self.ftP2Q(regIdx)
    
    def cd(self, targetReg: int, ctrlReg: int, ctrlQubit: int, alpha: Union[complex, float, int]) -> None:
        """Apply conditional displacement CD(α) controlled by qubit."""
        if isinstance(alpha, (int, float)):
            alpha = complex(alpha)
        
        # Step 1: Apply CD(i*Im(α)) in position space
        if abs(alpha.imag) > 1e-12:
            self._apply_conditional_phase_q(targetReg, ctrlReg, ctrlQubit, sqrt(2) * alpha.imag)
        
        # Step 2: Apply CD(Re(α)) in momentum space
        if abs(alpha.real) > 1e-12:
            self.ftQ2P(targetReg)
            self._apply_conditional_phase_q(targetReg, ctrlReg, ctrlQubit, -sqrt(2) * alpha.real)
            self.ftP2Q(targetReg)
    
    def _apply_conditional_phase_q(self, targetReg: int, ctrlReg: int, ctrlQubit: int, coeff: float) -> None:
        """Helper: Apply exp(i*coeff*Z*q) where Z acts on control qubit."""
        assert self.state is not None, "State not initialized"
        
        # Get position grid for target register
        q = self._get_position_grid(targetReg)
        
        # Compute phases
        phase_plus = torch.exp(1j * coeff * q).to(torch.cdouble)
        phase_minus = torch.exp(-1j * coeff * q).to(torch.cdouble)
        
        # Decompose control register dimension into qubits
        ctrl_dim = self.register_dims[ctrlReg]
        n_ctrl_qubits = self.qubit_counts[ctrlReg]
        
        # Move control and target registers to first dimensions
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_target = targetReg if targetReg != 0 else ctrlReg
        perm[1], perm[actual_target] = perm[actual_target], perm[1]
        
        state = self.state.permute(*perm)
        
        # Reshape control dimension to expose qubits
        new_shape = [2] * n_ctrl_qubits + list(state.shape[1:])
        state = state.reshape(new_shape)
        
        # Move control qubit to front
        qubit_perm = list(range(n_ctrl_qubits))
        qubit_perm[0], qubit_perm[ctrlQubit] = qubit_perm[ctrlQubit], qubit_perm[0]
        full_perm = qubit_perm + list(range(n_ctrl_qubits, len(new_shape)))
        state = state.permute(*full_perm)
        
        # Apply conditional phase
        # State shape: [2, *other_ctrl_qubits, target_dim, *other_regs]
        # phase_plus/minus shape: (target_dim,)
        # Reshape phases to broadcast: (1, ..., 1, target_dim, 1, ..., 1)
        phase_shape = [1] * (len(state.shape) - len(state.shape[n_ctrl_qubits:]) + 1)
        phase_shape[n_ctrl_qubits] = -1
        
        state[0] = state[0] * phase_plus.reshape(phase_shape)[0]
        state[1] = state[1] * phase_minus.reshape(phase_shape)[0]
        
        # Restore shape
        state = state.permute(*[qubit_perm.index(i) for i in range(n_ctrl_qubits)] + 
                            list(range(n_ctrl_qubits, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl_qubits:]))
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        
        self.state = state
    
    # ==================== Qubit Gates ====================
    
    def _apply_qubit_gate(self, regIdx: int, targetQubit: int, gate_matrix: torch.Tensor) -> None:
        """Apply a 2x2 gate matrix to a qubit within a register."""
        assert self.state is not None, "State not initialized"
        
        dim = self.register_dims[regIdx]
        n_qubits = self.qubit_counts[regIdx]
        
        # Move register to first dimension
        perm = list(range(self.num_registers))
        perm[0], perm[regIdx] = perm[regIdx], perm[0]
        state = self.state.permute(*perm)
        
        # Reshape to expose qubits
        qubit_dims = [2] * n_qubits
        other_dims = list(state.shape[1:])
        state = state.reshape(qubit_dims + other_dims)
        
        # Move target qubit to front
        qubit_perm = list(range(n_qubits))
        qubit_perm[0], qubit_perm[targetQubit] = qubit_perm[targetQubit], qubit_perm[0]
        full_perm = qubit_perm + list(range(n_qubits, len(state.shape)))
        state = state.permute(*full_perm)
        
        # Apply gate: reshape to (2, -1), multiply, reshape back
        original_shape = state.shape
        state = state.reshape(2, -1)
        state = gate_matrix @ state
        state = state.reshape(original_shape)
        
        # Restore qubit order
        state = state.permute(*[qubit_perm.index(i) for i in range(n_qubits)] + 
                            list(range(n_qubits, len(state.shape))))
        
        # Restore register shape
        state = state.reshape([dim] + other_dims)
        
        # Restore register order
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        
        self.state = state
    
    def x(self, regIdx: int, targetQubit: int) -> None:
        """Pauli X gate (implemented as Rx(π))."""
        self.rx(regIdx, targetQubit, pi)
    
    def y(self, regIdx: int, targetQubit: int) -> None:
        """Pauli Y gate (implemented as Ry(π))."""
        self.ry(regIdx, targetQubit, pi)
    
    def z(self, regIdx: int, targetQubit: int) -> None:
        """Pauli Z gate (implemented as Rz(π))."""
        self.rz(regIdx, targetQubit, pi)
    
    def rx(self, regIdx: int, targetQubit: int, theta: float) -> None:
        """Rotation around X axis."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        Rx = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.cdouble, device=self.device)
        self._apply_qubit_gate(regIdx, targetQubit, Rx)
    
    def ry(self, regIdx: int, targetQubit: int, theta: float) -> None:
        """Rotation around Y axis."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        Ry = torch.tensor([[c, -s], [s, c]], dtype=torch.cdouble, device=self.device)
        self._apply_qubit_gate(regIdx, targetQubit, Ry)
    
    def rz(self, regIdx: int, targetQubit: int, theta: float) -> None:
        """Rotation around Z axis."""
        Rz = torch.tensor([[np.exp(-1j * theta / 2), 0],
                          [0, np.exp(1j * theta / 2)]], dtype=torch.cdouble, device=self.device)
        self._apply_qubit_gate(regIdx, targetQubit, Rz)
    
    def h(self, regIdx: int, targetQubit: int) -> None:
        """Hadamard gate."""
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble, device=self.device) / sqrt(2)
        self._apply_qubit_gate(regIdx, targetQubit, H)
    
    # ==================== Phase Gates ====================
    
    def p(self, regIdx: int) -> None:
        """Apply parity gate: flips all basis states (|j⟩ → |N-1-j⟩)."""
        assert self.state is not None, "State not initialized"
        # Flip the register dimension
        self.state = torch.flip(self.state, dims=[regIdx])
    
    def cp(self, targetReg: int, ctrlReg: int, ctrlQubit: int) -> None:
        """Apply conditional parity."""
        assert self.state is not None, "State not initialized"
        
        ctrl_dim = self.register_dims[ctrlReg]
        n_ctrl_qubits = self.qubit_counts[ctrlReg]
        
        # Move control and target to front
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_target = targetReg if targetReg != 0 else ctrlReg
        perm[1], perm[actual_target] = perm[actual_target], perm[1]
        
        state = self.state.permute(*perm)
        
        # Reshape control dimension to expose qubits
        new_shape = [2] * n_ctrl_qubits + list(state.shape[1:])
        state = state.reshape(new_shape)
        
        # Move control qubit to front
        qubit_perm = list(range(n_ctrl_qubits))
        qubit_perm[0], qubit_perm[ctrlQubit] = qubit_perm[ctrlQubit], qubit_perm[0]
        full_perm = qubit_perm + list(range(n_ctrl_qubits, len(new_shape)))
        state = state.permute(*full_perm)
        
        # Flip target dimension for control=1 branch
        # State shape: [2, *other_ctrl_qubits, target_dim, *other_regs]
        # After indexing [1], shape is: [*other_ctrl_qubits, target_dim, *other_regs]
        # so target is at dimension n_ctrl_qubits - 1
        state[1] = torch.flip(state[1], dims=[n_ctrl_qubits - 1])
        
        # Restore shape
        state = state.permute(*[qubit_perm.index(i) for i in range(n_ctrl_qubits)] + 
                            list(range(n_ctrl_qubits, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl_qubits:]))
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        
        self.state = state
    
    def _apply_conditional_phase_q2(self, targetReg: int, ctrlReg: int, ctrlQubit: int, t: float) -> None:
        """Helper: Apply exp(i*t*Z*q^2) where Z acts on control qubit."""
        assert self.state is not None, "State not initialized"
        
        # Get position grid for target register
        q = self._get_position_grid(targetReg)
        
        # Compute phases
        phase_plus = torch.exp(1j * t * q ** 2).to(torch.cdouble)
        phase_minus = torch.exp(-1j * t * q ** 2).to(torch.cdouble)
        
        # Decompose control register dimension into qubits
        ctrl_dim = self.register_dims[ctrlReg]
        n_ctrl_qubits = self.qubit_counts[ctrlReg]
        
        # Move control and target registers to first dimensions
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_target = targetReg if targetReg != 0 else ctrlReg
        perm[1], perm[actual_target] = perm[actual_target], perm[1]
        
        state = self.state.permute(*perm)
        
        # Reshape control dimension to expose qubits
        new_shape = [2] * n_ctrl_qubits + list(state.shape[1:])
        state = state.reshape(new_shape)
        
        # Move control qubit to front
        qubit_perm = list(range(n_ctrl_qubits))
        qubit_perm[0], qubit_perm[ctrlQubit] = qubit_perm[ctrlQubit], qubit_perm[0]
        full_perm = qubit_perm + list(range(n_ctrl_qubits, len(new_shape)))
        state = state.permute(*full_perm)
        
        # Apply conditional phase
        phase_shape = [1] * (len(state.shape) - len(state.shape[n_ctrl_qubits:]) + 1)
        phase_shape[n_ctrl_qubits] = -1
        
        state[0] = state[0] * phase_plus.reshape(phase_shape)[0]
        state[1] = state[1] * phase_minus.reshape(phase_shape)[0]
        
        # Restore shape
        state = state.permute(*[qubit_perm.index(i) for i in range(n_ctrl_qubits)] + 
                            list(range(n_ctrl_qubits, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl_qubits:]))
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        
        self.state = state
    
    def cr(self, targetReg: int, ctrlReg: int, ctrlQubit: int, theta: float) -> None:
        """Apply conditional rotation CR(θ).
        
        Implements conditional rotation where |0⟩ branch gets R(θ) and |1⟩ gets R(-θ).
        Uses decomposition: CR(θ) = exp(-i/2 Z tan(θ/2) q^2) FT exp(-i/2 Z sin(θ) p^2) FT† exp(-i/2 Z tan(θ/2) q^2)
        """
        # Find nearest multiple of π/2
        ratio = theta / (pi / 2)
        theta0 = int(np.floor(ratio + 0.5)) * (pi / 2)
        remainder = theta - theta0
        
        # Apply R(θ₀) for the integer-multiple part
        quarter_turns = int(np.floor(ratio + 0.5)) % 4
        quarter_turns = (quarter_turns + 4) % 4
        
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
        
        # Apply small-angle remainder
        if abs(remainder) > 1e-15:
            tan_half = np.tan(remainder / 2)
            sin_theta = np.sin(remainder)
            
            self._apply_conditional_phase_q2(targetReg, ctrlReg, ctrlQubit, -0.5 * tan_half)
            self.ftQ2P(targetReg)
            self._apply_conditional_phase_q2(targetReg, ctrlReg, ctrlQubit, -0.5 * sin_theta)
            self.ftP2Q(targetReg)
            self._apply_conditional_phase_q2(targetReg, ctrlReg, ctrlQubit, -0.5 * tan_half)
    
    def sheer(self, regIdx: int, t: float) -> None:
        """Apply phase square gate: exp(i*t*q^2)."""
        x = self._get_position_grid(regIdx)
        phase = torch.exp(1j * t * x ** 2).to(torch.cdouble)
        self._apply_phase_to_register(regIdx, phase)
    
    def phaseCubic(self, regIdx: int, t: float) -> None:
        """Apply cubic phase gate: exp(i*t*q^3)."""
        x = self._get_position_grid(regIdx)
        phase = torch.exp(1j * t * x ** 3).to(torch.cdouble)
        self._apply_phase_to_register(regIdx, phase)
    
    # ==================== Bosonic Gates ====================
    
    def r(self, regIdx: int, theta: float) -> None:
        """Apply rotation gate R(θ) in phase space."""
        # Find nearest multiple of π/2
        ratio = theta / (pi / 2)
        theta0 = int(np.floor(ratio + 0.5)) * (pi / 2)
        remainder = theta - theta0
        
        # Apply R(θ₀) for the integer-multiple part
        quarter_turns = int(np.floor(ratio + 0.5)) % 4
        quarter_turns = (quarter_turns + 4) % 4
        
        if quarter_turns == 1:
            self.ftQ2P(regIdx)
        elif quarter_turns == 2:
            self.p(regIdx)
        elif quarter_turns == 3:
            self.ftP2Q(regIdx)
        
        # Apply small-angle remainder
        if abs(remainder) > 1e-15:
            self.sheer(regIdx, -0.5 * np.tan(remainder / 2))
            self.ftQ2P(regIdx)
            self.sheer(regIdx, -0.5 * np.sin(remainder))
            self.ftP2Q(regIdx)
            self.sheer(regIdx, -0.5 * np.tan(remainder / 2))
    
    def s(self, regIdx: int, r: float) -> None:
        """Apply squeezing gate S(r)."""
        exp_r = np.exp(r)
        exp_minus_r = np.exp(-r)
        t = np.exp(-r / 2.0) * np.sqrt(abs(1.0 - exp_minus_r))
        
        self.sheer(regIdx, 0.5 * t)
        self.ftQ2P(regIdx)
        self.sheer(regIdx, (1.0 - exp_minus_r) / (2.0 * t))
        self.ftP2Q(regIdx)
        self.sheer(regIdx, -0.5 * t * exp_r)
        self.ftQ2P(regIdx)
        self.sheer(regIdx, (exp_minus_r - 1.0) / (2.0 * t * exp_r))
        self.ftP2Q(regIdx)
    
    def cs(self, targetReg: int, ctrlReg: int, ctrlQubit: int, r: float) -> None:
        """Apply conditional squeezing gate CS(r).
        
        Conditional version of squeezing where |0⟩ branch gets S(r) and |1⟩ gets S(-r).
        """
        ch_r = np.cosh(r)
        sh_r = np.sinh(r)
        s = np.sqrt(2.0 * abs(np.sinh(0.5 * r)))
        
        # First
        self.sheer(targetReg, 0.5 * s * ch_r)
        self._apply_conditional_phase_q2(targetReg, ctrlReg, ctrlQubit, -0.5 * s * sh_r)
        # Second
        self.ftQ2P(targetReg)
        self.sheer(targetReg, 0.5 * (ch_r - 1) / s)
        self._apply_conditional_phase_q2(targetReg, ctrlReg, ctrlQubit, 0.5 * sh_r / s)
        self.ftP2Q(targetReg)
        # Third
        self.sheer(targetReg, -0.5 * s)
        # Fourth
        self.ftQ2P(targetReg)
        self.sheer(targetReg, 0.5 * (ch_r - 1) / s)
        self._apply_conditional_phase_q2(targetReg, ctrlReg, ctrlQubit, -0.5 * sh_r / s)
        self.ftP2Q(targetReg)
    
    def bs(self, reg1: int, reg2: int, theta: float) -> None:
        """Apply beam splitter gate BS(θ) between two registers.
        
        Implements: BS(θ) = exp(-i*tan(θ/4)*q1*q2) FT₁ FT₂ exp(-i*sin(θ/2)*p1*p2) FT₁† FT₂† exp(-i*tan(θ/4)*q1*q2)
        where FT₁ and FT₂ are Fourier transforms on both registers.
        """
        # Find nearest multiple of π
        ratio = theta / pi
        theta0 = int(np.floor(ratio + 0.5)) * pi
        remainder = theta - theta0
        
        # Apply BS(θ₀) for the integer-multiple part
        half_turns = int(np.floor(ratio + 0.5)) % 4
        half_turns = (half_turns + 4) % 4
        
        # BS(π) = FT₁ FT₂ SWAP, BS(2π) = Par₁ Par₂, BS(-π) = FT₁† FT₂† SWAP
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
        
        # Apply small-angle remainder
        if abs(remainder) > 1e-15:
            tan_quarter = np.tan(remainder / 4)
            sin_half = np.sin(remainder / 2)
            
            self.q1q2(reg1, reg2, -tan_quarter)
            self.ftQ2P(reg1)
            self.ftQ2P(reg2)
            self.q1q2(reg1, reg2, -sin_half)
            self.ftP2Q(reg1)
            self.ftP2Q(reg2)
            self.q1q2(reg1, reg2, -tan_quarter)
    
    def _apply_conditional_q1q2(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, coeff: float) -> None:
        """Helper: Apply exp(i*coeff*Z*q1*q2) where Z acts on control qubit."""
        assert self.state is not None, "State not initialized"
        
        # Get position grids
        q1 = self._get_position_grid(reg1)
        q2 = self._get_position_grid(reg2)
        
        # Compute phase matrices
        phase_matrix_plus = torch.exp(1j * coeff * q1[:, None] * q2[None, :]).to(torch.cdouble)
        phase_matrix_minus = torch.exp(-1j * coeff * q1[:, None] * q2[None, :]).to(torch.cdouble)
        
        # Decompose control register dimension into qubits
        ctrl_dim = self.register_dims[ctrlReg]
        n_ctrl_qubits = self.qubit_counts[ctrlReg]
        
        # Move control, reg1, reg2 to first dimensions
        perm = list(range(self.num_registers))
        perm[0], perm[ctrlReg] = perm[ctrlReg], perm[0]
        actual_reg1 = reg1 if reg1 != 0 else ctrlReg
        perm[1], perm[actual_reg1] = perm[actual_reg1], perm[1]
        actual_reg2_idx = perm.index(reg2)
        perm[2], perm[actual_reg2_idx] = perm[actual_reg2_idx], perm[2]
        
        state = self.state.permute(*perm)
        
        # Reshape control dimension to expose qubits
        new_shape = [2] * n_ctrl_qubits + list(state.shape[1:])
        state = state.reshape(new_shape)
        
        # Move control qubit to front
        qubit_perm = list(range(n_ctrl_qubits))
        qubit_perm[0], qubit_perm[ctrlQubit] = qubit_perm[ctrlQubit], qubit_perm[0]
        full_perm = qubit_perm + list(range(n_ctrl_qubits, len(new_shape)))
        state = state.permute(*full_perm)
        
        # Reshape phase matrices to broadcast
        shape_plus = [1] * len(state.shape)
        shape_plus[n_ctrl_qubits] = self.register_dims[reg1]
        shape_plus[n_ctrl_qubits + 1] = self.register_dims[reg2]
        
        state[0] = state[0] * phase_matrix_plus.reshape(shape_plus)[0]
        state[1] = state[1] * phase_matrix_minus.reshape(shape_plus)[0]
        
        # Restore shape
        state = state.permute(*[qubit_perm.index(i) for i in range(n_ctrl_qubits)] + 
                            list(range(n_ctrl_qubits, len(state.shape))))
        state = state.reshape([ctrl_dim] + list(state.shape[n_ctrl_qubits:]))
        state = state.permute(*[perm.index(i) for i in range(self.num_registers)])
        
        self.state = state
    
    def cbs(self, reg1: int, reg2: int, ctrlReg: int, ctrlQubit: int, theta: float) -> None:
        """Apply conditional beam splitter CBS(θ).
        
        Conditional version where |0⟩ branch gets BS(θ) and |1⟩ gets BS(-θ).
        """
        # Find nearest multiple of π
        ratio = theta / pi
        theta0 = int(np.floor(ratio + 0.5)) * pi
        remainder = theta - theta0
        
        # Apply CBS(θ₀) for the integer-multiple part
        half_turns = int(np.floor(ratio + 0.5)) % 4
        half_turns = (half_turns + 4) % 4
        
        # CBS implementations for large angles
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
        
        # Apply small-angle remainder
        if abs(remainder) > 1e-15:
            tan_quarter = np.tan(remainder / 4)
            sin_half = np.sin(remainder / 2)
            
            self._apply_conditional_q1q2(reg1, reg2, ctrlReg, ctrlQubit, -tan_quarter)
            self.ftQ2P(reg1)
            self.ftQ2P(reg2)
            self._apply_conditional_q1q2(reg1, reg2, ctrlReg, ctrlQubit, -sin_half)
            self.ftP2Q(reg1)
            self.ftP2Q(reg2)
            self._apply_conditional_q1q2(reg1, reg2, ctrlReg, ctrlQubit, -tan_quarter)
    
    def q1q2(self, reg1: int, reg2: int, coeff: float) -> None:
        """Apply Q1Q2 interaction gate: exp(i*coeff*q1*q2)."""
        assert self.state is not None, "State not initialized"
        
        # Get position grids
        q1 = self._get_position_grid(reg1)
        q2 = self._get_position_grid(reg2)
        
        # Compute phase matrix: exp(i*coeff*q1*q2)
        # Shape: (dim1, dim2)
        phase_matrix = torch.exp(1j * coeff * q1[:, None] * q2[None, :]).to(torch.cdouble)
        
        # Reshape phase_matrix to broadcast correctly
        shape = [1] * self.num_registers
        shape[reg1] = self.register_dims[reg1]
        shape[reg2] = self.register_dims[reg2]
        phase_broadcast = phase_matrix.reshape(shape)
        
        self.state = self.state * phase_broadcast
    
    # ==================== Other Gates ====================
    
    def swap(self, reg1: int, reg2: int) -> None:
        """Swap the contents of two registers.
        
        Note: Requires registers to have the same number of qubits.
        """
        assert self.state is not None, "State not initialized"
        
        # Check that registers have same number of qubits (matching CUDA behavior)
        if self.qubit_counts[reg1] != self.qubit_counts[reg2]:
            raise ValueError(f"SWAP requires registers with same number of qubits: "
                           f"reg{reg1} has {self.qubit_counts[reg1]}, "
                           f"reg{reg2} has {self.qubit_counts[reg2]}")
        
        # Transpose the two register dimensions
        perm = list(range(self.num_registers))
        perm[reg1], perm[reg2] = perm[reg2], perm[reg1]
        self.state = self.state.permute(*perm)
    
    def ftQ2P(self, regIdx: int) -> None:
        """Apply Fourier transform: position to momentum representation."""
        assert self.state is not None, "State not initialized"
        
        dx = self.grid_steps[regIdx]
        dim = self.register_dims[regIdx]
        phaseCoeff = pi * (dim - 1.0) / (dim * dx)
        
        # Pre-phase
        x = self._get_position_grid(regIdx)
        pre_phase = torch.exp(1j * phaseCoeff * x).to(torch.cdouble)
        self._apply_phase_to_register(regIdx, pre_phase)
        
        # FFT along register dimension
        self.state = torch.fft.fft(self.state, dim=regIdx)
        
        # Post-phase
        p = self._get_position_grid(regIdx)
        post_phase = torch.exp(1j * phaseCoeff * p).to(torch.cdouble)
        self._apply_phase_to_register(regIdx, post_phase)
        
        # Normalize
        self.state = self.state / sqrt(dim)
    
    def ftP2Q(self, regIdx: int) -> None:
        """Apply inverse Fourier transform: momentum to position representation."""
        assert self.state is not None, "State not initialized"
        
        dx = self.grid_steps[regIdx]
        dim = self.register_dims[regIdx]
        phaseCoeff = -pi * (dim - 1.0) / (dim * dx)
        
        # Pre-phase
        p = self._get_position_grid(regIdx)
        pre_phase = torch.exp(1j * phaseCoeff * p).to(torch.cdouble)
        self._apply_phase_to_register(regIdx, pre_phase)
        
        # IFFT along register dimension
        self.state = torch.fft.ifft(self.state, dim=regIdx, norm='forward')
        
        # Post-phase
        x = self._get_position_grid(regIdx)
        post_phase = torch.exp(1j * phaseCoeff * x).to(torch.cdouble)
        self._apply_phase_to_register(regIdx, post_phase)
        
        # Normalize
        self.state = self.state / sqrt(dim)
    
    # ==================== Measurement & Observables ====================
    
    def getState(self) -> np.ndarray:
        """Get full state vector as numpy array (flattened)."""
        assert self.state is not None, "State not initialized"
        return self.state.flatten().cpu().numpy()
    
    def getXGrid(self, regIdx: int) -> np.ndarray:
        """Get position grid points for register."""
        return self._get_position_grid(regIdx).cpu().numpy()
    
    def m(self, regIdx: int) -> np.ndarray:
        """Compute measurement probabilities for all basis states of a register."""
        assert self.state is not None, "State not initialized"
        
        # Sum over all dimensions except regIdx
        probs = torch.abs(self.state) ** 2
        dims_to_sum = [i for i in range(self.num_registers) if i != regIdx]
        for dim in sorted(dims_to_sum, reverse=True):
            probs = torch.sum(probs, dim=dim)
        
        return probs.cpu().numpy()
    
    def jointMeasure(self, reg1Idx: int, reg2Idx: int) -> np.ndarray:
        """Compute joint measurement probabilities for two registers."""
        assert self.state is not None, "State not initialized"
        
        # Sum over all dimensions except reg1Idx and reg2Idx
        probs = torch.abs(self.state) ** 2
        dims_to_sum = [i for i in range(self.num_registers) if i not in [reg1Idx, reg2Idx]]
        for dim in sorted(dims_to_sum, reverse=True):
            probs = torch.sum(probs, dim=dim)
        
        # Ensure correct dimension order
        if reg1Idx > reg2Idx:
            probs = probs.transpose(0, 1)
        
        return probs.cpu().numpy()
    
    def innerProduct(self) -> complex:
        """Compute inner product between current state and register tensor product."""
        from functools import reduce
        assert self.state is not None, "State not initialized"
        
        # Verify all registers are initialized
        for i in range(self.num_registers):
            assert self.register_arrays[i] is not None, f"Register {i} not initialized"
        
        # Build tensor product from register arrays using reduce
        if self.num_registers == 1:
            tensor_product = self.register_arrays[0].clone()
        else:
            tensor_product = reduce(lambda a, b: torch.kron(a, b),
                                   [arr.flatten() for arr in self.register_arrays])
        
        # Normalize tensor product
        norm = torch.sqrt(torch.sum(torch.abs(tensor_product) ** 2))
        tensor_product = tensor_product / norm
        
        # Compute <state | tensor_product>
        state_flat = self.state.flatten()
        inner = torch.sum(torch.conj(state_flat) * tensor_product)
        return complex(inner.cpu().item())
    
    def getNorm(self) -> float:
        """Compute norm of the state vector."""
        assert self.state is not None, "State not initialized"
        norm = torch.sqrt(torch.sum(torch.abs(self.state) ** 2))
        return float(norm.cpu().item())
    
    # ==================== Phase Space Functions ====================
    
    def getWignerSingleSlice(self, regIdx: int, slice_indices: List[int], wignerN: int = 101, 
                             wXMax: float = 5.0, wPMax: float = 5.0) -> np.ndarray:
        """Compute Wigner function for register at specific slice.
        
        Args:
            regIdx: Register index to compute Wigner function for
            slice_indices: Fixed basis states for other registers (slice_indices[regIdx] is ignored)
            wignerN: Grid size for output
            wXMax: Maximum position value
            wPMax: Maximum momentum value
        
        Returns:
            2D array of shape (wignerN, wignerN) with Wigner function values
        """
        assert self.state is not None, "State not initialized"
        
        # Extract slice by selecting specific indices for other registers
        # Move target register to first dimension
        perm = list(range(self.num_registers))
        perm[0], perm[regIdx] = perm[regIdx], perm[0]
        state = self.state.permute(*perm)
        
        # Select slice for other registers
        for i in range(1, self.num_registers):
            actual_reg = perm[i]
            idx = slice_indices[actual_reg]
            state = state.select(i, idx)
        
        # Now state has shape (target_dim,)
        psi = state.cpu().numpy()
        
        # Compute Wigner function using FFT method
        dim = self.register_dims[regIdx]
        dx = self.grid_steps[regIdx]
        
        # Build Wigner integrand for all x values
        wigner = np.zeros((wignerN, wignerN), dtype=np.float64)
        x_grid = np.linspace(-wXMax, wXMax, wignerN)
        p_grid = np.linspace(-wPMax, wPMax, wignerN)
        
        # Pre-compute FFT for all x values
        fft_results = np.zeros((wignerN, dim), dtype=np.complex128)
        for i, x in enumerate(x_grid):
            # Build integrand for this x: f[y] = conj(ψ(x+y)) * ψ(x-y)
            integrand = np.zeros(dim, dtype=np.complex128)
            
            for j in range(dim):
                y = (j - (dim - 1) / 2) * dx
                
                # Find indices for x+y and x-y (matches CUDA: round((x±y)/dx) + dim/2)
                idx_plus = int(round((x + y) / dx)) + dim // 2
                idx_minus = int(round((x - y) / dx)) + dim // 2
                
                if 0 <= idx_plus < dim and 0 <= idx_minus < dim:
                    integrand[j] = np.conj(psi[idx_plus]) * psi[idx_minus]
            
            # FFT to get Wigner at different p values (matches cuFFT INVERSE)
            fft_results[i, :] = np.fft.ifft(integrand) * dim
        
        # Extract Wigner values (CUDA convention: wigner[p, x])
        for k, p in enumerate(p_grid):
            for i, x in enumerate(x_grid):
                # Map momentum to FFT bin index (matches CUDA algorithm)
                # FFT freq: p_k = π*k/(dim*dx) for k=0..dim-1
                # After fftshift: p_{k'} = (k' - dim/2) * π/(dim*dx)
                dp = pi / (dim * dx)
                k_shifted = int(round(p / dp + dim / 2))
                k_shifted = max(0, min(dim - 1, k_shifted))
                
                # Undo fftshift to get FFT output index
                k_fft = (k_shifted + dim // 2) % dim
                
                # Phase correction using actual FFT bin frequency
                p_actual = (k_shifted - dim / 2) * dp
                phase_corr = np.exp(-1j * p_actual * (dim - 1) * dx)
                wigner[k, i] = np.real(phase_corr * fft_results[i, k_fft]) * dx / pi
        
        return wigner
    
    def getHusimiQFullMode(self, regIdx: int, qN: int = 101, qMax: float = 5.0, pMax: float = 5.0) -> np.ndarray:
        """Compute Husimi Q function for register by tracing out all others.
        
        The Husimi Q function is Q(q,p) = (1/π) ⟨α|ρ|α⟩ where α = (q + ip)/√2.
        
        Args:
            regIdx: Register index
            qN: Grid size for output
            qMax: Maximum position value
            pMax: Maximum momentum value
        
        Returns:
            2D array of shape (qN, qN) with Husimi Q function values
        """
        assert self.state is not None, "State not initialized"
        
        # Move target register to first dimension
        perm = list(range(self.num_registers))
        perm[0], perm[regIdx] = perm[regIdx], perm[0]
        state = self.state.permute(*perm)
        
        # Reshape: (target_dim, other_dims_flat)
        target_dim = self.register_dims[regIdx]
        other_size = self.total_size // target_dim
        state = state.reshape(target_dim, other_size)
        
        # Get state on CPU for numpy operations
        psi = state.cpu().numpy()
        dx = self.grid_steps[regIdx]
        
        # Build position grid
        x_values = np.array([(j - (target_dim - 1) / 2) * dx for j in range(target_dim)])
        
        # Compute Husimi Q function
        husimiQ = np.zeros((qN, qN), dtype=np.float64)
        q_grid = np.linspace(-qMax, qMax, qN)
        p_grid = np.linspace(-pMax, pMax, qN)
        
        PI_POW_NEG_QUARTER = pi ** (-0.25)
        
        # Build windowed signals for all q values and do batched FFT
        # This matches the CUDA algorithm more closely
        windowed_signals = np.zeros((qN, target_dim), dtype=np.float64)  # Real-valued accumulator
        
        for i, q in enumerate(q_grid):
            # Gaussian window (no momentum phase - handled by FFT)
            window = np.exp(-0.5 * (x_values - q) ** 2) * PI_POW_NEG_QUARTER * np.sqrt(dx)
            
            # Apply window to each slice and accumulate power spectrum
            for slice_idx in range(other_size):
                windowed = window * psi[:, slice_idx]
                # FFT (matches cuFFT FORWARD for Husimi Q)
                fft_result = np.fft.fft(windowed)
                windowed_signals[i, :] += np.abs(fft_result) ** 2
        
        # Extract Husimi Q values at requested p-grid (CUDA convention: Q[p, q])
        for j, p_val in enumerate(p_grid):
            for i, q in enumerate(q_grid):
                # Map momentum to FFT bin index
                # FFT freq: p_k = 2π*k/(dim*dx) for k=0..dim-1
                # After fftshift: p_{k'} = (k' - dim/2) * 2π/(dim*dx)
                dp = 2.0 * pi / (target_dim * dx)
                k_shifted = int(round(p_val / dp + target_dim / 2))
                k_shifted = max(0, min(target_dim - 1, k_shifted))
                
                # Undo fftshift to get FFT output index
                k_fft = (k_shifted + target_dim // 2) % target_dim
                
                husimiQ[j, i] = windowed_signals[i, k_fft] / pi
        
        return husimiQ
    
    # ==================== Utilities ====================
    
    def info(self) -> None:
        """Print system information."""
        mem_gb = (self.total_size * 16) / (1024 * 1024 * 1024)
        print(f"PyTorch Backend (Multi-dim Tensor, device: {self.device})")
        print(f"Number of registers: {self.num_registers}")
        print(f"State shape: {tuple(self.register_dims)}")
        print(f"Total state size: {self.total_size} elements ({mem_gb:.3f} GB)")
        for i in range(self.num_registers):
            dim = self.register_dims[i]
            dx = self.grid_steps[i]
            x_bound = sqrt(2 * pi * dim)
            print(f"  Register {i}: dim={dim}, "
                  f"qubits={self.qubit_counts[i]}, dx={dx:.6f}, x_bound={x_bound:.6f}")
    
    def plotWigner(self, regIdx: int, slice_indices: Optional[List[int]] = None, wignerN: int = 201, 
                    wignerMax: float = 5.0, cmap: str = 'RdBu', figsize: Tuple[int, int] = (7, 6), 
                    show: bool = True):
        """Plot Wigner function for a register.
        
        Args:
            regIdx: Register index
            slice_indices: If provided, compute single-slice Wigner; otherwise full mode
            wignerN: Grid resolution
            wignerMax: Maximum value for both position and momentum axes
            cmap: Colormap name
            figsize: Figure size
            show: Whether to display the plot
        
        Returns:
            Figure and axes objects
        """
        if slice_indices is not None:
            wigner = self.getWignerSingleSlice(regIdx, slice_indices, wignerN, wignerMax, wignerMax)
        else:
            wigner = self.getWignerFullMode(regIdx, wignerN, wignerMax, wignerMax)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        extent = (-wignerMax, wignerMax, -wignerMax, wignerMax)
        vmax = np.max(np.abs(wigner))
        
        im = ax.imshow(wigner.T, extent=extent, origin='lower', cmap=cmap,
                      vmin=-vmax, vmax=vmax, aspect='auto')
        
        ax.set_xlabel(r'Position $q$')
        ax.set_ylabel(r'Momentum $p$')
        ax.set_title(f'Wigner Function (Register {regIdx})')
        
        plt.colorbar(im, ax=ax, label=r'$W(q,p)$')
        
        if show:
            plt.show()
        
        return fig, ax
