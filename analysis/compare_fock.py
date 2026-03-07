"""Compare Fock Encoding with Qiskit Transpilation.

Compares displacement gate simulation error vs CNOT count:
- **WF encoding** (ours)
- **Fock Qiskit** (transpiled)

Produces: figures/compare_fock.pdf, figures/compare_fock.png
Returns: {'df_fock': DataFrame, 'df_wf': DataFrame}
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from ._common import _save_fig

# Plotting setup
plt.style.use(['science'])
plt.rcParams.update({
    'font.size': 24,
    'text.usetex': True,
})

# Device setup
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

# Simulation parameters
ALPHA = 2.0           # Displacement parameter
GAMMA_MAX = 127       # Maximum Fock state cutoff
K_MAX = 256           # Maximum Fock state dimension
N_VALUES_WF = [7, 8]  # Qubit numbers for WF encoding
N_VALUES_FOCK = [6, 7]  # Qubit numbers for Fock/Qiskit
OPTIMIZATION_LEVELS = [3]  # Qiskit transpiler optimization levels
SQ2 = np.sqrt(2)


def exact_displacement(K, alpha):
    """Compute exact displacement operator via matrix exponential."""
    H = torch.zeros((K, K), dtype=torch.complex128)
    for k in range(K-1):
        val = torch.sqrt(torch.tensor(k + 1, dtype=torch.float64))
        H[k, k+1] = val
        H[k+1, k] = val
    return torch.linalg.matrix_exp(1j * alpha * H)


def cumulative_error(V, D_exact, Gamma):
    """Compute cumulative error over Fock states 0 to Gamma."""
    diff = V[:Gamma+1, :Gamma+1] - D_exact[:Gamma+1, :Gamma+1]
    err_sq = torch.sum(torch.abs(diff)**2).item()
    return torch.sqrt(torch.tensor(err_sq)).item()


def compiled_fock_to_qiskit_circuit(n_qubits, alpha):
    """Convert exact Fock displacement operator to Qiskit circuit.
    
    Args:
        n_qubits: Number of qubits to encode the circuit
        alpha: Displacement parameter
    
    Returns:
        Qiskit QuantumCircuit object
    """
    # Dimension is 2^n_qubits
    dim = 1 << n_qubits
    
    # Get exact displacement in Fock basis
    U_exact = exact_displacement(dim, alpha).cpu().numpy()

    # Assert unitary
    assert np.allclose(U_exact.conj().T @ U_exact, np.eye(dim)), "U_exact is not unitary"
    
    # Create Qiskit operator from unitary
    operator = Operator(U_exact)
    
    # Initialize circuit and decompose the unitary
    circuit = QuantumCircuit(n_qubits)
    circuit.unitary(operator, range(n_qubits), label='D(alpha)')
    
    # Decompose the unitary gate into basic gates
    circuit = circuit.decompose()
    
    return circuit


def compile_with_qiskit(circuit, optimization_level=1):
    """Compile circuit using Qiskit transpiler and extract gate counts.
    
    Args:
        circuit: Qiskit QuantumCircuit object
        optimization_level: 0-3, higher is more optimized (default: 1)
    
    Returns:
        Transpiled circuit and CNOT count
    """
    # Transpile with optimization
    basis_gates = ['u3', 'cx']  # Standard gate set: single-qubit rotations + CNOT
    
    transpiled = transpile(
        circuit, 
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        seed_transpiler=42  # For reproducibility
    )
    
    # Count CNOT gates
    cnot_count = transpiled.count_ops().get('cx', 0)
    
    return transpiled, cnot_count


def apply_qiskit_circuit_to_fock(transpiled_circuit, n_qubits):
    """Extract unitary from Qiskit transpiled circuit.
    
    Args:
        transpiled_circuit: Qiskit transpiled QuantumCircuit object
        n_qubits: Number of qubits
    
    Returns:
        Unitary matrix as torch tensor
    """
    # Get unitary from circuit
    operator = Operator(transpiled_circuit)
    unitary_np = operator.data
    
    # Convert to torch
    dim = 1 << n_qubits
    U_compiled = torch.tensor(unitary_np[:dim, :dim],  # type: ignore
                              dtype=torch.complex128, 
                              device='cuda')
    
    return U_compiled


def discrete_qft(psi: torch.Tensor) -> torch.Tensor:
    """Discrete QFT with proper phase conventions."""
    N_total = psi.shape[0]
    global_phase = torch.exp(torch.tensor(-1j * torch.pi * (N_total - 1)**2 / (2 * N_total)))
    j = torch.arange(N_total, device=psi.device)
    pre_phase = torch.exp(1j * torch.pi * (N_total - 1) * j / N_total)
    psi_phased = psi * pre_phase
    psi_fft = torch.fft.fft(psi_phased, dim=-1, norm='ortho')
    k = torch.arange(N_total, device=psi.device)
    post_phase = torch.exp(1j * torch.pi * (N_total - 1) * k / N_total)
    psi_out = psi_fft * post_phase * global_phase
    return psi_out


def discrete_iqft(psi: torch.Tensor) -> torch.Tensor:
    """Inverse discrete QFT."""
    N_total = psi.shape[0]
    global_phase = torch.exp(torch.tensor(1j * torch.pi * (N_total - 1)**2 / (2 * N_total)))
    j = torch.arange(N_total, device=psi.device)
    pre_phase = torch.exp(-1j * torch.pi * (N_total - 1) * j / N_total)
    psi_phased = psi * pre_phase
    psi_ifft = torch.fft.ifft(psi_phased, dim=-1, norm='ortho')
    k = torch.arange(N_total, device=psi.device)
    post_phase = torch.exp(-1j * torch.pi * (N_total - 1) * k / N_total)
    psi_out = psi_ifft * post_phase * global_phase
    return psi_out


def apply_displacement_discrete(psi: torch.Tensor, alpha: float, x: torch.Tensor) -> torch.Tensor:
    """Apply displacement using WF encoding via QFT."""
    psi_p = discrete_qft(psi)
    phase = torch.exp(-1j * SQ2 * alpha * x)
    psi_p_displaced = psi_p * phase
    psi_q_displaced = discrete_iqft(psi_p_displaced)
    return psi_q_displaced


def wf_displacement_on_fock(N_total: int, alpha_val: float, Gamma: int) -> dict:
    """Compute WF encoding displacement error on Fock states.
    
    Args:
        N_total: Total grid size (must be power of 2)
        alpha_val: Displacement parameter
        Gamma: Fock state cutoff
    
    Returns:
        Dictionary with errors per Fock state and cumulative error
    """
    dx = torch.sqrt(torch.tensor(2.0 * torch.pi / N_total, device='cuda'))
    x = torch.arange(N_total, dtype=torch.float64, device='cuda')
    x = (x - (N_total - 1) / 2) * dx
    
    psi_q_prev = torch.zeros_like(x)
    psi_q = torch.exp(-0.5 * x**2) / (torch.pi ** 0.25) * torch.sqrt(dx)
    
    alpha_shift = alpha_val * SQ2
    x_shifted = x - alpha_shift
    psi_shifted_prev = torch.zeros_like(x)
    psi_shifted = torch.exp(-0.5 * x_shifted**2) / (torch.pi ** 0.25) * torch.sqrt(dx)
    
    errors = []
    for n in range(Gamma + 1):
        if n > 0:
            psi_q_prev, psi_q = psi_q, torch.sqrt(torch.tensor(2.0 / n)) * x * psi_q - torch.sqrt(torch.tensor((n - 1) / n)) * psi_q_prev
            psi_shifted_prev, psi_shifted = (
                psi_shifted,
                torch.sqrt(torch.tensor(2.0 / n)) * x_shifted * psi_shifted - 
                torch.sqrt(torch.tensor((n - 1) / n)) * psi_shifted_prev
            )
        
        psi_complex = psi_q.to(torch.complex128)
        psi_discrete = apply_displacement_discrete(psi_complex, alpha_val, x)
        psi_analytical = psi_shifted.to(torch.complex128)
        
        err = torch.linalg.norm(psi_discrete - psi_analytical).item()
        errors.append(err)
    
    cumulative_error = np.sqrt(np.sum(np.array(errors)**2))
    return {'errors': errors, 'cumulative_error': cumulative_error}


def cnot_wf(n):
    """Gate count estimation for WF encoding."""
    return 2 * n * (n - 1)


def run(fig_dir: str) -> dict:
    """Run comparison between Fock and WF encodings.
    
    Args:
        fig_dir: Directory to save figures
        
    Returns:
        Dictionary with 'df_fock' and 'df_wf' DataFrames
    """
    D_exact = exact_displacement(K_MAX, ALPHA)
    results_fock_qiskit = []
    results_wf = []

    print("  Running Fock Qiskit transpilation...")
    for n in tqdm(N_VALUES_FOCK, desc='compare_fock (Fock)'):
        for opt_level in OPTIMIZATION_LEVELS:
            try:
                circuit = compiled_fock_to_qiskit_circuit(n, ALPHA)
                transpiled, cnot = compile_with_qiskit(circuit, optimization_level=opt_level)
                U_qiskit = apply_qiskit_circuit_to_fock(transpiled, n)
                
                # Compute error for each Gamma value (capped at dim-1)
                dim = 1 << n
                for Gamma in range(min(GAMMA_MAX, dim - 1) + 1):
                    error = cumulative_error(U_qiskit, D_exact, Gamma)
                    results_fock_qiskit.append({'n': n, 'CNOT': cnot, 'error': error, 'opt_level': opt_level, 'Gamma': Gamma})
            except Exception as e:
                print(f"  Transpilation failed for n={n}, opt={opt_level}: {e}")

    print("  Running WF encoding...")
    for n in tqdm(N_VALUES_WF, desc='compare_fock (WF)'):
        N_total = 2**n
        cnot = cnot_wf(n)
        
        # Compute error for each Gamma value
        for Gamma in range(GAMMA_MAX + 1):
            result = wf_displacement_on_fock(N_total, ALPHA, Gamma)
            results_wf.append({'n': n, 'CNOT': cnot, 'error': result['cumulative_error'], 'Gamma': Gamma})

    df_fock = pd.DataFrame(results_fock_qiskit)
    df_wf = pd.DataFrame(results_wf)

    # Filter errors to range [1e-9, 1e-1]
    df_fock = df_fock[(df_fock['error'] >= 1e-9) & (df_fock['error'] <= 1e-1)]
    df_wf = df_wf[(df_wf['error'] >= 1e-9) & (df_wf['error'] <= 1e-1)]

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlabel(r'$\Gamma$')
    ax.set_ylabel(r'$\varepsilon$')
    ax.set_yscale('log')

    # Blue-purple series for WF encoding
    wf_ns = sorted(df_wf['n'].unique())
    wf_colors = plt.get_cmap('cool')(np.linspace(0.15, 0.75, len(wf_ns)))

    # Red-yellow series for Fock encoding
    fock_ns = sorted(df_fock['n'].unique())
    fock_colors = plt.get_cmap('autumn')(np.linspace(0.1, 0.75, len(fock_ns)))

    # Plot WF - error vs Gamma with CNOT count in legend
    for color, n in zip(wf_colors, wf_ns):
        df_sub = df_wf[df_wf['n'] == n].sort_values('Gamma')
        cnot = df_sub['CNOT'].iloc[0]
        ax.plot(df_sub['Gamma'], df_sub['error'], marker='o', linewidth=2, markersize=8,
                color=color, label=f'Ours, $n={n}$, $N_{{CNOT}}={cnot}$', alpha=0.85)

    # Plot Fock (Qiskit) - error vs Gamma with CNOT count in legend
    for color, n in zip(fock_colors, fock_ns):
        df_sub = df_fock[df_fock['n'] == n].sort_values('Gamma')
        cnot = df_sub['CNOT'].iloc[0]
        ax.plot(df_sub['Gamma'], df_sub['error'], marker='s', linewidth=2, markersize=7,
                color=color, label=f'Fock, $n={n}$, $N_{{CNOT}}={cnot}$', linestyle='--', alpha=0.85)

    ax.grid(True, alpha=0.3, which='major')
    ax.legend(loc=(0.14, 0.05), frameon=True, framealpha=0.5, fontsize=16)
    fig.tight_layout()
    _save_fig(fig, 'compare_fock', fig_dir)
    plt.close(fig)

    return {'df_fock': df_fock, 'df_wf': df_wf}
