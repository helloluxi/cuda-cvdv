"""
CVDV Benchmarks - CV-DV State Transfer Algorithm
Based on Phys. Rev. Lett. 128, 110503 (2022)

Benchmarks the core operations of the CV-to-DV state transfer algorithm.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt

# Add parent directory to path to import src package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src import CVDV


def run_cvdv_transfer_experiment(n_dv_qubits=4, cv_qubits=12, lam=0.29, return_plots=False):
    """
    Run CV-to-DV state transfer experiment once.
    
    Args:
        n_dv_qubits: Number of DV qubits for encoding
        cv_qubits: CV register size in qubits (2^cv_qubits grid points)
        lam: Interaction parameter
        return_plots: If True, include initial/final state data for plotting
    
    Returns:
        dict: Contains 'time' and optionally 'plots' with initial/final state data
    """
    t_start = time.perf_counter()
    # Define cat state parameters
    # Cat state: |cat⟩ = c0|α0⟩ + c1|α1⟩ where α = q/√2 + i·p/√2
    cat0_center = -1.0 * sqrt(2)  # q0 = -sqrt(2)
    cat1_center = 2.0 * sqrt(2)   # q1 = 2*sqrt(2)

    # Convert to coherent state amplitudes: α = q/√2
    alpha0 = cat0_center / sqrt(2)  # α0 = -1.0
    alpha1 = cat1_center / sqrt(2)  # α1 = 2.0

    # Cat state with equal coefficients (will be normalized automatically)
    cat_states = [(alpha0, 1.0), (alpha1, 1.0)]

    # Capture initial state if plots are requested
    if return_plots:
        sim_initial = CVDV([n_dv_qubits, cv_qubits])
        sim_initial.setUniform(0)
        sim_initial.setCat(1, cat_states)
        sim_initial.initStateVector()
        probs_dv_initial = sim_initial.m(0)
        wigner_cv_initial = sim_initial.getWignerFullMode(1, wignerN=201, wXMax=5, wPMax=5)

    # Initialize system
    sim = CVDV([n_dv_qubits, cv_qubits])
    sim.setUniform(0)
    sim.setCat(1, cat_states)
    
    sim.initStateVector()
    
    # Apply encoding circuit
    dvReg = 0
    cvReg = 1
    
    for k in range(1, n_dv_qubits + 1):
        qubitIdx = k - 1
        
        # V_k: exp(i·v_k·q·σ_y)
        v_k = -pi / (2 * lam * (1 << k))
        sim.rx(dvReg, qubitIdx, pi/2)
        sim.cd(cvReg, dvReg, qubitIdx, 1j * v_k / sqrt(2))
        sim.rx(dvReg, qubitIdx, -pi/2)
        
        # W_k: exp(i·w_k·p·σ_x)
        w_k = lam * (1 << k) / 2 * (-1 if k == n_dv_qubits else 1)
        sim.cd(cvReg, dvReg, qubitIdx, -w_k / sqrt(2))
    
    # Measure final state
    probs_dv_final = sim.m(0)
    
    t_total = time.perf_counter() - t_start
    
    result = {'time': t_total}
    
    if return_plots:
        wigner_cv_final = sim.getWignerFullMode(1, wignerN=201, wXMax=5, wPMax=5)
        result['plots'] = {
            'probs_dv_initial': probs_dv_initial,
            'wigner_cv_initial': wigner_cv_initial,
            'probs_dv_final': probs_dv_final,
            'wigner_cv_final': wigner_cv_final,
            'n_dv_qubits': n_dv_qubits
        }
    
    return result


def benchmark_cvdv_transfer(n_dv_qubits=4, cv_qubits=12, n_runs=10, warmup=2):
    """
    Benchmark CV-to-DV state transfer algorithm.
    
    Args:
        n_dv_qubits: Number of DV qubits for encoding (default: 4)
        cv_qubits: CV register size in qubits (2^cv_qubits grid points, default: 12)
        n_runs: Number of timing runs (default: 10)
        warmup: Number of warmup runs to discard (default: 2)
    
    Returns:
        dict: Timing results with mean, std, min, max
    """
    lam = 0.29  # Interaction parameter
    timings = []
    
    # Warmup runs
    for _ in range(warmup):
        run_cvdv_transfer_experiment(n_dv_qubits, cv_qubits, lam, return_plots=False)
    
    # Timed runs
    for _ in range(n_runs):
        result = run_cvdv_transfer_experiment(n_dv_qubits, cv_qubits, lam, return_plots=False)
        timings.append(result['time'])
    
    # Compute statistics
    results = {
        'mean': np.mean(timings),
        'std': np.std(timings),
        'min': np.min(timings),
        'max': np.max(timings),
        'all': timings,
        'config': {
            'n_dv_qubits': n_dv_qubits,
            'cv_qubits': cv_qubits,
            'n_runs': n_runs,
            'warmup': warmup
        }
    }
    
    return results


def print_results(results):
    """Pretty print benchmark results."""
    config = results['config']
    print("\n" + "="*60)
    print("CVDV State Transfer Benchmark Results")
    print("="*60)
    print(f"Configuration:")
    print(f"  DV qubits:    {config['n_dv_qubits']}")
    print(f"  CV qubits:    {config['cv_qubits']} (grid size: {2**config['cv_qubits']})")
    print(f"  Runs:         {config['n_runs']} (+ {config['warmup']} warmup)")
    print("-"*60)
    print(f"Total Time: {results['mean']*1000:7.2f} ± {results['std']*1000:5.2f} ms  "
          f"[{results['min']*1000:6.2f}, {results['max']*1000:6.2f}]")
    print("="*60 + "\n")


def visualize_cvdv_transfer(n_dv_qubits=4, cv_qubits=12):
    """
    Visualize CV-to-DV state transfer with initial and final state plots.
    
    Args:
        n_dv_qubits: Number of DV qubits for encoding
        cv_qubits: CV register size in qubits (2^cv_qubits grid points)
    
    Returns:
        tuple: (fig_initial, fig_final) matplotlib figure objects
    """
    lam = 0.29  # Interaction parameter
    
    # Run experiment once with plot data
    result = run_cvdv_transfer_experiment(n_dv_qubits, cv_qubits, lam, return_plots=True)
    plots = result['plots']
    
    probs_dv_initial = plots['probs_dv_initial']
    wigner_cv_initial = plots['wigner_cv_initial']
    probs_dv_final = plots['probs_dv_final']
    wigner_cv_final = plots['wigner_cv_final']
    
    # Plot initial state
    fig_initial, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(len(probs_dv_initial)), probs_dv_initial, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Basis state')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(f'Initial DV: |+⟩^⊗{n_dv_qubits}')
    axes[0].set_xlim(-0.5, len(probs_dv_initial) - 0.5)
    axes[0].grid(True, alpha=0.3)
    
    vmax = np.max(np.abs(wigner_cv_initial))
    im = axes[1].imshow(wigner_cv_initial, extent=[-5, 5, -5, 5], origin='lower',
                       cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[1].set_xlabel('q')
    axes[1].set_ylabel('p')
    axes[1].set_title('Initial CV')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    
    # Plot final state
    fig_final, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Apply shifted index mapping
    indices = np.arange(len(probs_dv_final))
    rev_indices = sum((indices >> i & 1) << (n_dv_qubits - 1 - i) for i in range(n_dv_qubits))
    shifted_idx = np.roll(rev_indices, len(probs_dv_final) // 2)[::-1]
    
    axes[0].bar(range(len(probs_dv_final)), probs_dv_final[shifted_idx], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Basis state')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Final DV (shifted)')
    axes[0].set_xlim(-0.5, len(probs_dv_final) - 0.5)
    axes[0].grid(True, alpha=0.3)
    
    vmax = np.max(np.abs(wigner_cv_final))
    im = axes[1].imshow(wigner_cv_final, extent=[-5, 5, -5, 5], origin='lower',
                       cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='equal')
    axes[1].set_xlabel('q')
    axes[1].set_ylabel('p')
    axes[1].set_title('Final CV')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    
    return fig_initial, fig_final


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark CVDV state transfer')
    parser.add_argument('--dv-qubits', type=int, default=4, 
                        help='Number of DV qubits (default: 4)')
    parser.add_argument('--cv-qubits', type=int, default=12,
                        help='CV register qubits (default: 12)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of timing runs (default: 10)')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup runs (default: 2)')
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_cvdv_transfer(
        n_dv_qubits=args.dv_qubits,
        cv_qubits=args.cv_qubits,
        n_runs=args.runs,
        warmup=args.warmup
    )
    
    # Print results
    print_results(results)
