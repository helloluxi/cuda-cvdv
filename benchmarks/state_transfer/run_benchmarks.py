
"""
Benchmark Runner - Compare CVDV vs Bosonic-Qiskit
"""

import os
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from bench_cvdv import benchmark_cvdv_transfer, print_results as print_cvdv, visualize_cvdv_transfer


def run_comparison(dv_qubits=4, cvdv_cv_qubits=[10, 11, 12], bosonic_cv_qubits=None, n_runs=10, warmup=2):
    """
    Run benchmarks for both CVDV and bosonic-qiskit across multiple configurations.
    
    Args:
        dv_qubits: Number of DV qubits
        cvdv_cv_qubits: List of CV register sizes (in qubits) for CVDV to test
        bosonic_cv_qubits: List of CV register sizes (in qubits) for Bosonic to test (defaults to cvdv_cv_qubits)
        n_runs: Number of timing runs
        warmup: Number of warmup runs
    """
    if bosonic_cv_qubits is None:
        bosonic_cv_qubits = cvdv_cv_qubits
    
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON: CUDA-CVDV vs Bosonic-Qiskit")
    print("="*70)
    
    # Try to import bosonic-qiskit
    try:
        from bench_bosonic import benchmark_bosonic_transfer, print_results as print_bosonic
        has_bosonic = True
    except ImportError:
        print("\nWarning: bosonic-qiskit not available, running CVDV only")
        has_bosonic = False
    
    # Use the union of all configs for plotting
    all_configs = sorted(set(cvdv_cv_qubits + bosonic_cv_qubits))
    
    # Store results for all configurations
    cvdv_times = {}
    cvdv_results_all = {}
    bosonic_times = {}
    bosonic_results_all = {}
    
    # Run CVDV benchmarks
    for cv_qubits in cvdv_cv_qubits:
        print(f"\n{'='*70}")
        print(f"[CVDV] Configuration: DV={dv_qubits} qubits, CV={cv_qubits} qubits (dim={2**cv_qubits})")
        print(f"{'='*70}")
        cvdv_results = benchmark_cvdv_transfer(dv_qubits, cv_qubits, n_runs, warmup)
        print_cvdv(cvdv_results)
        cvdv_times[cv_qubits] = cvdv_results['mean'] * 1000  # Convert to ms
        cvdv_results_all[cv_qubits] = cvdv_results
    
    # Run bosonic benchmarks
    if has_bosonic:
        for cv_qubits in bosonic_cv_qubits:
            print(f"\n{'='*70}")
            print(f"[Bosonic-Qiskit] Configuration: DV={dv_qubits} qubits, CV={cv_qubits} qubits (dim={2**cv_qubits})")
            print(f"{'='*70}")
            try:
                bosonic_results = benchmark_bosonic_transfer(dv_qubits, 2**cv_qubits, n_runs, warmup)
                if bosonic_results is not None:
                    print_bosonic(bosonic_results)
                    bosonic_times[cv_qubits] = bosonic_results['mean'] * 1000  # Convert to ms
                    bosonic_results_all[cv_qubits] = bosonic_results
                else:
                    bosonic_times[cv_qubits] = None
                    bosonic_results_all[cv_qubits] = None
            except Exception as e:
                print(f"Bosonic-qiskit benchmark failed: {e}")
                bosonic_times[cv_qubits] = None
    
    # Save JSON results
    save_json_results(dv_qubits, cvdv_results_all, bosonic_results_all if has_bosonic else None, n_runs, warmup)
    
    # Generate comparison plot
    plot_comparison(all_configs, cvdv_times, bosonic_times if has_bosonic else None)
    
    # Generate state visualizations for the last config
    if cvdv_cv_qubits:
        last_cvdv_config = cvdv_cv_qubits[-1]
        print(f"\n{'='*70}")
        print(f"Generating CVDV state visualization for CV={last_cvdv_config} qubits...")
        print(f"{'='*70}")
        try:
            fig_initial, fig_final = visualize_cvdv_transfer(dv_qubits, last_cvdv_config)
            output_dir = os.path.join(os.path.dirname(__file__), 'results')
            os.makedirs(output_dir, exist_ok=True)
            fig_initial.savefig(os.path.join(output_dir, 'cvdv_initial_state.png'), dpi=300, bbox_inches='tight')
            fig_final.savefig(os.path.join(output_dir, 'cvdv_final_state.png'), dpi=300, bbox_inches='tight')
            plt.close(fig_initial)
            plt.close(fig_final)
            print(f"✓ Saved: cvdv_initial_state.png")
            print(f"✓ Saved: cvdv_final_state.png")
        except Exception as e:
            print(f"Warning: CVDV visualization failed: {e}")
    
    if has_bosonic and bosonic_cv_qubits:
        last_bosonic_config = bosonic_cv_qubits[-1]
        print(f"\nGenerating Bosonic-Qiskit state visualization for CV={last_bosonic_config} qubits...")
        try:
            from bench_bosonic import visualize_bosonic_transfer
            fig_initial, fig_final = visualize_bosonic_transfer(dv_qubits, 2**last_bosonic_config)
            if fig_initial is not None and fig_final is not None:
                output_dir = os.path.join(os.path.dirname(__file__), 'results')
                fig_initial.savefig(os.path.join(output_dir, 'bosonic_initial_state.png'), dpi=300, bbox_inches='tight')
                fig_final.savefig(os.path.join(output_dir, 'bosonic_final_state.png'), dpi=300, bbox_inches='tight')
                plt.close(fig_initial)
                plt.close(fig_final)
                print(f"✓ Saved: bosonic_initial_state.png")
                print(f"✓ Saved: bosonic_final_state.png")
        except Exception as e:
            print(f"Warning: Bosonic visualization failed: {e}")


def save_json_results(dv_qubits, cvdv_results, bosonic_results=None, n_runs=10, warmup=2):
    """Save benchmark results to JSON file."""
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dv_qubits': dv_qubits,
            'n_runs': n_runs,
            'warmup': warmup
        },
        'cvdv': {}
    }
    
    # CVDV results (convert numpy types to native Python)
    for cv_qubits, res in cvdv_results.items():
        results['cvdv'][str(cv_qubits)] = {
            'cv_dimension': 2**cv_qubits,
            'mean_ms': float(res['mean'] * 1000),
            'std_ms': float(res['std'] * 1000),
            'min_ms': float(res['min'] * 1000),
            'max_ms': float(res['max'] * 1000)
        }
    
    # Bosonic results
    if bosonic_results:
        results['bosonic'] = {}
        for cv_qubits, res in bosonic_results.items():
            if res is not None:
                results['bosonic'][str(cv_qubits)] = {
                    'cv_dimension': 2**cv_qubits,
                    'mean_ms': float(res['mean'] * 1000),
                    'std_ms': float(res['std'] * 1000),
                    'min_ms': float(res['min'] * 1000),
                    'max_ms': float(res['max'] * 1000)
                }
    
    json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved: benchmark_results.json")


def plot_comparison(cv_qubit_configs, cvdv_times_dict, bosonic_times_dict=None):
    """Generate comparison bar chart with modern styling."""
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX for simplicity
    
    # Modern color palette
    colors = {
        'cvdv': '#2E86AB',      # Modern blue
        'bosonic': '#A23B72'    # Modern purple
    }
    
    width = 0.5
    
    # Create figure with modern style
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F9FA')
    
    # Arrange bars: bosonic (low to high), then cvdv (low to high)
    bar_configs = []
    bar_values = []
    bar_colors = []
    bar_labels_added = {'bosonic': False, 'cvdv': False}
    
    # Add bosonic bars (sorted by dimension low to high)
    if bosonic_times_dict:
        for cv_q in sorted(bosonic_times_dict.keys()):
            if bosonic_times_dict[cv_q] is not None:
                bar_configs.append(('Bosonic', 2**cv_q))
                bar_values.append(bosonic_times_dict[cv_q])
                bar_colors.append(colors['bosonic'])
    
    # Add cvdv bars (sorted by dimension low to high)
    for cv_q in sorted(cvdv_times_dict.keys()):
        if cvdv_times_dict[cv_q] is not None:
            bar_configs.append(('CVDV', 2**cv_q))
            bar_values.append(cvdv_times_dict[cv_q])
            bar_colors.append(colors['cvdv'])
    
    # Plot bars centered at their positions
    x_positions = np.arange(len(bar_configs))
    bars = ax.bar(x_positions, bar_values, width, color=bar_colors, alpha=0.85, 
                  edgecolor='white', linewidth=1.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Create legend manually
    from matplotlib.patches import Patch
    legend_elements = []
    if bosonic_times_dict and any(bosonic_times_dict.values()):
        legend_elements.append(Patch(facecolor=colors['bosonic'], alpha=0.85, 
                                     edgecolor='white', linewidth=1.5, label='Bosonic-Qiskit (CPU)'))
    if cvdv_times_dict:
        legend_elements.append(Patch(facecolor=colors['cvdv'], alpha=0.85,
                                     edgecolor='white', linewidth=1.5, label='CUDA-CVDV (GPU)'))
    
    # Styling
    ax.set_ylabel('Total Runtime (ms)', fontsize=12, fontweight='bold', color='#333')
    ax.set_xlabel('CV Mode Dimension', fontsize=12, fontweight='bold', color='#333')
    ax.set_title('Performance Comparison', 
                 fontsize=15, fontweight='bold', pad=20, color='#222')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{dim}' for lib, dim in bar_configs], fontsize=12)
    ax.legend(handles=legend_elements, fontsize=12, frameon=True, shadow=True, fancybox=True, loc='upper left')
    
    # Modern grid style
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    plt.tight_layout()
    
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n{'='*70}")
    print(f"Comparison plot saved to: {plot_file}")
    print(f"{'='*70}\n")
    plt.close(fig)  # Close figure to free memory


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Run benchmarks and compare CVDV vs bosonic-qiskit'
    )
    parser.add_argument('--dv-qubits', type=int, default=4,
                        help='Number of DV qubits (default: 4)')
    parser.add_argument('--cvdv-cv-qubits', type=int, nargs='+', default=[10, 11, 12],
                        help='CVDV: CV register qubits to test (default: 10 11 12)')
    parser.add_argument('--bosonic-cv-qubits', type=int, nargs='+', default=None,
                        help='Bosonic: CV register qubits to test (default: same as --cvdv-cv-qubits)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of timing runs (default: 10)')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup runs (default: 2)')
    args = parser.parse_args()
    
    run_comparison(
        dv_qubits=args.dv_qubits,
        cvdv_cv_qubits=args.cvdv_cv_qubits,
        bosonic_cv_qubits=args.bosonic_cv_qubits,
        n_runs=args.runs,
        warmup=args.warmup
    )
