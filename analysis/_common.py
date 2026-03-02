"""Shared utilities for all analysis modules."""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots  # noqa: F401

# Use CUDA backend for torch
torch.set_default_device('cuda')
torch.set_default_dtype(torch.float64)

# Use scienceplots style
plt.style.use(['science'])
plt.rcParams.update({'font.size': 24, 'text.usetex': True})

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _save_fig(fig, base_name: str, fig_dir: str):
    """Save figure as both PDF and PNG in figures directory."""
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f'{base_name}.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(fig_dir, f'{base_name}.png'), bbox_inches='tight', dpi=300)


def fock_recurrence(N: int):
    """Generate all N Fock wavefunctions on the position grid using torch.

    Returns (states, x, dx) where:
    - states: list of torch tensors (float64), each of length N
    - x: torch tensor of position grid points
    - dx: grid spacing (float)
    
    Stored as real (no imaginary part yet; caller applies (-i)^n for momentum).
    """
    dx = torch.sqrt(torch.tensor(2.0 * torch.pi / N))
    idx = torch.arange(N, dtype=torch.float64)
    x = (idx - (N - 1) * 0.5) * dx

    states = []
    psi_prev = torch.zeros(N, dtype=torch.float64)
    psi_curr = torch.exp(-0.5 * x ** 2) * (torch.pi ** -0.25) * torch.sqrt(dx)
    states.append(psi_curr.clone())

    for k in range(1, N):
        k_float = float(k)
        psi_next = (torch.sqrt(torch.tensor(2.0 / k_float)) * x * psi_curr
                    - torch.sqrt(torch.tensor((k_float - 1.0) / k_float)) * psi_prev)
        psi_prev, psi_curr = psi_curr, psi_next
        states.append(psi_curr.clone())

    return states, x, dx.item()


def linear_model(x, a, b):
    return a * x + b


def fit_ab(Gamma_vals, err_vals):
    """Fit log(err) = a*(Gamma+1/2) + b. Returns (a, b) or None."""
    if len(Gamma_vals) < 2:
        return None
    try:
        popt, _ = curve_fit(linear_model, np.asarray(Gamma_vals) + 0.5, np.log(err_vals))
        return tuple(popt)
    except Exception:
        return None


def fit_coeff_scaling(fit_params):
    """Given list of (N, a, b), fit a~c_a*N^{-1/2}+d_a and b~c_b*N^{1/2}+d_b.

    Returns dict with keys c_a, d_a, r2_a, c_b, d_b, r2_b.
    """
    N_vals = np.array([p[0] for p in fit_params], dtype=float)
    a_vals = np.array([p[1] for p in fit_params])
    b_vals = np.array([p[2] for p in fit_params])

    inv_sqrt_N = N_vals ** (-0.5)
    sqrt_N = N_vals ** 0.5

    popt_a, _ = curve_fit(linear_model, inv_sqrt_N, a_vals)
    c_a, d_a = popt_a
    res_a = a_vals - linear_model(inv_sqrt_N, c_a, d_a)
    r2_a = 1 - np.sum(res_a ** 2) / np.sum((a_vals - a_vals.mean()) ** 2)

    popt_b, _ = curve_fit(linear_model, sqrt_N, b_vals)
    c_b, d_b = popt_b
    res_b = b_vals - linear_model(sqrt_N, c_b, d_b)
    r2_b = 1 - np.sum(res_b ** 2) / np.sum((b_vals - b_vals.mean()) ** 2)

    return dict(c_a=c_a, d_a=d_a, r2_a=r2_a, c_b=c_b, d_b=d_b, r2_b=r2_b)


def _qft_ab(n_qubits: int):
    """Return (a, b) for QFT error bound coefficients (fitted for 5<=n<=10)."""
    a = 4.2125 * 2 ** (-n_qubits / 2) + 0.1027
    b = -5.7712 * 2 ** (n_qubits / 2) + 14.7006
    return a, b


def bound_qft(k, n_qubits: int):
    """Per-Fock QFT error upper bound: exp(a*(k+1/2) + b)."""
    a, b = _qft_ab(n_qubits)
    return np.exp(a * (np.asarray(k) + 0.5) + b)


def bound_disp(k, n_qubits: int, Re_alpha: float = 2.0):
    """Per-Fock displacement D(Re_alpha) error upper bound."""
    a, b = _qft_ab(n_qubits)
    k = np.asarray(k)
    term1 = np.exp(a * (k + 0.5) + b)
    term2 = np.exp(a * (np.sqrt(k + 0.5) + Re_alpha) ** 2 + b)
    return term1 + term2


def bound_rot(k, n_qubits: int, theta: float = np.pi / 4):
    """Per-Fock rotation R(theta) error upper bound."""
    a, b = _qft_ab(n_qubits)
    k = np.asarray(k)
    t = abs(np.tan(theta / 2))
    c1 = (1 + t) ** 2
    c2 = 1 + np.sin(theta) ** 2 * (1 + t) ** 2
    term1 = np.exp(a * c1 * (k + 0.5) + b)
    term2 = np.exp(a * c2 * (k + 0.5) + b)
    return term1 + term2


def bound_sq(k, n_qubits: int, r: float = 1.0):
    """Per-Fock squeezing S(r) error upper bound."""
    a, b = _qft_ab(n_qubits)
    k = np.asarray(k)
    sq = np.sqrt(np.exp(r) - 1)
    mu = [
        -np.exp(r / 2) * sq,
        np.exp(-r / 2) * sq,
        -np.exp(-r / 2) * sq,
        -np.exp(r / 2) * sq,
    ]
    s = [0, mu[0], mu[0] - mu[1], mu[0] - mu[1] - mu[2]]
    total = np.zeros_like(k, dtype=float)
    for sj in s:
        c = (1 + abs(sj)) ** 2
        total += np.exp(a * c * (k + 0.5) + b)
    return total


def bound_bs(k, n_qubits: int, theta: float = np.pi / 2):
    """Per-Fock beam-splitter BS(theta) error upper bound on input |k,0>."""
    a, b = _qft_ab(n_qubits)
    k = np.asarray(k)
    tau2 = np.tan(theta / 4) ** 2
    sig2 = np.sin(theta / 2) ** 2
    c12 = 1 + tau2
    c34 = 1 + tau2 + sig2
    term1 = np.exp(a * c12 * (k + 1) + b)
    term2 = np.exp(a * c34 * (k + 1) + b)
    return 2 * term1 + 2 * term2


def plot_err_vs_fock(df, x_col, y_col, group_col, ylabel, base_name, fig_dir,
                     bound_fn=None, legend_loc='lower right', n_qubits_col='N'):
    """Semilogy plot of per-Fock error vs Fock index with theoretical bound overlay.

    bound_fn: callable(k_array, n_qubits) -> bound_array, or None for no bound.
    Same color used for data (solid) and bound (dashed) per group.
    Returns empty list (no fit params).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in prop_cycle]

    for idx, (N, group) in enumerate(df.groupby(group_col)):
        color = colors[idx % len(colors)]
        ks = group[x_col].values
        Es = group[y_col].values
        n_q = int(round(np.log2(float(N))))
        ax.semilogy(ks, Es, marker='o', linewidth=2, markersize=6,
                    color=color, label=f'$n={n_q}$', alpha=0.85)
        if bound_fn is not None:
            k_fine = np.arange(ks[0], ks[-1] + 1)
            bnd = bound_fn(k_fine, n_q)
            ax.semilogy(k_fine, bnd, '--', linewidth=2.5, color=color, alpha=0.85,
                        label=f'$n={n_q}$ bound')

    ax.set_xlabel(r'Fock index $k$')
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, fontsize=18, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, base_name, fig_dir)
    plt.close(fig)
    return []


def plot_err_vs_gamma(df, x_col, y_col, group_col, ylabel, base_name, fig_dir, legend_loc='lower right'):
    """Semilogy plot of error vs Gamma with linear fits. Returns fit_params list."""
    fit_params = []
    fig, ax = plt.subplots(figsize=(12, 8))

    for N, group in df.groupby(group_col):
        Gs = group[x_col].values
        Es = group[y_col].values
        ax.semilogy(Gs, Es, marker='o', linewidth=2, markersize=6,
                    label=f'$N={int(N)}$', alpha=0.8)
        ab = fit_ab(Gs, Es)
        if ab is not None:
            Gf = np.linspace(Gs.min(), Gs.max(), 100)
            ax.semilogy(Gf, np.exp(linear_model(Gf + 0.5, *ab)), '--', linewidth=1.5, alpha=0.5)
            fit_params.append((N, ab[0], ab[1]))

    ax.set_xlabel(r'$\Gamma$')
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, base_name, fig_dir)
    plt.close(fig)
    return fit_params


def plot_param_sweep(sweep_rows: list, param_label: str, base_name: str, fig_dir: str,
                     ylabel: str = r'$\varepsilon(\Gamma)$', N_fixed: int | None = None):
    """Plot error vs Gamma for multiple gate param values (fixed N).

    sweep_rows: list of dicts with keys 'param', 'Gamma', 'err'
    Returns list of dicts {'param', 'N', 'gamma_max', 'eps_final'} for the report table.
    """
    params = sorted(set(r['param'] for r in sweep_rows))
    # Use a perceptually-uniform colormap spread across param values
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i / max(len(params) - 1, 1)) for i in range(len(params))]

    fig, ax = plt.subplots(figsize=(12, 8))
    summary = []
    for color, pval in zip(colors, params):
        pts = [(r['Gamma'], r['err']) for r in sweep_rows if r['param'] == pval]
        pts.sort()
        Gs = np.array([p[0] for p in pts])
        Es = np.array([p[1] for p in pts])
        ax.semilogy(Gs, Es, marker='o', linewidth=2, markersize=5,
                    color=color, label=f'${param_label}{pval:.3g}$', alpha=0.85)
        # linear fit overlay
        ab = fit_ab(Gs, Es)
        if ab is not None:
            Gf = np.linspace(Gs.min(), Gs.max(), 100)
            ax.semilogy(Gf, np.exp(linear_model(Gf + 0.5, *ab)), '--', linewidth=1.2,
                        color=color, alpha=0.5)
        N_val = N_fixed if N_fixed is not None else 0
        summary.append({'param': pval, 'N': N_val,
                        'gamma_max': int(Gs[-1]), 'eps_final': float(Es[-1])})

    ax.set_xlabel(r'$\Gamma$')
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right', fontsize=16, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    if N_fixed is not None:
        ax.set_title(f'$N={N_fixed}$', fontsize=20)
    fig.tight_layout()
    _save_fig(fig, base_name, fig_dir)
    plt.close(fig)
    return summary


def plot_coeff_scaling(fit_params, base_name, fig_dir):
    """Plot a vs N^{-1/2} and b vs N^{1/2}. Returns scaling dict."""
    scaling = fit_coeff_scaling(fit_params)
    N_vals = np.array([p[0] for p in fit_params], dtype=float)
    a_vals = np.array([p[1] for p in fit_params])
    b_vals = np.array([p[2] for p in fit_params])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    inv_sqrt_N = N_vals ** (-0.5)
    sqrt_N = N_vals ** 0.5

    ax1.plot(inv_sqrt_N, a_vals, 'o', markersize=10, label='Data')
    xf = np.linspace(inv_sqrt_N.min(), inv_sqrt_N.max(), 100)
    ax1.plot(xf, linear_model(xf, scaling['c_a'], scaling['d_a']), '--', linewidth=2, label='Fit')
    ax1.set_xlabel(r'$N^{-1/2}$')
    ax1.set_ylabel(r'$a$')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(sqrt_N, b_vals, 'o', markersize=10, label='Data')
    xf2 = np.linspace(sqrt_N.min(), sqrt_N.max(), 100)
    ax2.plot(xf2, linear_model(xf2, scaling['c_b'], scaling['d_b']), '--', linewidth=2, label='Fit')
    ax2.set_xlabel(r'$N^{1/2}$')
    ax2.set_ylabel(r'$b$')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, base_name, fig_dir)
    plt.close(fig)
    return scaling


def plot_param_fit_scaling(fit_params, param_label, base_name, fig_dir):
    """Plot fit coefficients a and b vs gate parameter.
    
    fit_params: list of (param_value, a, b) tuples from linear fits
    param_label: LaTeX label for parameter (e.g., r'\\alpha', r'\\theta', r'r')
    """
    if len(fit_params) < 2:
        return {}
    
    param_vals = np.array([p[0] for p in fit_params], dtype=float)
    a_vals = np.array([p[1] for p in fit_params])
    b_vals = np.array([p[2] for p in fit_params])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(param_vals, a_vals, 'o-', markersize=10, linewidth=2)
    ax1.set_xlabel(f'${param_label}$')
    ax1.set_ylabel(r'$a$ (slope)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(param_vals, b_vals, 'o-', markersize=10, linewidth=2)
    ax2.set_xlabel(f'${param_label}$')
    ax2.set_ylabel(r'$b$ (intercept)')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, base_name, fig_dir)
    plt.close(fig)
    return {'params': param_vals.tolist(), 'a': a_vals.tolist(), 'b': b_vals.tolist()}


def plot_eps_vs_param(sweep_rows: list, param_label: str, base_name: str, fig_dir: str,
                      gamma_values: list, ylabel: str = r'$\varepsilon$', N_fixed: int | None = None):
    """Plot error vs gate parameter for multiple fixed Gamma values (at fixed N).

    sweep_rows: list of dicts with keys 'param', 'Gamma', 'err'
    gamma_values: list of Gamma values to plot (will skip if not present in data)
    param_label: LaTeX label for x-axis parameter (e.g., r'\\alpha', r'\\theta')
    Returns summary dict.
    """
    # Use a perceptually-uniform colormap spread across gamma values
    cmap = plt.get_cmap('viridis')
    
    # Filter to only gamma values that exist in the data
    available_gammas = sorted(set(r['Gamma'] for r in sweep_rows))
    gamma_values_to_plot = [g for g in gamma_values if g in available_gammas]
    
    if not gamma_values_to_plot:
        # If none of the requested gammas exist, use a few evenly-spaced ones
        if len(available_gammas) >= 5:
            indices = np.linspace(0, len(available_gammas)-1, 5, dtype=int)
            gamma_values_to_plot = [available_gammas[i] for i in indices]
        else:
            gamma_values_to_plot = available_gammas
    
    colors = [cmap(i / max(len(gamma_values_to_plot) - 1, 1)) for i in range(len(gamma_values_to_plot))]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    for color, gamma in zip(colors, gamma_values_to_plot):
        # Find all points with this Gamma value
        pts = [(r['param'], r['err']) for r in sweep_rows 
               if r['Gamma'] == gamma]
        if not pts:
            continue
        pts.sort()
        params = np.array([p[0] for p in pts])
        errs = np.array([p[1] for p in pts])
        ax.semilogy(params, errs, marker='o', linewidth=2, markersize=6,
                    color=color, label=f'$\\Gamma={gamma}$', alpha=0.85)

    ax.set_xlabel(f'${param_label}$')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=16, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    if N_fixed is not None:
        ax.set_title(f'$N={N_fixed}$', fontsize=20)
    fig.tight_layout()
    _save_fig(fig, base_name, fig_dir)
    plt.close(fig)
    return {'gamma_values': gamma_values_to_plot, 'N': N_fixed}
