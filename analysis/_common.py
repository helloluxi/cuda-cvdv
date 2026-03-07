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


def fit_ab_upper(k_vals, err_vals):
    """OLS fit log(err) = a*k + b, then shift b up by max residual.

    Returns (a, b) such that exp(a*k + b) >= err_vals[i] for all i.
    Returns None if fit fails.
    """
    if len(k_vals) < 2:
        return None
    try:
        x = np.asarray(k_vals, dtype=float)
        y = np.log(np.asarray(err_vals, dtype=float))
        popt, _ = curve_fit(linear_model, x, y)
        a, b = popt
        residuals = y - linear_model(x, a, b)
        b += residuals.max()   # shift up so curve lies above all data
        return (a, b)
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
    ss_a = np.sum((a_vals - a_vals.mean()) ** 2)
    r2_a = float(1 - np.sum(res_a ** 2) / ss_a) if ss_a > 0 else float('nan')

    popt_b, _ = curve_fit(linear_model, sqrt_N, b_vals)
    c_b, d_b = popt_b
    res_b = b_vals - linear_model(sqrt_N, c_b, d_b)
    ss_b = np.sum((b_vals - b_vals.mean()) ** 2)
    r2_b = float(1 - np.sum(res_b ** 2) / ss_b) if ss_b > 0 else float('nan')

    return dict(c_a=c_a, d_a=d_a, r2_a=r2_a, c_b=c_b, d_b=d_b, r2_b=r2_b)


def plot_err_vs_gamma(df, x_col, y_col, group_col, ylabel, base_name, fig_dir, legend_loc='lower right'):
    """Semilogy plot of error vs Gamma with upper-bound linear fits. Returns fit_params list."""
    fit_params = []
    fig, ax = plt.subplots(figsize=(12, 8))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in prop_cycle]

    for idx, (N, group) in enumerate(df.groupby(group_col)):
        color = colors[idx % len(colors)]
        Gs = group[x_col].values
        Es = group[y_col].values
        ax.semilogy(Gs, Es, marker='o', linewidth=2, markersize=6,
                    color=color, label=f'$N={int(N)}$', alpha=0.8)
        ab = fit_ab_upper(Gs, Es)
        if ab is not None:
            fit_params.append((N, ab[0], ab[1]))

    ax.set_xlabel(r'$\Gamma$')
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, base_name, fig_dir)
    plt.close(fig)
    return fit_params


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
