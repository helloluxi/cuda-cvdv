"""Displacement D(alpha=2) error analysis.

Discrete: ftQ2P -> exp(-i*sqrt(2)*alpha*p) phase -> ftP2Q applied to each Fock state.
Analytic: shifted Fock state psi_n(x - alpha*sqrt(2)).
Produces: figures/disp_err.pdf
Returns: {'fit_params': [(N, a, b), ...], 'scaling': {...}}
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from ._common import fit_ab_upper, plot_coeff_scaling, _save_fig

from src.torchCvdv import TorchCvdv
from src.separable import SeparableState

N_TOTALS = [64, 128, 256, 512]
ALPHA = 4.0   # max of sweep range — gives worst-case upper bound
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1


def _analytic_states(N: int, alpha: float):
    """Shifted Fock states psi_n(x - alpha*sqrt(2)) on the grid using torch."""
    dx = torch.sqrt(torch.tensor(2.0 * torch.pi / N))
    idx = torch.arange(N, dtype=torch.float64)
    x_shift = (idx - (N - 1) * 0.5) * dx - alpha * torch.sqrt(torch.tensor(2.0))

    states = []
    psi_prev = torch.zeros(N, dtype=torch.float64)
    psi_curr = torch.exp(-0.5 * x_shift ** 2) * (torch.pi ** -0.25) * torch.sqrt(dx)
    states.append(psi_curr.clone())
    for k in range(1, N):
        k_float = float(k)
        psi_next = (torch.sqrt(torch.tensor(2.0 / k_float)) * x_shift * psi_curr
                    - torch.sqrt(torch.tensor((k_float - 1.0) / k_float)) * psi_prev)
        psi_prev, psi_curr = psi_curr, psi_next
        states.append(psi_curr.clone())
    return states


def _sweep(N: int, alpha: float) -> list:
    """Per-Fock errors (not cumulative). Returns list of individual errors per Fock index."""
    q = int(round(torch.log2(torch.tensor(float(N))).item()))

    ana_states = _analytic_states(N, alpha)

    per_fock_errors = []

    for n in range(N):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = TorchCvdv([q], device='cuda')
        sim.initStateVector(sep)
        sim.d(0, float(alpha))
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)

        psi_analytic = ana_states[n].to(torch.cdouble)
        err = float(torch.linalg.norm(psi_discrete - psi_analytic))
        per_fock_errors.append(err)

        if err >= STOP_ERR:
            break

    return per_fock_errors


def _plot_disp_err_with_fit(df, fig_dir):
    """Plot per-Fock displacement error with per-N exponential upper bound overlay."""
    fig, ax = plt.subplots(figsize=(12, 8))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in prop_cycle]

    fit_params = []
    for idx, (N, group) in enumerate(df.groupby('N')):
        color = colors[idx % len(colors)]
        ks = group['k'].values
        Es = group['err'].values
        n_q = int(round(np.log2(float(N))))

        ax.semilogy(ks, Es, marker='o', linewidth=2, markersize=6,
                    color=color, label=f'$n={n_q}$', alpha=0.85)

        ab = fit_ab_upper(ks, Es)
        if ab is not None:
            a, b = ab
            fit_params.append((N, a, b))

    ax.set_xlabel(r'Fock index $\Gamma$')
    ax.set_ylabel(r'$\varepsilon_{D(\alpha)}(|\Gamma\rangle)$')
    ax.legend(loc='lower right', ncol=1)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, 'disp_err', fig_dir)
    plt.close(fig)
    return fit_params


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='disp_err') as pbar:
        for N in N_TOTALS:
            errors = _sweep(N, ALPHA)
            for k, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'k': k, 'err': eps})
            pbar.set_postfix({'N': N, 'k_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = _plot_disp_err_with_fit(df, fig_dir)
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'disp_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
