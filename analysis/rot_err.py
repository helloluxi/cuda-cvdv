"""Rotation R(theta=pi/4) error analysis.

R(theta) = exp(-i*tan(theta/2)/2*q^2) * exp(-i*sin(theta)/2*p^2) * exp(-i*tan(theta/2)/2*q^2)
Analytic: psi_n(x) * exp(-i*(n+0.5)*theta)
Produces: figures/rot_err.pdf
Returns: {'fit_params': [(N, a, b), ...], 'scaling': {...}}
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, fit_ab_upper, plot_coeff_scaling, _save_fig

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
THETA = float(torch.pi / 4)
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1


def _sweep(N: int, theta: float) -> list:
    """Per-Fock errors (not cumulative). Returns list of individual errors per Fock index."""
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    pos_states, _, _ = fock_recurrence(N)

    per_fock_errors = []

    for n, psi_pos in enumerate(pos_states):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        sim.r(0, theta)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)

        phase = torch.exp(torch.tensor(-1j * (n + 0.5) * theta, dtype=torch.cdouble))
        psi_analytic = psi_pos.to(torch.cdouble) * phase
        err = float(torch.linalg.norm(psi_discrete - psi_analytic))
        per_fock_errors.append(err)

        if err >= STOP_ERR:
            break

    return per_fock_errors


def _plot_rot_err_with_fit(df, fig_dir):
    """Plot per-Fock rotation error with per-N exponential upper bound overlay."""
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
    ax.set_ylabel(r'$\varepsilon_{R(\theta)}(|\Gamma\rangle)$')
    ax.legend(loc='lower center', bbox_to_anchor=(0.7, 0.0), ncol=1)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, 'rot_err', fig_dir)
    plt.close(fig)
    return fit_params


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='rot_err') as pbar:
        for N in N_TOTALS:
            errors = _sweep(N, THETA)
            for k, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'k': k, 'err': eps})
            pbar.set_postfix({'N': N, 'k_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = _plot_rot_err_with_fit(df, fig_dir)
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'rot_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
