"""QFT per-Fock error analysis.

L2 error of ftQ2P on individual Fock states |n> vs analytic target (-i)^n * |n>.
No accumulation: each Fock state is tested independently.
Fits log(eps) = a*(k+1/2) + b per qubit count from experimental data, then overlays
the fitted exponential as the theoretical bound curve.
Produces: figures/qft_err_per_fock.png
Returns: {'fit_params': [(N, a, b), ...]}
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, fit_ab_upper, plot_coeff_scaling, _save_fig

from src import CVDV, SeparableState  # type: ignore

QUBIT_COUNTS = [5, 6, 7, 8, 9, 10]   # N = 32 .. 1024
PRECISION_CUTOFF = 1e-9
STOP_ERR_SQ = 1e-1                    # stop when individual err^2 >= this


def _sweep(q: int) -> list:
    N = 1 << q
    states, x, dx = fock_recurrence(N)

    per_fock_errors = []

    for n, psi_pos in enumerate(states):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        sim.ftQ2P(0)
        psi_qft = torch.tensor(sim.getState(), dtype=torch.cdouble)

        phase_factor = torch.tensor((-1j) ** (n % 4), dtype=torch.cdouble)
        psi_p = psi_pos.to(torch.cdouble) * phase_factor
        err_sq = float(torch.sum(torch.abs(psi_qft - psi_p) ** 2))
        per_fock_errors.append(torch.sqrt(torch.tensor(err_sq)).item())

        if err_sq >= STOP_ERR_SQ:
            break

    return per_fock_errors


def _plot_qft_err_with_fit(df, fig_dir):
    """Plot per-Fock QFT error with per-N exponential fit overlay."""
    fig, ax = plt.subplots(figsize=(12, 8))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in prop_cycle]

    fit_params = []
    for idx, (N, group) in enumerate(df.groupby('N')):
        color = colors[idx % len(colors)]
        ks = group['n'].values
        Es = group['err'].values
        n_q = int(round(np.log2(float(N))))

        ax.semilogy(ks, Es, marker='o', linewidth=2, markersize=6,
                    color=color, label=f'$n={n_q}$', alpha=0.85)

        ab = fit_ab_upper(ks, Es)
        if ab is not None:
            a, b = ab
            fit_params.append((N, a, b))
            k_fit = np.linspace(ks[0], ks[-1], 200)
            ax.semilogy(k_fit, np.exp(a * k_fit + b), '--', color=color, linewidth=1.5)

    ax.set_xlabel(r'Fock index $k$')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{QFT}}(|k\rangle)$')
    ax.legend(loc='lower center', bbox_to_anchor=(0.7, 0.0), ncol=1)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, 'qft_err_per_fock', fig_dir)
    plt.close(fig)
    return fit_params


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(QUBIT_COUNTS), desc='qft_err_per_fock') as pbar:
        for q in QUBIT_COUNTS:
            N = 1 << q
            errors = _sweep(q)
            for n, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'n': n, 'err': eps})
            pbar.set_postfix({'N': N, 'n_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = _plot_qft_err_with_fit(df, fig_dir)
    scaling = plot_coeff_scaling(fit_params, 'qft_per_fock_coeff_scaling', fig_dir)
    scaling['x_var'] = 'k'

    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
