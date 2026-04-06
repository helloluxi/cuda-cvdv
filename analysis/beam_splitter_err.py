"""Beam splitter BS(theta=pi/2) error analysis.

2-mode state |n>|0> -> discrete BS via q1q2 + 2D QFT -> compare to analytic binomial expansion.
Discrete: exp(-i*tan(theta/4)*q1*q2) * ftQ2P(both) * exp(-i*sin(theta/2)*p1*p2) * ftP2Q(both) * exp(-i*tan(theta/4)*q1*q2)
Analytic: BS(theta)|n>|0> = sum_k sqrt(C(n,k)) cos^k(theta/2)*(-i*sin(theta/2))^{n-k} |k>|n-k>
Produces: figures/beam_splitter_err.pdf
Returns: {'fit_params': [(N, a, b), ...], 'scaling': {...}}
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from ._common import fock_recurrence, fit_ab_upper, plot_coeff_scaling, _save_fig

from src.torchCvdv import TorchCvdv
from src.separable import SeparableState

N_TOTALS = [32, 64, 128, 256]
THETA = float(torch.pi / 2)
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1
MIN_N = 4


def _sweep(N: int, theta: float) -> list:
    """Per-Fock errors (not cumulative). Returns list of individual errors per Fock index."""
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    fock_states, x, dx = fock_recurrence(N)

    per_fock_errors = []
    cos_h = torch.cos(torch.tensor(theta / 2)).item()
    sin_h = torch.sin(torch.tensor(theta / 2)).item()

    for n in range(N):
        # --- discrete ---
        sep = SeparableState([q, q], device='cpu')
        sep.setFock(0, n)
        sep.setFock(1, 0)
        sim = TorchCvdv([q, q], device='cuda')
        sim.initStateVector(sep)
        sim.bs(0, 1, theta)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble).reshape(N, N)   # [mode0, mode1]

        # --- analytic binomial ---
        psi_analytic = torch.zeros((N, N), dtype=torch.cdouble)
        for k in range(n + 1):
            binom = int(comb(n, k, exact=True))
            amp = torch.sqrt(torch.tensor(float(binom))) * (cos_h ** k) * ((-1j * sin_h) ** (n - k))
            psi_k = fock_states[k].to(torch.cdouble)
            psi_nk = fock_states[n - k].to(torch.cdouble)
            psi_analytic += amp * torch.outer(psi_k, psi_nk)

        err = float(torch.linalg.norm(psi_discrete - psi_analytic))
        per_fock_errors.append(err)

        if err >= STOP_ERR and n >= MIN_N:
            break

    return per_fock_errors


def _plot_bs_err_with_fit(df, fig_dir):
    """Plot per-Fock beam splitter error with per-N exponential upper bound overlay."""
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
    ax.set_ylabel(r'$\varepsilon_{BS(\theta)}(|\Gamma,0\rangle)$')
    ax.legend(loc='lower right', ncol=1)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    _save_fig(fig, 'beam_splitter_err', fig_dir)
    plt.close(fig)
    return fit_params


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='beam_splitter_err') as pbar:
        for N in N_TOTALS:
            errors = _sweep(N, THETA)
            for k, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'k': k, 'err': eps})
            pbar.set_postfix({'N': N, 'k_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = _plot_bs_err_with_fit(df, fig_dir)
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'bs_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
