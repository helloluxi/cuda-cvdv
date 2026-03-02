"""Displacement D(alpha=2) error analysis.

Discrete: ftQ2P -> exp(-i*sqrt(2)*alpha*p) phase -> ftP2Q applied to each Fock state.
Analytic: shifted Fock state psi_n(x - alpha*sqrt(2)).
Produces: figures/disp_err.pdf
Returns: {'fit_params': [], 'scaling': {}}
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ._common import plot_err_vs_fock, bound_disp, plot_eps_vs_param

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
ALPHA = 2.0
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1

# Param sweep: fix N, vary alpha
SWEEP_N = 256
SWEEP_NUM_GATE_PARAM = 101
ALPHA_VALUES = (torch.arange(1, 1+SWEEP_NUM_GATE_PARAM) / SWEEP_NUM_GATE_PARAM * 4.0).tolist()
SWEEP_LIST_GAMMA = [80, 100, 120, 140, 160]


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
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        sim.d(0, float(alpha))
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)

        psi_analytic = ana_states[n].to(torch.cdouble)
        err = float(torch.linalg.norm(psi_discrete - psi_analytic))
        per_fock_errors.append(err)

        if err >= STOP_ERR:
            break

    return per_fock_errors


def _run_param_sweep(fig_dir: str) -> list:
    rows = []
    with tqdm(total=len(ALPHA_VALUES), desc='disp param sweep') as pbar:
        for alpha in ALPHA_VALUES:
            errors = _sweep(SWEEP_N, alpha)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'param': alpha, 'Gamma': gamma, 'err': eps})
            pbar.update(1)

    # Plot eps vs alpha for fixed Gamma values
    plot_eps_vs_param(
        rows, param_label=r'\alpha', base_name='disp_eps_vs_alpha',
        fig_dir=fig_dir, gamma_values=SWEEP_LIST_GAMMA,
        ylabel=r'$\varepsilon_{D(\alpha)}$', N_fixed=SWEEP_N
    )

    return []


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
    plot_err_vs_fock(
        df, 'k', 'err', 'N',
        ylabel=r'$\varepsilon_{D(2)}(|k\rangle)$',
        base_name='disp_err',
        fig_dir=fig_dir,
        bound_fn=lambda k, n_q: bound_disp(k, n_q, Re_alpha=ALPHA),
    )
    _run_param_sweep(fig_dir)
    return {'fit_params': [], 'scaling': {}, 'df': df}
