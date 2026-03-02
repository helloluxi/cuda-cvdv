"""Squeezing S(r=1) error analysis.

S(r) decomposed as 4-phase QFT circuit (see dvsim-code/squeeze_err.ipynb).
Analytic: squeezed Fock state - recurrence on x*exp(r).
Produces: figures/squeeze_err.pdf
Returns: {'fit_params': [], 'scaling': {}}
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ._common import plot_err_vs_fock, bound_sq, plot_eps_vs_param

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
R = 1.0
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1
MIN_N = 4   # must reach at least this before stopping (matches notebook)

# Param sweep: fix N, vary r
SWEEP_N = 256
SWEEP_NUM_GATE_PARAM = 101
R_VALUES = (torch.arange(1, 1+SWEEP_NUM_GATE_PARAM) / SWEEP_NUM_GATE_PARAM * 2.0).tolist()
SWEEP_LIST_GAMMA = [80, 100, 120, 140, 160]


def _analytic_squeezed_states(N: int, r: float):
    """S(r)|n> in position basis: recurrence on squeezed_x = x*exp(r), scaled by exp(r/2)."""
    dx = torch.sqrt(torch.tensor(2.0 * torch.pi / N))
    idx = torch.arange(N, dtype=torch.float64)
    x_sq = (idx - (N - 1) * 0.5) * dx * torch.exp(torch.tensor(r))   # squeezed coordinate
    norm_factor = torch.exp(torch.tensor(r / 2))

    states = []
    psi_prev = torch.zeros(N, dtype=torch.float64)
    psi_curr = norm_factor * torch.exp(-0.5 * x_sq ** 2) * (torch.pi ** -0.25) * torch.sqrt(dx)
    states.append(psi_curr.clone())
    for k in range(1, N):
        k_float = float(k)
        psi_next = (torch.sqrt(torch.tensor(2.0 / k_float)) * x_sq * psi_curr
                    - torch.sqrt(torch.tensor((k_float - 1.0) / k_float)) * psi_prev)
        psi_prev, psi_curr = psi_curr, psi_next
        states.append(psi_curr.clone())
    return states


def _sweep(N: int, r: float) -> list:
    """Per-Fock errors (not cumulative). Returns list of individual errors per Fock index."""
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    ana_states = _analytic_squeezed_states(N, r)

    per_fock_errors = []

    for n in range(N):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        sim.s(0, r)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)

        psi_analytic = ana_states[n].to(torch.cdouble)
        err = float(torch.linalg.norm(psi_discrete - psi_analytic))
        per_fock_errors.append(err)

        if err >= STOP_ERR and n >= MIN_N:
            break

    return per_fock_errors


def _run_param_sweep(fig_dir: str) -> list:
    rows = []
    with tqdm(total=len(R_VALUES), desc='squeeze param sweep') as pbar:
        for r in R_VALUES:
            errors = _sweep(SWEEP_N, r)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'param': r, 'Gamma': gamma, 'err': eps})
            pbar.update(1)
    
    # Plot eps vs r for fixed Gamma values
    plot_eps_vs_param(
        rows, param_label=r'r', base_name='squeeze_eps_vs_r',
        fig_dir=fig_dir, gamma_values=SWEEP_LIST_GAMMA,
        ylabel=r'$\varepsilon_{S(r)}$', N_fixed=SWEEP_N
    )
    
    return []


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='squeeze_err') as pbar:
        for N in N_TOTALS:
            errors = _sweep(N, R)
            for k, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'k': k, 'err': eps})
            pbar.set_postfix({'N': N, 'k_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    plot_err_vs_fock(
        df, 'k', 'err', 'N',
        ylabel=r'$\varepsilon_{S(1)}(|k\rangle)$',
        base_name='squeeze_err',
        fig_dir=fig_dir,
        bound_fn=lambda k, n_q: bound_sq(k, n_q, r=R),
    )
    _run_param_sweep(fig_dir)
    return {'fit_params': [], 'scaling': {}, 'df': df}
