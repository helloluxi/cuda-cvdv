"""Rotation R(theta=pi/4) error analysis.

R(theta) = exp(-i*tan(theta/2)/2*q^2) * exp(-i*sin(theta)/2*p^2) * exp(-i*tan(theta/2)/2*q^2)
Analytic: psi_n(x) * exp(-i*(n+0.5)*theta)
Produces: figures/rot_err.pdf
Returns: {'fit_params': [], 'scaling': {}}
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_fock, bound_rot, plot_eps_vs_param

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
THETA = float(torch.pi / 4)
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1

# Param sweep: fix N, vary theta
SWEEP_N = 256
SWEEP_NUM_GATE_PARAM = 101
THETA_VALUES = (torch.arange(1, 1+SWEEP_NUM_GATE_PARAM) / SWEEP_NUM_GATE_PARAM * torch.pi / 4).tolist()
SWEEP_LIST_GAMMA = [140, 150, 160, 170, 180]


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


def _run_param_sweep(fig_dir: str) -> list:
    rows = []
    with tqdm(total=len(THETA_VALUES), desc='rot param sweep') as pbar:
        for theta in THETA_VALUES:
            errors = _sweep(SWEEP_N, theta)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'param': theta, 'Gamma': gamma, 'err': eps})
            pbar.update(1)
    
    # Plot eps vs theta for fixed Gamma values
    plot_eps_vs_param(
        rows, param_label=r'\theta', base_name='rot_eps_vs_theta',
        fig_dir=fig_dir, gamma_values=SWEEP_LIST_GAMMA,
        ylabel=r'$\varepsilon_{R(\theta)}$', N_fixed=SWEEP_N
    )
    
    return []


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
    plot_err_vs_fock(
        df, 'k', 'err', 'N',
        ylabel=r'$\varepsilon_{R(\pi/4)}(|k\rangle)$',
        base_name='rot_err',
        fig_dir=fig_dir,
        bound_fn=lambda k, n_q: bound_rot(k, n_q, theta=THETA),
    )
    _run_param_sweep(fig_dir)
    return {'fit_params': [], 'scaling': {}, 'df': df}
