"""Beam splitter BS(theta=pi/2) error analysis.

2-mode state |n>|0> -> discrete BS via q1q2 + 2D QFT -> compare to analytic binomial expansion.
Discrete: exp(-i*tan(theta/4)*q1*q2) * ftQ2P(both) * exp(-i*sin(theta/2)*p1*p2) * ftP2Q(both) * exp(-i*tan(theta/4)*q1*q2)
Analytic: BS(theta)|n>|0> = sum_k sqrt(C(n,k)) cos^k(theta/2)*(-i*sin(theta/2))^{n-k} |k>|n-k>
Produces: figures/beam_splitter_err.pdf
Returns: {'fit_params': [], 'scaling': {}}
"""

import torch
import numpy as np
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_fock, bound_bs, plot_eps_vs_param

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [32, 64, 128, 256]
THETA = float(torch.pi / 2)
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1
MIN_N = 4

# Param sweep: fix N, vary theta
SWEEP_N = 64
SWEEP_NUM_GATE_PARAM = 101
THETA_BS_VALUES = (torch.arange(1, 1+SWEEP_NUM_GATE_PARAM) / SWEEP_NUM_GATE_PARAM * torch.pi / 2).tolist()
SWEEP_LIST_GAMMA = [20, 30, 40]


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
        sim = CVDV([q, q], backend='torch-cuda')
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


def _run_param_sweep(fig_dir: str) -> list:
    rows = []
    with tqdm(total=len(THETA_BS_VALUES), desc='bs param sweep') as pbar:
        for theta in THETA_BS_VALUES:
            errors = _sweep(SWEEP_N, theta)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'param': theta, 'Gamma': gamma, 'err': eps})
            pbar.update(1)
    
    # Plot eps vs theta for fixed Gamma values
    plot_eps_vs_param(
        rows, param_label=r'\theta', base_name='bs_eps_vs_theta',
        fig_dir=fig_dir, gamma_values=SWEEP_LIST_GAMMA,
        ylabel=r'$\varepsilon_{BS(\theta)}$', N_fixed=SWEEP_N
    )
    
    return []


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
    plot_err_vs_fock(
        df, 'k', 'err', 'N',
        ylabel=r'$\varepsilon_{BS(\pi/2)}(|k,0\rangle)$',
        base_name='beam_splitter_err',
        fig_dir=fig_dir,
        bound_fn=lambda k, n_q: bound_bs(k, n_q, theta=THETA),
    )
    _run_param_sweep(fig_dir)
    return {'fit_params': [], 'scaling': {}, 'df': df}
