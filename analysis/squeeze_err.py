"""Squeezing S(r=1) error analysis.

S(r) decomposed as 4-phase QFT circuit (see dvsim-code/squeeze_err.ipynb).
Analytic: squeezed Fock state — recurrence on x·exp(r).
Produces: figures/squeeze_err.pdf
Returns: {'fit_params': [...], 'scaling': {...}}
"""

import torch
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_gamma, plot_coeff_scaling

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
R = 1.0
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1
MIN_N = 4   # must reach at least this before stopping (matches notebook)


def _analytic_squeezed_states(N: int, r: float):
    """S(r)|n⟩ in position basis: recurrence on squeezed_x = x·exp(r), ×exp(r/2)."""
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


def _apply_squeezing(sim: CVDV, r: float):
    e_r = torch.exp(torch.tensor(r))
    e_mr = torch.exp(torch.tensor(-r))
    t = torch.exp(torch.tensor(-r / 2)) * torch.sqrt(torch.abs(1 - e_mr))

    x = torch.tensor(sim.getXGrid(0), dtype=torch.cdouble)
    sim.state = sim.state * torch.exp(1j * t / 2 * x ** 2)
    sim.ftQ2P(0)
    p = torch.tensor(sim.getXGrid(0), dtype=torch.cdouble)
    sim.state = sim.state * torch.exp(1j * (1 - e_mr) / (2 * t) * p ** 2)
    sim.ftP2Q(0)
    x2 = torch.tensor(sim.getXGrid(0), dtype=torch.cdouble)
    sim.state = sim.state * torch.exp(-1j * t / 2 * e_r * x2 ** 2)
    sim.ftQ2P(0)
    p2 = torch.tensor(sim.getXGrid(0), dtype=torch.cdouble)
    sim.state = sim.state * torch.exp(1j * (e_mr - 1) / (2 * t * e_r) * p2 ** 2)
    sim.ftP2Q(0)


def _sweep(N: int, r: float) -> list:
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    pos_states, _, _ = fock_recurrence(N)
    ana_states = _analytic_squeezed_states(N, r)

    cumulative_errors = []
    err_cum = 0.0

    for n in range(N):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        _apply_squeezing(sim, r)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)

        psi_analytic = ana_states[n].to(torch.cdouble)
        err_cum += float(torch.linalg.norm(psi_discrete - psi_analytic))
        cumulative_errors.append(err_cum)

        if err_cum >= STOP_ERR and n >= MIN_N:
            break

    return cumulative_errors


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='squeeze_err') as pbar:
        for N in N_TOTALS:
            errors = _sweep(N, R)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'Gamma': gamma, 'err': eps})
            pbar.set_postfix({'N': N, 'n_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = plot_err_vs_gamma(
        df, 'Gamma', 'err', 'N',
        ylabel=r'Bounds of $\epsilon_{S(1.0)}$',
        base_name='squeeze_err',
        fig_dir=fig_dir,
    )
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'squeeze_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
