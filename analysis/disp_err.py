"""Displacement D(α=2) error analysis.

Discrete: ftQ2P → exp(-i√2·α·p) phase → ftP2Q applied to each Fock state.
Analytic: shifted Fock state ψ_n(x - α√2).
Produces: figures/disp_err.pdf
Returns: {'fit_params': [...], 'scaling': {...}}
"""

import torch
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_gamma, plot_coeff_scaling

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
ALPHA = 2.0
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1


def _analytic_states(N: int, alpha: float):
    """Shifted Fock states ψ_n(x - α√2) on the grid using torch."""
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
    q = int(round(torch.log2(torch.tensor(float(N))).item()))

    pos_states, _, _ = fock_recurrence(N)
    ana_states = _analytic_states(N, alpha)

    cumulative_errors = []
    err_cum = 0.0

    for n in range(N):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        # D(alpha): real alpha → ftQ2P, phase exp(-i√2·alpha·p), ftP2Q
        sim.ftQ2P(0)
        p = sim.getXGrid(0)           # same grid in momentum space
        phase = torch.exp(-1j * torch.sqrt(torch.tensor(2.0)) * alpha * torch.tensor(p, dtype=torch.cdouble))
        sim.state = sim.state * phase
        sim.ftP2Q(0)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)

        psi_analytic = ana_states[n].to(torch.cdouble)
        err_cum += float(torch.linalg.norm(psi_discrete - psi_analytic))
        cumulative_errors.append(err_cum)

        if err_cum >= STOP_ERR:
            break

    return cumulative_errors


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='disp_err') as pbar:
        for N in N_TOTALS:
            errors = _sweep(N, ALPHA)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'Gamma': gamma, 'err': eps})
            pbar.set_postfix({'N': N, 'n_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = plot_err_vs_gamma(
        df, 'Gamma', 'err', 'N',
        ylabel=r'Bounds of $\epsilon_{D(2.0)}$',
        base_name='disp_err',
        fig_dir=fig_dir,
    )
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'disp_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
