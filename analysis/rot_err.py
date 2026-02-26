"""Rotation R(θ=π/4) error analysis.

R(θ) = exp(-i·tan(θ/2)/2·q²) · exp(-i·sin(θ)/2·p²) · exp(-i·tan(θ/2)/2·q²)
Analytic: ψ_n(x) * exp(-i·(n+0.5)·θ)
Produces: figures/rot_err.pdf
Returns: {'fit_params': [...], 'scaling': {...}}
"""

import torch
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_gamma, plot_coeff_scaling

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
THETA = float(torch.pi / 4)
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1


def _apply_rotation(sim: CVDV, q: int, theta: float):
    theta_t = torch.tensor(theta)
    x = torch.tensor(sim.getXGrid(0), dtype=torch.cdouble)
    # First q² shear
    sim.state = sim.state * torch.exp(-1j * torch.tan(theta_t / 2) / 2 * x ** 2)
    sim.ftQ2P(0)
    p = torch.tensor(sim.getXGrid(0), dtype=torch.cdouble)
    sim.state = sim.state * torch.exp(-1j * torch.sin(theta_t) / 2 * p ** 2)
    sim.ftP2Q(0)
    # Second q² shear
    x2 = torch.tensor(sim.getXGrid(0), dtype=torch.cdouble)
    sim.state = sim.state * torch.exp(-1j * torch.tan(theta_t / 2) / 2 * x2 ** 2)


def _sweep(N: int, theta: float) -> list:
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    pos_states, _, _ = fock_recurrence(N)

    cumulative_errors = []
    err_cum = 0.0

    for n, psi_pos in enumerate(pos_states):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        _apply_rotation(sim, q, theta)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)

        phase = torch.exp(torch.tensor(-1j * (n + 0.5) * theta, dtype=torch.cdouble))
        psi_analytic = psi_pos.to(torch.cdouble) * phase
        err_cum += float(torch.linalg.norm(psi_discrete - psi_analytic))
        cumulative_errors.append(err_cum)

        if err_cum >= STOP_ERR:
            break

    return cumulative_errors


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='rot_err') as pbar:
        for N in N_TOTALS:
            errors = _sweep(N, THETA)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'Gamma': gamma, 'err': eps})
            pbar.set_postfix({'N': N, 'n_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = plot_err_vs_gamma(
        df, 'Gamma', 'err', 'N',
        ylabel=r'Bounds of $\epsilon_{R(\pi/4)}$',
        base_name='rot_err',
        fig_dir=fig_dir,
    )
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'rot_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
