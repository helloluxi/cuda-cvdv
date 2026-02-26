"""QFT error analysis.

Cumulative L2 error of ftQ2P on Fock states |n⟩ vs analytic target (-i)^n * |n⟩.
Produces: figures/qft_err.pdf, figures/qft_coeff_scaling.pdf
Returns: {'fit_params': [(N, a, b), ...], 'scaling': {...}}
"""

import torch
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_gamma, plot_coeff_scaling

from src import CVDV, SeparableState  # type: ignore

QUBIT_COUNTS = [5, 6, 7, 8, 9, 10]   # N = 32 .. 1024
PRECISION_CUTOFF = 1e-9
STOP_ERR_SQ = 1e-1                    # stop when err_cum^2 >= this


def _sweep(q: int) -> list:
    N = 1 << q
    states, x, dx = fock_recurrence(N)

    cumulative_errors = []
    err_cum_sq = 0.0

    for n, psi_pos in enumerate(states):
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        sim.ftQ2P(0)
        psi_qft = torch.tensor(sim.getState(), dtype=torch.cdouble)

        # Use torch for analytic state and error computation
        phase_factor = torch.tensor((-1j) ** (n % 4), dtype=torch.cdouble)
        psi_p = psi_pos.to(torch.cdouble) * phase_factor
        err_cum_sq += float(torch.sum(torch.abs(psi_qft - psi_p) ** 2))
        cumulative_errors.append(torch.sqrt(torch.tensor(err_cum_sq)).item())

        if err_cum_sq >= STOP_ERR_SQ:
            break

    return cumulative_errors


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(QUBIT_COUNTS), desc='qft_err') as pbar:
        for q in QUBIT_COUNTS:
            N = 1 << q
            errors = _sweep(q)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'Gamma': gamma, 'err': eps})
            pbar.set_postfix({'N': N, 'n_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)

    fit_params = plot_err_vs_gamma(
        df, 'Gamma', 'err', 'N',
        ylabel=r'Bounds of $\epsilon_F$',
        base_name='qft_err',
        fig_dir=fig_dir,
        legend_loc='lower right',
    )

    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'qft_coeff_scaling', fig_dir)

    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
