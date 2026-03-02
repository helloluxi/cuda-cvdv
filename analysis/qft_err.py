"""QFT per-Fock error analysis.

L2 error of ftQ2P on individual Fock states |n> vs analytic target (-i)^n * |n>.
No accumulation: each Fock state is tested independently.
Produces: figures/qft_err_per_fock.png, figures/qft_per_fock_coeff_scaling.png
Returns: {'fit_params': [(N, a, b), ...], 'scaling': {...}}
"""

import torch
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_fock, bound_qft

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

    plot_err_vs_fock(
        df, 'n', 'err', 'N',
        ylabel=r'$\varepsilon_{\mathrm{QFT}}(|k\rangle)$',
        base_name='qft_err_per_fock',
        fig_dir=fig_dir,
        bound_fn=bound_qft,
        legend_loc='lower right',
    )

    return {'fit_params': [], 'scaling': {}, 'df': df}
