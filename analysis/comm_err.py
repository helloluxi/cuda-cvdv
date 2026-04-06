"""Commutator [q, p] = i error analysis.

Measures: ||x·iQFT(x·QFT(ψ)) - iQFT(x·QFT(x·ψ)) - i·ψ|| on each Fock state.
Same grid sizes as qft_err.
Produces: figures/comm_err.pdf, figures/comm_coeff_scaling.pdf
Returns: {'fit_params': [...], 'scaling': {...}}
"""

import torch
import pandas as pd
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_gamma, plot_coeff_scaling

from src.torchCvdv import TorchCvdv
from src.separable import SeparableState

Ns = [32, 64, 128, 256, 512, 1024]
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1


def _make_sim(sim_class, q, psi_arr):
    """Create a sim with state set to psi_arr without normalization."""
    sep = SeparableState([q], device='cpu')
    sep.setZero(0)
    sim = sim_class([q], device='cuda')
    sim.initStateVector(sep)
    if isinstance(psi_arr, torch.Tensor):
        sim.state = psi_arr.detach().clone().to(dtype=torch.cdouble)
    else:
        sim.state = torch.tensor(psi_arr, dtype=torch.cdouble)
    return sim


def _qft(sim_class, q, psi_arr):
    """Apply ftQ2P to a state array, return result array."""
    sim = _make_sim(sim_class, q, psi_arr)
    sim.ftQ2P(0)
    return torch.tensor(sim.getState(), dtype=torch.cdouble)


def _iqft(sim_class, q, psi_arr):
    sim = _make_sim(sim_class, q, psi_arr)
    sim.ftP2Q(0)
    return torch.tensor(sim.getState(), dtype=torch.cdouble)


def _sweep(N: int) -> list:
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    pos_states, x, dx = fock_recurrence(N)

    cumulative_errors = []
    err_cum = 0.0

    for n, psi_pos in enumerate(pos_states):
        psi = psi_pos.to(torch.cdouble)
        x_c = x.to(torch.cdouble)

        # x·iQFT(x·QFT(ψ))
        t1 = _iqft(TorchCvdv, q, x_c * _qft(TorchCvdv, q, psi))
        # iQFT(x·QFT(x·ψ))
        t2 = _iqft(TorchCvdv, q, x_c * _qft(TorchCvdv, q, x_c * psi))
        # commutator residual
        err_vec = x_c * t1 - t2 - 1j * psi

        err_cum += float(torch.linalg.norm(err_vec))
        cumulative_errors.append(err_cum)

        if err_cum >= STOP_ERR:
            break

    return cumulative_errors


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(Ns), desc='comm_err') as pbar:
        for N in Ns:
            errors = _sweep(N)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'N': N, 'Gamma': gamma, 'err': eps})
            pbar.set_postfix({'N': N, 'n_max': len(errors) - 1, 'eps': f'{errors[-1]:.2e}'})
            pbar.update(1)

    df = pd.DataFrame(rows)
    fit_params = plot_err_vs_gamma(
        df, 'Gamma', 'err', 'N',
        ylabel=r'$\epsilon_{\max}$',
        base_name='comm_err',
        fig_dir=fig_dir,
    )
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'comm_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
