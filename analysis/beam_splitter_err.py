"""Beam splitter BS(θ=π/2) error analysis.

2-mode state |n⟩|0⟩ → discrete BS via q1q2 + 2D QFT → compare to analytic binomial expansion.
Discrete: exp(-i·tan(θ/4)·q1·q2) · ftQ2P(both) · exp(-i·sin(θ/2)·p1·p2) · ftP2Q(both) · exp(-i·tan(θ/4)·q1·q2)
Analytic: BS(θ)|n⟩|0⟩ = Σ_k √C(n,k) cos^k(θ/2)·(-i·sin(θ/2))^{n-k} |k⟩|n-k⟩
Produces: figures/beam_splitter_err.pdf
Returns: {'fit_params': [...], 'scaling': {...}}
"""

import torch
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from ._common import fock_recurrence, plot_err_vs_gamma, plot_coeff_scaling

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [32, 64, 128, 256]
THETA = float(torch.pi / 2)
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1
MIN_N = 4


def _apply_bs(sim: CVDV, theta: float):
    """Apply BS(theta) to a 2-register CVDV sim (both registers same size)."""
    # q1q2 phase
    tan_val = torch.tan(torch.tensor(theta / 4)).item()
    sim.q1q2(0, 1, -tan_val)
    # QFT both modes
    sim.ftQ2P(0)
    sim.ftQ2P(1)
    # p1p2 phase (q1q2 in momentum basis)
    sin_val = torch.sin(torch.tensor(theta / 2)).item()
    sim.q1q2(0, 1, -sin_val)
    # iQFT both modes
    sim.ftP2Q(0)
    sim.ftP2Q(1)
    # second q1q2 phase
    sim.q1q2(0, 1, -tan_val)


def _sweep(N: int, theta: float) -> list:
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    fock_states, x, dx = fock_recurrence(N)

    cumulative_errors = []
    err_cum = 0.0
    cos_h = torch.cos(torch.tensor(theta / 2)).item()
    sin_h = torch.sin(torch.tensor(theta / 2)).item()

    for n in range(N):
        # --- discrete ---
        sep = SeparableState([q, q], device='cpu')
        sep.setFock(0, n)
        sep.setFock(1, 0)
        sim = CVDV([q, q], backend='torch-cuda')
        sim.initStateVector(sep)
        _apply_bs(sim, theta)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble).reshape(N, N)   # [mode0, mode1]

        # --- analytic binomial ---
        psi_analytic = torch.zeros((N, N), dtype=torch.cdouble)
        for k in range(n + 1):
            binom = int(comb(n, k, exact=True))
            amp = torch.sqrt(torch.tensor(float(binom))) * (cos_h ** k) * ((-1j * sin_h) ** (n - k))
            psi_k = fock_states[k].to(torch.cdouble)
            psi_nk = fock_states[n - k].to(torch.cdouble)
            psi_analytic += amp * torch.outer(psi_k, psi_nk)

        err_cum += float(torch.linalg.norm(psi_discrete - psi_analytic))
        cumulative_errors.append(err_cum)

        if err_cum >= STOP_ERR and n >= MIN_N:
            break

    return cumulative_errors


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(N_TOTALS), desc='beam_splitter_err') as pbar:
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
        ylabel=r'Bounds of $\epsilon_{BS(\pi/2)}$',
        base_name='beam_splitter_err',
        fig_dir=fig_dir,
    )
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'bs_coeff_scaling', fig_dir)
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df}
