"""QFT error analysis on displaced Fock states D(α)|n⟩.

Discrete: D(α) → ftQ2P applied to each Fock state.
Analytic: FT[ψ_n(x - α√2)] × (-i)^n in momentum basis.
Produces: figures/qft_disp_err.pdf, figures/qft_disp_coeff_scaling.pdf,
          figures/qft_disp_eps_vs_alpha.pdf, figures/qft_disp_vs_gamma.pdf
Returns: {'fit_params': [...], 'scaling': {...}, 'param_sweep': [...]}
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ._common import plot_err_vs_gamma, plot_coeff_scaling, plot_eps_vs_param, plot_param_sweep, fit_ab

from src import CVDV, SeparableState  # type: ignore

N_TOTALS = [64, 128, 256, 512]
ALPHA = 2.0
PRECISION_CUTOFF = 1e-9
STOP_ERR = 0.1

# Param sweep: fix N, vary alpha
SWEEP_N = 256
SWEEP_NUM_GATE_PARAM = 101
ALPHA_VALUES = (torch.arange(1, 1+SWEEP_NUM_GATE_PARAM) / SWEEP_NUM_GATE_PARAM * 4.0).tolist()
SWEEP_LIST_GAMMA = [80, 100, 120, 140, 160]


def _analytic_qft_displaced_states(N: int, alpha: float):
    """Analytic QFT[D(α)|n⟩] in momentum basis.
    
    Start with displaced Fock ψ_n(x - α√2), then apply discrete Fourier transform.
    Note: (-i)^n phase is NOT applied - that's only for pure Fock states.
    Returns list of momentum wavefunctions (complex).
    """
    dx = torch.sqrt(torch.tensor(2.0 * torch.pi / N))
    
    # Position grid (shifted)
    idx = torch.arange(N, dtype=torch.float64)
    x_shift = (idx - (N - 1) * 0.5) * dx - alpha * torch.sqrt(torch.tensor(2.0))
    
    # Generate displaced Fock states in position basis
    pos_states = []
    psi_prev = torch.zeros(N, dtype=torch.float64)
    psi_curr = torch.exp(-0.5 * x_shift ** 2) * (torch.pi ** -0.25) * torch.sqrt(dx)
    pos_states.append(psi_curr.clone())
    
    for n in range(1, N):
        n_float = float(n)
        psi_next = (torch.sqrt(torch.tensor(2.0 / n_float)) * x_shift * psi_curr
                    - torch.sqrt(torch.tensor((n_float - 1.0) / n_float)) * psi_prev)
        psi_prev, psi_curr = psi_curr, psi_next
        pos_states.append(psi_curr.clone())
    
    # Apply QFT: Fourier transform (zero-centered convention matching ftQ2P)
    mom_states = []
    for psi_pos in pos_states:
        psi_pos_complex = psi_pos.to(torch.cdouble)
        psi_qft = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(psi_pos_complex)))
        mom_states.append(psi_qft)
    
    return mom_states


def _sweep(N: int, alpha: float) -> list:
    """Run QFT on displaced Fock states for all n, accumulate error."""
    q = int(round(torch.log2(torch.tensor(float(N))).item()))
    
    ana_states = _analytic_qft_displaced_states(N, alpha)
    
    cumulative_errors = []
    err_cum = 0.0
    
    for n in range(N):
        # Prepare discrete: D(α)|n⟩ → QFT
        sep = SeparableState([q], device='cpu')
        sep.setFock(0, n)
        sim = CVDV([q], backend='torch-cuda')
        sim.initStateVector(sep)
        sim.d(0, float(alpha))
        sim.ftQ2P(0)
        psi_discrete = torch.tensor(sim.getState(), dtype=torch.cdouble)
        
        # Compare with analytic
        psi_analytic = ana_states[n]
        err_cum += float(torch.linalg.norm(psi_discrete - psi_analytic))
        cumulative_errors.append(err_cum)
        
        if err_cum >= STOP_ERR:
            break
    
    return cumulative_errors


def _run_param_sweep(fig_dir: str) -> list:
    """Sweep alpha values at fixed N, produce two plots:
    1. eps vs Gamma for different alpha values
    2. eps vs alpha for fixed Gamma values
    """
    rows = []
    with tqdm(total=len(ALPHA_VALUES), desc='qft_disp param sweep') as pbar:
        for alpha in ALPHA_VALUES:
            errors = _sweep(SWEEP_N, alpha)
            for gamma, eps in enumerate(errors):
                if eps >= PRECISION_CUTOFF:
                    rows.append({'param': alpha, 'Gamma': gamma, 'err': eps})
            pbar.update(1)
    
    # Plot 1: eps vs Gamma for different alpha values
    summary = plot_param_sweep(
        rows, param_label=r'\alpha', base_name='qft_disp_vs_gamma',
        fig_dir=fig_dir, ylabel=r'$\varepsilon_{\mathrm{QFT}[D(\alpha)|n\rangle]}$',
        N_fixed=SWEEP_N
    )
    
    # Plot 2: eps vs alpha for fixed Gamma values
    plot_eps_vs_param(
        rows, param_label=r'\alpha', base_name='qft_disp_eps_vs_alpha',
        fig_dir=fig_dir, gamma_values=SWEEP_LIST_GAMMA,
        ylabel=r'$\varepsilon_{\mathrm{QFT}[D(\alpha)|n\rangle]}$', N_fixed=SWEEP_N
    )
    
    return summary


def run(fig_dir: str) -> dict:
    """Main analysis: error vs Gamma for different N values."""
    rows = []
    with tqdm(total=len(N_TOTALS), desc='qft_disp_err') as pbar:
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
        ylabel=r'Bounds of $\varepsilon_{\mathrm{QFT}[D(2.0)|n\rangle]}$',
        base_name='qft_disp_err',
        fig_dir=fig_dir,
    )
    
    scaling = {}
    if len(fit_params) >= 2:
        scaling = plot_coeff_scaling(fit_params, 'qft_disp_coeff_scaling', fig_dir)
    
    param_sweep = _run_param_sweep(fig_dir)
    
    return {'fit_params': fit_params, 'scaling': scaling, 'df': df, 'param_sweep': param_sweep}
