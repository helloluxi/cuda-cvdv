"""Advantage diagram.

Uses QFT error fit coefficients to compute Gamma at target error, then plots
tau_hybrid = c_n * Gamma * tau_qubit vs tau_qubit for each encoding n.
Produces: figures/advantage.pdf
Returns: {'Gamma_ls': [...], 'c_ls': [...], 'n_ls': [...]}
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(['science'])
plt.rcParams.update({'font.size': 24})

# Hard-coded fit from qft_err (matches dvsim-code paper values)
# log(eps) ≈ (4.333/sqrt(N) + 0.118)*Gamma - 6.279*sqrt(N) + 18.5
_C_A = 4.333
_D_A = 0.118
_C_B = -6.279
_D_B = 18.5

TARGET_ERROR = 1e-4
N_LS = np.array([6, 8, 10, 12])
X_LS = np.array([1e-5, 1e-3])


def _gamma_for_n(n: int, target_error: float, c_a=_C_A, d_a=_D_A, c_b=_C_B, d_b=_D_B) -> int:
    N = 1 << n
    sqrt_N = np.sqrt(N)
    ln_eps = np.log(target_error)
    gamma = (ln_eps - (c_b * sqrt_N + d_b)) / (c_a / sqrt_N + d_a)
    return max(1, round(gamma))


def run(fig_dir: str, qft_fit_params=None) -> dict:
    # If caller passes updated fit params from this run, recompute coefficients
    c_a, d_a, c_b, d_b = _C_A, _D_A, _C_B, _D_B
    if qft_fit_params and len(qft_fit_params) >= 2:
        from ._common import fit_coeff_scaling
        sc = fit_coeff_scaling(qft_fit_params)
        c_a, d_a = sc['c_a'], sc['d_a']
        c_b, d_b = sc['c_b'], sc['d_b']

    N_ls = 2 ** N_LS
    sqrt_N_ls = np.sqrt(N_ls)
    Gamma_ls = np.array([
        max(1, round((np.log(TARGET_ERROR) - (c_b * np.sqrt(1 << n) + d_b))
                     / (c_a / np.sqrt(1 << n) + d_a)))
        for n in N_LS
    ])
    c_ls = 2 * N_LS * (N_LS + 1)   # gate count per mode: 2n(n+1) CNOTs

    x_log = np.log10(X_LS)
    y_log_all = [np.log10(c_ls[i] * X_LS) for i in range(len(N_LS))]
    y_min = min(v.min() for v in y_log_all) - 0.1
    y_max = max(v.max() for v in y_log_all) + 0.1
    x_min = x_log[0] - 0.1
    x_max = x_log[1] + 0.1

    fig, ax = plt.subplots(figsize=(10, 8))

    # Gradient background (red→white→blue)
    bg = 30
    xb = np.linspace(x_min, x_max, bg)
    yb = np.linspace(y_min, y_max, bg)
    Xb, Yb = np.meshgrid(xb, yb)
    xn = (Xb - x_min) / (x_max - x_min)
    yn = (Yb - y_min) / (y_max - y_min)
    g = (yn - xn + 1) / 2
    colors = np.zeros((bg, bg, 3))
    colors[:, :, 0] = np.clip(np.exp(-5 * g ** 2) + np.exp(-5 * (g - 0.5) ** 2), 0, 1)
    colors[:, :, 1] = np.clip(np.exp(-5 * (g - 0.5) ** 2), 0, 1)
    colors[:, :, 2] = np.clip(np.exp(-5 * (1 - g) ** 2) + np.exp(-5 * (g - 0.5) ** 2), 0, 1)
    ax.imshow(colors, extent=[x_min, x_max, y_min, y_max],
              aspect='auto', origin='lower', zorder=1, alpha=0.5, interpolation='bilinear')

    for idx, n in enumerate(N_LS):
        y_log_line = np.log10(c_ls[idx] * X_LS)
        line, = ax.plot(x_log, y_log_line, linewidth=3, zorder=10)
        # Label at lower-left
        x_lbl = x_log[0] + 0.05 * (x_log[1] - x_log[0])
        y_lbl = np.log10(c_ls[idx]) + x_lbl
        # Compute angle in display coordinates so label is parallel to line
        p0 = ax.transData.transform((x_log[0], y_log_line[0]))
        p1 = ax.transData.transform((x_log[1], y_log_line[1]))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(x_lbl, y_lbl + 0.05, f'$n={n},\\,\\Gamma={Gamma_ls[idx]}$',
                rotation=angle, rotation_mode='anchor',
                color=line.get_color(), fontweight='bold', zorder=15)

    xticks = np.arange(int(x_log[0]), int(x_log[1]) + 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'$10^{{{p}}}$' for p in xticks])
    yticks = np.arange(int(y_min), int(y_max) + 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'$10^{{{p}}}$' for p in yticks])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r'$\tau_{\mathrm{qubit}}$')
    ax.set_ylabel(r'$\tau_{\mathrm{hybrid}}$')
    fig.tight_layout()
    from ._common import _save_fig
    import os
    _save_fig(fig, 'advantage', fig_dir)
    plt.close(fig)

    return {'Gamma_ls': Gamma_ls.tolist(), 'c_ls': c_ls.tolist(), 'n_ls': N_LS.tolist(),
            'target_error': TARGET_ERROR}
