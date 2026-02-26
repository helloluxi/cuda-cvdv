"""Fock state norm check.

Computes ||Enc_Q(|n⟩_F)|| for n = 0..N-1 across grid sizes.
Produces: figures/norm_check.pdf
Returns: {'df': DataFrame}
"""

import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots  # noqa: F401

from ._common import fock_recurrence, _save_fig
import os

# Use scienceplots style
plt.style.use(['science'])
plt.rcParams.update({'font.size': 24, 'text.usetex': True})

Ns = [32, 64, 128, 256, 512, 1024]


def run(fig_dir: str) -> dict:
    rows = []
    with tqdm(total=len(Ns), desc='check_norm') as pbar:
        for N in Ns:
            states, _, _ = fock_recurrence(N)
            for gamma, psi in enumerate(states):
                norm = float(torch.linalg.norm(psi))
                rows.append({'N': N, 'Gamma': gamma, 'norm': norm})
            pbar.set_postfix({'N': N})
            pbar.update(1)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Norm = 1')
    for N, group in df.groupby('N'):
        ax.plot(group['Gamma'].values, group['norm'].values,
                marker='o', linewidth=2, markersize=4, label=f'$N={int(N)}$', alpha=0.8)
    ax.set_xlabel(r'$\Gamma$')
    ax.set_ylabel(r'$\|\mathrm{Enc}_Q(|\Gamma\rangle_F)\|$')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.15])
    fig.tight_layout()
    _save_fig(fig, 'norm_check', fig_dir)
    plt.close(fig)

    return {'df': df}
