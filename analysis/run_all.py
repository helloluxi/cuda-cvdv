"""Run all error analyses and write a single report.md.

Usage:
    python analysis/run_all.py [--skip <module> ...]

Outputs:
    analysis/figures/*.pdf  — all figures
    analysis/report.md      — combined report
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
# Make analysis/ importable as a package
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

FIG_DIR = os.path.join(_HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── module registry ──────────────────────────────────────────────────────────
MODULES = [
    ('check_norm',        'check_norm',        'Fock State Norm Check'),
    ('qft_err',           'qft_err',           'QFT Error'),
    ('comm_err',          'comm_err',          'Commutator [q,p]=i Error'),
    ('disp_err',          'disp_err',          'Displacement D(2) Error'),
    ('rot_err',           'rot_err',           'Rotation R(π/4) Error'),
    ('squeeze_err',       'squeeze_err',       'Squeezing S(1) Error'),
    ('beam_splitter_err', 'beam_splitter_err', 'Beam Splitter BS(π/2) Error'),
    ('compare_fock',      'compare_fock',      'Compare Fock vs WF Encoding'),
    ('plot_advantage',    'plot_advantage',    'Advantage Diagram'),
]

# Figures produced per module (PNG for report, PDF archived separately)
FIGURES = {
    'check_norm':        ['norm_check.png'],
    'qft_err':           ['qft_err.png', 'qft_coeff_scaling.png'],
    'comm_err':          ['comm_err.png', 'comm_coeff_scaling.png'],
    'disp_err':          ['disp_err.png', 'disp_coeff_scaling.png'],
    'rot_err':           ['rot_err.png', 'rot_coeff_scaling.png'],
    'squeeze_err':       ['squeeze_err.png', 'squeeze_coeff_scaling.png'],
    'beam_splitter_err': ['beam_splitter_err.png', 'bs_coeff_scaling.png'],
    'compare_fock':      ['compare_fock.png'],
    'plot_advantage':    ['advantage.png'],
}


def _fmt_scaling(sc: dict) -> str:
    if not sc:
        return '_no scaling fit_'
    return (f'a ≈ {sc["c_a"]:.4f}·N⁻¹/² {sc["d_a"]:+.4f}  (R²={sc["r2_a"]:.4f}),  '
            f'b ≈ {sc["c_b"]:.4f}·N¹/²  {sc["d_b"]:+.4f}  (R²={sc["r2_b"]:.4f})')


def _fmt_fit_table(fit_params: list) -> str:
    if not fit_params:
        return '_no fit params_\n'
    lines = ['| N | a | b |', '|---|---|---|']
    for N, a, b in fit_params:
        lines.append(f'| {int(N)} | {a:.4e} | {b:.4f} |')
    return '\n'.join(lines) + '\n'


def _section(title, key, result, elapsed) -> str:
    figs = FIGURES.get(key, [])
    fig_links = '\n'.join(
        f'![{f}](analysis/figures/{f})' for f in figs
        if os.path.exists(os.path.join(FIG_DIR, f))
    )
    body = f'## {title}\n\n'
    if fig_links:
        body += fig_links + '\n\n'

    fp = result.get('fit_params')
    sc = result.get('scaling', {})
    if fp is not None:
        body += '**Per-N linear fits** `log(ε) = a·Γ + b`:\n\n'
        body += _fmt_fit_table(fp)
        if sc:
            body += f'\n**Coefficient scaling**: {_fmt_scaling(sc)}\n'
    return body + '\n'


def main(skip=None):
    skip = set(skip or [])
    all_results = {}

    print(f'Output directory: {FIG_DIR}\n')

    for key, modname, title in MODULES:
        if key in skip:
            print(f'[SKIP] {title}')
            continue
        print(f'[RUN ] {title} ...')
        t0 = time.time()
        import importlib
        mod = importlib.import_module(f'analysis.{modname}')
        kwargs = {'fig_dir': FIG_DIR}
        if key == 'plot_advantage' and 'qft_err' in all_results:
            kwargs['qft_fit_params'] = all_results['qft_err'].get('fit_params')
        result = mod.run(**kwargs)
        elapsed = time.time() - t0
        all_results[key] = result
        all_results[key]['_elapsed'] = elapsed
        print(f'       done in {elapsed:.1f}s')

    # ── write report ─────────────────────────────────────────────────────────
    report_path = os.path.join(_ROOT, 'ERROR-ANALYSIS.md')
    with open(report_path, 'w') as f:
        f.write('# CVDV Discretization Error Analysis\n\n')
        for key, modname, title in MODULES:
            if key in skip or key not in all_results:
                continue
            r = all_results[key]
            f.write(_section(title, key, r, r.get('_elapsed', 0.0)))

    print(f'\nReport written to: {report_path}')
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', nargs='*', default=[], help='module keys to skip')
    args = parser.parse_args()
    main(skip=args.skip)
