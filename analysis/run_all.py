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
    ('qft_disp_err',      'qft_disp_err',      'QFT on Displaced Fock States'),
    ('qft_squeeze_err',   'qft_squeeze_err',   'QFT on Squeezed Fock States'),
    ('comm_err',          'comm_err',          'Commutator Error'),
    ('disp_err',          'disp_err',          'Displacement D(2) Error'),
    ('rot_err',           'rot_err',           'Rotation R(pi/4) Error'),
    ('squeeze_err',       'squeeze_err',       'Squeezing S(1) Error'),
    ('beam_splitter_err', 'beam_splitter_err', 'Beam Splitter BS(pi/2) Error'),
    ('compare_fock',      'compare_fock',      'Compare Fock vs WF Encoding'),
    ('plot_advantage',    'plot_advantage',    'Advantage Diagram'),
]

# Figures produced per module (PNG for report, PDF archived separately)
FIGURES = {
    'check_norm':                  ['norm_check.png'],
    'qft_err':                     ['qft_err_per_fock.png', 'qft_per_fock_coeff_scaling.png'],
    'qft_disp_err':                ['qft_disp_err.png', 'qft_disp_coeff_scaling.png'],
    'qft_disp_err_param_sweep':    ['qft_disp_vs_gamma.png', 'qft_disp_eps_vs_alpha.png'],
    'qft_squeeze_err':             ['qft_squeeze_err.png', 'qft_squeeze_coeff_scaling.png'],
    'qft_squeeze_err_param_sweep': ['qft_squeeze_vs_gamma.png', 'qft_squeeze_eps_vs_r.png'],
    'comm_err':                    ['comm_err.png', 'comm_coeff_scaling.png'],
    'disp_err':                    ['disp_err.png', 'disp_coeff_scaling.png'],
    'disp_err_param_sweep':        ['disp_eps_vs_alpha.png'],
    'rot_err':                     ['rot_err.png', 'rot_coeff_scaling.png'],
    'rot_err_param_sweep':         ['rot_eps_vs_theta.png'],
    'squeeze_err':                 ['squeeze_err.png', 'squeeze_coeff_scaling.png'],
    'squeeze_err_param_sweep':     ['squeeze_eps_vs_r.png'],
    'beam_splitter_err':           ['beam_splitter_err.png', 'bs_coeff_scaling.png'],
    'beam_splitter_err_param_sweep': ['bs_eps_vs_theta.png'],
    'compare_fock':                ['compare_fock.png'],
    'plot_advantage':              ['advantage.png'],
}


def _fmt_latex_formula(fit_params: list, scaling: dict) -> str:
    """Render combined fit as a LaTeX display equation."""
    if not fit_params or not scaling:
        return ''
    ca = scaling['c_a']; da = scaling['d_a']
    cb = scaling['c_b']; db = scaling['d_b']
    def _pm(v): return f'+ {v:.4f}' if v >= 0 else f'- {abs(v):.4f}'
    return (
        '$$\n'
        r'\log \varepsilon \;\approx\; '
        rf'\Bigl({ca:.4f}\,N^{{-1/2}} {_pm(da)}\Bigr)\,\Bigl(\Gamma+\tfrac{{1}}{{2}}\Bigr)'
        rf'\;+\; {cb:.4f}\,N^{{1/2}} {_pm(db)}'
        '\n$$\n'
    )


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
    if fp is not None and sc:
        latex = _fmt_latex_formula(fp, sc)
        if latex:
            body += '**Fitted formula** `log(eps) = a(N)*(Gamma+1/2) + b(N)`:\n\n' + latex

    # param-sweep subsection
    psw_figs = FIGURES.get(key + '_param_sweep', [])
    if psw_figs:
        body += '\n### Gate-parameter sweep (fixed N)\n\n'
        for f in psw_figs:
            if os.path.exists(os.path.join(FIG_DIR, f)):
                body += f'![{f}](analysis/figures/{f})\n'
        body += '\n'

    return body + '\n'


def main(skip=None, only=None):
    only = set(only) if only else set()
    skip = set() if only else set(skip or [])
    all_results = {}

    print(f'Output directory: {FIG_DIR}\n')

    for key, modname, title in MODULES:
        if only and key not in only:
            print(f'[SKIP] {title}')
            continue
        if key in skip:
            print(f'[SKIP] {title}')
            continue
        print(f'[RUN]  {title} ...')
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
            if (only and key not in only) or key in skip or key not in all_results:
                continue
            r = all_results[key]
            f.write(_section(title, key, r, r.get('_elapsed', 0.0)))

    print(f'\nReport written to: {report_path}')
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', nargs='*', default=[], help='module keys to skip')
    parser.add_argument('--only', nargs='*', default=[], help='only run these module keys (ignores --skip)')
    args = parser.parse_args()
    main(skip=args.skip, only=args.only)
