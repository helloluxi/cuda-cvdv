"""Benchmark individual gate operations across register configurations."""
import sys, os, time, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))
from src import CVDV
import matplotlib.pyplot as plt

WARMUP = 2
RUNS = 10
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def make_sim(regs):
    sim = CVDV(regs)
    for i, q in enumerate(regs):
        sim.setUniform(i) if q == 1 else sim.setFock(i, 0)
    sim.initStateVector()
    return sim


def bench(regs, gate_fn):
    sim = make_sim(regs)
    for _ in range(WARMUP):
        gate_fn(sim)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        gate_fn(sim)
        times.append(time.perf_counter() - t0)
    return times


# Config definitions: (name, title, cv_qubits_list, regs_fn, gates_dict)
CONFIGS = [
    ('1cv', '1 CV', [12, 14, 16], lambda q: [q], {
        'Displace':        lambda s: s.d(0, 1.0),
        'Rotate':        lambda s: s.r(0, 0.5),
        'Squeeze':        lambda s: s.s(0, 0.3),
        'FT':       lambda s: s.ftQ2P(0),
        'Wigner':   lambda s: s.getWignerFullMode(0, wignerN=51),
        'Husimi-Q': lambda s: s.getHusimiQFullMode(0, qN=51),
    }),
    ('1dv_1cv', '1 DV + 1 CV', [12, 14, 16], lambda q: [1, q], {
        'Qubit Rz':       lambda s: s.rz(0, 0, 0.5),
        'FT':       lambda s: s.ftQ2P(1),
        'C-Displace':       lambda s: s.cd(1, 0, 0, 1.0),
        'C-Rotate':       lambda s: s.cr(1, 0, 0, 0.5),
        'C-Squeeze':       lambda s: s.cs(1, 0, 0, 0.3),
        'Qubit Measure':        lambda s: s.m(0),
        'Wigner':   lambda s: s.getWignerFullMode(1, wignerN=51),
        'Husimi-Q': lambda s: s.getHusimiQFullMode(1, qN=51),
    }),
    ('2cv', '2 CV', [8, 9, 10], lambda q: [q, q], {
        'Displace':        lambda s: s.d(0, 1.0),
        'Rotate':        lambda s: s.r(0, 0.5),
        'Squeeze':        lambda s: s.s(0, 0.3),
        'FT':       lambda s: s.ftQ2P(0),
        'Beam Splitter':       lambda s: s.bs(0, 1, 0.5),
        'Wigner':   lambda s: s.getWignerFullMode(0, wignerN=51),
        'Husimi-Q': lambda s: s.getHusimiQFullMode(0, qN=51),
    }),
    ('1dv_2cv', '1 DV + 2 CV', [8, 9, 10], lambda q: [1, q, q], {
        'Qubit Rz':       lambda s: s.rz(0, 0, 0.5),
        'FT':       lambda s: s.ftQ2P(1),
        'C-Displace':       lambda s: s.cd(1, 0, 0, 1.0),
        'C-Rotate':       lambda s: s.cr(1, 0, 0, 0.5),
        'C-Squeeze':       lambda s: s.cs(1, 0, 0, 0.3),
        'Beam Splitter':       lambda s: s.bs(1, 2, 0.5),
        'C-Beam Splitter':      lambda s: s.cbs(1, 2, 0, 0, 0.5),
        'Qubit Measure':        lambda s: s.m(0),
        'Wigner':   lambda s: s.getWignerFullMode(1, wignerN=51),
        'Husimi-Q': lambda s: s.getHusimiQFullMode(1, qN=51),
    }),
]


def plot_config(cfg_name, title, gate_names, cv_qubits, data):
    n_gates = len(gate_names)
    x = np.arange(n_gates)
    width = 0.25
    colors = ['#4878d0', '#ee854a', '#6acc64']

    fig, ax = plt.subplots(figsize=(max(8, n_gates * 1.0), 5))
    for i, q in enumerate(cv_qubits):
        means = [np.mean(data[q][g]) * 1000 for g in gate_names]
        stds = [np.std(data[q][g]) * 1000 for g in gate_names]
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=stds,
                      label=f'CV Dimension: {1<<q}', color=colors[i], capsize=3)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.2f}' if val >= 1 else f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, rotation=45)

    ax.set_ylabel('Runtime (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(gate_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f'bench_{cfg_name}.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for cfg_name, title, cv_qubits, regs_fn, gates in CONFIGS:
        print(f"\n{'='*50}\n{title}\n{'='*50}")
        gate_names = list(gates.keys())
        cfg_data = {}

        for q in cv_qubits:
            regs = regs_fn(q)
            print(f"\n  cv-qubit={q}  regs={regs}")
            cfg_data[q] = {}
            for gname, gfn in gates.items():
                times = bench(regs, gfn)
                cfg_data[q][gname] = times
                print(f"    {gname:10s} {np.mean(times)*1000:8.3f} +/- {np.std(times)*1000:.3f} ms")

        all_results[cfg_name] = {
            'title': title, 'cv_qubits': cv_qubits, 'gates': gate_names,
            'data': {str(q): {g: cfg_data[q][g] for g in gate_names} for q in cv_qubits},
        }
        plot_config(cfg_name, title, gate_names, cv_qubits, cfg_data)
        print(f"  -> bench_{cfg_name}.png")

    with open(os.path.join(RESULTS_DIR, 'bench_ops.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Results in {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
