"""
benchmarks/measure_kernels/bench_measure.py

Micro-benchmark for the reduction-kernel paths in cvdvMeasureMultiple
and cvdvGetNorm:

  1. kernelAbsSquareReduce  outSize==1   (full norm, blockReduceSum)
  2. kernelAbsSquareReduce  outSize<=4096 (shared-memory histogram)
  3. kernelAbsSquareReduce  outSize>4096  (scatter atomicAdd)

Sweeps over:
  - total qubit count  (state vector size)
  - register layout    (which register is selected → stride of reduction axis)
  - numRegs            (1 vs 2 → single- vs multi-register path)

Reports median kernel time in microseconds for each configuration.

Usage:
  python benchmarks/measure_kernels/bench_measure.py
  python benchmarks/measure_kernels/bench_measure.py --runs 20
"""

import sys, os, ctypes, math, time
import numpy as np
from ctypes import c_int, c_double, POINTER, c_void_p, c_size_t
from time import perf_counter

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.dirname(os.path.dirname(BENCH_DIR))
BUILD_DIR = os.path.join(REPO_DIR, 'build')

N_WARMUP = 2
N_RUNS   = 10


# ── load libs ────────────────────────────────────────────────────────────────

def _load_cudart():
    for name in ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12',
                 '/usr/local/cuda/lib64/libcudart.so'):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaDeviceSynchronize.restype  = c_int
            lib.cudaMalloc.argtypes            = [POINTER(c_void_p), c_size_t]
            lib.cudaMalloc.restype             = c_int
            lib.cudaMemcpy.argtypes            = [c_void_p, c_void_p, c_size_t, c_int]
            lib.cudaMemcpy.restype             = c_int
            lib.cudaFree.argtypes              = [c_void_p]
            lib.cudaFree.restype               = c_int
            return lib
        except OSError:
            pass
    raise RuntimeError('Could not load libcudart.')


def _load_cvdv():
    lib_path = os.path.join(BUILD_DIR, 'libcvdv.so')
    if not os.path.exists(lib_path):
        raise RuntimeError(f'Library not found at {lib_path}. Run `make build` first.')
    lib = ctypes.CDLL(lib_path)

    lib.cvdvCreate.argtypes            = [c_int, POINTER(c_int)]
    lib.cvdvCreate.restype             = c_void_p
    lib.cvdvDestroy.argtypes           = [c_void_p]
    lib.cvdvDestroy.restype            = None
    lib.cvdvInitFromSeparable.argtypes = [c_void_p, POINTER(c_void_p), c_int]
    lib.cvdvInitFromSeparable.restype  = None
    lib.cvdvGetNorm.argtypes           = [c_void_p]
    lib.cvdvGetNorm.restype            = c_double
    lib.cvdvMeasureMultiple.argtypes   = [c_void_p, POINTER(c_int), c_int, POINTER(c_double)]
    lib.cvdvMeasureMultiple.restype    = None
    lib.cvdvMeasureMultipleCT.argtypes  = [c_void_p, POINTER(c_int), c_int, POINTER(c_double)]
    lib.cvdvMeasureMultipleCT.restype   = None
    return lib


USE_CUTENSOR = False


# ── state helpers ────────────────────────────────────────────────────────────

TWO_PI = 2.0 * math.pi

def _make_coherent(nq: int, alpha_re: float = 1.0, alpha_im: float = 0.0) -> np.ndarray:
    dim = 1 << nq
    dx  = math.sqrt(TWO_PI / dim)
    q0  = math.sqrt(2.0) * alpha_re
    p0  = math.sqrt(2.0) * alpha_im
    ks  = np.arange(dim, dtype=np.float64)
    x   = (ks - (dim - 1) * 0.5) * dx
    env = np.exp(-0.5 * (x - q0) ** 2)
    psi = env * np.exp(1j * p0 * x)
    psi /= np.linalg.norm(psi)
    return psi.astype(np.complex128)


def _upload(cuda, arr: np.ndarray):
    n_bytes = arr.nbytes
    ptr = c_void_p()
    cuda.cudaMalloc(ctypes.byref(ptr), c_size_t(n_bytes))
    cuda.cudaMemcpy(ptr, arr.ctypes.data_as(c_void_p),
                    c_size_t(n_bytes), c_int(1))
    return ptr


def _make_ctx(lib, cuda, qubits_list):
    """Create a context with the given per-register qubit counts."""
    num_reg = len(qubits_list)
    qarr = (c_int * num_reg)(*qubits_list)
    ctx = lib.cvdvCreate(num_reg, qarr)

    states = [_make_coherent(nq) for nq in qubits_list]
    dev_ptrs = [_upload(cuda, s) for s in states]
    ptr_arr = (c_void_p * num_reg)(*dev_ptrs)
    lib.cvdvInitFromSeparable(ctx, ptr_arr, num_reg)
    for p in dev_ptrs:
        cuda.cudaFree(p)
    return ctx


# ── timing ────────────────────────────────────────────────────────────────────

def _time_op(cuda, fn, *args):
    times = []
    for i in range(N_WARMUP + N_RUNS):
        cuda.cudaDeviceSynchronize()
        t0 = perf_counter()
        fn(*args)
        cuda.cudaDeviceSynchronize()
        t1 = perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1e6)
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


# ── benchmark configs ────────────────────────────────────────────────────────

# Each config: (label, qubits_list, regIdxs_to_measure)
# We vary:
#   - total state size (small/medium/large)
#   - which register is selected (LSB vs MSB position → stride)
#   - single vs multi register selection
#   - output size relative to MEASURE_SHARED_HIST_MAX (4096)

CONFIGS = [
    # ── GetNorm (outSize=1, blockReduceSum path) ────────────────────────
    ("norm  2^10 total",     [10],             None),   # 1K state
    ("norm  2^16 total",     [16],             None),   # 64K state
    ("norm  2^20 total",     [20],             None),   # 1M state

    # ── Single register, small output (≤4096) → shared histogram ────────
    ("1reg r0(4qb)  [2,4,6]",    [2, 4, 6],   [1]),   # outSize=16
    ("1reg r1(6qb)  [2,6,6]",    [2, 6, 6],   [1]),   # outSize=64
    ("1reg r1(10qb) [2,10,6]",   [2, 10, 6],  [1]),   # outSize=1024
    ("1reg r1(12qb) [2,12,6]",   [2, 12, 6],  [1]),   # outSize=4096 (boundary)

    # ── Single register, large output (>4096) → scatter atomicAdd ───────
    ("1reg r1(13qb) [2,13,6]",   [2, 13, 6],  [1]),   # outSize=8192
    ("1reg r1(14qb) [2,14,6]",   [2, 14, 6],  [1]),   # outSize=16384
    ("1reg r1(16qb) [2,16,6]",   [2, 16, 6],  [1]),   # outSize=65536

    # ── Single register, MSB position (large stride) ────────────────────
    ("1reg r2(6qb)  [6,6,2]",    [6, 6, 2],   [2]),   # outSize=4, small
    ("1reg r0(10qb) [10,6,2]",   [10, 6, 2],  [0]),   # outSize=1024, small
    ("1reg r0(14qb) [14,6,2]",   [14, 6, 2],  [0]),   # outSize=16384, large

    # ── Multi register → scatter atomicAdd ──────────────────────────────
    ("2reg [0,1] [2,6,6]",       [2, 6, 6],   [0, 1]),  # outSize=256
    ("2reg [1,2] [2,6,6]",       [2, 6, 6],   [1, 2]),  # outSize=4096
    ("2reg [1,2] [2,10,6]",      [2, 10, 6],  [1, 2]),  # outSize=65536
]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global N_RUNS, USE_CUTENSOR
    n_runs = N_RUNS
    for arg in sys.argv[1:]:
        if arg.startswith('--runs='):
            n_runs = int(arg.split('=')[1])
        elif arg.startswith('--runs'):
            n_runs = int(sys.argv[sys.argv.index(arg) + 1])
        elif arg == '--cutensor':
            USE_CUTENSOR = True

    N_RUNS = n_runs

    cuda = _load_cudart()
    lib  = _load_cvdv()

    COL_LABEL = 32
    COL_PATH  = 14
    COL_MED   = 12
    COL_MIN   = 12
    COL_MAX   = 12
    COL_OUT   = 10

    hdr = (f'{"Config":<{COL_LABEL}}  {"Kernel Path":>{COL_PATH}}  '
           f'{"outSize":>{COL_OUT}}  '
           f'{"Median(us)":>{COL_MED}}  {"Min(us)":>{COL_MIN}}  {"Max(us)":>{COL_MAX}}')
    sep = '-' * len(hdr)

    backend = 'cuTENSOR' if USE_CUTENSOR else 'custom kernel'
    print(f'\nMeasure Kernel Benchmark  ({N_WARMUP} warm-up + {N_RUNS} timed runs)  [{backend}]')
    print(f'Total state sizes from 2^4 to 2^24')
    print()
    print(hdr)
    print(sep)

    results = []

    for label, qubits_list, reg_idxs in CONFIGS:
        ctx = _make_ctx(lib, cuda, qubits_list)
        total_qbt = sum(qubits_list)
        total_size = 1 << total_qbt

        if reg_idxs is None:
            # ── cvdvGetNorm ──────────────────────────────────────────────
            out_size = 1
            path = "norm"
            med, mn, mx = _time_op(cuda, lib.cvdvGetNorm, ctx)
        else:
            # ── cvdvMeasureMultiple ──────────────────────────────────────
            num_regs = len(reg_idxs)
            out_size = 1
            for ri in reg_idxs:
                out_size *= (1 << qubits_list[ri])

            if out_size == 1:
                path = "norm"
            elif out_size <= 4096:
                path = "shared-hist"
            else:
                path = "scatter"

            regs_c = (c_int * num_regs)(*reg_idxs)
            probs = (c_double * out_size)()
            measure_fn = lib.cvdvMeasureMultipleCT if USE_CUTENSOR else lib.cvdvMeasureMultiple
            med, mn, mx = _time_op(cuda, measure_fn,
                                    ctx, regs_c, c_int(num_regs), probs)

        lib.cvdvDestroy(ctx)

        row = (f'{label:<{COL_LABEL}}  {path:>{COL_PATH}}  '
               f'{out_size:>{COL_OUT}d}  '
               f'{med:>{COL_MED}.1f}  {mn:>{COL_MIN}.1f}  {mx:>{COL_MAX}.1f}')
        print(row)
        results.append((label, path, out_size, med, mn, mx))

    print(sep)
    print()

    # ── Summary: group by kernel path ────────────────────────────────────
    print('Summary by kernel path (median, across all configs):')
    by_path = {}
    for label, path, out_size, med, mn, mx in results:
        by_path.setdefault(path, []).append((label, out_size, med))

    for path in ['norm', 'shared-hist', 'scatter']:
        entries = by_path.get(path, [])
        if not entries:
            continue
        medians = [e[2] for e in entries]
        print(f'  {path:>{COL_PATH}}  n={len(entries):>2}  '
              f'median={np.median(medians):.1f}us  '
              f'range=[{min(medians):.1f}, {max(medians):.1f}]us')

    print()


if __name__ == '__main__':
    main()
