"""
profiling/bench.py
Python-level C API timing for CVDV.

Times each C API call using time.perf_counter with cudaDeviceSynchronize
fencing, N=10 timed runs after 1 warm-up, reports median time per op.

Usage:
  python profiling/bench.py            # time + compare vs committed baseline
  python profiling/bench.py --no-save  # time only, skip writing CSV
"""

import sys, os, ctypes, math, shutil
import numpy as np
from ctypes import c_int, c_double, POINTER
from time import perf_counter

PROFILING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR      = os.path.dirname(PROFILING_DIR)
BUILD_DIR     = os.path.join(REPO_DIR, 'build')
CURRENT_CSV   = os.path.join(PROFILING_DIR, 'current',   'bench.csv')
COMMITTED_CSV = os.path.join(PROFILING_DIR, 'committed', 'bench.csv')

N_WARMUP = 1
N_RUNS   = 10


# ── load libs ────────────────────────────────────────────────────────────────

def _load_cudart():
    for name in ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12',
                 '/usr/local/cuda/lib64/libcudart.so'):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaDeviceSynchronize.restype  = c_int
            return lib
        except OSError:
            pass
    raise RuntimeError('Could not load libcudart. Is CUDA installed?')


def _load_cvdv():
    lib_path = os.path.join(BUILD_DIR, 'libcvdv.so')
    if not os.path.exists(lib_path):
        raise RuntimeError(f'Library not found at {lib_path}. Run `make build` first.')
    lib = ctypes.CDLL(lib_path)

    lib.cvdvCreate.argtypes            = [c_int, POINTER(c_int)]
    lib.cvdvCreate.restype             = ctypes.c_void_p
    lib.cvdvDestroy.argtypes           = [ctypes.c_void_p]
    lib.cvdvDestroy.restype            = None
    lib.cvdvInitFromSeparable.argtypes = [ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_void_p), c_int]
    lib.cvdvInitFromSeparable.restype  = None

    lib.cvdvDisplacement.argtypes             = [ctypes.c_void_p, c_int, c_double, c_double]
    lib.cvdvDisplacement.restype              = None
    lib.cvdvRotation.argtypes                 = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvRotation.restype                  = None
    lib.cvdvSqueeze.argtypes                  = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvSqueeze.restype                   = None
    lib.cvdvPhaseSquare.argtypes              = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvPhaseSquare.restype               = None
    lib.cvdvPhaseCubic.argtypes               = [ctypes.c_void_p, c_int, c_double]
    lib.cvdvPhaseCubic.restype                = None
    lib.cvdvFtQ2P.argtypes                    = [ctypes.c_void_p, c_int]
    lib.cvdvFtQ2P.restype                     = None
    lib.cvdvQ1Q2Gate.argtypes                 = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvQ1Q2Gate.restype                  = None
    lib.cvdvBeamSplitter.argtypes             = [ctypes.c_void_p, c_int, c_int, c_double]
    lib.cvdvBeamSplitter.restype              = None
    lib.cvdvSwapRegisters.argtypes            = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvSwapRegisters.restype             = None
    lib.cvdvHadamard.argtypes                 = [ctypes.c_void_p, c_int, c_int]
    lib.cvdvHadamard.restype                  = None
    lib.cvdvPauliRotation.argtypes            = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvPauliRotation.restype             = None
    lib.cvdvParity.argtypes                   = [ctypes.c_void_p, c_int]
    lib.cvdvParity.restype                    = None
    lib.cvdvConditionalDisplacement.argtypes  = [ctypes.c_void_p, c_int, c_int, c_int,
                                                  c_double, c_double]
    lib.cvdvConditionalDisplacement.restype   = None
    lib.cvdvConditionalRotation.argtypes      = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalRotation.restype       = None
    lib.cvdvConditionalSqueeze.argtypes       = [ctypes.c_void_p, c_int, c_int, c_int, c_double]
    lib.cvdvConditionalSqueeze.restype        = None
    lib.cvdvConditionalParity.argtypes        = [ctypes.c_void_p, c_int, c_int, c_int]
    lib.cvdvConditionalParity.restype         = None
    lib.cvdvConditionalBeamSplitter.argtypes  = [ctypes.c_void_p, c_int, c_int,
                                                  c_int, c_int, c_double]
    lib.cvdvConditionalBeamSplitter.restype   = None
    lib.cvdvGetNorm.argtypes                  = [ctypes.c_void_p]
    lib.cvdvGetNorm.restype                   = c_double
    lib.cvdvGetWignerFullMode.argtypes        = [ctypes.c_void_p, c_int, POINTER(c_double),
                                                  c_int, c_double, c_double]
    lib.cvdvGetWignerFullMode.restype         = None
    lib.cvdvGetHusimiQFullMode.argtypes       = [ctypes.c_void_p, c_int, POINTER(c_double),
                                                  c_int, c_double, c_double]
    lib.cvdvGetHusimiQFullMode.restype        = None
    lib.cvdvMeasure.argtypes                  = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvMeasure.restype                   = None
    return lib


# ── state helpers ─────────────────────────────────────────────────────────────

TWO_PI = 2.0 * math.pi

def _make_uniform(nq: int) -> np.ndarray:
    dim = 1 << nq
    arr = np.full(dim, 1.0 / math.sqrt(dim), dtype=np.complex128)
    return arr

def _make_coherent(nq: int, alpha_re: float, alpha_im: float) -> np.ndarray:
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

def _upload(lib_cuda, arr: np.ndarray):
    n_bytes = arr.nbytes
    ptr = ctypes.c_void_p()
    lib_cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(n_bytes))
    lib_cuda.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_size_t(n_bytes), ctypes.c_int(1))  # H2D = 1
    return ptr

def _load_cudart_full():
    """Load cudart with malloc/memcpy/free for state upload."""
    for name in ('libcudart.so', 'libcudart.so.13', 'libcudart.so.12',
                 '/usr/local/cuda/lib64/libcudart.so'):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaDeviceSynchronize.restype  = c_int
            lib.cudaMalloc.argtypes            = [ctypes.POINTER(ctypes.c_void_p),
                                                   ctypes.c_size_t]
            lib.cudaMalloc.restype             = c_int
            lib.cudaMemcpy.argtypes            = [ctypes.c_void_p, ctypes.c_void_p,
                                                   ctypes.c_size_t, c_int]
            lib.cudaMemcpy.restype             = c_int
            lib.cudaFree.argtypes              = [ctypes.c_void_p]
            lib.cudaFree.restype               = c_int
            return lib
        except OSError:
            pass
    raise RuntimeError('Could not load libcudart.')


def _make_ctx(lib, cuda):
    qubits = (c_int * 3)(1, 10, 10)
    ctx = lib.cvdvCreate(3, qubits)

    states = [_make_uniform(1), _make_coherent(10, 2.0, 1.0), _make_coherent(10, 1.0, 0.5)]
    dev_ptrs = []
    for s in states:
        dev_ptrs.append(_upload(cuda, s))

    ptr_arr = (ctypes.c_void_p * 3)(*dev_ptrs)
    lib.cvdvInitFromSeparable(ctx, ptr_arr, 3)
    for p in dev_ptrs:
        cuda.cudaFree(p)
    return ctx


# ── timing ────────────────────────────────────────────────────────────────────

def _time_op(cuda, fn, *args):
    """Return median time in microseconds over N_RUNS timed calls."""
    times = []
    for i in range(N_WARMUP + N_RUNS):
        cuda.cudaDeviceSynchronize()
        t0 = perf_counter()
        fn(*args)
        cuda.cudaDeviceSynchronize()
        t1 = perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1e6)
    return float(np.median(times))


def run_bench(lib, cuda):
    """Run all ops, return list of (op_name, time_us)."""
    results = []

    def rec(name, fn, *args):
        ctx = _make_ctx(lib, cuda)
        t = _time_op(cuda, fn, ctx, *args)
        lib.cvdvDestroy(ctx)
        results.append((name, t))

    # single-mode CV
    rec('cvdvDisplacement',            lib.cvdvDisplacement,           c_int(1), c_double(1.0), c_double(0.5))
    rec('cvdvRotation',                lib.cvdvRotation,               c_int(1), c_double(0.3))
    rec('cvdvSqueeze',                 lib.cvdvSqueeze,                c_int(1), c_double(0.5))
    rec('cvdvPhaseSquare',             lib.cvdvPhaseSquare,            c_int(1), c_double(0.1))
    rec('cvdvPhaseCubic',              lib.cvdvPhaseCubic,             c_int(1), c_double(0.05))
    rec('cvdvFtQ2P',                   lib.cvdvFtQ2P,                  c_int(1))

    # two-mode CV
    rec('cvdvQ1Q2Gate',                lib.cvdvQ1Q2Gate,               c_int(1), c_int(2), c_double(0.3))
    rec('cvdvBeamSplitter',            lib.cvdvBeamSplitter,           c_int(1), c_int(2), c_double(0.5))
    rec('cvdvSwapRegisters',           lib.cvdvSwapRegisters,          c_int(1), c_int(2))

    # qubit (DV)
    rec('cvdvHadamard',                lib.cvdvHadamard,               c_int(0), c_int(0))
    rec('cvdvPauliRotation',           lib.cvdvPauliRotation,          c_int(0), c_int(0), c_int(0), c_double(0.5))
    rec('cvdvParity',                  lib.cvdvParity,                 c_int(1))

    # hybrid CV-DV
    rec('cvdvConditionalDisplacement', lib.cvdvConditionalDisplacement,c_int(1), c_int(0), c_int(0), c_double(1.0), c_double(0.5))
    rec('cvdvConditionalRotation',     lib.cvdvConditionalRotation,    c_int(1), c_int(0), c_int(0), c_double(0.3))
    rec('cvdvConditionalSqueeze',      lib.cvdvConditionalSqueeze,     c_int(1), c_int(0), c_int(0), c_double(0.5))
    rec('cvdvConditionalParity',       lib.cvdvConditionalParity,      c_int(1), c_int(0), c_int(0))
    rec('cvdvConditionalBeamSplitter', lib.cvdvConditionalBeamSplitter,c_int(1), c_int(2), c_int(0), c_int(0), c_double(0.5))

    # readout
    rec('cvdvGetNorm',                 lib.cvdvGetNorm)

    N = 51
    buf = (c_double * (N * N))()
    rec('cvdvGetWignerFullMode',       lib.cvdvGetWignerFullMode,      c_int(1), buf, c_int(N), c_double(5.0), c_double(5.0))
    rec('cvdvGetHusimiQFullMode',      lib.cvdvGetHusimiQFullMode,     c_int(1), buf, c_int(N), c_double(5.0), c_double(5.0))

    dim = 1 << 10
    probs = (c_double * dim)()
    rec('cvdvMeasure',                 lib.cvdvMeasure,                c_int(1), probs)

    return results


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('op,time_us\n')
        for op, t in rows:
            f.write(f'{op},{t:.3f}\n')


def read_csv(path):
    out = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            op, t = line.strip().split(',')
            out[op] = float(t)
    return out


# ── comparison table ──────────────────────────────────────────────────────────

def print_table(current_rows, committed=None):
    COL_OP  = 36
    COL_T   = 12

    if committed:
        hdr = (f'{"Op":<{COL_OP}}  {"Before(us)":>{COL_T}}  {"After(us)":>{COL_T}}  {"Speedup":>8}')
    else:
        hdr = (f'{"Op":<{COL_OP}}  {"Time(us)":>{COL_T}}')

    print(f'\n{hdr}')
    print('-' * len(hdr))

    for op, t_new in current_rows:
        if committed:
            t_old = committed.get(op)
            if t_old is not None:
                sp_str = f'{t_old / t_new:.2f}x' if t_new else 'n/a'
                print(f'{op:<{COL_OP}}  {t_old:>{COL_T}.1f}  {t_new:>{COL_T}.1f}  {sp_str:>8}')
            else:
                print(f'{op:<{COL_OP}}  {"(new)":>{COL_T}}  {t_new:>{COL_T}.1f}  {"n/a":>8}')
        else:
            print(f'{op:<{COL_OP}}  {t_new:>{COL_T}.1f}')
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    no_save = '--no-save' in sys.argv

    cuda = _load_cudart_full()
    lib  = _load_cvdv()

    print(f'Benchmarking C API  ({N_WARMUP} warm-up + {N_RUNS} timed runs, reporting median)')
    results = run_bench(lib, cuda)

    if not no_save:
        write_csv(CURRENT_CSV, results)
        print(f'Written: {CURRENT_CSV}')

    committed = None
    if os.path.exists(COMMITTED_CSV):
        committed = read_csv(COMMITTED_CSV)
    else:
        print('No committed baseline — showing absolute times.')

    print_table(results, committed)


if __name__ == '__main__':
    main()
