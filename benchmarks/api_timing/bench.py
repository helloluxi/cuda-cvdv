"""
benchmarks/api_timing/bench.py
Python-level C API timing for CVDV, with PyTorch-CUDA baseline column.

Times each C API call using time.perf_counter with cudaDeviceSynchronize
fencing, N=10 timed runs after 1 warm-up, reports median time per op.
Also times the equivalent torch-cuda operation for each gate (where applicable)
and reports a "vs Torch" speedup column.

Usage:
  python benchmarks/api_timing/bench.py            # time + compare vs committed baseline
  python benchmarks/api_timing/bench.py --no-save  # time only, skip writing CSV
  python benchmarks/api_timing/bench.py --no-torch # skip torch-cuda baseline
"""

import sys, os, ctypes, math, shutil
import numpy as np
from ctypes import c_int, c_double, POINTER
from time import perf_counter

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.dirname(os.path.dirname(BENCH_DIR))
BUILD_DIR     = os.path.join(REPO_DIR, 'build')
CURRENT_CSV   = os.path.join(BENCH_DIR, 'current',   'bench.csv')
COMMITTED_CSV = os.path.join(BENCH_DIR, 'committed', 'bench.csv')

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
    lib.cvdvGetPhotonNumber.argtypes          = [ctypes.c_void_p, c_int]
    lib.cvdvGetPhotonNumber.restype           = c_double
    lib.cvdvFtP2Q.argtypes                    = [ctypes.c_void_p, c_int]
    lib.cvdvFtP2Q.restype                     = None
    lib.cvdvGetWigner.argtypes                = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetWigner.restype                 = None
    lib.cvdvGetHusimiQ.argtypes               = [ctypes.c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetHusimiQ.restype                = None
    lib.cvdvMeasureMultiple.argtypes          = [ctypes.c_void_p, POINTER(c_int), c_int, POINTER(c_double)]
    lib.cvdvMeasureMultiple.restype           = None
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


# ── torch bench ──────────────────────────────────────────────────────────────

def _make_torch_sim():
    """Create a CVDV torch-cuda sim matching _make_ctx's registers."""
    sys.path.insert(0, REPO_DIR)
    from src.torchCvdv import TorchCvdv
    from src.separable import SeparableState
    sim = TorchCvdv([1, 10, 10], device='cuda')
    sep = SeparableState([1, 10, 10])
    sep.setUniform(0)
    sep.setCoherent(1, 2.0 + 1.0j)
    sep.setCoherent(2, 1.0 + 0.5j)
    sim.initStateVector(sep)
    return sim


def _time_torch_op(fn, *args):
    """Return median time in microseconds for a torch-cuda op."""
    import torch
    times = []
    for i in range(N_WARMUP + N_RUNS):
        torch.cuda.synchronize()
        t0 = perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = perf_counter()
        if i >= N_WARMUP:
            times.append((t1 - t0) * 1e6)
    return float(np.median(times))


def run_torch_bench():
    """Time each C API op's PyTorch-CUDA equivalent. Returns {op: time_us}."""
    results = {}

    def rec(c_name, method_name, *args):
        sim = _make_torch_sim()
        t = _time_torch_op(getattr(sim, method_name), *args)
        del sim
        results[c_name] = t

    rec('cvdvDisplacement',            'd',              1, 1.0+0.5j)
    rec('cvdvRotation',                'r',              1, 0.3)
    rec('cvdvSqueeze',                 's',              1, 0.5)
    rec('cvdvPhaseSquare',             'sheer',          1, 0.1)
    rec('cvdvPhaseCubic',              'phaseCubic',     1, 0.05)
    rec('cvdvFtQ2P',                   'ftQ2P',          1)
    rec('cvdvFtP2Q',                   'ftP2Q',          1)
    rec('cvdvQ1Q2Gate',                'q1q2',           1, 2, 0.3)
    rec('cvdvBeamSplitter',            'bs',             1, 2, 0.5)
    rec('cvdvSwapRegisters',           'swap',           1, 2)
    rec('cvdvHadamard',                'h',              0, 0)
    rec('cvdvPauliRotation',           'rx',             0, 0, 0.5)
    rec('cvdvParity',                  'p',              1)
    rec('cvdvConditionalDisplacement', 'cd',             1, 0, 0, 1.0+0.5j)
    rec('cvdvConditionalRotation',     'cr',             1, 0, 0, 0.3)
    rec('cvdvConditionalSqueeze',      'cs',             1, 0, 0, 0.5)
    rec('cvdvConditionalParity',       'cp',             1, 0, 0)
    rec('cvdvConditionalBeamSplitter', 'cbs',            1, 2, 0, 0, 0.5)
    rec('cvdvGetNorm',                 'm')
    rec('cvdvGetPhotonNumber',         'getPhotonNumber', 1)
    rec('cvdvGetWigner',               'getWigner',       1)
    rec('cvdvGetHusimiQ',              'getHusimiQ',      1)
    rec('cvdvMeasureMultiple',         'm',               1)

    return results


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
    rec('cvdvFtP2Q',                   lib.cvdvFtP2Q,                  c_int(1))

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
    rec('cvdvGetPhotonNumber',         lib.cvdvGetPhotonNumber,        c_int(1))

    cv_dim = 1 << 10
    buf = (c_double * (cv_dim * cv_dim))()
    rec('cvdvGetWigner',               lib.cvdvGetWigner,              c_int(1), buf)
    rec('cvdvGetHusimiQ',              lib.cvdvGetHusimiQ,             c_int(1), buf)

    dim = 1 << 10
    probs = (c_double * dim)()
    regs_c = (c_int * 1)(1)
    rec('cvdvMeasureMultiple',         lib.cvdvMeasureMultiple,        regs_c, c_int(1), probs)

    return results


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def write_csv(path, rows, torch_dict=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('op,time_us,torch_us\n')
        for op, t in rows:
            tu = torch_dict.get(op) if torch_dict else None
            torch_str = f'{tu:.3f}' if tu is not None else ''
            f.write(f'{op},{t:.3f},{torch_str}\n')


def read_csv(path):
    """Return {op: time_us} (reads 2nd column only for backward compat)."""
    out = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(',')
            out[parts[0]] = float(parts[1])
    return out


def read_torch_from_csv(path):
    """Return {op: torch_us} from the 3rd column, or {} if column absent."""
    out = {}
    try:
        with open(path) as f:
            header = next(f).strip().split(',')
            if len(header) < 3 or header[2] != 'torch_us':
                return {}
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2]:
                    out[parts[0]] = float(parts[2])
    except OSError:
        pass
    return out


# ── comparison table ──────────────────────────────────────────────────────────

def print_table(current_rows, committed=None, torch_dict=None):
    COL_OP  = 36
    COL_T   = 12

    if committed:
        hdr = (f'{"Op":<{COL_OP}}  {"Before(us)":>{COL_T}}  {"After(us)":>{COL_T}}  {"Speedup":>8}')
    else:
        hdr = (f'{"Op":<{COL_OP}}  {"Time(us)":>{COL_T}}')
    if torch_dict:
        hdr += f'  {"Torch(us)":>{COL_T}}  {"vs Torch":>8}'

    print(f'\n{hdr}')
    print('-' * len(hdr))

    for op, t_new in current_rows:
        if committed:
            t_old = committed.get(op)
            if t_old is not None:
                sp_str = f'{t_old / t_new:.2f}x' if t_new else 'n/a'
                row = f'{op:<{COL_OP}}  {t_old:>{COL_T}.1f}  {t_new:>{COL_T}.1f}  {sp_str:>8}'
            else:
                row = f'{op:<{COL_OP}}  {"(new)":>{COL_T}}  {t_new:>{COL_T}.1f}  {"n/a":>8}'
        else:
            row = f'{op:<{COL_OP}}  {t_new:>{COL_T}.1f}'
        if torch_dict:
            t_torch = torch_dict.get(op)
            if t_torch is not None:
                vs_str = f'{t_torch / t_new:.2f}x' if t_new else 'n/a'
                row += f'  {t_torch:>{COL_T}.1f}  {vs_str:>8}'
            else:
                row += f'  {"n/a":>{COL_T}}  {"n/a":>8}'
        print(row)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    no_save   = '--no-save'   in sys.argv
    no_torch  = '--no-torch'  in sys.argv

    cuda = _load_cudart_full()
    lib  = _load_cvdv()

    print(f'Benchmarking C API  ({N_WARMUP} warm-up + {N_RUNS} timed runs, reporting median)')
    results = run_bench(lib, cuda)

    torch_dict = None
    if not no_torch:
        try:
            import torch  # noqa: F401
            print(f'Benchmarking PyTorch-CUDA  ({N_WARMUP} warm-up + {N_RUNS} timed runs, reporting median)')
            torch_dict = run_torch_bench()
        except Exception as e:
            print(f'PyTorch bench skipped: {e}')

    if not no_save:
        write_csv(CURRENT_CSV, results, torch_dict)
        print(f'Written: {CURRENT_CSV}')

    committed = None
    if os.path.exists(COMMITTED_CSV):
        committed = read_csv(COMMITTED_CSV)
    else:
        print('No committed baseline — showing absolute times.')

    print_table(results, committed, torch_dict)


if __name__ == '__main__':
    main()
