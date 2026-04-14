"""
benchmarks/measure_kernels/profile_measure.py

Profiling harness for kernelAbsSquareReduce.

Runnable standalone (Python end-to-end breakdown):
    python benchmarks/measure_kernels/profile_measure.py

Runnable under ncu (targeted sections, skip 3 warm-ups):
    ncu --target-processes all \\
        --launch-skip 3 --launch-count 1 \\
        --section SpeedOfLight --section Occupancy \\
        --section MemoryWorkloadAnalysis --section SchedulerStatistics \\
        -o benchmarks/measure_kernels/current/ncu_targeted \\
        python benchmarks/measure_kernels/profile_measure.py

Runnable under nsys (full CUDA timeline):
    nsys profile --gpu-metrics-devices=all \\
        -o benchmarks/measure_kernels/current/nsys_trace \\
        python benchmarks/measure_kernels/profile_measure.py

Context: [1, 10, 10] registers → totalSize = 2^21, outSize = 1024 (reg1).
This matches the todo.md baseline config (217 μs cuTENSOR, current target).
"""

import sys, os, ctypes, math, time
import numpy as np
from ctypes import c_int, c_double, POINTER, c_void_p, c_size_t

REPO_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD_DIR = os.path.join(REPO_DIR, 'build')
TWO_PI    = 2.0 * math.pi

N_WARMUP = 3
N_RUNS   = 20


# ── NVTX (optional, used by nsys for timeline annotations) ───────────────────

def _make_nvtx():
    for name in ('libnvToolsExt.so', 'libnvToolsExt.so.1',
                 '/usr/local/cuda/lib64/libnvToolsExt.so'):
        try:
            lib = ctypes.CDLL(name)
            lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
            lib.nvtxRangePushA.restype  = ctypes.c_int
            lib.nvtxRangePop.argtypes   = []
            lib.nvtxRangePop.restype    = ctypes.c_int
            return lib
        except OSError:
            pass
    return None

_nvtx = _make_nvtx()

def nvtx_push(name: str):
    if _nvtx: _nvtx.nvtxRangePushA(name.encode())

def nvtx_pop():
    if _nvtx: _nvtx.nvtxRangePop()


# ── Library loading ───────────────────────────────────────────────────────────

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
        raise RuntimeError(f'Library not found: {lib_path}. Run `make build`.')
    lib = ctypes.CDLL(lib_path)
    lib.cvdvCreate.argtypes            = [c_int, POINTER(c_int)]
    lib.cvdvCreate.restype             = c_void_p
    lib.cvdvDestroy.argtypes           = [c_void_p]
    lib.cvdvDestroy.restype            = None
    lib.cvdvInitFromSeparable.argtypes = [c_void_p, POINTER(c_void_p), c_int]
    lib.cvdvInitFromSeparable.restype  = None
    lib.cvdvMeasureMultiple.argtypes   = [c_void_p, POINTER(c_int), c_int, POINTER(c_double)]
    lib.cvdvMeasureMultiple.restype    = None
    lib.cvdvGetNorm.argtypes           = [c_void_p]
    lib.cvdvGetNorm.restype            = c_double
    lib.cvdvMeasureMultipleCT.argtypes = [c_void_p, POINTER(c_int), c_int, POINTER(c_double)]
    lib.cvdvMeasureMultipleCT.restype  = None
    return lib


# ── State helpers ─────────────────────────────────────────────────────────────

def _make_coherent(nq: int, alpha: complex = 1.0) -> np.ndarray:
    dim  = 1 << nq
    dx   = math.sqrt(TWO_PI / dim)
    ks   = np.arange(dim, dtype=np.float64)
    x    = (ks - (dim - 1) * 0.5) * dx
    psi  = np.exp(-0.5 * (x - math.sqrt(2) * alpha.real) ** 2)
    psi  = psi * np.exp(1j * math.sqrt(2) * alpha.imag * x)
    psi /= np.linalg.norm(psi)
    return psi.astype(np.complex128)


def _upload(cuda, arr: np.ndarray) -> c_void_p:
    ptr = c_void_p()
    cuda.cudaMalloc(ctypes.byref(ptr), c_size_t(arr.nbytes))
    cuda.cudaMemcpy(ptr, arr.ctypes.data_as(c_void_p),
                    c_size_t(arr.nbytes), c_int(1))  # H2D = 1
    return ptr


def _make_ctx(lib, cuda, qubits_list):
    n = len(qubits_list)
    qarr = (c_int * n)(*qubits_list)
    ctx  = lib.cvdvCreate(n, qarr)

    states   = [_make_coherent(nq) for nq in qubits_list]
    dev_ptrs = [_upload(cuda, s) for s in states]
    ptr_arr  = (c_void_p * n)(*dev_ptrs)
    lib.cvdvInitFromSeparable(ctx, ptr_arr, n)
    for p in dev_ptrs:
        cuda.cudaFree(p)
    return ctx


# ── Python end-to-end timer ───────────────────────────────────────────────────

def _time_breakdown(cuda, fn, *args, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """
    Returns (t_fn_us, t_sync_before_us, t_sync_after_us) arrays.

    The cvdvMeasureMultiple / cvdvGetNorm calls include an internal synchronous
    cudaMemcpy D2H, so the function itself blocks until GPU work completes.
    t_fn includes: cudaMemset + kernel + D2H memcpy.
    t_sync_before/-after should be ~0 μs after warm-up.
    """
    t_fn, t_sb, t_sa = [], [], []
    for i in range(n_warmup + n_runs):
        t0 = time.perf_counter()
        cuda.cudaDeviceSynchronize()
        t1 = time.perf_counter()

        nvtx_push('cvdvMeasureMultiple' if fn.__name__ != 'cvdvGetNorm' else 'cvdvGetNorm')
        fn(*args)
        nvtx_pop()
        t2 = time.perf_counter()

        cuda.cudaDeviceSynchronize()
        t3 = time.perf_counter()

        if i >= n_warmup:
            t_sb.append((t1 - t0) * 1e6)
            t_fn.append((t2 - t1) * 1e6)
            t_sa.append((t3 - t2) * 1e6)

    return np.array(t_fn), np.array(t_sb), np.array(t_sa)


def _report(label, t_fn, t_sb, t_sa):
    total = t_fn + t_sb + t_sa
    print(f'\n  {label}')
    print(f'    fn (kernel+memset+D2H):  median={np.median(t_fn):7.1f} μs'
          f'   min={np.min(t_fn):7.1f}   max={np.max(t_fn):7.1f}')
    print(f'    sync-before (overhead):  median={np.median(t_sb):7.1f} μs')
    print(f'    sync-after  (overhead):  median={np.median(t_sa):7.1f} μs')
    print(f'    total wall time:         median={np.median(total):7.1f} μs')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    use_cutensor = '--cutensor' in sys.argv

    cuda = _load_cudart()
    lib  = _load_cvdv()

    # Baseline context: [1, 10, 10] → totalSize = 2^21, outSize = 1024 (reg1)
    QUBITS = [1, 10, 10]
    ctx    = _make_ctx(lib, cuda, QUBITS)

    # ── MeasureMultiple: reg1 (outSize = 1024) ──────────────────────────────
    probs = np.zeros(1024, dtype=np.float64)
    regs  = (c_int * 1)(1)

    if use_cutensor:
        def measure_fn():
            lib.cvdvMeasureMultipleCT(ctx, regs, 1,
                                      probs.ctypes.data_as(POINTER(c_double)))
        measure_fn.__name__ = 'cvdvMeasureMultipleCT'
        label = 'cvdvMeasureMultipleCT reg1, outSize=1024 (cuTENSOR)'
    else:
        def measure_fn():
            lib.cvdvMeasureMultiple(ctx, regs, 1,
                                    probs.ctypes.data_as(POINTER(c_double)))
        measure_fn.__name__ = 'cvdvMeasureMultiple'
        label = 'cvdvMeasureMultiple  reg1, outSize=1024'

    backend = 'cuTENSOR' if use_cutensor else 'custom kernel'
    print(f'\nProfile harness  ({N_WARMUP} warm-up + {N_RUNS} timed runs)  [{backend}]')
    print(f'Context: {QUBITS} qubits, totalSize=2^{sum(QUBITS)}, outSize=1024')
    print(f'Pass --cutensor to switch backend')
    print()

    nvtx_push('warmup-measure')
    t_fn, t_sb, t_sa = _time_breakdown(cuda, measure_fn)
    nvtx_pop()
    _report(label, t_fn, t_sb, t_sa)

    # ── GetNorm (outSize=1 path, for comparison) ──────────────────────────────
    def norm_fn():
        lib.cvdvGetNorm(ctx)

    norm_fn.__name__ = 'cvdvGetNorm'

    nvtx_push('warmup-norm')
    t_fn_n, t_sb_n, t_sa_n = _time_breakdown(cuda, norm_fn)
    nvtx_pop()
    _report('cvdvGetNorm          (outSize=1, blockReduceSum)', t_fn_n, t_sb_n, t_sa_n)

    # ── Additional config: reg0 (outSize=2, tiny outSize) ────────────────────
    probs4 = np.zeros(4, dtype=np.float64)
    regs2  = (c_int * 1)(0)

    if use_cutensor:
        def measure_small():
            lib.cvdvMeasureMultipleCT(ctx, regs2, 1,
                                      probs4.ctypes.data_as(POINTER(c_double)))
    else:
        def measure_small():
            lib.cvdvMeasureMultiple(ctx, regs2, 1,
                                    probs4.ctypes.data_as(POINTER(c_double)))

    measure_small.__name__ = 'cvdvMeasureMultiple'

    t_fn_s, t_sb_s, t_sa_s = _time_breakdown(cuda, measure_small)
    _report(f'{"CT" if use_cutensor else ""}MeasureMultiple  reg0, outSize=2', t_fn_s, t_sb_s, t_sa_s)

    lib.cvdvDestroy(ctx)

    print()
    print('Done. To profile under ncu:')
    print(f'  make profile-measure')
    print('To profile under nsys:')
    print(f'  make nsys-profile-measure')


if __name__ == '__main__':
    main()
