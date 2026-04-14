"""
benchmarks/wigner_husimi/profile_husimi.py

Profiling harness for Husimi kernels.
Run with:
  python benchmarks/wigner_husimi/profile_husimi.py
or under ncu:
  ncu --target-processes all --launch-skip 3 --launch-count 1 \
      --section SpeedOfLight --section Occupancy \
      --section MemoryWorkloadAnalysis --section SchedulerStatistics \
      -o benchmarks/wigner_husimi/current/ncu_targeted \
      python benchmarks/wigner_husimi/profile_husimi.py
"""

import os
import ctypes
import math
import time
import numpy as np
from ctypes import c_int, c_double, c_void_p, c_size_t, POINTER

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD_DIR = os.path.join(REPO_DIR, "build")

N_WARMUP = 3
N_RUNS = 20
TWO_PI = 2.0 * math.pi


def _make_nvtx():
    for name in ("libnvToolsExt.so", "libnvToolsExt.so.1", "/usr/local/cuda/lib64/libnvToolsExt.so"):
        try:
            lib = ctypes.CDLL(name)
            lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
            lib.nvtxRangePushA.restype = ctypes.c_int
            lib.nvtxRangePop.argtypes = []
            lib.nvtxRangePop.restype = ctypes.c_int
            return lib
        except OSError:
            pass
    return None


_nvtx = _make_nvtx()


def nvtx_push(name: str):
    if _nvtx:
        _nvtx.nvtxRangePushA(name.encode())


def nvtx_pop():
    if _nvtx:
        _nvtx.nvtxRangePop()


def _load_cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaDeviceSynchronize.argtypes = []
            lib.cudaDeviceSynchronize.restype = c_int
            lib.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t]
            lib.cudaMalloc.restype = c_int
            lib.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
            lib.cudaMemcpy.restype = c_int
            lib.cudaFree.argtypes = [c_void_p]
            lib.cudaFree.restype = c_int
            return lib
        except OSError:
            pass
    raise RuntimeError("Could not load libcudart")


def _load_cvdv():
    lib_path = os.path.join(BUILD_DIR, "libcvdv.so")
    if not os.path.exists(lib_path):
        raise RuntimeError(f"Library not found: {lib_path}. Run `make build`.")
    lib = ctypes.CDLL(lib_path)
    lib.cvdvCreate.argtypes = [c_int, POINTER(c_int)]
    lib.cvdvCreate.restype = c_void_p
    lib.cvdvDestroy.argtypes = [c_void_p]
    lib.cvdvDestroy.restype = None
    lib.cvdvInitFromSeparable.argtypes = [c_void_p, POINTER(c_void_p), c_int]
    lib.cvdvInitFromSeparable.restype = None
    lib.cvdvGetHusimiQOverlap.argtypes = [c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetHusimiQOverlap.restype = None
    lib.cvdvGetHusimiQWigner.argtypes = [c_void_p, c_int, POINTER(c_double)]
    lib.cvdvGetHusimiQWigner.restype = None
    return lib


def _make_coherent(nq: int, alpha_re: float = 0.8, alpha_im: float = 0.3) -> np.ndarray:
    dim = 1 << nq
    dx = math.sqrt(TWO_PI / dim)
    q0 = math.sqrt(2.0) * alpha_re
    p0 = math.sqrt(2.0) * alpha_im
    ks = np.arange(dim, dtype=np.float64)
    x = (ks - (dim - 1) * 0.5) * dx
    env = np.exp(-0.5 * (x - q0) ** 2)
    psi = env * np.exp(1j * p0 * x)
    psi /= np.linalg.norm(psi)
    return psi.astype(np.complex128)


def _upload(cuda, arr: np.ndarray):
    ptr = c_void_p()
    cuda.cudaMalloc(ctypes.byref(ptr), c_size_t(arr.nbytes))
    cuda.cudaMemcpy(ptr, arr.ctypes.data_as(c_void_p), c_size_t(arr.nbytes), c_int(1))
    return ptr


def _make_ctx(lib, cuda, qubits_list):
    n = len(qubits_list)
    qarr = (c_int * n)(*qubits_list)
    ctx = lib.cvdvCreate(n, qarr)
    states = [_make_coherent(nq) for nq in qubits_list]
    dev_ptrs = [_upload(cuda, s) for s in states]
    ptr_arr = (c_void_p * n)(*dev_ptrs)
    lib.cvdvInitFromSeparable(ctx, ptr_arr, n)
    for p in dev_ptrs:
        cuda.cudaFree(p)
    return ctx


def _time_breakdown(cuda, fn, *args):
    t_fn, t_sb, t_sa = [], [], []
    for i in range(N_WARMUP + N_RUNS):
        t0 = time.perf_counter()
        cuda.cudaDeviceSynchronize()
        t1 = time.perf_counter()

        nvtx_push(fn.__name__)
        fn(*args)
        nvtx_pop()
        t2 = time.perf_counter()

        cuda.cudaDeviceSynchronize()
        t3 = time.perf_counter()

        if i >= N_WARMUP:
            t_sb.append((t1 - t0) * 1e6)
            t_fn.append((t2 - t1) * 1e6)
            t_sa.append((t3 - t2) * 1e6)

    return np.array(t_fn), np.array(t_sb), np.array(t_sa)


def _report(label, t_fn, t_sb, t_sa):
    total = t_fn + t_sb + t_sa
    print(f"\n  {label}")
    print(f"    fn (kernel+fft+D2H):   median={np.median(t_fn):7.1f} us   min={np.min(t_fn):7.1f}   max={np.max(t_fn):7.1f}")
    print(f"    sync-before overhead:  median={np.median(t_sb):7.1f} us")
    print(f"    sync-after overhead:   median={np.median(t_sa):7.1f} us")
    print(f"    total wall time:       median={np.median(total):7.1f} us")


def main():
    cuda = _load_cudart()
    lib = _load_cvdv()

    # target profile size: dim=128, slices=128
    qubits = [7, 7]
    reg_idx = 1
    ctx = _make_ctx(lib, cuda, qubits)
    dim = 1 << qubits[reg_idx]
    out = np.zeros(dim * dim, dtype=np.float64)
    out_ptr = out.ctypes.data_as(POINTER(c_double))

    def run_overlap():
        lib.cvdvGetHusimiQOverlap(ctx, c_int(reg_idx), out_ptr)

    def run_wigner():
        lib.cvdvGetHusimiQWigner(ctx, c_int(reg_idx), out_ptr)

    run_overlap.__name__ = "cvdvGetHusimiQOverlap"
    run_wigner.__name__ = "cvdvGetHusimiQWigner"

    print(f"\nHusimi profile harness ({N_WARMUP} warm-up + {N_RUNS} timed runs)")
    print(f"Context qubits={qubits}, target reg={reg_idx}, dim={dim}")

    t_fn_o, t_sb_o, t_sa_o = _time_breakdown(cuda, run_overlap)
    _report("Overlap full-batch", t_fn_o, t_sb_o, t_sa_o)

    t_fn_w, t_sb_w, t_sa_w = _time_breakdown(cuda, run_wigner)
    _report("Wigner FFT2", t_fn_w, t_sb_w, t_sa_w)

    lib.cvdvDestroy(ctx)

    print("\nDone. For targeted NCU: make profile-husimi")


if __name__ == "__main__":
    main()
