"""
benchmarks/wigner_husimi/bench_husimi.py

Compare Husimi implementations:
  1) cvdvGetHusimiQOverlap (full-batch overlap)
  2) cvdvGetHusimiQWigner  (Wigner -> FFT2 Gaussian -> IFFT2)

Usage:
  python benchmarks/wigner_husimi/bench_husimi.py
  python benchmarks/wigner_husimi/bench_husimi.py --runs=20
"""

import os
import sys
import math
import ctypes
import numpy as np
from time import perf_counter
from ctypes import c_int, c_double, c_void_p, c_size_t, POINTER

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD_DIR = os.path.join(REPO_DIR, "build")

N_WARMUP = 2
N_RUNS = 10
TWO_PI = 2.0 * math.pi


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


def _time_op(cuda, fn, *args):
    vals = []
    for i in range(N_WARMUP + N_RUNS):
        cuda.cudaDeviceSynchronize()
        t0 = perf_counter()
        fn(*args)
        cuda.cudaDeviceSynchronize()
        t1 = perf_counter()
        if i >= N_WARMUP:
            vals.append((t1 - t0) * 1e6)
    return float(np.median(vals)), float(np.min(vals)), float(np.max(vals))


def main():
    global N_RUNS
    for arg in sys.argv[1:]:
        if arg.startswith("--runs="):
            N_RUNS = int(arg.split("=", 1)[1])

    cuda = _load_cudart()
    lib = _load_cvdv()

    # (label, qubits_per_reg, target_reg)
    configs = [
        ("dim=64  slices=64", [6, 6], 1),
        ("dim=128 slices=128", [7, 7], 1),
        ("dim=256 slices=256", [8, 8], 1),
    ]

    print(f"\nHusimi benchmark ({N_WARMUP} warm-up + {N_RUNS} timed runs)")
    print("Compare overlap(full-batch) vs wigner_fft2 on same state")
    print()
    print(f"{'Config':<20}  {'Method':<18}  {'Median(us)':>12}  {'Min(us)':>12}  {'Max(us)':>12}")
    print("-" * 82)

    for label, qubits, reg_idx in configs:
        ctx = _make_ctx(lib, cuda, qubits)
        dim = 1 << qubits[reg_idx]
        out = np.zeros(dim * dim, dtype=np.float64)
        out_ptr = out.ctypes.data_as(POINTER(c_double))

        med_o, min_o, max_o = _time_op(cuda, lib.cvdvGetHusimiQOverlap, ctx, c_int(reg_idx), out_ptr)
        med_w, min_w, max_w = _time_op(cuda, lib.cvdvGetHusimiQWigner, ctx, c_int(reg_idx), out_ptr)

        print(f"{label:<20}  {'overlap_fullbatch':<18}  {med_o:>12.1f}  {min_o:>12.1f}  {max_o:>12.1f}")
        print(f"{label:<20}  {'wigner_fft2':<18}  {med_w:>12.1f}  {min_w:>12.1f}  {max_w:>12.1f}")

        speedup = med_w / med_o if med_o > 0 else float("inf")
        winner = "overlap_fullbatch" if med_o <= med_w else "wigner_fft2"
        print(f"{'':<20}  {'winner':<18}  {winner}  (ratio wigner/overlap={speedup:.2f}x)")
        print()

        lib.cvdvDestroy(ctx)


if __name__ == "__main__":
    main()
