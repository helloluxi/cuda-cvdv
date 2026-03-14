#!/bin/bash
# Nsight profiling pipeline for CVDV kernels.
#
# Usage:
#   ./profiling/run.sh    # profile → speedup table
#   ./profiling/save.sh   # promote current → committed baseline
set -e
REPO="$(dirname "$0")/.."
pushd "$REPO" > /dev/null

NCU_METRICS="gpu__time_duration.sum,dram__bytes_read.sum,sm__warps_active.avg.pct_of_peak_suspended_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l2tex__t_sector_hit_rate.pct"

WORKLOAD="
import os
import sys; sys.path.insert(0, '.')
from src import CVDV
from src.separable import SeparableState

# reg 0: 1 qubit (DV)  |  reg 1: 10 qubits (CV)  |  reg 2: 10 qubits (CV)
regs = [1, 10, 10]
sep = SeparableState(regs)
sep.setUniform(0); sep.setCoherent(1, 2+1j); sep.setCoherent(2, 1+0.5j)
sim = CVDV(regs); sim.initStateVector(sep)

# --- single-mode CV ---
sim.d(1, 1+0.5j)
sim.r(1, 0.3)
sim.s(1, 0.5)
sim.sheer(1, 0.1)
sim.phaseCubic(1, 0.05)
sim.p(1)
sim.ftQ2P(1)
sim.ftP2Q(1)

# --- two-mode CV ---
sim.q1q2(1, 2, 0.3)
sim.bs(1, 2, 0.5)
sim.swap(1, 2)

# --- qubit (DV) ---
sim.h(0, 0)
sim.rx(0, 0, 0.5)   # cvdvPauliRotation — same kernel for ry/rz/x/y/z

# --- hybrid ---
sim.cd(1, 0, 0, 1+0.5j)
sim.cr(1, 0, 0, 0.3)
sim.cs(1, 0, 0, 0.5)
sim.cp(1, 0, 0)
sim.cbs(1, 2, 0, 0, 0.5)

# --- readout ---
sim.getNorm()
sim.getWignerFullMode(1, wignerN=51)
sim.getHusimiQFullMode(1, qN=51)

# --- measurement (destructive, last) ---
sim.m(1)
"

echo "--- ncu profile → profiling/current/results.ncu-rep ---"
ncu --metrics "$NCU_METRICS" --target-processes all -f \
    -o profiling/current/results \
    python -c "$WORKLOAD"

echo "--- export CSV ---"
ncu --import profiling/current/results.ncu-rep --csv \
    > profiling/current/results.csv

echo "--- compare ---"
python profiling/compare.py | tee profiling/current/compare.log

popd > /dev/null
