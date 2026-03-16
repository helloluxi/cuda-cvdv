#!/bin/bash
# Nsight profiling pipeline for CVDV kernels.
#
# Usage:
#   ./profiling/run.sh              # profile all kernels → speedup table
#   ./profiling/run.sh displacement # profile only kernels matching regex
#   ./profiling/save.sh             # promote current → committed baseline
set -e
REPO="$(dirname "$0")/.."
pushd "$REPO" > /dev/null

NVCC=/usr/local/cuda-13.1/bin/nvcc
WORKLOAD_SRC=profiling/workload.cu
WORKLOAD_BIN=profiling/workload

NCU_METRICS="gpu__time_duration.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active"

echo "[0/3] building workload  →  $WORKLOAD_BIN"
$NVCC -O2 -std=c++17 \
    "$WORKLOAD_SRC" \
    -o "$WORKLOAD_BIN" \
    -L"$REPO/build" -lcvdv -lcufft \
    -Xlinker -rpath,"$REPO/build"

KERNEL_FILTER=""
[ -n "$1" ] && KERNEL_FILTER="--kernel-name $1"

echo "[1/3] profiling  →  profiling/current/results.ncu-rep"
LD_LIBRARY_PATH="$REPO/build:$LD_LIBRARY_PATH" \
ncu --metrics "$NCU_METRICS" --target-processes all -f $KERNEL_FILTER \
    -o profiling/current/results \
    "$WORKLOAD_BIN" > /dev/null

echo "[2/3] exporting  →  profiling/current/results.csv"
ncu --import profiling/current/results.ncu-rep --csv > profiling/current/results.csv

echo "[3/3] comparing"
python profiling/compare.py | tee profiling/current/compare.log

popd > /dev/null
