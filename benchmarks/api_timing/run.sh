#!/bin/bash
# C API timing benchmark pipeline.
#
# Usage:
#   ./benchmarks/api_timing/run.sh
set -e
REPO="$(dirname "$0")/../.."
pushd "$REPO" > /dev/null

echo "[1/1] benching C API  →  benchmarks/api_timing/current/bench.csv"
LD_LIBRARY_PATH="$REPO/build:$LD_LIBRARY_PATH" \
python benchmarks/api_timing/bench.py | tee benchmarks/api_timing/current/bench.log

popd > /dev/null
