#!/bin/bash
# Promote current profiling results to committed baseline.
#
# Usage:
#   ./benchmarks/kernel_profiling/save.sh
set -e
REPO="$(dirname "$0")/../.."
pushd "$REPO" > /dev/null

cp benchmarks/kernel_profiling/current/results.csv benchmarks/kernel_profiling/committed/results.csv
echo "Saved: benchmarks/kernel_profiling/current/results.csv → benchmarks/kernel_profiling/committed/results.csv"

echo "Commit benchmarks/kernel_profiling/committed/ to record this baseline."

popd > /dev/null
