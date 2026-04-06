#!/bin/bash
# Promote current API timing results to committed baseline.
#
# Usage:
#   ./benchmarks/api_timing/save.sh
set -e
REPO="$(dirname "$0")/../.."
pushd "$REPO" > /dev/null

cp benchmarks/api_timing/current/bench.csv benchmarks/api_timing/committed/bench.csv
echo "Saved: benchmarks/api_timing/current/bench.csv → benchmarks/api_timing/committed/bench.csv"

echo "Commit benchmarks/api_timing/committed/ to record this baseline."

popd > /dev/null
