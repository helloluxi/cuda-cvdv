#!/bin/bash
# Promote current profiling results to committed baseline.
#
# Usage:
#   ./profiling/save.sh
set -e
REPO="$(dirname "$0")/.."
pushd "$REPO" > /dev/null

cp profiling/current/results.csv profiling/committed/results.csv
echo "Saved: profiling/current/results.csv → profiling/committed/results.csv"
echo "Commit profiling/committed/results.csv to record this as the new baseline."

popd > /dev/null
