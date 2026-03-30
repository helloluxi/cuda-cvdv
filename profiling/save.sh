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

if [ -f profiling/current/bench.csv ]; then
    cp profiling/current/bench.csv profiling/committed/bench.csv
    echo "Saved: profiling/current/bench.csv → profiling/committed/bench.csv"
fi

echo "Commit profiling/committed/ to record these as the new baseline."

popd > /dev/null
