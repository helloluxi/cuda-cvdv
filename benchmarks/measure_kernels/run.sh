#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python benchmarks/measure_kernels/bench_measure.py "$@"
