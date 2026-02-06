#!/bin/bash
set -e
pushd "$(dirname "$0")" > /dev/null
source ../venv/bin/activate
rm -f results/* && python run_benchmarks.py --dv-qubits 4 --cvdv-cv-qubits 10 12 14 --bosonic-cv-qubits 5 6 7 --runs 10
popd > /dev/null