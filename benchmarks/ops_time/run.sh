#!/bin/bash
set -e
pushd "$(dirname "$0")" > /dev/null
source ../../venv/bin/activate
rm -f results/*.png results/*.json && python bench_ops.py
popd > /dev/null
