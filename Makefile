.PHONY: build test clean help bench bench-state-transfer bench-api bench-kernel \
       bench-measure bench-measure-ct profile-measure nsys-profile-measure \
       bench-husimi profile-husimi save-bench

help:
	@echo "CVDV Quantum Simulator - Build & Test Commands"
	@echo ""
	@echo "  make build    - Compile CUDA library"
	@echo "  make test     - Build and run all tests"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make bench-state-transfer - Run state-transfer benchmark"
	@echo "  make bench-api           - Run C API timing benchmark"
	@echo "  make bench-kernel        - Run Nsight kernel profiling"
	@echo "  make bench-measure            - Run measure-kernel benchmark (custom)"
	@echo "  make bench-measure-ct         - Run measure-kernel benchmark (cuTENSOR)"
	@echo "  make profile-measure          - NCU targeted sections (fast, 4 passes)"
	@echo "  make bench-husimi             - Compare Husimi overlap vs Wigner routes"
	@echo "  make profile-husimi           - NCU targeted profile for Husimi routes"
	@echo "  make nsys-profile-measure     - nsys timeline: CPU+GPU CUDA API timing"
	@echo "  make bench-all           - Run all benchmark tasks"
	@echo ""
	@echo "Add new kernel targets: bench-<name> and profile-<name>"
	@echo "  bench-<name>:    python benchmarks/<name>/bench_<name>.py"
	@echo "  profile-<name>:  ncu ... python benchmarks/<name>/bench_<name>.py"
	@echo "  make help     - Show this help message"

build:
	@echo "Building CUDA library..."
	@mkdir -p build
	@cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc && make -j$$(nproc)
	@echo "Build successful: $$(pwd)/build/libcvdv.so"

test: build
	@echo "Running tests..."
	pytest tests/ -v

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -f cuda.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

bench: build bench-state-transfer bench-api bench-kernel

bench-state-transfer:
	@bash ./benchmarks/state_transfer/run.sh

bench-api:
	@bash ./benchmarks/api_timing/run.sh

bench-kernel:
	@bash ./benchmarks/kernel_profiling/run.sh

bench-measure: build
	@python benchmarks/measure_kernels/bench_measure.py

bench-measure-ct: build
	@python benchmarks/measure_kernels/bench_measure.py --cutensor

bench-husimi: build
	@python benchmarks/wigner_husimi/bench_husimi.py

profile-husimi: build
	@mkdir -p benchmarks/wigner_husimi/current
	ncu --target-processes all \
	    --launch-skip $(or $(SKIP),3) --launch-count 1 \
	    --section SpeedOfLight --section Occupancy \
	    --section MemoryWorkloadAnalysis --section SchedulerStatistics \
	    -o benchmarks/wigner_husimi/current/ncu_targeted -f \
	    python benchmarks/wigner_husimi/profile_husimi.py
	@echo ""
	@echo "── Roofline summary ──────────────────────────────────────────"
	ncu --import benchmarks/wigner_husimi/current/ncu_targeted.ncu-rep \
	    --print-summary per-kernel 2>/dev/null || true
	@echo "View: ncu-ui benchmarks/wigner_husimi/current/ncu_targeted.ncu-rep"

# Targeted NCU profile (4 passes, fast — use for roofline classification first)
profile-measure: build
	@mkdir -p benchmarks/measure_kernels/current
	ncu --target-processes all \
	    --launch-skip $(or $(SKIP),3) --launch-count 1 \
	    --section SpeedOfLight --section Occupancy \
	    --section MemoryWorkloadAnalysis --section SchedulerStatistics \
	    -o benchmarks/measure_kernels/current/ncu_targeted -f \
	    python benchmarks/measure_kernels/profile_measure.py
	@echo ""
	@echo "── Roofline summary ──────────────────────────────────────────"
	ncu --import benchmarks/measure_kernels/current/ncu_targeted.ncu-rep \
	    --print-summary per-kernel 2>/dev/null || true
	@echo ""
	@echo "To escalate to full profile:"
	@echo "  ncu --set full --kernel-name 'regex:kernelAbsSquareReduce' \\"
	@echo "      -o benchmarks/measure_kernels/current/ncu_full \\"
	@echo "      python benchmarks/measure_kernels/profile_measure.py"
	@echo "View: ncu-ui benchmarks/measure_kernels/current/ncu_targeted.ncu-rep"

# nsys CUDA timeline profile (cheap, shows CPU/GPU overlap, memcpy, memset gaps)
nsys-profile-measure: build
	@mkdir -p benchmarks/measure_kernels/current
	nsys profile \
	    --gpu-metrics-devices=all \
	    --trace=cuda,nvtx \
	    -o benchmarks/measure_kernels/current/nsys_trace -f true \
	    python benchmarks/measure_kernels/profile_measure.py
	@echo ""
	@echo "── GPU kernel summary ────────────────────────────────────────"
	nsys stats --report cuda_gpu_kern_sum \
	    benchmarks/measure_kernels/current/nsys_trace.nsys-rep 2>/dev/null | \
	    grep -v "^Generating\|^Processing\|^NOTICE\|It is assumed\|Consider" || true
	@echo ""
	@echo "── CUDA API summary ──────────────────────────────────────────"
	nsys stats --report cuda_api_sum \
	    benchmarks/measure_kernels/current/nsys_trace.nsys-rep 2>/dev/null | \
	    grep -v "^Generating\|^Processing\|^NOTICE\|It is assumed\|Consider" || true
	@echo ""
	@echo "── GPU memory op summary ─────────────────────────────────────"
	nsys stats --report cuda_gpu_mem_time_sum \
	    benchmarks/measure_kernels/current/nsys_trace.nsys-rep 2>/dev/null | \
	    grep -v "^Generating\|^Processing\|^NOTICE\|It is assumed\|Consider" || true
	@echo "View: nsys-ui benchmarks/measure_kernels/current/nsys_trace.nsys-rep"

save-bench:
	@bash ./benchmarks/api_timing/save.sh
	@bash ./benchmarks/kernel_profiling/save.sh
