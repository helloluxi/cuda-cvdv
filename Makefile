.PHONY: build test clean help bench bench-state-transfer bench-api bench-kernel bench-all

help:
	@echo "CVDV Quantum Simulator - Build & Test Commands"
	@echo ""
	@echo "  make build    - Compile CUDA library"
	@echo "  make test     - Build and run all tests"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make bench-state-transfer - Run state-transfer benchmark"
	@echo "  make bench-api           - Run C API timing benchmark"
	@echo "  make bench-kernel        - Run Nsight kernel profiling"
	@echo "  make bench-all           - Run all benchmark tasks"
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

bench: bench-all

bench-state-transfer:
	@bash ./benchmarks/state_transfer/run.sh

bench-api:
	@bash ./benchmarks/api_timing/run.sh

bench-kernel:
	@bash ./benchmarks/kernel_profiling/run.sh

bench-all: bench-state-transfer bench-api bench-kernel