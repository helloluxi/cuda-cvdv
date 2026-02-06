.PHONY: build test clean help

help:
	@echo "CVDV Quantum Simulator - Build & Test Commands"
	@echo ""
	@echo "  make build    - Compile CUDA library"
	@echo "  make test     - Build and run all tests"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this help message"

build:
	@echo "Building CUDA library..."
	bash run.sh

test: build
	@echo "Running tests..."
	pytest tests/ -v

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -f cuda.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
