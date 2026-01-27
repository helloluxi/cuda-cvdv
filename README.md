# CVDV: CUDA-Accelerated Hybrid CV-DV Quantum Simulator

A high-performance CUDA library for simulating hybrid continuous-variable (CV) and discrete-variable (DV) quantum systems using position wave function encoding.

> **Reference**: This project demonstrates classical simulation techniques from arXiv:xxxx.xxxxx (TODO), which establishes qubit circuits for simulating hybrid CV-DV processors. While the paper focuses on quantum simulation, this CUDA implementation shows the potential for efficient classical dense state vector simulation of hybrid systems.

## Why Position Wave Function Encoding?

### The Problem with Fock Basis

Traditional CV quantum simulators use **Fock basis** encoding $|\psi\rangle = \sum_n c_n |n\rangle$, which presents fundamental computational challenge that Gaussian operations (displacement, squeezing, beam splitters) have **dense, non-sparse matrix representations**

### Position Wave Function Advantages

**This codebase encodes CV states by sampling the position wave function** $\psi(q)$ on a discrete grid:

$$|\psi\rangle \mapsto \sqrt{\lambda} \sum_{j=0}^{N-1} \psi(\lambda \tilde{j}) |j\rangle$$

where $\tilde{j} = j - (N-1)/2$ is the shifted index and $\lambda = \sqrt{2\pi/N}$ is the grid spacing.

This encoding offers:

1. **Exact Position-Space Operations**: Operators of the form $e^{it\hat{q}}$ and $e^{it\hat{q}^2}$ are **diagonal** in position encoding—implemented as simple phase multiplications with zero error
2. **Efficient Gaussian Gates**: Displacement, squeezing, rotation, and beam splitters decompose into elementary $e^{it\hat{q}_j\hat{q}_k}$ and $e^{it\hat{p}_j\hat{p}_k}$ operations, each requiring only $O(n^2)$ elementary gates
3. **Controlled Error Source**: Errors arise **only** from QFT basis switching between position and momentum representations—not from Trotter or other approximations
4. **CUDA-Friendly Parallelism**: Each grid point is processed independently, achieving near-optimal GPU utilization

## Data Format

### Register-Based Architecture

CVDV uses a unified **register abstraction** where all quantum systems are discrete registers with dimension $2^n$:

```
Register 0: 2^numQubits[0] levels
Register 1: 2^numQubits[1] levels
...
```

The full state vector is the tensor product of all registers:
$$|\Psi\rangle = |\psi_0\rangle \otimes |\psi_1\rangle \otimes \cdots$$

### Grid Discretization

For CV operations, the position-space grid is automatically configured:
- Grid dimension: $\text{N} = 2^n$
- Grid step: $dx = \sqrt{2\pi / N}$
- Position range: $x \in [-\sqrt{\pi N/2}, \sqrt{\pi N/2}]$

## Installation

This codebase is only tested on linux with

- CUDA Toolkit 13.1
- CMake 3.28.3
- Python 3.12.3

This compiles the CUDA library to `build/libcvdv.so`. The Python wrapper automatically rebuilds on import.

## Usage

### Basic Example

```python
from interface import CVDV

# Create system with two registers: 1 qubit + 10-qubit CV mode
sim = CVDV([1, 10])  # Register 0: 2 levels, Register 1: 1024 grid points

# Initialize states
sim.setZero(0)       # |0⟩ for qubit
sim.setCoherent(1, 2+1j)  # Coherent state |α=2+i⟩

# Build tensor product state (REQUIRED before operations)
sim.initStateVector()

# Apply operations
sim.hadamard(0, 0)           # Hadamard on qubit
sim.cd(1, 0, 0, 1.5)         # Conditional displacement
sim.displacement(1, -1+0.5j) # Unconditional displacement

# Visualize
wigner = sim.getWignerSingleSlice(1, [-1], wignerN=201, wXMax=5, wPMax=5)
```

### Initialization Pattern

**Critical**: Follow this 3-step pattern:

1. **Create** instance with register sizes
2. **Initialize** each register (`setZero`, `setCoherent`, `setFock`, `setFocks`, `setUniform`)
3. **Build** tensor product with `initStateVector()`

### Available Operations

| Category | Operations |
|----------|------------|
| **CV Gates** | `displacement(α)` ($e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$), `squeeze(r)` ($e^{r(\hat{a}^2 - \hat{a}^{\dagger 2})/2}$), `rotation(θ)` ($e^{i\theta \hat{a}^\dagger \hat{a}}$), `sheer(t)` ($e^{it\hat{q}^2/2}$) |
| **DV Gates** | `hadamard()` ($H$), `pauliRotation(axis, θ)` (axis=0,1,2: $R_x(\theta), R_y(\theta), R_z(\theta)$) |
| **Hybrid** | `cd(α)` ($e^{Z(\alpha\hat{a}^\dagger - \alpha^*\hat{a})}$) |
| **Two-Mode** | `beamSplitter(θ)`, `q1q2(coeff)` ($e^{i t \hat{q}_1 \hat{q}_2}$) |
| **Transforms** | `ftQ2P()`, `ftP2Q()` (Fourier transforms) |
| **Measurement** | `measure()`, `jointMeasure()`, `getState()` |
| **Visualization** | `getWignerSingleSlice()`, `getWignerFullMode()`, `getHusimiQFullMode()` |

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| [test.ipynb](test.ipynb) | Core functionality: vacuum/coherent/Fock states, displacement, Fourier transforms, conditional displacement, Wigner function visualization |
| [test_cvdv_transfer.ipynb](test_cvdv_transfer.ipynb) | CV-to-DV state transfer protocol (arXiv:2106.12272) |
| [test_qcst.ipynb](test_qcst.ipynb) | Quantum coherent state transform (arXiv: 2412.12871) |

## Debugging

Debug logs are written to `cuda.log` in the project root (cleared on each `CVDV()` instantiation).

## License

MIT License

Xi Lu
