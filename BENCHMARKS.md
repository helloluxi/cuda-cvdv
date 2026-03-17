# Benchmarks

All benchmarks run on **NVIDIA RTX 4070 Laptop GPU**.

## CV-to-DV State Transfer

The position encoding approach enables universal transfer of CV modes into qubits, where $\psi(q_j)$ are directly encoded into qubit register amplitudes ([Phys. Rev. Lett. 128, 110503 (2022)](https://link.aps.org/doi/10.1103/PhysRevLett.128.110503)):

$$|\psi\rangle_{\text{CV}} = \int \psi(q) |q\rangle dq \mapsto \sqrt{\lambda} \sum_{j=0}^{N-1} \psi(\lambda\tilde{j}) |j\rangle_{\text{DV}}$$

### Performance vs bosonic-qiskit (CPU, Fock basis)

![Performance Comparison](benchmarks/state_transfer/results/comparison.png)

CUDA-CVDV scales efficiently to dimension **16384** (14 qubits) in **12.4 ms**, while bosonic-qiskit already takes **1079 ms** at the much smaller dimension 128 (7 qubits) — an **87× speedup** at 128× larger scale. Bosonic-qiskit's runtime grows significantly beyond dimension 128 due to dense matrix operations in Fock basis.

### Cat State Transfer Visualization

**Initial State**

![Initial State](benchmarks/state_transfer/results/cvdv_initial_state.png)

**Final State**

![Final State](benchmarks/state_transfer/results/cvdv_final_state.png)

### Run

```bash
./benchmarks/state_transfer/run.sh
```

Results saved to `benchmarks/state_transfer/results/` (`benchmark_results.json` + plots).
