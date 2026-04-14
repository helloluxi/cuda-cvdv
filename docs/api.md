# API Reference

All public classes and methods in `src/`.

---

## `SeparableState`

Holds per-register initial states before building the full tensor-product state vector. All registers must be initialized before calling `CVDV.initStateVector()`.

```python
from src.separable import SeparableState
sep = SeparableState([1, 10])   # list of qubit counts, one per register
```

| Method | Description |
|--------|-------------|
| `setZero(regIdx)` | Vacuum / qubit `\|0⟩` state |
| `setCoherent(regIdx, alpha)` | Coherent state `\|α⟩`, `alpha` complex |
| `setFock(regIdx, n)` | Single Fock state `\|n⟩` |
| `setFocks(regIdx, coeffs)` | Superposition of Fock states; `coeffs[n]` = amplitude of `\|n⟩` |
| `setCoeffs(regIdx, coeffs)` | Arbitrary position-basis amplitudes (length must equal `2**nqubits`) |
| `setCat(regIdx, alpha, n_components, ...)` | Cat state superposition of coherent states |
| `setUniform(regIdx)` | Uniform superposition over all grid points |
| `validate()` | Raises if any register is uninitialized |

---

## `CVDV`

Main simulator class. Dispatches to CUDA (`libcvdv.so`), `torch-cuda`, or `torch-cpu` depending on `backend`.

```python
from src.cudaCvdv import CudaCvdv
from src.torchCvdv import TorchCvdv
sim = CudaCvdv(numQubits_list)               # CUDA lib callers only
# or
sim = TorchCvdv(numQubits_list, device='cuda', dtype='complex128')  # pure torch
```

`numQubits_list`: list of integers, one per register. Register `i` has `2**numQubits_list[i]` grid points.

### Initialization

| Method | Description |
|--------|-------------|
| `initStateVector(sep)` | Build full tensor-product state from a `SeparableState` |

### CV Gates

All angles/parameters are floats. `regIdx` is the register index.

| Method | Operator | Description |
|--------|----------|-------------|
| `d(regIdx, alpha)` | $D(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$ | Displacement; `alpha` complex |
| `r(regIdx, theta)` | $e^{i\theta\hat{a}^\dagger\hat{a}}$ | Rotation in phase space |
| `s(regIdx, r)` | $e^{r(\hat{a}^2 - \hat{a}^{\dagger 2})/2}$ | Squeezing |
| `sheer(regIdx, t)` | $e^{it\hat{q}^2}$ | Quadratic phase / shear |
| `phaseCubic(regIdx, t)` | $e^{it\hat{q}^3}$ | Cubic phase gate |
| `p(regIdx)` | $e^{i\pi\hat{a}^\dagger\hat{a}}$ | Parity |
| `ftQ2P(regIdx)` | QFT | Position → momentum basis (cuFFT) |
| `ftP2Q(regIdx)` | QFT⁻¹ | Momentum → position basis (cuFFT) |

### DV (Qubit) Gates

`regIdx` selects the DV register; `targetQubit` selects the qubit within it.

| Method | Description |
|--------|-------------|
| `h(regIdx, targetQubit)` | Hadamard |
| `x(regIdx, targetQubit)` | Pauli X |
| `y(regIdx, targetQubit)` | Pauli Y |
| `z(regIdx, targetQubit)` | Pauli Z |
| `rx(regIdx, targetQubit, theta)` | $R_x(\theta)$ |
| `ry(regIdx, targetQubit, theta)` | $R_y(\theta)$ |
| `rz(regIdx, targetQubit, theta)` | $R_z(\theta)$ |

### Hybrid Gates

CV operation conditioned on a DV qubit (applies $+$ or $-$ version based on qubit state).

| Method | Operator | Description |
|--------|----------|-------------|
| `cd(targetReg, ctrlReg, ctrlQubit, alpha)` | $e^{Z(\alpha\hat{a}^\dagger - \alpha^*\hat{a})}$ | Conditional displacement |
| `cr(targetReg, ctrlReg, ctrlQubit, theta)` | $e^{iZ\theta\hat{a}^\dagger\hat{a}}$ | Conditional rotation |
| `cs(targetReg, ctrlReg, ctrlQubit, r)` | Conditional squeezing | |
| `cp(targetReg, ctrlReg, ctrlQubit)` | Conditional parity | |

### Two-Mode Gates

| Method | Description |
|--------|-------------|
| `bs(reg1, reg2, theta)` | Beam splitter $e^{i\theta(\hat{a}_1^\dagger\hat{a}_2 + \text{h.c.})}$ |
| `cbs(reg1, reg2, ctrlReg, ctrlQubit, theta)` | Conditional beam splitter |
| `q1q2(reg1, reg2, coeff)` | $e^{i\,\text{coeff}\,\hat{q}_1\hat{q}_2}$ |
| `swap(reg1, reg2)` | Swap two registers |

### Measurement & State Access

| Method | Returns | Description |
|--------|---------|-------------|
| `m()` | float | $\|\Psi\|$ (should be 1.0) |
| `m(i)` | `ndarray (dim_i,)` float | Marginal probability distribution for register `i` |
| `m(i, j, ...)` or `m([i, j, ...])` | `ndarray (d_i, d_j, ...)` float | Joint probability over selected registers |
| `getState()` | `ndarray` complex128 | Full state vector (all registers, flattened) |
| `getXGrid(regIdx)` | `ndarray (dim,)` float | Position grid $x_k$ for register |
| `getFidelity(sep)` | float | $|\langle\psi_\text{sep}|\Psi\rangle|^2$ with a `SeparableState` |
| `info()` | — | Print register sizes and VRAM usage |

### Phase-Space Distributions

| Method | Returns | Description |
|--------|---------|-------------|
| `getWigner(regIdx)` | `ndarray (dim, dim)` | Wigner function on full native grid |
| `getWigner(regIdx, bound)` | `ndarray (N, N)` | Wigner clipped to $q, p \in [-\text{bound}, +\text{bound}]$ |
| `getHusimiQ(regIdx)` | `ndarray (dim, dim)` | Husimi Q function on full native grid |
| `getHusimiQ(regIdx, bound)` | `ndarray (N, N)` | Husimi Q clipped to $q, p \in [-\text{bound}, +\text{bound}]$ |
| `plotWigner(regIdx, bound, cmap, figsize, show)` | `(fig, ax)` | Plot Wigner function (default `bound=5.0`) |
| `plotHusimiQ(regIdx, bound, cmap, figsize, show)` | `(fig, ax)` | Plot Husimi Q function (default `bound=5.0`) |

When `bound` is given, the output is clipped to grid points within $[-\text{bound}, +\text{bound}]$ on both axes. The Wigner p-axis uses spacing $\Delta p_W = \pi / (N \cdot dx)$; the Husimi p-axis uses $\Delta p_H = 2\pi / (N \cdot dx) = dx$.

**Normalization conventions:**

- Wigner peak for coherent state $|\alpha\rangle$: $W(q_0, p_0) = \frac{dx}{\pi} e^{-(q-q_0)^2-(p-p_0)^2}$
- Husimi peak: $Q(q_0, p_0) = \frac{1}{\pi} e^{-\frac{1}{2}(q-q_0)^2 - \frac{1}{2}(p-p_0)^2}$

where $q_0 = \sqrt{2}\,\text{Re}\,\alpha$, $p_0 = \sqrt{2}\,\text{Im}\,\alpha$.

---

## Grid & Normalization Notes

- Grid step: `dx = sqrt(2π / dim)`, `dim = 2**nqubits`
- Position grid: `x[k] = (k - (dim-1)/2) * dx`
- Wigner p-spacing: `dp_W = π / (dim * dx)`
- Husimi p-spacing: `dp_H = 2π / (dim * dx) = dx`
- State stored with **discrete norm** `Σ|ψ[k]|² = 1` (not `∫|ψ|²dx = 1`)
