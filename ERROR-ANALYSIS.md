# CVDV Discretization Error Analysis

## Fock State Norm Check

![norm_check.png](analysis/figures/norm_check.png)


## QFT Error

![qft_err_per_fock.png](analysis/figures/qft_err_per_fock.png)
![qft_per_fock_coeff_scaling.png](analysis/figures/qft_per_fock_coeff_scaling.png)


## QFT on Displaced Fock States

![qft_disp_err.png](analysis/figures/qft_disp_err.png)


### Gate-parameter sweep (fixed N)

![qft_disp_vs_gamma.png](analysis/figures/qft_disp_vs_gamma.png)
![qft_disp_eps_vs_alpha.png](analysis/figures/qft_disp_eps_vs_alpha.png)


## QFT on Squeezed Fock States

![qft_squeeze_err.png](analysis/figures/qft_squeeze_err.png)
![qft_squeeze_coeff_scaling.png](analysis/figures/qft_squeeze_coeff_scaling.png)

**Fitted formula** `log(eps) = a(N)*(Gamma+1/2) + b(N)`:

$$
\log \varepsilon \;\approx\; \Bigl(0.0729\,N^{-1/2} + 0.3877\Bigr)\,\Bigl(\Gamma+\tfrac{1}{2}\Bigr)\;+\; 0.0752\,N^{1/2} + 1.4076
$$

### Gate-parameter sweep (fixed N)

![qft_squeeze_vs_gamma.png](analysis/figures/qft_squeeze_vs_gamma.png)
![qft_squeeze_eps_vs_r.png](analysis/figures/qft_squeeze_eps_vs_r.png)


## Commutator Error

![comm_err.png](analysis/figures/comm_err.png)
![comm_coeff_scaling.png](analysis/figures/comm_coeff_scaling.png)

**Fitted formula** `log(eps) = a(N)*(Gamma+1/2) + b(N)`:

$$
\log \varepsilon \;\approx\; \Bigl(6.3275\,N^{-1/2} + 0.1203\Bigr)\,\Bigl(\Gamma+\tfrac{1}{2}\Bigr)\;+\; -7.8531\,N^{1/2} + 28.9305
$$

## Displacement D(2) Error

![disp_err.png](analysis/figures/disp_err.png)
![disp_coeff_scaling.png](analysis/figures/disp_coeff_scaling.png)


### Gate-parameter sweep (fixed N)

![disp_eps_vs_alpha.png](analysis/figures/disp_eps_vs_alpha.png)


## Rotation R(pi/4) Error

![rot_err.png](analysis/figures/rot_err.png)
![rot_coeff_scaling.png](analysis/figures/rot_coeff_scaling.png)


### Gate-parameter sweep (fixed N)

![rot_eps_vs_theta.png](analysis/figures/rot_eps_vs_theta.png)


## Squeezing S(1) Error

![squeeze_err.png](analysis/figures/squeeze_err.png)
![squeeze_coeff_scaling.png](analysis/figures/squeeze_coeff_scaling.png)


### Gate-parameter sweep (fixed N)

![squeeze_eps_vs_r.png](analysis/figures/squeeze_eps_vs_r.png)


## Beam Splitter BS(pi/2) Error

![beam_splitter_err.png](analysis/figures/beam_splitter_err.png)
![bs_coeff_scaling.png](analysis/figures/bs_coeff_scaling.png)


### Gate-parameter sweep (fixed N)

![bs_eps_vs_theta.png](analysis/figures/bs_eps_vs_theta.png)


## Compare Fock vs WF Encoding

![compare_fock.png](analysis/figures/compare_fock.png)


## Advantage Diagram

![advantage.png](analysis/figures/advantage.png)


