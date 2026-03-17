# Progress Log

## Iteration Log

### Iteration 2 — Task 12: Bootstrap Uncertainty
- Added `compute_bootstrap_uncertainty()` to `02_map_to_hnoca.py` — resamples cells within each condition, returns per-condition per-cell-type variance
- Wired into `run_mapping_pipeline()` and `__main__` block (saves `gp_noise_variance_{prefix}.csv`)
- Added `noise_variance` param to `fit_gp_botorch()` — passes `train_Yvar` to `SingleTaskGP` for heteroscedastic noise
- Added `noise_variance_csv` param to `run_gpbo_loop()` + `--bootstrap-noise` CLI flag
- ILR variance propagation uses mean-variance approximation (conservative, avoids expensive Jacobian)
- Noise floor clamped to 1e-6 to avoid numerical issues
- 4 new tests: shape/positive, more-cells-less-variance, reproducibility, GP fitting with noise
- Total: 575 tests passing (was 571)

### Iteration 1 — Task 11: LassoBO
- Added `_fit_lassobo()` function to `04_gpbo_loop.py` — L1-regularized MAP variable selection
- Added `--lassobo` and `--lassobo-alpha` CLI flags
- Wired through `fit_gp_botorch()` and `run_gpbo_loop()`
- Added LassoBO model type to diagnostics
- 7 new tests in `TestLassoBO` class (all passing)
- Total: 571 tests passing (was 564)
