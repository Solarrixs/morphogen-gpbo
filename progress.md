# Progress Log

## Iteration Log

### Iteration 1 — Task 11: LassoBO
- Added `_fit_lassobo()` function to `04_gpbo_loop.py` — L1-regularized MAP variable selection
- Added `--lassobo` and `--lassobo-alpha` CLI flags
- Wired through `fit_gp_botorch()` and `run_gpbo_loop()`
- Added LassoBO model type to diagnostics
- 7 new tests in `TestLassoBO` class (all passing)
- Total: 571 tests passing (was 564)
