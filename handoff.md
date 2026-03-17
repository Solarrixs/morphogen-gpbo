# Handoff to Iteration 2
> Written at end of each iteration for the next fresh-context agent.

## Last Completed
Task 11: LassoBO — L1-regularized MAP variable selection (AISTATS 2025). Added `--lassobo` flag to `04_gpbo_loop.py` with `fit_lassobo()` function. 7 new tests, 571 total passing. Simplify pass hoisted loop invariants and added convergence check.

## Next Up
Task 12: Bootstrap uncertainty — Compute bootstrap CIs on cell type fractions from per-cell KNN probabilities in `02_map_to_hnoca.py`. Propagate as heteroscedastic GP noise via `FixedNoiseGP` in `04_gpbo_loop.py`.
- **Acceptance**: per-condition noise estimates saved; FixedNoiseGP used when noise available; 3+ new tests

## Warnings
- LassoBO with high alpha (>0.5) can cause PSD issues on small datasets (15 points). Default 0.1 is safe.
- LassoBO removes default BoTorch priors (LogNormalPrior on lengthscales/noise) to avoid validation errors during Adam optimization. The L1 penalty serves as the regularizer instead.
- `04_gpbo_loop.py` is very large (~2200+ lines). Read specific sections rather than the whole file.
- Multiple GP paths exist: standard MAP, SAASBO, TVR, per-type, LassoBO, multi-fidelity, Mixed (timing). Bootstrap noise should integrate with as many as practical (at minimum: standard MAP path).

## Key Context
- Branch: ralph/production-readiness-phase2
- Tests: 571 passing (`python -m pytest gopro/tests/ -v`)
- Venv: `source .venv/bin/activate`
- Import constants from `gopro.config`, use `.copy()` before mutating DataFrames
- ralph-task.md has the subtask list

## Remaining
- 4 tasks todo (12: bootstrap uncertainty, 13: entropy center, 14: /simplify, 15: /bug-hunter)
- 0 blocked
- 11 complete
