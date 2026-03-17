# Handoff Document
> Written at end of each iteration for the next fresh-context agent.

## Last Completed
Task 11: LassoBO — L1-regularized lengthscale estimation (AISTATS 2025). Added `--lassobo` flag to 04_gpbo_loop.py. 7 new tests, 571 total passing.

## Next Up
Task 12: Bootstrap uncertainty — per-condition noise estimates from KNN probabilities, propagated as heteroscedastic GP noise via FixedNoiseGP.

## Warnings
- LassoBO with high alpha (>0.5) can cause PSD issues on small datasets (15 points). Default 0.1 is safe.
- LassoBO removes default BoTorch priors (LogNormalPrior on lengthscales/noise) to avoid validation errors during Adam optimization. The L1 penalty serves as the regularizer instead.

## Key Context
- Branch: ralph/production-readiness-phase2
- Tests: 571 passing (`python -m pytest gopro/tests/ -v`)
- Venv: `source .venv/bin/activate`
- Tasks 1-11 done, remaining: 12 (bootstrap uncertainty), 13 (entropy center), 14 (/simplify), 15 (/bug-hunter)
