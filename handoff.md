# Handoff to Iteration 3
> Written at end of each iteration for the next fresh-context agent.

## Last Completed
Task 12: Bootstrap uncertainty — Added `compute_bootstrap_uncertainty()` to `02_map_to_hnoca.py`. Wired `train_Yvar` through `fit_gp_botorch()` and `run_gpbo_loop()` with `--bootstrap-noise` CLI flag. Simplify pass vectorized loop, fixed ILR variance, removed double-reindex. 575 tests passing.

## Next Up
Task 13: Data-driven entropy center — Replace arbitrary 0.55 entropy weight in composite fidelity (`03_fidelity_scoring.py`) with Braun reference mean entropy.
- **Acceptance**: entropy center matches Braun reference; 2+ new tests

## Warnings
- `04_gpbo_loop.py` is very large (~2300+ lines). Read specific sections rather than the whole file.
- Bootstrap noise only applies to standard MAP and per-type-GP paths (not SAASBO, LassoBO, multi-fidelity).
- ILR variance propagation uses mean-variance approximation (conservative upper bound).
- Noise floor clamped to 1e-6 to avoid numerical issues.
- `03_fidelity_scoring.py` entropy center is at line ~395 (Gaussian penalty centered at 0.55). The fix should compute mean normalized entropy from Braun fetal brain reference profiles and use that as center.

## Key Context
- Branch: ralph/production-readiness-phase2
- Tests: 575 passing (`python -m pytest gopro/tests/ -v`)
- Venv: `source .venv/bin/activate`
- Import constants from `gopro.config`, use `.copy()` before mutating DataFrames
- ralph-task.md has the subtask list

## Remaining
- 3 tasks todo (13: entropy center, 14: /simplify, 15: /bug-hunter)
- 0 blocked
- 12 complete
