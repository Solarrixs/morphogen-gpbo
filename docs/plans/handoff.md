# Handoff to Iteration 6

## Last Completed: TODO-3 — Add Day 72 OOD warning (§1.3)
Added `CELLFLOW_MAX_TRAINING_DAY = 36` to `gopro/config.py`. Added `_warn_ood_harvest_days()` helper in `06_cellflow_virtual.py` that checks `log_harvest_day` column and logs a warning when any protocol exceeds CellFlow's training range (days 1-36). Warning fires in `predict_cellflow()` before either CellFlow or heuristic path. 5 new tests in `TestOODHarvestDayWarning`. 610 tests passing.

## Next Up: TODO-4 — Handle CellFlow conservative prediction bias (§1.3)
- Add variance inflation factor to CellFlow predictions
- Add `--cellflow-variance-inflation` CLI flag (default 2.0)
- Apply inflation to CellFlow-predicted fractions' variance before merging into MF-GP
- Acceptance: inflation applied to CellFlow predictions; test verifies scaling; 2+ new tests
- File: `gopro/06_cellflow_virtual.py`

Alternative: §1.9 Sanchis-Calleja data ingestion (HIGH priority, 3× training data increase) or §1.4 GP model improvements.

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict
- CellFlow imports: use `jax` and `jax.random`, NOT `torch` or `jax.numpy` (jnp not needed in current code)

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~120+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (610 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 in progress (TODO-1, TODO-3 done; TODO-4 remains)
- Config: `gopro/config.py` — all constants (including `CELLFLOW_MAX_TRAINING_DAY`)
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~101 tasks todo, 0 blocked, ~24 complete
