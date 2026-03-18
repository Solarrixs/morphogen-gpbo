# Handoff to Iteration 8

## Last Completed: TODO-9 — Configurable pseudocount for ILR transform (§1.4 started)
Added `pseudocount` parameter to `ilr_transform()`, threaded through `_multiplicative_replacement()`, `fit_gp_botorch()`, `fit_tvr_models()`, `compute_ensemble_disagreement()`, and `run_gpbo_loop()`. Added `--pseudocount` CLI flag (default: None → uses multiplicative replacement default of ~3.8e-4 for 17 cell types). 3 new tests, 620 total passing.

## Next Up: §1.4 GP Model Improvements — TODO-28 (selective log-scaling)
- Add `LOG_SCALE_COLUMNS` list to config.py
- Apply log1p transform to those columns in `build_training_set()`
- Provide inverse for recommendations
- Acceptance: config constant defined; transform applied; inverse available; 3+ new tests

Alternative: TODO-29 (MLL optimization restarts) or TODO-30 (explicit GP priors).

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict
- CellFlow imports: use `jax` and `jax.random`, NOT `torch` or `jax.numpy`
- OOD warning uses `CELLFLOW_MAX_TRAINING_DAY` from config — don't hardcode 36
- Variance inflation uses `CELLFLOW_DEFAULT_VARIANCE_INFLATION` from config — don't hardcode 2.0
- Variance inflation lives in `06_cellflow_virtual.py` only — `04_gpbo_loop.py` calls it, does NOT duplicate the logic
- Pseudocount default is None (uses multiplicative replacement default) — don't hardcode a value

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~120+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (620 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 started (TODO-9 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~100 tasks todo, 0 blocked, ~29 complete
