# Handoff to Iteration 7

## Last Completed: TODO-4 — Handle CellFlow conservative prediction bias (§1.3)
Added `CELLFLOW_DEFAULT_VARIANCE_INFLATION = 2.0` to config.py. Added `inflate_cellflow_variance()` helper to `06_cellflow_virtual.py` that amplifies deviations from global mean, clamps to non-negative, and re-normalises. Wired into `predict_cellflow()` via `variance_inflation` parameter. Added `--cellflow-variance-inflation` CLI flag to `04_gpbo_loop.py`, threaded through `run_gpbo_loop()` → `merge_multi_fidelity_data()`. Inline inflation also applied in merge for CellFlow data (fidelity=0.0). 7 new tests, 617 total passing.

## Next Up: §1.4 GP Model Improvements — TODO-9 (pseudocount handling)
- Verify and fix pseudocount handling before ILR transform
- Add `--pseudocount` CLI flag (default 1e-6)
- File: `04_gpbo_loop.py`
- Acceptance: pseudocount applied consistently; test with zero-fraction row passes; 2+ new tests

Alternative: §1.9 Sanchis-Calleja data ingestion (HIGH priority, 3× training data increase).

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict
- CellFlow imports: use `jax` and `jax.random`, NOT `torch` or `jax.numpy`
- OOD warning uses `CELLFLOW_MAX_TRAINING_DAY` from config — don't hardcode 36
- Variance inflation uses `CELLFLOW_DEFAULT_VARIANCE_INFLATION` from config — don't hardcode 2.0

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~120+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (617 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE (TODO-1, TODO-3, TODO-4 all done)
- Config: `gopro/config.py` — all constants (including `CELLFLOW_MAX_TRAINING_DAY`, `CELLFLOW_DEFAULT_VARIANCE_INFLATION`)
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~100 tasks todo, 0 blocked, ~28 complete
