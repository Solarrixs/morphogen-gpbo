# Handoff to Iteration 5

## Last Completed: TODO-1 — Fix CellFlow JAX vs PyTorch API mismatch (§1.3)
Replaced PyTorch imports (`torch`, `torch.no_grad()`) with JAX (`jax`, `jax.random`) in `_predict_with_cellflow()`. CellFlow is JAX-based — JAX doesn't track gradients by default, so no `no_grad` equivalent needed. Added `rng_key` parameter to `model.predict()` for reproducible sampling. 605 tests passing.

## Next Up: §1.3 CellFlow Integration Fixes (continued)
- **TODO-3:** Add Day 72 out-of-distribution warning (CellFlow trained on days 1-36 only)
- **TODO-4:** Handle CellFlow conservative prediction bias — variance inflation

Or alternatively, jump to §1.4 GP Model Improvements or §1.9 (Sanchis-Calleja data ingestion).

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict for new code

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~120+ tasks total across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (605 passing)
- §1.2 is COMPLETE (all 3 critical bugs fixed: TODO-24, TODO-25, TODO-26)
- §1.3 TODO-1 now done; TODO-3 and TODO-4 remain
- Config: `gopro/config.py` — all constants
- Import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~104 tasks todo, 0 blocked, ~21 complete
