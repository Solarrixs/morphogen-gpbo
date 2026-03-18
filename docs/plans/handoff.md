# Handoff to Iteration 4

## Last Completed: TODO-26 — Fix CellFlow dose encoding (§1.2 FINAL)
Changed `encode_protocol_cellflow()` to use `math.log1p(conc)` instead of raw `conc` for dose encoding. This completes all §1.2 critical bug fixes. 602 tests passing.

## Next Up: §1.3 CellFlow Integration Fixes
- **TODO-1:** Fix CellFlow JAX vs PyTorch API mismatch in `_predict_with_cellflow()`
- **TODO-3:** Add Day 72 out-of-distribution warning (CellFlow trained on days 1-36 only)
- **TODO-4:** Handle CellFlow conservative prediction bias — variance inflation

Or alternatively, jump to §1.9 (Sanchis-Calleja data ingestion) for high-priority data integration.

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict for new code

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~120+ tasks total across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (602 passing)
- §1.2 is now COMPLETE (all 3 critical bugs fixed: TODO-24, TODO-25, TODO-26)
- Config: `gopro/config.py` — all constants
- Import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~105 tasks todo, 0 blocked, ~20 complete
