# Handoff to Iteration 3

## Last Completed: TODO-25 — R²-based 3-zone fidelity routing
- Switched from Spearman correlation (threshold 0.3) to R² coefficient of determination
- 3-zone routing: R²>0.90→skip MF-BO, R²<0.80→single fidelity, 0.80-0.90→MF-BO
- Added `FIDELITY_R2_THRESHOLDS` dict to `config.py`, legacy aliases preserved
- Added `_compute_r_squared()` helper to `04_gpbo_loop.py`
- Default method changed from "spearman" to "r_squared"; legacy methods still work
- 599 tests passing

## Next Up: TODO-26 — Fix CellFlow dose encoding
- **File:** `06_cellflow_virtual.py` L129-187
- **What:** Change dose encoding from raw `dose * onehot` to `log1p(dose) * onehot`
- **Acceptance:** encoding uses log1p; test verifies log1p transform; zero dose maps to zero vector; 2+ new tests

## Warnings
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are now legacy aliases derived from `FIDELITY_R2_THRESHOLDS` — use the dict for new code
- `visualize_report.py` still imports the legacy aliases for hline annotations — fine as-is
- After TODO-26, all §1.2 critical bugs will be resolved

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~142 tasks total)
- Config: `gopro/config.py` — `FIDELITY_R2_THRESHOLDS` dict with "drop" and "skip" keys
- Tests: `python -m pytest gopro/tests/ -v` (599 passing)

## Remaining: 125 tasks todo, 0 blocked, 17 complete
