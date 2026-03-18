# Handoff to Iteration 3

## Last Completed: TODO-25 — R²-based 3-zone fidelity routing + simplify pass
Switched from Spearman correlation to R² with 3-zone routing (R²>0.90→skip MF-BO, R²<0.80→single fidelity, 0.80-0.90→MF-BO). Added `FIDELITY_R2_THRESHOLDS` dict to config.py, `_compute_r_squared()` helper. Simplify pass cleaned method validation and dead imports. 599 tests passing.

## Next Up: TODO-26 — Fix CellFlow dose encoding (§1.2, CRITICAL)
- **File:** `gopro/06_cellflow_virtual.py` L129-187
- **What:** Change dose encoding from raw `dose * onehot` to `log1p(dose) * onehot`
- **Acceptance:** encoding uses log1p; test verifies log1p transform; zero dose maps to zero vector; 2+ new tests
- This is the LAST §1.2 critical bug. After this, move to §1.3 (CellFlow integration fixes) or §1.9 (Sanchis-Calleja data ingestion).

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict for new code
- `visualize_report.py` still uses legacy aliases for hline annotations — fine as-is

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~120+ tasks total across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (599 passing)
- Config: `gopro/config.py` — all constants; `FIDELITY_R2_THRESHOLDS = {"drop": 0.80, "skip": 0.90}`
- Import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~108 tasks todo, 0 blocked, ~17 complete
