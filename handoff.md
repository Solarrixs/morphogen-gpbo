# Handoff to Iteration 4

## Last Completed: TODO-26 — Fix CellFlow dose encoding + simplify pass
Changed `encode_protocol_cellflow()` to use `log1p(conc)` instead of raw concentration. Added `concentration_scale` metadata field. 3 new tests. All §1.2 critical MF-GP bugs now resolved (TODO-24, 25, 26). 602 tests passing.

## Next Up: §1.3 TODO-1 — Fix CellFlow JAX vs PyTorch mismatch
- **File:** `gopro/06_cellflow_virtual.py`, `_predict_with_cellflow()`
- **What:** CellFlow uses JAX internally; update prediction function to use JAX API instead of PyTorch
- **Acceptance:** prediction works with CellFlow's JAX backend; 2+ new tests
- **Alternative high-priority:** §1.9 — Ingest 97 Sanchis-Calleja conditions as fidelity 0.8-0.9 (3× training data increase). Requires `SanchisCallejaParser` + step 02 run + `merge_multi_fidelity_data()` wiring.

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict
- CellFlow integration (§1.3) may need actual CellFlow package installed to test properly — TODO-3 and TODO-4 may be blocked without it
- §1.9 data ingestion requires 22 GB patterning screen download (may be deferred to GPU/server)

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~120+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (602 passing)
- Config: `gopro/config.py` — all constants; `FIDELITY_R2_THRESHOLDS = {"drop": 0.80, "skip": 0.90}`
- Import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs
- §1.1 COMPLETE (15/15), §1.2 COMPLETE (3/3)

## Remaining: ~105 tasks todo, 0 blocked, ~20 complete
