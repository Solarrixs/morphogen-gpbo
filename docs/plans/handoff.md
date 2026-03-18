# Handoff to Iteration 9

## Last Completed: TODO-28 — Selective log-scaling for concentration dimensions (§1.4)
Added `LOG_SCALE_COLUMNS` to `config.py` (all `_uM` columns, excluding `log_harvest_day`). Added `_apply_log_scale()` and `_inverse_log_scale()` helpers to `04_gpbo_loop.py`. Wired into `run_gpbo_loop()` with `--log-scale` CLI flag. Transform applies log1p before bounds computation and GP fitting; expm1 inverse applied to recommendations before output. 7 new tests, 627 total passing.

## Next Up: §1.4 GP Model Improvements — TODO-29 (MLL optimization restarts)
- Add `--mll-restarts N` CLI flag
- Use `fit_gpytorch_mll` with multiple restarts, keep best
- Acceptance: multiple restarts run; best MLL selected; test verifies improvement over single restart; 2+ new tests

Alternative: TODO-30 (explicit GP priors) or TODO-31 (FixedNoiseGP with heteroscedastic noise).

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict
- CellFlow imports: use `jax` and `jax.random`, NOT `torch` or `jax.numpy`
- OOD warning uses `CELLFLOW_MAX_TRAINING_DAY` from config — don't hardcode 36
- Variance inflation uses `CELLFLOW_DEFAULT_VARIANCE_INFLATION` from config — don't hardcode 2.0
- Variance inflation lives in `06_cellflow_virtual.py` only — `04_gpbo_loop.py` calls it, does NOT duplicate the logic
- Pseudocount default is None (uses multiplicative replacement default) — don't hardcode a value
- `ilr_transform()` now has `return_safe` param — use it to get pre-replacement safe fractions without duplicate computation
- Log-scale transform lives in `run_gpbo_loop()` only — applied after data merge, before bounds computation; inverse applied to recommendations before output

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (627 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 started (TODO-9, TODO-28 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~98 tasks todo, 0 blocked, ~31 complete
