# Handoff to Iteration 12

## Last Completed: TODO-29 — MLL optimization restarts
Added `_fit_mll_with_restarts()` helper and `--mll-restarts N` CLI flag. Multiple random restarts for MLL optimization with best-fit selection. Wired through all MAP fitting paths (standard, MixedSingleTaskGP, per-type-GP, multi-fidelity). 631 tests passing.

## Next Up: §1.4 GP Model Improvements — TODO-30 (explicit GP priors)
- Add MVN prior on lengthscales (centered on data range), Gamma(3,6) on noise
- Acceptance: priors set before fitting; test verifies prior objects attached; 2+ new tests

Alternative: TODO-31 (FixedNoiseGP with heteroscedastic noise) or TODO-32 (Sobol QMC sampler).

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIDELITY_CORRELATION_THRESHOLD` and `FIDELITY_SKIP_MFBO_THRESHOLD` are legacy aliases — use `FIDELITY_R2_THRESHOLDS` dict
- CellFlow imports: use `jax` and `jax.random`, NOT `torch` or `jax.numpy`
- OOD warning uses `CELLFLOW_MAX_TRAINING_DAY` from config — don't hardcode 36
- Variance inflation uses `CELLFLOW_DEFAULT_VARIANCE_INFLATION` from config — don't hardcode 2.0
- Variance inflation lives in `06_cellflow_virtual.py` only — `04_gpbo_loop.py` calls it, does NOT duplicate the logic
- Pseudocount default is None (uses multiplicative replacement default) — don't hardcode a value
- `ilr_transform()` has `return_safe` param — use it to get pre-replacement safe fractions without duplicate computation
- Log-scale: `_apply_log_scale()` handles missing columns internally — do NOT pre-filter column lists before calling it
- `LOG_SCALE_COLUMNS` = all `_uM` columns from `MORPHOGEN_COLUMNS` — no exclusions needed
- `_fit_mll_with_restarts` handles multi-output MLL by summing per-output values
- MLL restarts only apply to MAP paths — SAASBO and LassoBO are unaffected

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (631 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 started (TODO-9, TODO-28, TODO-29 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~95 tasks todo, 0 blocked, ~34 complete
