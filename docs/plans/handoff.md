# Handoff to Iteration 14

## Last Completed: TODO-30 — Explicit GP priors
Added `_set_noise_prior()` (Gamma(3,6)) and `_set_explicit_priors()` (lengthscale + noise) helpers. Wired through `--explicit-priors` CLI flag → `run_gpbo_loop` → `fit_gp_botorch`. 3 new tests, 634 tests passing.

## Next Up: §1.4 GP Model Improvements — TODO-31 (FixedNoiseGP with heteroscedastic noise) or TODO-32 (Sobol QMC sampler)
- TODO-31: FixedNoiseGP with per-observation heteroscedastic noise — `--fixed-noise` flag, compute noise from bootstrap replicates, clamp min 0.02
- TODO-32: Sobol QMC sampler — `--mc-samples N` flag (default 512, max 2048), use `SobolQMCNormalSampler`

Alternative: TODO-27 (input warping), TODO-5 (per-fidelity ARD), TODO-6 (zero-passing kernel).

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
- `_fit_mll_with_restarts` handles multi-output MLL by summing per-output values; validates n_restarts >= 1
- MLL restarts only apply to MAP paths — SAASBO and LassoBO are unaffected
- `explicit_priors` only applies to MAP paths — SAASBO/LassoBO have their own priors
- `_set_explicit_priors` calls `_set_dim_scaled_lengthscale_prior` (LogNormal) + `_set_noise_prior` (Gamma(3,6))
- When `explicit_priors=True`, it replaces the default lengthscale-only prior in standard/per-type paths

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (634 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 started (TODO-9, TODO-28, TODO-29, TODO-30 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~93 tasks todo, 0 blocked, ~37 complete
