# Handoff to Iteration 18

## Last Completed: TODO-32 — Sobol QMC sampler for acquisition
Added `--mc-samples N` CLI flag (default 512, max 2048) to `04_gpbo_loop.py`. All three acquisition function constructors (`qLogExpectedImprovement` ×2, `qLogNoisyExpectedHypervolumeImprovement`) now receive a `SobolQMCNormalSampler` instead of the default IID sampler. Parameter threaded through `recommend_next_experiments()` → `run_gpbo_loop()` → CLI. 4 new tests, 642 tests passing.

## Next Up: TODO-27 — Input warping (Kumaraswamy CDF, `--input-warp`)
- Add `--input-warp` CLI flag
- Wrap GP with BoTorch `Warp` input transform
- File: `gopro/04_gpbo_loop.py`
- Acceptance: warping applied when flag set; test verifies transform; 2+ new tests

Alternative: TODO-5 (per-fidelity ARD lengthscales) or TODO-6 (zero-passing kernel)

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIXED_NOISE_MIN_VARIANCE` (0.02) is intentionally much higher than old clamp (1e-6) per Cosenza 2022
- `--fixed-noise` auto-discovers `data/gp_noise_variance_amin_kelley.csv`; falls back to Y column variance
- `explicit_priors` only applies to MAP paths — SAASBO/LassoBO have their own priors
- Log-scale: `_apply_log_scale()` handles missing columns internally — do NOT pre-filter
- `_fit_mll_with_restarts` validates n_restarts >= 1; only applies to MAP paths
- Variance inflation lives in `06_cellflow_virtual.py` only — `04_gpbo_loop.py` calls it, does NOT duplicate
- CellFlow uses JAX (`jax.random`), NOT torch
- `mc_samples` is clamped to [1, 2048] inside `recommend_next_experiments()`

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (642 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 in progress (6/13 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~87 tasks todo, 0 blocked, ~43 complete
