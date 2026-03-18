# Handoff to Iteration 19

## Last Completed: TODO-27 — Input warping (Kumaraswamy CDF, `--input-warp`)
Added `--input-warp` CLI flag to `04_gpbo_loop.py`. When enabled, applies a learnable Kumaraswamy CDF warp to all continuous input dimensions before normalization via `ChainedInputTransform(warp=Warp(...), normalize=Normalize(...))`. Warping is applied on the standard MAP and per-type-GP paths; ignored for SAASBO, LassoBO, and multi-fidelity (logged when skipped). Helper `_build_input_transform(d, warp, cat_dims)` centralizes input transform construction. 5 new tests, 647 tests passing.

## Next Up: TODO-5 — Per-fidelity ARD lengthscales
- Implement `g(x) + delta(x,m)` GP structure for multi-fidelity
- Separate base kernel + fidelity-specific residual
- File: `04_gpbo_loop.py`
- Acceptance: MF-GP uses per-fidelity ARD; lengthscale extraction works; 3+ new tests

Alternative: TODO-6 (zero-passing kernel) or TODO-11 (ILR vs ALR comparison)

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
- `input_warp` only applies to MAP paths (standard + per-type-GP); ignored for SAASBO, LassoBO, MF-GP

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (647 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 in progress (7/13 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~86 tasks todo, 0 blocked, ~44 complete
