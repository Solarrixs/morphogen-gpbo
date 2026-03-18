# Handoff to Iteration 16

## Last Completed: TODO-31 — FixedNoiseGP with heteroscedastic noise (§1.4)
Added `--fixed-noise` CLI flag with auto-discovery of bootstrap variance CSV. When no CSV exists, computes uniform noise from Y column variance. Noise clamped at `FIXED_NOISE_MIN_VARIANCE = 0.02` (was 1e-6). 638 tests passing.

## Next Up: TODO-32 — Sobol QMC sampler (§1.4)
- Add `--mc-samples N` CLI flag (default 512, max 2048)
- Use `SobolQMCNormalSampler` in acquisition function
- File: `04_gpbo_loop.py`
- Acceptance: Sobol sampler used in acquisition; test verifies sample count; 2+ new tests

Alternative: TODO-27 (input warping — Kumaraswamy CDF, `--input-warp`)

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `FIXED_NOISE_MIN_VARIANCE` (0.02) is much higher than the previous clamp (1e-6) — this is intentional per Cosenza 2022
- `--fixed-noise` auto-discovers `data/gp_noise_variance_amin_kelley.csv`; falls back to Y column variance
- `--bootstrap-noise` (explicit path) still works and takes priority over auto-discovery
- `explicit_priors` only applies to MAP paths — SAASBO/LassoBO have their own priors
- Log-scale: `_apply_log_scale()` handles missing columns internally — do NOT pre-filter
- `_fit_mll_with_restarts` validates n_restarts >= 1; only applies to MAP paths
- Variance inflation lives in `06_cellflow_virtual.py` only — `04_gpbo_loop.py` calls it, does NOT duplicate
- CellFlow uses JAX (`jax.random`), NOT torch

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (638 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 started (TODO-9, TODO-28, TODO-29, TODO-30, TODO-31 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~88 tasks todo, 0 blocked, ~42 complete
