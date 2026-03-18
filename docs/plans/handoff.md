# Handoff to Iteration 20

## Last Completed: TODO-5 — Per-fidelity ARD lengthscales (`--per-fidelity-ard`)
Implemented `g(x) + delta(x,m)` kernel structure for multi-fidelity GP as alternative to `LinearTruncatedFidelityKernel`. Base kernel `k_base(x,x')` has shared Matern 5/2 ARD; residual `k_fidelity(m,m') * k_residual(x,x')` has per-fidelity ARD via `IndexKernel`. Fidelity floats converted to integer task indices. 7 new tests, 654 total passing.

Key additions:
- `_fidelity_to_task_idx()`: float fidelity → int task index mapping
- `_build_per_fidelity_ard_model()`: constructs SingleTaskGP with additive kernel
- `_extract_per_fidelity_ard_lengthscales()`: extracts base + residual ARD lengthscales
- `--per-fidelity-ard` CLI flag, wired through `run_gpbo_loop` → `fit_gp_botorch`

## Next Up: TODO-6 — Zero-passing kernel
- Modified RBF enforcing `k(0,x)=0` for concentration inputs (GPerturb)
- Custom kernel class inheriting from `MaternKernel`
- File: `04_gpbo_loop.py`
- Acceptance: kernel returns 0 when either input is zero-vector; test verifies property; 2+ new tests

Alternative: TODO-11 (ILR vs ALR comparison) or TODO-12 (contextual parameter support)

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `per_fidelity_ard` only applies to multi-fidelity path; ignored when only 1 fidelity level
- `per_fidelity_ard` converts fidelity to integer task indices (NOT `_remap_fidelity`)
- `_extract_lengthscales` returns base (shared) lengthscales for per-fidelity ARD models
- Lengthscale column alignment handles per-fidelity ARD (fewer dims than X.columns)
- `input_warp` only applies to MAP paths (standard + per-type-GP); ignored for SAASBO, LassoBO, MF-GP
- `explicit_priors` only applies to MAP paths — SAASBO/LassoBO have their own priors
- Log-scale: `_apply_log_scale()` handles missing columns internally — do NOT pre-filter
- Variance inflation lives in `06_cellflow_virtual.py` only — `04_gpbo_loop.py` calls it, does NOT duplicate
- CellFlow uses JAX (`jax.random`), NOT torch
- `mc_samples` is clamped to [1, 2048] inside `recommend_next_experiments()`

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (654 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 in progress (8/13 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~85 tasks todo, 0 blocked, ~45 complete
