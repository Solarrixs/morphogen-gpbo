# Handoff to Iteration 22

## Last Completed: TODO-6 — Zero-passing kernel (`--zero-passing`)
Implemented `ZeroPassingKernel` wrapping any GPyTorch kernel with a smooth mask `phi(x) = 1 - exp(-||x_conc||^2 / eps)` that forces `k(0,x) = 0` for concentration inputs (GPerturb, Xing & Yau 2025). Kernel inherits from `gpytorch.kernels.Kernel` via lazy factory to defer imports. Applied in standard and per-type-GP MAP paths. 8 new tests, 662 total passing.

## Next Up: TODO-7 — Desirability-based feasibility gate
- `D(x) = phi(x) * y_bar(x)` gates infeasible regions (Cosenza 2022)
- File: `04_gpbo_loop.py`

Alternative: TODO-11 (ILR vs ALR), TODO-12 (contextual parameters), TODO-36 (carry-forward controls)

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory and `INDEX.md` are untracked — not committed
- `ZeroPassingKernel` is lazily created via `_get_zero_passing_kernel_class()` factory — never reference `ZeroPassingKernel` global directly (it starts as None)
- `zero_passing` only applies to MAP paths (standard + per-type-GP); ignored for SAASBO, LassoBO, MF-GP
- `per_fidelity_ard` converts fidelity to integer task indices (NOT `_remap_fidelity`)
- `per_fidelity_ard` only applies to multi-fidelity path; ignored when only 1 fidelity level
- `input_warp` only applies to MAP paths (standard + per-type-GP); ignored for SAASBO, LassoBO, MF-GP
- `explicit_priors` only applies to MAP paths — SAASBO/LassoBO have their own priors
- Log-scale: `_apply_log_scale()` handles missing columns internally — do NOT pre-filter
- Variance inflation lives in `06_cellflow_virtual.py` only — `04_gpbo_loop.py` calls it, does NOT duplicate
- CellFlow uses JAX (`jax.random`), NOT torch
- `mc_samples` is clamped to [1, 2048] inside `recommend_next_experiments()`

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (662 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 in progress (9/13 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~83 tasks todo, 0 blocked, ~47 complete
