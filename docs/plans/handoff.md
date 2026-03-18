# Handoff to Iteration 24

## Last Completed: TODO-11 — ILR vs ALR comparison (`--alr`)
Implemented `alr_transform()` and `alr_inverse()` functions in `04_gpbo_loop.py` as alternatives to ILR. ALR uses the last composition component as reference: z_j = log(y_j / y_D). Simpler than ILR (no Helmert basis) but introduces asymmetry. Added `--alr` CLI flag, wired through `run_gpbo_loop()` → `fit_gp_botorch()` → `fit_tvr_models()` → `recommend_next_experiments()` → `compute_ensemble_disagreement()`. ALR and ILR are mutually exclusive (ALR takes precedence). Delta-method variance propagation updated for ALR Jacobian. Differentiable ALR inverse added to acquisition scalarization. 7 new tests, 679 total passing.

## Next Up: TODO-8 — Spike-and-slab output sparsity
- scCODA-style continuous relaxation
- File: `04_gpbo_loop.py`
- Note: marked "high effort" in §5 (deferred). Consider TODO-10 (Dirichlet) instead.

Alternative: TODO-10 (Dirichlet-Multinomial), TODO-12 (contextual parameters), TODO-36 (carry-forward controls)

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
- `desirability_gate` generates 2x candidates then filters — only modifies recommendation selection, not GP fitting
- `ANTAGONIST_PAIRS` covers WNT, BMP, SHH, TGFb — add Notch if DAPT antagonist conflicts become relevant
- `use_alr` and `use_ilr` are mutually exclusive — when `use_alr=True`, `use_ilr` is set to False inside `run_gpbo_loop`
- ALR uses last composition component as reference — asymmetric by construction

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (679 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 in progress (11/13 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~81 tasks todo, 0 blocked, ~49 complete
