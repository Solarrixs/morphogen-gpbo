# Handoff to Iteration 23

## Last Completed: TODO-7 — Desirability-based feasibility gate (`--desirability-gate`)
Implemented `compute_desirability()` function and `ANTAGONIST_PAIRS` constant in `04_gpbo_loop.py`. The gate penalises biologically implausible protocols where both agonist and antagonist morphogens for the same signaling pathway are present at non-trivial concentrations (Cosenza 2022). D(x) = phi(x) * acq(x) where phi(x) is a product of per-pathway penalties. Generates 2x candidates when active, scores and keeps top N. Added `--desirability-gate` CLI flag threaded through `run_gpbo_loop()` → `recommend_next_experiments()`. 10 new tests, 672 total passing.

## Next Up: TODO-8 — Spike-and-slab output sparsity
- scCODA-style continuous relaxation
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
- `desirability_gate` generates 2x candidates then filters — only modifies recommendation selection, not GP fitting
- `ANTAGONIST_PAIRS` covers WNT, BMP, SHH, TGFb — add Notch if DAPT antagonist conflicts become relevant

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Task plan: `docs/task_plan.md` (~100+ tasks across 5 sections)
- Tests: `python -m pytest gopro/tests/ -v` (672 passing)
- §1.1 COMPLETE, §1.2 COMPLETE, §1.3 COMPLETE, §1.4 in progress (10/13 done)
- Config: `gopro/config.py` — all constants
- Conventions: import from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~82 tasks todo, 0 blocked, ~48 complete
