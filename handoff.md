# Handoff to Iteration 4

## Last Completed: FBaxis_rank Regionalization + Simplify (Sanchis-Calleja 2025, Idea #12)
- `BRAIN_REGION_AP_POSITIONS` mapping 9 HNOCA regions to A-P axis (0=forebrain, 1=hindbrain)
- `compute_fbaxis_rank()` scores conditions by weighted A-P position
- `build_ap_target_profile()` creates Gaussian-weighted region profiles for GP-BO targeting
- `--target-region ap_axis --target-fbaxis 0.7` targets hindbrain protocols
- Simplify pass fixed 3 issues: magic string, zero-row semantics, registry sync
- 521 tests passing, 0 failures

## Next Up: Phase C Idea #8 — Additive + Interaction Kernel
- **Task**: Replace Matern ARD with k_additive + k_interaction structure (NAIAD 2025). Reduces effective params from O(d^2) to O(d). Add --kernel additive_interaction flag.
- **Acceptance**: GP fits with new kernel; 4+ tests
- **Key files**: `gopro/04_gpbo_loop.py` (fit_gp_botorch)
- **Reference**: `docs/plans/ideas_from_naiad_2025.md`

## Warnings
- `build_ap_target_profile()` uses Gaussian weighting with default width=0.15 — narrow enough to distinguish adjacent regions but broad enough for smooth optimization
- TVR `_TVRPosterior` is a lightweight wrapper — may need adaptation for advanced acquisition functions beyond qLogEI
- `refine_target_profile()` returns original unchanged if <3 overlapping conditions
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (521 passing)
- Task list: `ralph-task.md` (15 subtasks total)
- All competitive landscape ideas: `docs/plans/competitive_landscape_ideas_index.md`
- Per-idea specs: `docs/plans/ideas_from_*.md`

## Remaining: 12 tasks todo, 0 blocked, 3 complete
