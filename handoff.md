# Handoff to Iteration 5

## Last Completed: Phase C Idea #8 — Additive + Interaction Kernel (NAIAD 2025)
- `_build_additive_interaction_kernel(d)` creates sum-of-1D additive + full ARD interaction kernel
- Interaction outputscale initialized to 0.1 (prior toward additivity)
- `--kernel additive_interaction` CLI flag on `04_gpbo_loop.py`
- `kernel_type` parameter wired through `fit_gp_botorch()` and `run_gpbo_loop()`
- `_extract_lengthscales()` updated to extract from additive+interaction structure
- 526 tests passing, 0 failures

## Next Up: Phase C Idea #9 — Adaptive Complexity Schedule
- **Task**: Round 1: shared lengthscale. Round 2+: per-dim ARD. Round 3+: SAASBO. Auto-select based on N/d ratio.
- **Acceptance**: complexity auto-selected; 3+ tests
- **Key files**: `gopro/04_gpbo_loop.py` (fit_gp_botorch), `gopro/config.py`
- **Reference**: `docs/plans/ideas_from_naiad_2025.md` (Idea 1)

## Warnings
- `_build_additive_interaction_kernel` only applies to the standard SingleTaskGP path (not multi-fidelity or SAASBO)
- Interaction kernel's ARD lengthscales are used for importance ranking via `_extract_lengthscales`
- `kernel_type="additive_interaction"` skips the dim-scaled lengthscale prior (the additive structure provides its own regularization)
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (526 passing)
- Task list: `ralph-task.md` (15 subtasks total)
- All competitive landscape ideas: `docs/plans/competitive_landscape_ideas_index.md`
- Per-idea specs: `docs/plans/ideas_from_*.md`

## Remaining: 11 tasks todo, 0 blocked, 4 complete
