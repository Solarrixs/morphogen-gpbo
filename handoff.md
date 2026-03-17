# Handoff to Iteration 7

## Last Completed: Phase C Idea #10 — Morphogen Timing Window Encoding
- Categorical timing columns (CHIR99021_window, SAG_window, BMP4_window) added to morphogen matrix
- MixedSingleTaskGP used when `--timing-windows` flag is set
- /simplify pass fixed 4 quality issues
- 7 tests added, 541 total passing, 0 failures

## Next Up: Phase C Idea #11 — Per-cell-type GP models
- **Task**: Fit separate GP per cell type (MAP path, GPerturb 2025). Per-output lengthscale matrix for interpretability. Compare to current multi-output approach.
- **Acceptance**: per-type GPs produce predictions; 4+ tests
- **Key files**: `gopro/04_gpbo_loop.py`
- **Reference**: `docs/plans/ideas_from_gperturb_2025.md`

## Warnings
- MixedSingleTaskGP only activates on the standard GP path (not multi-fidelity, SAASBO, or TVR)
- Timing window columns are only added when `--timing-windows` is set (not default)
- `_compute_active_bounds()` runs BEFORE timing columns are added; timing bounds are added separately
- Timing columns that are constant across all conditions are automatically dropped
- The `cat_dims` parameter is passed through to `fit_gp_botorch()` — ignored when None
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments
- SAASBO + multi-output + scalarized qLogEI is guarded with NotImplementedError (iteration 5 fix)

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (541 passing)
- Task list: `ralph-task.md` (9 tasks todo, 0 blocked, 6 complete)
- Iterations 1-6: TVR, target profile refinement, FBaxis_rank, additive+interaction kernel, adaptive complexity, timing windows

## Remaining: 9 tasks todo, 0 blocked, 6 complete
