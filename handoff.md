# Handoff to Iteration 8

## Last Completed: Phase C Idea #11 — Per-cell-type GP models (GPerturb 2025)
- Separate SingleTaskGP per output dimension via ModelListGP (MAP path only)
- `_extract_per_output_lengthscales()` returns (d x n_outputs) morphogen sensitivity matrix
- `--per-type-gp` CLI flag, wired through `run_gpbo_loop(per_type_gp=True)`
- Simplify pass fixed 4 quality issues (commit 47a2f6d)
- 6 tests added, 547 total passing, 0 failures

## Next Up: Phase D Idea #13 — Per-round fidelity monitoring
- **Task**: Re-evaluate cross-fidelity correlation each round. Auto-fallback to single-fidelity if correlation degrades. Add trend to visualization report.
- **Acceptance**: monitoring runs per round; 2+ tests
- **Key files**: `gopro/04_gpbo_loop.py` (existing `validate_fidelity_correlation()`), `gopro/visualize_report.py`
- **Tip**: `validate_fidelity_correlation()` already exists — extend it with per-round tracking and auto-fallback logic

## Warnings
- Per-type GP only activates on the standard MAP path (not SAASBO, multi-fidelity, TVR, or MixedSingleTaskGP)
- Per-type GP requires `train_Y.shape[1] > 1`; single-output falls back to standard SingleTaskGP
- SAASBO + multi-output + scalarized qLogEI is guarded with NotImplementedError (iteration 5 fix)
- TVR + `high_variance` replicate strategy strips fidelity from `active_cols` (iteration 5 fix)
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (547 passing)
- Task list: `ralph-task.md` (8 tasks todo, 0 blocked, 7 complete)
- Iterations 1-7: TVR, target profile refinement, FBaxis_rank, additive+interaction kernel, adaptive complexity, timing windows, per-type GP

## Remaining: 8 tasks todo, 0 blocked, 7 complete
