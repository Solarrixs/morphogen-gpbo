# Handoff to Iteration 6

## Last Completed: Phase C Idea #9 — Adaptive Complexity Schedule tests + simplify fixes
- 8 tests added covering all 3 regimes (shared/ARD/SAASBO), boundary conditions, custom thresholds, zero-dim safety
- Simplify pass removed unused `round_number` param from `_select_kernel_complexity()`, added fail-loud thresholds
- 534 tests passing, 0 failures

## Next Up: Phase C Idea #10 — Morphogen Timing Window Encoding
- **Task**: Add temporal window categorical dimensions (early/mid/late patterning) to morphogen matrix. Use MixedSingleTaskGP for mixed continuous+categorical.
- **Acceptance**: timing dims in training data; 3+ tests
- **Key files**: `gopro/04_gpbo_loop.py`, `gopro/morphogen_parser.py`
- **Reference**: `docs/plans/ideas_from_sanchis_calleja_2025.md`

## Warnings
- Additive+interaction kernel only applies to standard SingleTaskGP path (not multi-fidelity or SAASBO)
- Adaptive complexity overrides both `kernel_type` and `use_saasbo` when `--adaptive-complexity` is set
- MixedSingleTaskGP requires `cat_dims` parameter — identify which columns are categorical vs continuous
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (534 passing)
- Task list: `ralph-task.md` (10 subtasks remaining, 5 complete)
- Iterations 1-5: TVR, target profile refinement, FBaxis_rank, additive+interaction kernel, adaptive complexity

## Remaining: 10 tasks todo, 0 blocked, 5 complete
