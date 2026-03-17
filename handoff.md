# Handoff to Iteration 4
> Written at end of each iteration for the next fresh-context agent.

## Last Completed
Task 13: Data-driven entropy center — Added `compute_braun_entropy_center()` to `03_fidelity_scoring.py`. Replaces hardcoded 0.55 with mean normalized entropy from Braun fetal brain region profiles. Wired `entropy_center` param through `compute_composite_fidelity()`, `score_all_conditions()`, and `run_fidelity_scoring()`. 580 tests passing.

## Next Up
Task 14: /simplify pass on all changes
- **Acceptance**: all tests pass after fixes

Then Task 15: /bug-hunter final sweep
- **Acceptance**: no confirmed critical bugs remain

## Warnings
- `04_gpbo_loop.py` is very large (~2300+ lines). Read specific sections rather than the whole file.
- Bootstrap noise only applies to standard MAP and per-type-GP paths (not SAASBO, LassoBO, multi-fidelity).
- `_DEFAULT_ENTROPY_CENTER = 0.55` is the fallback when `entropy_center=None` (backward compatible).
- Existing tests that pass `norm_entropy=0.55` without `entropy_center` still work because of the fallback.

## Key Context
- Branch: ralph/production-readiness-phase2
- Tests: 580 passing (`python -m pytest gopro/tests/ -v`)
- Venv: `source .venv/bin/activate`
- Import constants from `gopro.config`, use `.copy()` before mutating DataFrames
- ralph-task.md has the subtask list

## Remaining
- 2 tasks todo (14: /simplify, 15: /bug-hunter)
- 0 blocked
- 13 complete
