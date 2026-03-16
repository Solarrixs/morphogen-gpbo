# Handoff to Iteration 3

## Last Completed: Target Profile Refinement (DeMeo 2025) — Phase B Idea #4
- `refine_target_profile()` in `gopro/04_gpbo_loop.py` with softmax-learned interpolation
- Simplify pass fixed 4 issues (dead code, vectorized cosine sim, redundant copy/normalization)
- 510 tests passing, 0 failures

## Next Up: Phase B Idea #12 — FBaxis_rank Regionalization
- **Task**: Extract A-P axis score from Sanchis-Calleja data. Add as continuous optimization target alongside discrete region profiles.
- **Acceptance**: `--target-region` accepts "ap_axis" value; 3+ tests
- **Key files**: `gopro/region_targets.py` (add AP axis support) + `gopro/04_gpbo_loop.py` (wire in)
- **Reference**: `docs/plans/ideas_from_sanchis_calleja_2025.md`

## Warnings
- `refine_target_profile()` returns original unchanged if <3 overlapping conditions
- Cosine-similarity proxy in `run_gpbo_loop()` is an approximation — requires `target_profile` to be set
- TVR `_TVRPosterior` is a lightweight wrapper — may need adaptation for advanced acquisition functions beyond qLogEI
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (510 passing)
- Task list: `ralph-task.md` (15 subtasks total)
- All competitive landscape ideas: `docs/plans/competitive_landscape_ideas_index.md`
- Per-idea specs: `docs/plans/ideas_from_*.md`

## Remaining: 13 tasks todo, 0 blocked, 2 complete
