# Handoff to Iteration 3

## Last Completed: Target Profile Refinement (DeMeo 2025) — Phase B Idea #4
- `refine_target_profile()` added to `gopro/04_gpbo_loop.py`
- Wired into `run_gpbo_loop(refine_target=True, refine_lr=0.3)` and CLI `--refine-target`/`--refine-lr`
- 7 new tests, 510 total passing
- Softmax-based learned profile from Pearson correlations; interpolation with original target

## Next Up: Phase B Idea #12 — FBaxis_rank Regionalization
- **Task**: Extract A-P axis score from Sanchis-Calleja data. Add as continuous optimization target alongside discrete region profiles.
- **Acceptance**: `--target-region` accepts "ap_axis" value; 3+ tests
- **Key file**: `gopro/region_targets.py` (add AP axis support) + `gopro/04_gpbo_loop.py` (wire in)
- **Reference**: `docs/plans/ideas_from_sanchis_calleja_2025.md`

## Warnings
- `refine_target_profile()` uses softmax of Pearson correlations for the learned profile. With very few conditions or highly uniform data, the learned profile may not be informative (function returns original unchanged if <3 overlapping conditions).
- The cosine-similarity proxy used in `run_gpbo_loop()` for fidelity scores is an approximation — it requires `target_profile` to be set.
- Import constants from `gopro.config` — never hardcode paths or columns.
- Use `.copy()` before mutating DataFrames passed as arguments.

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (510 passing)
- Task list: `ralph-task.md` (15 subtasks total)
- All competitive landscape ideas documented in `docs/plans/competitive_landscape_ideas_index.md`
- Per-idea specs in `docs/plans/ideas_from_*.md`

## Remaining: 13 tasks todo, 0 blocked, 2 complete
