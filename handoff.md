# Handoff to Iteration 2

## Last Completed: TVR (Targeted Variance Reduction) — Phase B Idea #2
- `fit_tvr_models()`, `TVRModelEnsemble`, `_TVRPosterior` added to `gopro/04_gpbo_loop.py`
- Wired into `run_gpbo_loop(use_tvr=True)` and CLI `--tvr` flag
- 8 new tests, 503 total passing
- Simplify pass fixed 6 issues (cost scaling, no_grad, rsample, dead code)

## Next Up: Phase B Idea #4 — Target Profile Refinement
- **Task**: After Round 1, update `target_profile` using observed best compositions interpolated toward Braun reference
- **Wire into**: `run_gpbo_loop()` as `--refine-target` flag
- **Acceptance**: Refined target differs from original; 3+ new tests
- **Key file**: `gopro/04_gpbo_loop.py` (add refinement logic) + `gopro/region_targets.py` (target profile utilities)

## Warnings
- TVR `_TVRPosterior` is a lightweight wrapper, not a full GPyTorchPosterior. Works with `qLogExpectedImprovement` via `sample()` but may need adaptation for more advanced acquisition functions.
- Multi-output GP noise is a tensor, not scalar — use `.mean().item()` for logging.
- Import constants from `gopro.config` — never hardcode paths or columns.
- Use `.copy()` before mutating DataFrames passed as arguments.

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (503 passing)
- Task list: `ralph-task.md` (15 subtasks total)
- All competitive landscape ideas documented in `docs/plans/competitive_landscape_ideas_index.md`
- Per-idea specs in `docs/plans/ideas_from_*.md`

## Remaining: 14 tasks todo, 0 blocked, 1 complete
