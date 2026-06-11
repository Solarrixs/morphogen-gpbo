# Handoff to Iteration 2

## Last Completed: I-3 — TVR gradient warning docstring + acquisition test
Added docstring warning to `TVRModelEnsemble.posterior()` that argmin breaks gradient chain. Added 2 tests: one verifying acquisition optimization doesn't silently fail with TVR, one checking improvement direction. Simplify pass fixed test to capture baseline before optimization.

## Next Up: I-4 — Deduplicate antagonist pairs (YAML single source of truth)
- **Files:** `gopro/04_gpbo_loop.py` (has `ANTAGONIST_PAIRS` dict), `gopro/agents/scorer.py` (also has antagonist logic), `gopro/agents/pathway_rules.yaml` (should be single source)
- **What:** Remove `ANTAGONIST_PAIRS` dict from `04_gpbo_loop.py`, have both `04_gpbo_loop.py` and `agents/scorer.py` load from `pathway_rules.yaml`
- **Acceptance:** `ANTAGONIST_PAIRS` dict removed from `04_gpbo_loop.py`, both consumers load from YAML, all tests pass

## Warnings
- Branch is `ralph/fix-review-issues` (not `ralph/production-readiness-phase2`)
- ralph-task.md defines all 8 subtasks — 1 done (I-3), 7 remaining (I-4, I-1, S-1 through S-5)
- task_plan.md at repo root tracks code review fixes; docs/task_plan.md is the canonical full project plan
- `ANTAGONIST_PAIRS` in `04_gpbo_loop.py` was added for `compute_desirability()` — make sure the YAML loader works for both desirability gate and scorer contexts
- `_inflate_cellflow_variance` in `04_gpbo_loop.py` uses `importlib.util` to lazy-import step 06 — I-1 wants this extracted

## Key Context
- Branch: `ralph/fix-review-issues` (2 commits ahead of main)
- Tests: `python -m pytest gopro/tests/ -v` (866+ passing, 9 pre-existing failures)
- All §1 pipeline TODOs complete — these are post-merge code review cleanup items
- Config: `gopro/config.py` — all constants; `gopro/agents/pathway_rules.yaml` — antagonist rules

## Remaining: 7 tasks todo, 0 blocked, 1 complete
