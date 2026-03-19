# Handoff to Iteration 8

## Last Completed: S-4 — Fix _fidelity_to_task_idx dtype to torch.long
- Changed `dtype=fidelity_values.dtype` → `dtype=torch.long` in `gopro/04_gpbo_loop.py:394`
- Added dtype assertions to both existing tests
- Commit: `71aa6b2`

## Next Up: S-5 — Add 12+ unit tests for new features
- TVRModelEnsemble (2+ tests)
- ZeroPassingKernel (2+ tests)
- generate_validation_plate (2+ tests)
- generate_confirmation_plate (2+ tests)
- apply_desirability_gate (2+ tests)
- confidence_to_noise_variance (2+ tests)
- Acceptance: 12+ new tests added, all pass

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `ralph-task.md` is untracked
- 9 pre-existing test failures (gruffi/scgpt/numba) — ignore these
- `ZeroPassingKernel` uses lazy factory `_get_zero_passing_kernel_class()` — never reference global
- CellFlow uses JAX (`jax.random`), NOT torch

## Key Context
- Branch: `ralph/fix-review-issues`
- Task definition: `ralph-task.md` (8 subtasks from code review; 7 complete, 1 remaining)
- Tests: `.venv/bin/python -m pytest gopro/tests/ -v`
- Config: `gopro/config.py` — all constants; import from here, never hardcode

## Remaining: 1 task todo, 0 blocked, 7 complete
