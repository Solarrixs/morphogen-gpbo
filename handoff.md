# Handoff Document
> Written at end of each iteration for the next fresh-context agent.

## Last Completed
Task 1: TVR (Targeted Variance Reduction) — 8 new tests, all 503 pass

## Next Up
Task 2: Target profile refinement (--refine-target flag)

## Warnings
- TVR `_TVRPosterior` is a lightweight wrapper, not a full GPyTorchPosterior. Works with `qLogExpectedImprovement` via `sample()` but may need adaptation for more advanced acquisition functions.
- Multi-output GP noise is a tensor, not scalar — use `.mean().item()` for logging.

## Key Context
- Branch: ralph/production-readiness-phase2
- Tests: 503 gopro (was 495, +8 TVR)
- TVR implementation: `fit_tvr_models()`, `TVRModelEnsemble`, `_TVRPosterior` in 04_gpbo_loop.py
- TVR is wired into `run_gpbo_loop(use_tvr=True)` and CLI `--tvr` flag
- TVR fits separate SingleTaskGP per fidelity level, selects by lowest cost-scaled variance
- When TVR is active, fidelity column is excluded from recommendation columns/bounds
