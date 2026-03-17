# Handoff to Iteration 10

## Last Completed: Phase D Idea #16 — Convergence diagnostics
- `compute_convergence_diagnostics()` tracks 3 metrics: mean posterior std, max acquisition value, recommendation spread
- Persistent CSV at `convergence_diagnostics.csv` with idempotent re-run support
- Adaptive batch suggestion when acquisition decays <10% of round 1 AND spread <0.05
- `build_convergence_diagnostics_figure()` added to visualize_report.py (3-panel subplot)
- Wired into `run_gpbo_loop()`: called after recommendations, metrics added to gp_diagnostics CSV
- Constants in `gopro/config.py`: CONVERGENCE_ACQUISITION_DECAY_THRESHOLD, CONVERGENCE_CLUSTER_SPREAD_THRESHOLD, CONVERGENCE_POSTERIOR_EVAL_POINTS
- 561 tests passing, 0 failures

## Next Up: Phase D Idea #17 — Ensemble disagreement
- **Task**: Multi-restart GP fitting with recommendation stability scoring. Flag unstable recommendations.
- **Acceptance**: stability score in diagnostics; 3+ tests
- **Key files**: `gopro/04_gpbo_loop.py` (fit_gp_botorch or new function), diagnostics dict
- **Tip**: Fit N independent GPs from different random restarts, generate recommendations from each, compute pairwise agreement metrics (Kendall's tau on lengthscales, cosine similarity between recommended vectors). Report stability score in diagnostics CSV.

## Warnings
- Per-type GP only activates on the standard MAP path (not SAASBO, multi-fidelity, TVR, or MixedSingleTaskGP)
- SAASBO + multi-output + scalarized qLogEI is guarded with NotImplementedError
- TVR + `high_variance` replicate strategy strips fidelity from `active_cols`
- Convergence diagnostics uses SobolEngine for posterior variance estimation (512 points default)
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (561 passing)
- Task list: `ralph-task.md` (6 tasks todo, 0 blocked, 9 complete)
- Iterations 1-9: TVR, target profile refinement, FBaxis_rank, additive+interaction kernel, adaptive complexity, timing windows, per-type GP, per-round fidelity monitoring, convergence diagnostics

## Remaining: 6 tasks todo, 0 blocked, 9 complete
