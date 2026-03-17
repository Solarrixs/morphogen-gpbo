# Handoff to Iteration 9

## Last Completed: Phase D Idea #13 — Per-round fidelity monitoring
- `monitor_fidelity_per_round()` tracks cross-fidelity correlation across rounds in `fidelity_monitoring.csv`
- Detects sustained degradation (2+ consecutive declining rounds) and triggers auto-fallback to single-fidelity
- Wired into `run_gpbo_loop()`: after validation gate, before merge decision
- `build_fidelity_trend_figure()` in visualize_report.py shows correlation trend per source
- Simplify pass extracted fidelity constants to config.py, removed redundant sort and str() wrapping
- 555 tests passing, 0 failures

## Next Up: Phase D Idea #16 — Convergence diagnostics
- **Task**: Track posterior variance, acquisition decay, recommendation clustering. Adaptive batch sizing. Add to gp_model_diagnostics CSV and viz report.
- **Acceptance**: diagnostics in CSV; 4+ tests
- **Key files**: `gopro/04_gpbo_loop.py` (diagnostics dict near line 2250), `gopro/visualize_report.py`
- **Tip**: The diagnostics dict is already written to `gp_diagnostics_round{N}.csv` — extend it with posterior_variance_mean, acquisition_value_decay, recommendation_cluster_spread

## Warnings
- Per-type GP only activates on the standard MAP path (not SAASBO, multi-fidelity, TVR, or MixedSingleTaskGP)
- Per-type GP requires `train_Y.shape[1] > 1`; single-output falls back to standard SingleTaskGP
- SAASBO + multi-output + scalarized qLogEI is guarded with NotImplementedError (iteration 5 fix)
- TVR + `high_variance` replicate strategy strips fidelity from `active_cols` (iteration 5 fix)
- Fidelity monitoring constants now in `gopro/config.py` (FIDELITY_DEGRADATION_WINDOW, etc.)
- Import constants from `gopro.config` — never hardcode paths or columns
- Use `.copy()` before mutating DataFrames passed as arguments

## Key Context
- Branch: `ralph/production-readiness-phase2`
- Tests: `source .venv/bin/activate && python -m pytest gopro/tests/ -v` (555 passing)
- Task list: `ralph-task.md` (7 tasks todo, 0 blocked, 8 complete)
- Iterations 1-8: TVR, target profile refinement, FBaxis_rank, additive+interaction kernel, adaptive complexity, timing windows, per-type GP, per-round fidelity monitoring

## Remaining: 7 tasks todo, 0 blocked, 8 complete
