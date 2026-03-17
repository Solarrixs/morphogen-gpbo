# Progress Log

## Iteration Log

### Iteration 8 (2026-03-16)
- **Task**: Task 8 — Per-round fidelity monitoring (Phase D Idea #13)
- **Status**: PASS
- **Changes**:
  - `gopro/04_gpbo_loop.py`: Added `monitor_fidelity_per_round()` — tracks cross-fidelity correlation per round in persistent CSV, detects degradation trends, auto-fallback to single-fidelity when correlation declines for 2+ consecutive rounds. Wired into `run_gpbo_loop()` after validation gate.
  - `gopro/visualize_report.py`: Added `build_fidelity_trend_figure()` — Plotly line chart showing correlation trend per fidelity source with threshold reference lines. Wired into `generate_report()`.
  - `gopro/tests/test_fidelity_validation.py`: Added 5 tests in `TestMonitorFidelityPerRound` class
  - `gopro/tests/test_unit.py`: Added 1 test for `build_fidelity_trend_figure`
- **Tests**: 555 passed (was 547)

### Iteration 1 (2026-03-16)
- **Task**: Task 1 — TVR (Targeted Variance Reduction)
- **Status**: PASS
- **Changes**:
  - `gopro/04_gpbo_loop.py`: Added `fit_tvr_models()`, `TVRModelEnsemble`, `_TVRPosterior`, `--tvr` CLI flag, wired into `run_gpbo_loop(use_tvr=True)`
  - `gopro/tests/test_unit.py`: Added 8 tests in `TestTVR` class
- **Tests**: 503 passed (was 495)
