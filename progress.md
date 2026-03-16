# Progress Log

## Iteration Log

### Iteration 1 (2026-03-16)
- **Task**: Task 1 — TVR (Targeted Variance Reduction)
- **Status**: PASS
- **Changes**:
  - `gopro/04_gpbo_loop.py`: Added `fit_tvr_models()`, `TVRModelEnsemble`, `_TVRPosterior`, `--tvr` CLI flag, wired into `run_gpbo_loop(use_tvr=True)`
  - `gopro/tests/test_unit.py`: Added 8 tests in `TestTVR` class
- **Tests**: 503 passed (was 495)
