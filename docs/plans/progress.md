# Progress Log

## Iteration 2 — 2026-03-18
- Task: TODO-25 — R²-based 3-zone fidelity routing
- Result: pass
- Changes:
  - `config.py`: Added `FIDELITY_R2_THRESHOLDS = {"drop": 0.80, "skip": 0.90}` dict; old `FIDELITY_CORRELATION_THRESHOLD`/`FIDELITY_SKIP_MFBO_THRESHOLD` now aliases
  - `04_gpbo_loop.py`: Added `_compute_r_squared()` helper; rewrote `validate_fidelity_correlation()` to use R² by default with 3-zone routing; imports updated
  - `visualize_report.py`: Added `FIDELITY_R2_THRESHOLDS` import (legacy aliases still used for hlines)
  - `tests/test_fidelity_validation.py`: Rewrote test file — 29 tests total (was 19). New classes: `TestComputeRSquared` (4 tests), `TestFidelityR2Thresholds` (5 tests), `TestThreeZoneRouting` (4 tests). Updated existing tests for R² semantics.
- Tests: 599 passing (was 586). +13 net new tests.
- Notes: All §1.2 critical bugs except TODO-26 (CellFlow dose encoding) now resolved.

## Iteration 1 — 2026-03-18T00:00:44Z
- Task: TODO-24 — Remap fidelity encodings for MF-GP kernel (§1.2) + /simplify hardening pass
- Result: pass
- Commits:
  - `61bc4bb` [ralph-1] TODO-24: Remap fidelity encodings for MF-GP kernel
  - `4313729` [ralph-simplify] Harden fidelity remap: use isclose, warn on unknown values, single-pass mask
- Files changed: `gopro/04_gpbo_loop.py` (26 lines changed), `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md`, data CSVs updated
- Quality: Simplify pass hardened fidelity remap with `isclose` for float comparison, added warning on unknown fidelity values, consolidated to single-pass mask. Docs updated across audit report, architecture, competitive landscape index, and README.
- Notes: TODO-24 complete. Next priorities are TODO-25 (R²-based fidelity thresholds) and TODO-26 (CellFlow dose encoding). Both are §1.2 critical MF-GP fixes.

## Iteration 5 (Ralph Pipeline) — 2026-03-17
- Task: TODO-24 — Remap fidelity encodings for MF-GP kernel (§1.2)
- Result: pass
- Changes:
  - Added `FIDELITY_KERNEL_REMAP` / `FIDELITY_KERNEL_UNMAP` constants and `_remap_fidelity()` / `_unmap_fidelity()` helpers to `04_gpbo_loop.py`
  - MF-GP fitting path now remaps fidelity values from {0.0, 0.5, 1.0} → {1/3, 1/2, 2/3} before passing to `SingleTaskMultiFidelityGP`, preventing `LinearTruncatedFidelityKernel` boundary collapse
  - `recommend_next_experiments()` sets fidelity bounds to remapped value (2/3) instead of raw 1.0
  - Unknown fidelity values handled via linear interpolation fallback
- Tests: 586 passing (was 580). 6 new tests in `TestFidelityKernelRemap`.
- Files: `gopro/04_gpbo_loop.py`, `gopro/tests/test_unit.py`

## Post-Ralph Audit — 2026-03-17
- Task: Manual audit of all competitive landscape changes
- Result: 3 bugs found and fixed
- Fixes:
  - **CRITICAL**: TVR cost-scaling inverted (`var / cost` → `var * cost`). Was always selecting expensive model, making `--tvr` useless.
  - **MEDIUM**: `_select_replicate_conditions` now filters `fidelity == 1.0` — was potentially selecting unexecutable virtual conditions.
  - **LOW**: Removed redundant outer `ScaleKernel` on additive kernel (over-parameterized but didn't produce wrong results).
- Validated as correct: Aitchison distance, multiplicative replacement, GP warm-start, adaptive complexity, FBaxis A-P positions, cross-fidelity Spearman, noise estimation
- Tests: 561 → 561 (no new tests, existing tests updated for kernel structure)
- Commit: `db8eda2`

## Iteration 4 — 2026-03-17
- Task: /bug-hunter final sweep (task_plan §1.1)
- Result: pass
- Agents: 4 parallel analysis agents (bug-finder, optimizer, refactorer, test-coverage)
- Findings: 76 total (6 critical, 38 warning, 32 info). 7 fixed this run.
- Fixes applied:
  - A-C-001: KernelSpec namedtuple replaces twin `effective_saasbo`/`effective_kernel_type` variables
  - BF7-W-1: Lengthscale zero guard in importance reporting
  - BF7-C-1: `n_composition_parts > 1` consistency fix
  - OP-A10: Sobol seed=42 for reproducible convergence metrics
  - OP-A3 + RF D-W-009: `_helmert_basis` cached + torch delegates to numpy
  - OP-D2: Removed dead `compute_condition_region_fractions`
- Tests: 580 passing (was 575)
- Remaining criticals: 5 test coverage gaps (not code bugs)
- Reports: `.bug-hunter/SUMMARY.md`, `.bug-hunter/reports/`

## Iteration 3 — 2026-03-17T11:25:08Z
- Task: Data-driven entropy center (task_plan §1.1) + /simplify pass + docs consolidation
- Result: pass
- Commits:
  - `02219d4` [ralph-3] Task 13: Data-driven entropy center — Braun reference mean entropy replaces hardcoded 0.55
  - `4dfdf23` [ralph-simplify] Vectorize entropy loop, extract sigma constant
  - `ffd98d5` [ralph-meta-3] Version history update
  - `578f3f6` Consolidate docs: single task_plan.md, remove 27 stale files
- Files changed: 37 files (+1,438 / -8,171 lines) — major docs consolidation removed 27 stale files, unified into `docs/task_plan.md`
- Quality: Simplify pass vectorized entropy loop and extracted sigma constant. Docs reduced from scattered plans/handoffs/progress across root + docs/plans/ to a single consolidated task_plan.md.
- Notes: Old progress.md, handoff.md, findings.md, version-history.md all deleted in consolidation. This is a fresh progress log. Next iteration should tackle /bug-hunter or the critical TODOs (24-26).
