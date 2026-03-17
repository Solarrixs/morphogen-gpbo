# Handoff to Iteration 5

## Last Completed: Bug-hunter final sweep + simplify pass (§1.1 complete)
All 15 production readiness items in §1.1 are done. Bug-hunter found 76 issues (6 critical, 38 warning, 32 info); 7 fixed this iteration including KernelSpec namedtuple dedup, zero-guards in viz, Sobol seed, Helmert caching, dead code removal. 580 tests passing.

## Next Up: TODO-24 — Remap fidelity encodings for MF-GP kernel (§1.2, CRITICAL)
fidelity=1.0 collapses the inter-fidelity kernel. Remap internally: 0.0→1/3, 0.5→1/2, 1.0→2/3. Add `_remap_fidelity()` helper in `04_gpbo_loop.py` ~L800.
**Acceptance:** MF-GP kernel sees fidelity values in (0,1) exclusive range; existing tests pass; 3+ new tests for remap function and round-trip.

Then TODO-25 (R²-based 3-zone fidelity routing) and TODO-26 (CellFlow log1p dose encoding) — all three are §1.2 critical bugs.

## Warnings
- Data CSVs in `data/` are modified but uncommitted (convergence_diagnostics, gp_diagnostics, gp_recommendations)
- `papers/` directory is untracked — not committed
- 5 bug-hunter criticals remain but are test coverage gaps, not code bugs
- Branch `ralph/production-readiness-phase2` — no PR yet

## Key Context
- 580 tests passing, run with `python -m pytest gopro/tests/ -v`
- Canonical task plan: `docs/task_plan.md` (also mirrored in `ralph-task.md`)
- Critical files for §1.2: `gopro/04_gpbo_loop.py` (TODO-24, TODO-25), `gopro/06_cellflow_virtual.py` (TODO-26), `gopro/config.py` (TODO-25 thresholds)
- Import constants from `gopro.config`, use `get_logger(__name__)`, `.copy()` before mutating DFs

## Remaining: ~55 tasks todo, 0 blocked, ~15 complete
