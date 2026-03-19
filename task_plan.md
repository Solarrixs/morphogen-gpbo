# Task Plan
> Goal: Pipeline Fixes & GP-BO Modeling Improvements
> Created: 2026-03-18T04:28:29Z
> Last updated: 2026-03-19

## Tasks

### Phase A: Critical Bug Fixes — COMPLETE
- [x] TODO-24: Remap fidelity encodings for MF-GP kernel
- [x] TODO-25: R²-based 3-zone fidelity routing
- [x] TODO-26: Fix CellFlow dose encoding to use log1p

### Phase B: CellFlow Integration Fixes — COMPLETE
- [x] TODO-1: Fix CellFlow JAX vs PyTorch API mismatch
- [x] TODO-3: Add Day 72 out-of-distribution warning — `_warn_ood_harvest_days()` implemented
- [x] TODO-4: Handle CellFlow conservative prediction bias — `confidence_to_noise_variance()` + `allow_fallback` gate

### Phase C–H: All pipeline TODOs complete. See docs/task_plan.md for full details (§1.3–§1.9 all marked COMPLETE).

### Paper work: See docs/task_plan.md §4 for detailed paper TODOs (fact-checks, figures, methods expansion).

## Architecture Decisions
| Decision | Rationale |
|----------|-----------|

## Errors Encountered
| Error | Task | Attempt | Resolution |
|-------|------|---------|------------|
