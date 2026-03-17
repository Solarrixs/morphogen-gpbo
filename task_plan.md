# Task Plan
> Goal: Finish remaining competitive landscape ideas (Tasks 11-13) + Final QA (Tasks 14-15)
> Created: 2026-03-17T09:57:01Z

## Tasks
- [x] Task 11: LassoBO — Lasso-regularized lengthscale estimation as SAASBO alternative (AISTATS 2025). Add --lassobo flag to 04_gpbo_loop.py. Much faster variable selection without NUTS. | Acceptance: LassoBO fits GP; 7 new tests
- [x] Task 12: Bootstrap uncertainty — Compute bootstrap CIs on cell type fractions from per-cell KNN probabilities. Propagate as heteroscedastic GP noise via SingleTaskGP(train_Yvar). | Acceptance: per-condition noise estimates; 4 new tests
- [x] Task 13: Data-driven entropy center — Replace arbitrary 0.55 entropy weight in composite fidelity (03_fidelity_scoring.py) with Braun reference mean entropy. | Acceptance: entropy center matches Braun; 5 new tests
- [ ] Task 14: /simplify pass on all changes | Acceptance: all tests pass after fixes
- [ ] Task 15: /bug-hunter final sweep | Acceptance: no confirmed critical bugs remain

## Architecture Decisions
| Decision | Rationale |
|----------|-----------|

## Errors Encountered
| Error | Task | Attempt | Resolution |
|-------|------|---------|------------|
