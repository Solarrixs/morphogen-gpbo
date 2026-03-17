# Task Plan
> Goal: Implement remaining competitive landscape ideas + deferred scientific fixes
> Created: 2026-03-16T07:35:29Z
> Updated: 2026-03-17

## Tasks
- [x] Task 1: TVR (Targeted Variance Reduction) — per-fidelity GP ensemble, cost-scaled variance. --tvr flag. (+8 tests, 503 total)
- [x] Task 2: Target profile refinement — refine_target_profile() with softmax interpolation. --refine-target flag. (+7 tests, 510 total)
- [x] Task 3: FBaxis_rank regionalization — continuous A-P axis targeting. --target-region ap_axis. (+11 tests, 521 total)
- [x] Task 4: Additive + interaction kernel — k_additive + k_interaction (NAIAD 2025). --kernel additive_interaction. (+5 tests, 526 total)
- [x] Task 5: Adaptive complexity schedule — auto-select shared/ARD/SAASBO by N/d ratio. (+8 tests, 534 total)
- [x] Task 6: Morphogen timing window encoding — categorical early/mid/late dims, MixedSingleTaskGP. (~5 tests, 539 total)
- [x] Task 7: Per-cell-type GP models — separate GP per cell type (GPerturb 2025). (+6 tests, 547 total)
- [x] Task 8: Per-round fidelity monitoring — auto-fallback on correlation degradation. (+6 tests, 555 total)
- [x] Task 9: Convergence diagnostics — posterior variance, acquisition decay, recommendation clustering. (~6 tests, 561 total)
- [ ] Task 10: Ensemble disagreement — Multi-restart GP with stability scoring. | Acceptance: stability score in diagnostics; 3+ tests
- [ ] Task 11: LassoBO — Lasso-regularized lengthscale estimation as SAASBO alternative. --lassobo flag. | Acceptance: LassoBO fits GP; 4+ tests
- [ ] Task 12: Bootstrap uncertainty — Bootstrap CIs on cell type fractions → heteroscedastic GP noise. | Acceptance: per-condition noise estimates; 3+ tests
- [ ] Task 13: Data-driven entropy center — Replace 0.55 entropy weight with Braun reference mean entropy. | Acceptance: entropy center matches Braun; 2+ tests
- [ ] Task 14: /simplify pass on all Phase B/C/D changes | Acceptance: all tests pass after fixes
- [ ] Task 15: /bug-hunter final sweep | Acceptance: no confirmed critical bugs remain

## Status: 9/15 tasks complete, 561 tests passing

## Architecture Decisions
| Decision | Rationale |
|----------|-----------|
| TVR over SingleTaskMultiFidelityGP | McDonald 2025: independent GPs per fidelity are more robust with small data than learned inter-fidelity covariance |
| Additive + interaction kernel | NAIAD 2025: reduces effective params from O(d^2) to O(d), encodes morphogen independence prior |
| Adaptive complexity by N/d | Avoids overfitting: shared lengthscale with <5 points/dim, ARD with 5-15, SAASBO with >15 |
| Per-cell-type GPs | GPerturb 2025: per-output lengthscale matrix reveals which morphogens matter for each cell type |
| Aitchison distance over cosine | Cosine violates compositional geometry; Aitchison is the correct metric (Aitchison 1986) |
| Multiplicative replacement over 1e-10 | 1e-10 pseudo-count creates log-ratios of -23 dominating GP training; CoDA-correct delta ~3.8e-4 |

## Errors Encountered
| Error | Task | Attempt | Resolution |
|-------|------|---------|------------|
| Stalled claude -p | Task 5 (run 2) | 1 | Killed after 2h20m, restarted from iteration 5 |
