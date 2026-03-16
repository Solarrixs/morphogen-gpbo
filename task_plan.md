# Task Plan
> Goal: Implement remaining competitive landscape ideas + deferred scientific fixes
> Created: 2026-03-16T07:35:29Z

## Tasks
- [x] Task 1: TVR (Targeted Variance Reduction) — Fit separate GPs per fidelity, scale variance by inverse cost, select model with lowest scaled variance. Add --tvr flag. | Acceptance: --tvr produces recommendations; 5+ new tests
- [ ] Task 2: Target profile refinement — Update target_profile using observed best compositions interpolated toward Braun reference. --refine-target flag. | Acceptance: refined target differs from original; 3+ tests
- [ ] Task 3: FBaxis_rank regionalization — Extract A-P axis score from Sanchis-Calleja data as continuous optimization target. | Acceptance: --target-region accepts "ap_axis"; 3+ tests
- [ ] Task 4: Additive + interaction kernel — k_additive + k_interaction structure (NAIAD 2025). --kernel additive_interaction flag. | Acceptance: GP fits with new kernel; 4+ tests
- [ ] Task 5: Adaptive complexity schedule — Auto-select GP complexity based on N/d ratio. | Acceptance: complexity auto-selected; 3+ tests
- [ ] Task 6: Morphogen timing window encoding — Add temporal categorical dimensions. MixedSingleTaskGP. | Acceptance: timing dims in training data; 3+ tests
- [ ] Task 7: Per-cell-type GP models — Separate GP per cell type (GPerturb 2025). | Acceptance: per-type GPs produce predictions; 4+ tests
- [ ] Task 8: Per-round fidelity monitoring — Re-evaluate cross-fidelity correlation each round. Auto-fallback. | Acceptance: monitoring runs per round; 2+ tests
- [ ] Task 9: Convergence diagnostics — Track posterior variance, acquisition decay, recommendation clustering. | Acceptance: diagnostics in CSV; 4+ tests
- [ ] Task 10: Ensemble disagreement — Multi-restart GP with stability scoring. | Acceptance: stability score in diagnostics; 3+ tests
- [ ] Task 11: LassoBO — Lasso-regularized lengthscale estimation as SAASBO alternative. --lassobo flag. | Acceptance: LassoBO fits GP; 4+ tests
- [ ] Task 12: Bootstrap uncertainty — Bootstrap CIs on cell type fractions → heteroscedastic GP noise. | Acceptance: per-condition noise estimates; 3+ tests
- [ ] Task 13: Data-driven entropy center — Replace 0.55 entropy weight with Braun reference mean entropy. | Acceptance: entropy center matches Braun; 2+ tests
- [ ] Task 14: /simplify pass on all Phase B/C/D changes | Acceptance: all tests pass after fixes
- [ ] Task 15: /bug-hunter final sweep | Acceptance: no confirmed critical bugs remain

## Architecture Decisions
| Decision | Rationale |
|----------|-----------|

## Errors Encountered
| Error | Task | Attempt | Resolution |
|-------|------|---------|------------|
