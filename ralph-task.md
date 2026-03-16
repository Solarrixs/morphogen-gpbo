# Task: Implement Remaining Competitive Landscape + Deferred Scientific Fixes

## Complexity: complex

## Context

Branch: ralph/production-readiness-phase2 (already pushed)
Tests: 495 gopro + 53 literature = 548 total, 0 failures
Venv: source .venv/bin/activate
Test cmd: python -m pytest gopro/tests/ -v

Phase A (cross-fidelity validation, cost ratios, warm-start, replicates) is DONE.
Scientific validation P0/P1 is DONE.
Production readiness (12 phases) is DONE.

Read docs/plans/competitive_landscape_ideas_index.md for the full 52-idea priority list.
Read docs/plans/ideas_from_*.md for per-idea implementation specs with DOIs and code repos.
Read docs/handoff/2026-03-16-ralph-loop-handoff.md for full context.

## Rules

- Run tests after EVERY subtask: python -m pytest gopro/tests/ -v
- Tests must be GENUINE — test real problems, never rig to pass
- Import constants from gopro.config — never hardcode paths or columns
- Use .copy() before mutating DataFrames
- Auto-detect GPU: torch.device("cuda" if torch.cuda.is_available() else "cpu")
- Read files BEFORE modifying them
- Keep solutions minimal — don't over-engineer

## Subtasks

- [ ] Phase B Idea #2: TVR (Targeted Variance Reduction) — Fit separate GPs per fidelity, scale variance by inverse cost, select model with lowest scaled variance at each candidate point. Add --tvr flag to 04_gpbo_loop.py. | Acceptance: --tvr produces recommendations; 5+ new tests
- [ ] Phase B Idea #4: Target profile refinement — After Round 1, update target_profile using observed best compositions interpolated toward Braun reference. Wire into run_gpbo_loop() as --refine-target flag. | Acceptance: refined target differs from original; 3+ tests
- [ ] Phase B Idea #12: FBaxis_rank regionalization — Extract A-P axis score from Sanchis-Calleja data. Add as continuous optimization target alongside discrete region profiles. | Acceptance: --target-region accepts "ap_axis" value; 3+ tests
- [x] Phase C Idea #8: Additive + interaction kernel — Replace Matern ARD with k_additive + k_interaction structure (NAIAD 2025). Reduces effective params from O(d^2) to O(d). Add --kernel additive_interaction flag. | Acceptance: GP fits with new kernel; 4+ tests
- [x] Phase C Idea #9: Adaptive complexity schedule — Round 1: shared lengthscale. Round 2+: per-dim ARD. Round 3+: SAASBO. Auto-select based on N/d ratio. | Acceptance: complexity auto-selected; 3+ tests
- [ ] Phase C Idea #10: Morphogen timing window encoding — Add temporal window categorical dimensions (early/mid/late patterning) to morphogen matrix. Use MixedSingleTaskGP for mixed continuous+categorical. | Acceptance: timing dims in training data; 3+ tests
- [ ] Phase C Idea #11: Per-cell-type GP models — Fit separate GP per cell type (MAP path, GPerturb 2025). Per-output lengthscale matrix for interpretability. Compare to current multi-output approach. | Acceptance: per-type GPs produce predictions; 4+ tests
- [ ] Phase D Idea #13: Per-round fidelity monitoring — Re-evaluate cross-fidelity correlation each round. Auto-fallback to single-fidelity if correlation degrades. Add trend to visualization report. | Acceptance: monitoring runs per round; 2+ tests
- [ ] Phase D Idea #16: Convergence diagnostics — Track posterior variance, acquisition decay, recommendation clustering. Adaptive batch sizing. Add to gp_model_diagnostics.csv and viz report. | Acceptance: diagnostics in CSV; 4+ tests
- [ ] Phase D Idea #17: Ensemble disagreement — Multi-restart GP fitting with recommendation stability scoring. Flag unstable recommendations. | Acceptance: stability score in diagnostics; 3+ tests
- [ ] Deferred P1-2: LassoBO — Implement Lasso-regularized lengthscale estimation as SAASBO alternative (AISTATS 2025). Much faster variable selection without NUTS overhead. Add --lassobo flag. | Acceptance: LassoBO fits GP; 4+ tests
- [ ] Deferred P2-1: Bootstrap uncertainty — Compute bootstrap confidence intervals on cell type fractions. Propagate as heteroscedastic GP noise via FixedNoiseGP. | Acceptance: per-condition noise estimates; 3+ tests
- [ ] Deferred P2-2: Data-driven entropy center — Replace arbitrary 0.55 entropy weight in composite fidelity with Braun reference mean entropy. | Acceptance: entropy center matches Braun; 2+ tests
- [ ] /simplify pass on all Phase B/C/D changes — Run 3-agent code review (reuse, quality, efficiency) on the diff. Fix HIGH/MEDIUM issues. | Acceptance: all tests pass after fixes
- [ ] /bug-hunter final sweep — Launch adversarial QA swarm across entire gopro/ codebase. Verify findings, fix confirmed criticals. | Acceptance: no confirmed critical bugs remain
