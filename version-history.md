# Version History
> Tracks every iteration's git state. Find when things broke, revert to working commits.

| Iter | Commit | Status | Branch | Summary |
|------|--------|--------|--------|---------|
| 1 | 7f59baf | PASS | ralph/production-readiness-phase2 | TVR (Targeted Variance Reduction) — per-fidelity GP ensemble, 8 new tests (503 total). Simplify pass fixed 6 issues. |
| 1 | `f7c84a6` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 6 TVR issues: cost scaling, no_grad, missing rsample, dead code |
| 2 | `e3a1b90` | PASS | ralph/production-readiness-phase2 | Target profile refinement (DeMeo 2025), 7 new tests (510 total) |
| 2 | `08f3823` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 4 issues in refine_target_profile: dead code, vectorize cosine sim, drop redundant copy/normalization |
| 3 | `5a264b9` | PASS | ralph/production-readiness-phase2 | FBaxis_rank regionalization — A-P axis targeting, 11 new tests (521 total) |
| 3 | `af2d0ec` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 3 issues in FBaxis_rank: magic string, zero-row semantics, registry sync |
| 3 | `80643a8` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 3 issues in FBaxis_rank: magic string, zero-row semantics, registry sync |
| final | `1de4b09` | verified | ralph/production-readiness-phase2 | Bug hunter verified |
| 4 | `51212d5` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 3 quality issues in additive kernel: Literal type, robust ARD detection, dedup guard |
| final | `35c37c2` | verified | ralph/production-readiness-phase2 | Bug hunter verified |
| 5 | (pending) | PASS | ralph/production-readiness-phase2 | Adaptive complexity schedule tests — 8 new tests (534 total) |
| 5 | `28e7789` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 2 quality issues in adaptive complexity: remove unused param, fail-loud thresholds |
| final | `143f3db` | verified | ralph/production-readiness-phase2 | Bug hunter verified |
| 6 | `af96884` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 4 quality issues in timing window encoding |
| 7 | `af45a7f` | PASS | ralph/production-readiness-phase2 | Per-cell-type GP models (GPerturb 2025, Idea #11) — 6 new tests (547 total) |
| 7 | `47a2f6d` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 4 quality issues in per-type GP models |
| 7 | `72d5876` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix 4 quality issues in per-type GP models |
| 8 | `d1890d0` | PASS | ralph/production-readiness-phase2 | Per-round fidelity monitoring — 6 new tests (555 total) |
| 8 | `99e8b58` | working | ralph/production-readiness-phase2 | [ralph-simplify] Extract fidelity constants, remove redundant sort and str() wrapping |
