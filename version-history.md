| 17 | `b39acc4` | working | ralph/production-readiness-phase2 | [ralph-17] TODO-11: ALR transform as alternative to ILR (--alr), 679 tests |
| 3 | `48212c6` | working | ralph/production-readiness-phase2 | TODO-26: Fix CellFlow dose encoding to use log1p, 602 tests |
| 3 | `df3ea3a` | working | ralph/production-readiness-phase2 | [ralph-simplify] Add concentration_scale field to CellFlow encoding |
| 4 | `8bd0fe1` | working | ralph/production-readiness-phase2 | Bug-hunter final sweep: KernelSpec + 6 fixes, 580 tests |
| 3 | `578f3f6` | working | ralph/production-readiness-phase2 | Consolidate docs: single task_plan.md, remove 27 stale files |
| 4 | `d21fb52` | working | ralph/production-readiness-phase2 | [ralph-simplify] Hoist KernelSpec to module level, harden zero-guards in viz |
| 1 | `4313729` | working | ralph/production-readiness-phase2 | [ralph-simplify] Harden fidelity remap: use isclose, warn on unknown values, single-pass mask |
| final | `c8e82b1` | verified | ralph/production-readiness-phase2 | Bug hunter verified |
| 2 | `7e198ff` | working | ralph/production-readiness-phase2 | TODO-25: R²-based 3-zone fidelity routing, 599 tests |
| 2 | `faee5b2` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix method validation, extract metric dispatch |
| 2 | `02c948c` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix method validation, extract metric dispatch, remove dead import |
| final | `513eed8` | verified | ralph/production-readiness-phase2 | Bug hunter verified |
| 3 | `daac41a` | working | ralph/production-readiness-phase2 | [ralph-simplify] Add concentration_scale field to CellFlow encoding |
| final | `4acab09` | verified | ralph/production-readiness-phase2 | Bug hunter verified |
| 4 | `6625794` | working | ralph/production-readiness-phase2 | [ralph-simplify] Remove unused jnp import, DRY up JAX test mocks |
| 5 | `d227e73` | working | ralph/production-readiness-phase2 | [ralph-simplify] DRY up OOD warning tests: hoist import, use logger.name, remove noise columns |
| 6 | `7d753e2` | working | ralph/production-readiness-phase2 | [ralph-simplify] DRY up variance inflation: single canonical implementation, fix double-application risk |
| 7 | `18ae6e8` | working | ralph/production-readiness-phase2 | [ralph-simplify] DRY up ilr_transform: return_safe option eliminates duplicate _multiplicative_replacement call |
| 8 | `da16394` | working | ralph/production-readiness-phase2 | [ralph-simplify] Remove redundant filter and double column-existence check in log-scale |
| 9 | `46fb2c9` | working | ralph/production-readiness-phase2 | [ralph-simplify] Validate n_restarts >= 1 and log HP randomisation failures |
| 10 | `b9d6bb7` | working | ralph/production-readiness-phase2 | [ralph-simplify] Thread explicit_priors through fit_tvr_models |
| 11 | `1f1caee` | working | ralph/production-readiness-phase2 | [ralph-simplify] DRY TestFixedNoise tests, fix argparse % formatting |
| 12 | `09e43d1` | working | ralph/production-readiness-phase2 | [ralph-simplify] Warn on mc_samples clamping, DRY test setup |
| 13 | `4bcd31d` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix input_warp review issues: log spam, MixedGP consistency |
| 14 | `f81dc9b` | working | ralph/production-readiness-phase2 | [ralph-simplify] Fix per-fidelity ARD review issues: acquisition bounds, double extraction, heuristic |
| 15 | `73e0d42` | working | ralph/production-readiness-phase2 | [ralph-simplify] Remove dead code from ZeroPassingKernel: orphaned _phi and forward methods after return statement |
| 16 | `8b6c435` | working | ralph/production-readiness-phase2 | [ralph-simplify] Remove redundant compute_desirability call, add per-pathway debug logging |
| 17 | `f9ebf5d` | working | ralph/production-readiness-phase2 | [ralph-simplify] Clean up ALR transform: remove overly strict guard, vectorize variance loop, fix docstrings |
| 1 | `1a958ab` | working | ralph/fix-review-issues | [ralph-simplify] Fix TVR test: capture baseline before optimization, assert improvement direction |
| 2 | `0438a12` | working | ralph/fix-review-issues | [ralph-simplify] Reuse scorer.py YAML loader instead of duplicating it |
| 3 | `574e099` | working | ralph/fix-review-issues | [ralph-3] I-1: Inline _inflate_cellflow_variance — remove importlib from 04_gpbo_loop.py |
| 5 | pending | working | ralph/fix-review-issues | [ralph-5] S-2: Optimize score_gene_signatures — avoid full AnnData copy |
| 4 | `9727e60` | working | ralph/fix-review-issues | [ralph-simplify] Hoist ToyMorphogenFunction out of inner loop, rename arr→vals |
