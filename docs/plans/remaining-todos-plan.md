# Remaining TODOs Implementation Plan

> Created: 2026-03-19
> Branch: ralph/production-readiness-phase2
> Baseline: 736 tests passing, 9 pre-existing failures

## Batch 1: Quick Wins (parallel, no dependencies)

- [ ] **TODO-37**: Failed experiment handling — `status` column filter in `build_training_set()`. ~40 LoC, 2 tests.
- [ ] **TODO-36**: Carry-forward top-K controls — `--n-controls K` flag, reuse `_select_replicate_conditions(strategy="high_value")`. ~30 LoC, 2 tests.
- [ ] **TODO-52**: Hit threshold via MAD cutoff — `compute_hit_threshold()` in `03_fidelity_scoring.py`. ~40 LoC, 2 tests.

## Batch 2: Contextual BO (sequential)

- [ ] **TODO-12**: Contextual parameter support — `--contextual-cols` + BoTorch `fixed_features`. ~80 LoC, 3 tests.
- [ ] **TODO-41**: Harvest day as discrete contextual — wire `log_harvest_day` through TODO-12 infra. ~40 LoC, 2 tests.

## Batch 3: NEST-Score Objective (sequential)

- [ ] **TODO-15/49**: NEST-Score implementation — new `gopro/signature_utils.py` with `compute_nest_score()`. ~120 LoC, 4 tests.
- [ ] **TODO-16**: Atlas-derived maturity signatures — `score_maturity_signatures()` using `scanpy.tl.score_genes`. ~60 LoC, 3 tests.
- [ ] **TODO-51**: Scrambled-signature permutation controls — 1000 permutations, p-value. ~40 LoC, 2 tests.

## Batch 4: Benchmarking Foundation

- [ ] **TODO-53**: Compositional toy morphogen function — new `gopro/benchmarks/`. Simplex output. ~100 LoC, 3 tests.
- [ ] **TODO-55**: ARD-derived Lipschitz diagnostic — `L_d ~ sigma_f / l_d` + posterior std. ~30 LoC, 2 tests.

## Deferred (needs external data, GPU, or lower priority)

- TODO-38: LHD gap-filling (Round 2+ only)
- TODO-39: Confirmation plate map (needs TODO-36 first)
- TODO-17+50: Signature refinement (needs TODO-15 data)
- TODO-42: Cost-aware desirability (lower priority)
- TODO-54: Noise-robustness sweep (needs TODO-53)
- TODO-56: Multi-dose validation (wet-lab protocol)
- TODO-14: Noise characterization (partially done via confidence_noise)
- Sanchis-Calleja MF wire-up (needs h5ad processing)
