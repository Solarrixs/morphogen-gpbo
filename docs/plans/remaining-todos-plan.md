# Remaining TODOs Implementation Plan

> Created: 2026-03-19
> Branch: ralph/production-readiness-phase2
> Baseline: 736 tests passing, 9 pre-existing failures
> Final: 782 tests passing, 9 pre-existing failures

## Batch 1: Quick Wins — COMPLETE (commit 667941f)

- [x] **TODO-37**: Failed experiment handling — `status` column filter in `build_training_set()`. 3 tests.
- [x] **TODO-36**: Carry-forward top-K controls — `--n-controls K` flag. 5 tests.
- [x] **TODO-52**: Hit threshold via MAD cutoff — `compute_hit_threshold()`. 6 tests.

## Batch 2: Contextual BO — COMPLETE (commit 6ddd6ea)

- [x] **TODO-12**: Contextual parameter support — `--contextual-cols` + BoTorch `fixed_features`. 3 tests.
- [x] **TODO-41**: Harvest day as contextual variable — wired through TODO-12 infra. (included in TODO-12 tests)

## Batch 3: NEST-Score Objective — COMPLETE (commit 6ddd6ea)

- [x] **TODO-15/49**: NEST-Score implementation — `gopro/signature_utils.py` with `compute_nest_score()`. 4 tests.
- [x] **TODO-16**: Gene signature scoring — `score_gene_signatures()` using `scanpy.tl.score_genes`. 2 tests.
- [x] **TODO-51**: Scrambled-signature permutation controls — `n_permutations` parameter with p-values. (included in TODO-16 tests)

## Batch 4: Benchmarking Foundation — COMPLETE (commit 6ddd6ea)

- [x] **TODO-53**: Compositional toy morphogen function — `gopro/benchmarks/toy_morphogen_function.py`. Simplex output. 13 tests.
- [x] **TODO-55**: ARD-derived Lipschitz diagnostic — `compute_ard_lipschitz()`. 5 tests.

## Deferred (needs external data, GPU, or lower priority)

- TODO-38: LHD gap-filling (Round 2+ only)
- TODO-39: Confirmation plate map (needs TODO-36 first — now unblocked)
- TODO-17+50: Signature refinement (needs TODO-15 data — now unblocked)
- TODO-42: Cost-aware desirability (lower priority)
- TODO-54: Noise-robustness sweep (needs TODO-53 — now unblocked)
- TODO-56: Multi-dose validation (wet-lab protocol)
- TODO-14: Noise characterization (partially done via confidence_noise)
- Sanchis-Calleja MF wire-up (needs h5ad processing)
