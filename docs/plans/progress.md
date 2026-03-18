# Progress Log

## Iteration 23 — 2026-03-18
- Task: TODO-11 — ILR vs ALR comparison (`--alr`) (§1.4 GP Model Improvements)
- Result: pass
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `alr_transform()`, `alr_inverse()` functions; `use_alr` param threaded through `fit_gp_botorch`, `fit_tvr_models`, `recommend_next_experiments`, `compute_ensemble_disagreement`, `run_gpbo_loop`; `--alr` CLI flag; ALR delta-method variance propagation; differentiable ALR inverse for acquisition scalarization; mutual exclusivity guard
  - `gopro/tests/test_unit.py` — Added `TestALRTransform` class with 7 tests (dimension reduction, roundtrip, zeros, reference component, return_safe, uniform, sums-to-one)
  - `docs/task_plan.md` — Marked TODO-11 complete, updated test count to 679
  - `docs/plans/handoff.md`, `docs/plans/progress.md` — updated
- Tests: 679 passing (was 672)
- Notes: §1.4 GP Model Improvements: 11/13 done. Remaining: TODO-8 (spike-and-slab, high effort), TODO-10 (Dirichlet).

## Iteration 22 — 2026-03-18T09:08:26Z
- Task: TODO-7 — Desirability-based feasibility gate (`--desirability-gate`) (§1.4 GP Model Improvements) + simplify pass (redundant compute_desirability call removal, per-pathway debug logging)
- Result: pass
- Commits:
  - `757eb62` [ralph-16] TODO-7: Desirability-based feasibility gate (--desirability-gate)
  - `8b6c435` [ralph-simplify] Remove redundant compute_desirability call, add per-pathway debug logging
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `compute_desirability()`, `ANTAGONIST_PAIRS` constant; `--desirability-gate` CLI flag; generates 2x candidates, scores and keeps top N; simplify pass removed redundant call, added per-pathway debug logging (+18/-1 lines net)
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass removed redundant `compute_desirability` call and added per-pathway debug logging for troubleshooting. 9 files changed, +281/-119 lines.
- Tests: 672 passing (was 662)
- Notes: §1.4 GP Model Improvements: 10/13 done (TODO-5, TODO-6, TODO-7, TODO-9, TODO-27, TODO-28, TODO-29, TODO-30, TODO-31, TODO-32). Remaining: TODO-8 (spike-and-slab), TODO-10 (Dirichlet), TODO-11 (ILR vs ALR).

## Iteration 21 — 2026-03-18T08:50:35Z
- Task: TODO-6 — Zero-passing kernel (`--zero-passing`) (§1.4 GP Model Improvements) + simplify pass (dead code removal)
- Result: pass
- Commits:
  - `4962c7d` [ralph-15] TODO-6: Zero-passing kernel (--zero-passing)
  - `73e0d42` [ralph-simplify] Remove dead code from ZeroPassingKernel: orphaned _phi and forward methods after return statement
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `ZeroPassingKernel` via lazy factory `_get_zero_passing_kernel_class()`; smooth mask `phi(x) = 1 - exp(-||x_conc||^2 / eps)` forces k(0,x)=0 for concentration inputs; `--zero-passing` CLI flag; wired into standard and per-type-GP MAP paths; simplify pass removed orphaned `_phi` and `forward` methods after return statement (-27 lines)
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass removed dead code (orphaned `_phi` and `forward` methods that were unreachable after a return statement in ZeroPassingKernel). 9 files changed, +252/-143 lines.
- Tests: 662 passing (was 654)
- Notes: §1.4 GP Model Improvements: 9/13 done (TODO-5, TODO-6, TODO-9, TODO-27, TODO-28, TODO-29, TODO-30, TODO-31, TODO-32). Remaining: TODO-7 (desirability gate), TODO-8 (spike-and-slab), TODO-10 (Dirichlet), TODO-11 (ILR vs ALR).

## Iteration 20 — 2026-03-18T08:31:16Z
- Task: TODO-5 — Per-fidelity ARD lengthscales (`--per-fidelity-ard`) (§1.4 GP Model Improvements) + simplify pass
- Result: pass
- Commits:
  - `887a0b5` [ralph-14] TODO-5: Per-fidelity ARD lengthscales (--per-fidelity-ard)
  - `f81dc9b` [ralph-simplify] Fix per-fidelity ARD review issues: acquisition bounds, double extraction, heuristic
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `_fidelity_to_task_idx()`, `_build_per_fidelity_ard_model()`, `_extract_per_fidelity_ard_lengthscales()`; `--per-fidelity-ard` CLI flag; simplify pass fixed acquisition bounds, double extraction, heuristic (+63/-48 lines)
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass fixed acquisition bounds for per-fidelity ARD path, removed double lengthscale extraction, improved heuristic. 9 files changed, +274/-143 lines.
- Tests: 654 passing (was 647)
- Notes: §1.4 GP Model Improvements: 8/13 done (TODO-5, TODO-9, TODO-27, TODO-28, TODO-29, TODO-30, TODO-31, TODO-32). Remaining: TODO-6 (zero-passing kernel), TODO-7 (desirability gate), TODO-8 (spike-and-slab), TODO-10 (Dirichlet), TODO-11 (ILR vs ALR).

## Iteration 18 — 2026-03-18T08:01:25Z
- Task: TODO-27 — Kumaraswamy CDF input warping (`--input-warp`) (§1.4 GP Model Improvements) + simplify pass
- Result: pass
- Commits:
  - `f1caa6b` [ralph-13] TODO-27: Kumaraswamy CDF input warping (--input-warp)
  - `4bcd31d` [ralph-simplify] Fix input_warp review issues: log spam, MixedGP consistency
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `_build_input_transform(d, warp, cat_dims)` helper; `--input-warp` CLI flag; Kumaraswamy CDF warp via `ChainedInputTransform(warp=Warp(...), normalize=Normalize(...))` on MAP and per-type-GP paths; simplify pass fixed log spam and MixedGP consistency (+9/-9 lines)
  - `gopro/tests/test_unit.py` — 5 new tests for input warping
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass fixed log spam (warping skip messages downgraded) and MixedGP consistency (input transform now applied uniformly). 9 files changed, +228/-120 lines.
- Tests: 647 passing (was 642)
- Notes: §1.4 GP Model Improvements: 7/13 done (TODO-9, TODO-27, TODO-28, TODO-29, TODO-30, TODO-31, TODO-32). Remaining: TODO-5 (per-fidelity ARD), TODO-6 (zero-passing kernel), TODO-7 (desirability gate), TODO-8 (spike-and-slab), TODO-10 (Dirichlet), TODO-11 (ILR vs ALR).

## Iteration 17 — 2026-03-18T07:43:21Z
- Task: TODO-32 — Sobol QMC sampler for acquisition functions (§1.4 GP Model Improvements) + simplify pass
- Result: pass
- Commits:
  - `2aa572e` [ralph-12] TODO-32: Sobol QMC sampler for acquisition functions
  - `09e43d1` [ralph-simplify] Warn on mc_samples clamping, DRY test setup
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `SobolQMCNormalSampler` to all 3 acquisition constructors; `--mc-samples N` CLI flag (default 512, max 2048); clamping with warning (+4 lines in simplify)
  - `gopro/tests/test_unit.py` — 4 new tests for Sobol sampler; DRY'd test setup (+54/-36 lines net)
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass added warning on mc_samples clamping, DRY'd up test setup. 10 files changed, +246/-136 lines.
- Tests: 642 passing (was 638)
- Notes: §1.4 GP Model Improvements: 6/13 done (TODO-9, TODO-28, TODO-29, TODO-30, TODO-31, TODO-32). Next: TODO-27 (input warping), TODO-5 (per-fidelity ARD), or TODO-6 (zero-passing kernel).

## Iteration 16 — 2026-03-18T07:25:03Z
- Task: TODO-31 simplify pass — DRY TestFixedNoise tests, fix argparse % formatting
- Result: pass
- Commits:
  - `1f1caee` [ralph-simplify] DRY TestFixedNoise tests, fix argparse % formatting
  - `c577b1f` [ralph-11] TODO-31: FixedNoiseGP with heteroscedastic noise
- Files changed:
  - `gopro/04_gpbo_loop.py` — Fixed argparse % formatting (+10/-10 lines)
  - `gopro/tests/test_unit.py` — DRY'd TestFixedNoise tests (+100/-100 lines net refactor)
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass DRY'd up TestFixedNoise test class (reduced duplication), fixed argparse metavar % formatting issue. 9 files changed, +228/-192 lines.
- Notes: §1.4 GP Model Improvements: 5/13 done (TODO-9, TODO-28, TODO-29, TODO-30, TODO-31). Next: TODO-32 (Sobol QMC sampler), TODO-27 (input warping), or TODO-5 (per-fidelity ARD).

## Iteration 15 — 2026-03-18
- Task: TODO-31 — FixedNoiseGP with heteroscedastic noise (§1.4)
- Result: pass
- Tests: 634 → 638 (+4 new tests)
- Changes:
  - `gopro/config.py` — Added `FIXED_NOISE_MIN_VARIANCE = 0.02` constant
  - `gopro/04_gpbo_loop.py` — Imported `FIXED_NOISE_MIN_VARIANCE`; changed train_Yvar clamp from 1e-6 to 0.02; added `fixed_noise` param to `run_gpbo_loop()` with auto-discovery of bootstrap CSV at `data/gp_noise_variance_amin_kelley.csv` and uniform-noise fallback from Y column variance; added `--fixed-noise` CLI flag
  - `gopro/tests/test_unit.py` — Added `TestFixedNoise` class with 4 tests (config export, noise clamp, uniform fallback, zero-noise clamp)
  - `docs/task_plan.md` — Marked TODO-31 complete, updated test count to 638
- Notes: §1.4 GP Model Improvements: TODO-9, TODO-28, TODO-29, TODO-30, TODO-31 done (5/13). Next: TODO-32 (Sobol QMC sampler), TODO-27 (input warping), or TODO-5 (per-fidelity ARD).

## Iteration 14 — 2026-03-18T07:09:16Z
- Task: Simplify pass on TODO-30 (explicit GP priors)
- Result: pass
- Commits:
  - `b9d6bb7` [ralph-simplify] Thread explicit_priors through fit_tvr_models
  - `326ba87` [ralph-10] TODO-30: Explicit GP priors (lengthscale + noise)
- Files changed:
  - `gopro/04_gpbo_loop.py` — Threaded `explicit_priors` param through `fit_tvr_models()` (+8 lines)
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass threaded `explicit_priors` through TVR model fitting path that was missed in initial implementation. Docs updated.
- Tests: 634 passing (unchanged)
- Notes: §1.4 GP Model Improvements: TODO-9, TODO-28, TODO-29, TODO-30 all done (4/13). Next: TODO-31 (FixedNoiseGP), TODO-32 (Sobol QMC), or TODO-27 (input warping).

## Iteration 13 — 2026-03-18
- Task: TODO-30 — Explicit GP priors (Cosenza 2022)
- Result: pass
- Tests: 631 → 634 (+3 new tests)
- Changes:
  - `gopro/04_gpbo_loop.py` — Added `_set_noise_prior()` (Gamma(3,6) on likelihood noise), `_set_explicit_priors()` (combines lengthscale + noise priors); added `explicit_priors` param to `fit_gp_botorch` and `run_gpbo_loop`; wired into all MAP factory paths (standard, MF, Mixed, per-type); added `--explicit-priors` CLI flag
  - `gopro/tests/test_unit.py` — Added `TestExplicitPriors` class with 3 tests (noise prior attachment, combined priors, integration with fit_gp_botorch)

## Iteration 12 — 2026-03-18T06:42:41Z
- Task: Simplify pass on TODO-29 (MLL restarts hardening)
- Result: pass
- Commits:
  - `46fb2c9` [ralph-simplify] Validate n_restarts >= 1 and log HP randomisation failures
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `n_restarts >= 1` validation guard; added logging for HP randomisation failures in `_fit_mll_with_restarts()`
- Quality: Hardened `_fit_mll_with_restarts()` with input validation and failure logging. No new tests needed (existing tests cover the behavior).
- Tests: 631 passing (unchanged)
- Notes: §1.4 GP Model Improvements: TODO-9, TODO-28, TODO-29 all done. Next: TODO-30 (explicit GP priors), TODO-31 (FixedNoiseGP), or TODO-32 (Sobol QMC sampler).

## Iteration 11 — 2026-03-18
- Task: TODO-29 — MLL optimization restarts
- Result: pass
- Tests: 627 → 631 (+4 new tests)
- Changes:
  - `gopro/04_gpbo_loop.py` — Added `_fit_mll_with_restarts()` helper; added `mll_restarts` param to `fit_gp_botorch` and `run_gpbo_loop`; refactored all MAP fitting paths (standard, MixedSingleTaskGP, per-type-GP, multi-fidelity) to use factory+restarts pattern; added `--mll-restarts` CLI flag
  - `gopro/tests/test_unit.py` — Added `TestMLLRestarts` class with 4 tests (single restart, multiple restarts, all-fail, integration with fit_gp_botorch)

## Iteration 10 — 2026-03-18T06:09:19Z
- Task: Simplify pass on TODO-28 (log-scale code cleanup)
- Result: pass
- Commits:
  - `da16394` [ralph-simplify] Remove redundant filter and double column-existence check in log-scale
- Files changed:
  - `gopro/config.py` — Removed redundant `and col != "log_harvest_day"` filter (`log_harvest_day` doesn't end in `_uM` anyway)
  - `gopro/04_gpbo_loop.py` — Moved `log_scaled_cols` computation after `_apply_log_scale()` calls; pass `LOG_SCALE_COLUMNS` directly to helper (which already handles missing columns internally)
- Quality: Removed double column-existence check — `_apply_log_scale()` already filters to columns present in the DataFrame, so pre-filtering was redundant. Also removed a tautological exclusion in config.
- Tests: 627 passing (unchanged)
- Notes: Simplify-only iteration, no new functionality. §1.4 GP Model Improvements: TODO-9 and TODO-28 done. Next: TODO-29 (MLL restarts), TODO-30 (explicit GP priors), or TODO-31 (FixedNoiseGP).

## Iteration 9 — 2026-03-18
- Task: TODO-28 — Selective log-scaling for concentration dimensions (§1.4 GP Model Improvements)
- Result: pass
- Files changed:
  - `gopro/config.py` — Added `LOG_SCALE_COLUMNS` (all `_uM` columns, excluding `log_harvest_day`)
  - `gopro/04_gpbo_loop.py` — Added `_apply_log_scale()`, `_inverse_log_scale()` helpers; added `log_scale` param to `run_gpbo_loop()`; added `--log-scale` CLI flag; imported `LOG_SCALE_COLUMNS` from config
  - `gopro/tests/test_unit.py` — 7 new tests in `TestLogScale`
  - `docs/task_plan.md` — Marked TODO-28 complete, updated test count to 627
- Tests: 627 passing (was 620)
- Notes: §1.4 GP Model Improvements: TODO-9 and TODO-28 done. Next: TODO-29 (MLL restarts), TODO-30 (explicit GP priors), or TODO-31 (FixedNoiseGP).

## Iteration 8 — 2026-03-18T05:46:59Z
- Task: TODO-9 — Configurable pseudocount for ILR zero-replacement (§1.4 GP Model Improvements)
- Result: pass
- Commits:
  - `4529b34` [ralph-7] TODO-9: Configurable pseudocount for ILR zero-replacement
  - `18ae6e8` [ralph-simplify] DRY up ilr_transform: return_safe option eliminates duplicate _multiplicative_replacement call
- Files changed:
  - `gopro/04_gpbo_loop.py` — Added `pseudocount` parameter to `ilr_transform()`, threaded through `_multiplicative_replacement()`, `fit_gp_botorch()`, `fit_tvr_models()`, `compute_ensemble_disagreement()`, `run_gpbo_loop()`; added `--pseudocount` CLI flag; simplify pass added `return_safe` option to eliminate duplicate `_multiplicative_replacement` call
  - `gopro/tests/test_unit.py` — 3 new tests for pseudocount handling; simplify pass cleaned up test fixtures
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass DRY'd up `ilr_transform` with `return_safe` option, eliminating duplicate `_multiplicative_replacement` call path. No issues found.
- Notes: §1.4 GP Model Improvements started. TODO-9 complete. Next: TODO-28 (selective log-scaling), TODO-29 (MLL restarts), or TODO-30 (explicit GP priors).

## Iteration 7 — 2026-03-18T05:24:53Z
- Task: TODO-4 — Handle CellFlow conservative prediction bias (§1.3) + DRY simplify pass
- Result: pass
- Commits:
  - `a96f612` [ralph-6] TODO-4: Add CellFlow variance inflation for conservative prediction bias
  - `7d753e2` [ralph-simplify] DRY up variance inflation: single canonical implementation, fix double-application risk
- Files changed:
  - `gopro/config.py` — Added `CELLFLOW_DEFAULT_VARIANCE_INFLATION = 2.0` constant
  - `gopro/06_cellflow_virtual.py` — Added `inflate_cellflow_variance()` helper; wired into `predict_cellflow()` via `variance_inflation` parameter
  - `gopro/04_gpbo_loop.py` — Added `--cellflow-variance-inflation` CLI flag; threaded through `run_gpbo_loop()` → `merge_multi_fidelity_data()`; DRY'd up to single canonical implementation, fixed double-application risk
  - `gopro/tests/test_phase4_5.py` — 7 new tests in `TestCellFlowVarianceInflation`; simplify pass DRY'd up fixtures
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Tests: 617 passing (was 610)
- Quality: Simplify pass consolidated variance inflation to single canonical implementation in `06_cellflow_virtual.py`, removed inline duplicate in `04_gpbo_loop.py` merge path that risked double-application. No issues found.
- Notes: §1.3 CellFlow Integration Fixes now COMPLETE (TODO-1, TODO-3, TODO-4 all done). §1.1, §1.2, §1.3 all COMPLETE. Next: §1.4 GP model improvements (TODO-9 pseudocount) or §1.9 Sanchis-Calleja data ingestion (high priority, 3× training data).

## Iteration 6 — 2026-03-18T06:00:00Z
- Task: TODO-4 — Handle CellFlow conservative prediction bias (§1.3)
- Result: pass
- Files changed:
  - `gopro/config.py` — Added `CELLFLOW_DEFAULT_VARIANCE_INFLATION = 2.0` constant
  - `gopro/06_cellflow_virtual.py` — Added `inflate_cellflow_variance()` helper; wired into `predict_cellflow()` via `variance_inflation` parameter
  - `gopro/04_gpbo_loop.py` — Added `--cellflow-variance-inflation` CLI flag; threaded through `run_gpbo_loop()` → `merge_multi_fidelity_data()`; inline inflation in merge for fidelity=0.0 data
  - `gopro/tests/test_phase4_5.py` — 7 new tests in `TestCellFlowVarianceInflation`
  - `docs/task_plan.md` — Marked TODO-4 complete, updated test count to 617
- Tests: 617 passing (was 610)
- Notes: §1.3 CellFlow Integration Fixes now COMPLETE (TODO-1, TODO-3, TODO-4 all done). Next: §1.4 GP model improvements (TODO-9 pseudocount) or §1.9 Sanchis-Calleja data ingestion.

## Iteration 5 — 2026-03-18T04:58:11Z
- Task: TODO-3 — Add CellFlow OOD harvest day warning (§1.3)
- Result: pass
- Commits:
  - `4ca0939` [ralph-5] TODO-3: Add CellFlow OOD harvest day warning
  - `d227e73` [ralph-simplify] DRY up OOD warning tests: hoist import, use logger.name, remove noise columns
- Files changed:
  - `gopro/config.py` — Added `CELLFLOW_MAX_TRAINING_DAY = 36` constant
  - `gopro/06_cellflow_virtual.py` — Added `_warn_ood_harvest_days()` helper; fires warning in `predict_cellflow()` when harvest day > 36
  - `gopro/tests/test_phase4_5.py` — 5 new tests in `TestOODHarvestDayWarning`; simplify pass DRY'd up imports and removed noise columns
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Tests: 610 passing (was 605)
- Quality: Simplify pass hoisted shared import, switched to `logger.name` assertion, removed noise columns from test fixtures. No issues found.
- Notes: §1.3 now has 2/3 items complete (TODO-1, TODO-3). TODO-4 (variance inflation) remains. §1.9 Sanchis-Calleja ingest is high-priority alternative for next iteration.

## Iteration 4 — 2026-03-18T04:44:37Z
- Task: TODO-1 — Fix CellFlow JAX vs PyTorch API mismatch (§1.3)
- Result: pass
- Commits:
  - `59b1be6` [ralph-4] TODO-1: Fix CellFlow JAX vs PyTorch API mismatch
  - `6625794` [ralph-simplify] Remove unused jnp import, DRY up JAX test mocks
- Files changed:
  - `gopro/06_cellflow_virtual.py` — Replaced `import torch` + `torch.no_grad()` with `import jax` + `jax.random.PRNGKey`/`split` in `_predict_with_cellflow()`. Added `rng_key` param to `model.predict()`. Simplify pass removed unused `jnp` import.
  - `gopro/tests/test_phase4_5.py` — 3 new tests: `test_uses_jax_not_torch`, `test_rng_key_differs_per_batch`, `test_fallback_clustering_when_no_cell_type`. Simplify pass DRY'd up JAX mock fixtures.
  - `docs/task_plan.md` — Marked TODO-1 complete, updated test count to 605
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Tests: 605 passing (was 602)
- Quality: Simplify pass removed unused `jnp` import from cellflow module, DRY'd up JAX test mocks into shared fixtures. No issues found.
- Notes: CellFlow (Klein et al., bioRxiv 2025) is JAX/Flax-based. Previous code incorrectly imported PyTorch. §1.3 has 2 remaining items: TODO-3 (OOD warning) and TODO-4 (variance inflation). §1.9 (Sanchis-Calleja ingest) is high-priority alternative.

## Iteration 3 — 2026-03-18T03:08:23Z
- Task: TODO-26 — Fix CellFlow dose encoding (§1.2 FINAL) + simplify pass
- Result: pass
- Commits:
  - `48212c6` [ralph-3] TODO-26: Fix CellFlow dose encoding to use log1p
  - `df3ea3a` [ralph-simplify] Add concentration_scale field to CellFlow encoding
- Files changed:
  - `gopro/06_cellflow_virtual.py` — Changed `encode_protocol_cellflow()` concentration from raw `conc` to `math.log1p(conc)`, added `concentration_scale` field
  - `gopro/tests/test_phase4_5.py` — 3 new tests: `test_concentration_uses_log1p`, `test_log1p_zero_dose_maps_to_zero`, `test_log1p_preserves_ordering`
  - `docs/task_plan.md` — Marked TODO-26 complete, updated §1.2 status to COMPLETE
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated docs
  - `ralph-pipeline.sh` — pipeline script updates
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Tests: 602 passing (was 599)
- Quality: Simplify pass added `concentration_scale` metadata field to CellFlow encoding for traceability. Docs updated across audit report, architecture, competitive landscape, and README.
- Notes: All §1.2 critical bugs now resolved (TODO-24, TODO-25, TODO-26). §1.1 and §1.2 both COMPLETE. Next: §1.3 CellFlow integration fixes, §1.4 GP model improvements, or §1.9 data ingestion (high priority).

## Iteration 2 — 2026-03-18T01:39:37Z
- Task: TODO-25 — R²-based 3-zone fidelity routing
- Result: pass
- Commits:
  - `7e198ff` [ralph-2] TODO-25: R²-based 3-zone fidelity routing
  - `faee5b2` [ralph-simplify] Fix method validation, extract metric dispatch, remove dead import
- Files changed:
  - `gopro/config.py` — Added `FIDELITY_R2_THRESHOLDS` dict
  - `gopro/04_gpbo_loop.py` — Added `_compute_r_squared()`, rewrote `validate_fidelity_correlation()` with 3-zone routing (+33/-26 lines)
  - `gopro/visualize_report.py` — Removed dead import (-1 line)
  - `gopro/tests/test_fidelity_validation.py` — 29 tests (was 19): new `TestComputeRSquared`, `TestFidelityR2Thresholds`, `TestThreeZoneRouting`
  - `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md` — updated for R² semantics
  - `data/convergence_diagnostics.csv`, `data/gp_diagnostics_round1.csv`, `data/gp_recommendations_round1.csv` — regenerated
- Quality: Simplify pass extracted metric dispatch, fixed method validation edge case, removed dead `FIDELITY_R2_THRESHOLDS` import from visualize_report.py. No issues found by QA.
- Notes: All §1.2 critical bugs except TODO-26 (CellFlow dose encoding) now resolved. 599 tests passing.

## Iteration 1 — 2026-03-18T00:00:44Z
- Task: TODO-24 — Remap fidelity encodings for MF-GP kernel (§1.2) + /simplify hardening pass
- Result: pass
- Commits:
  - `61bc4bb` [ralph-1] TODO-24: Remap fidelity encodings for MF-GP kernel
  - `4313729` [ralph-simplify] Harden fidelity remap: use isclose, warn on unknown values, single-pass mask
- Files changed: `gopro/04_gpbo_loop.py` (26 lines changed), `docs/AUDIT_REPORT.md`, `docs/architecture.md`, `docs/competitive_landscape_ideas_index.md`, `gopro/README.md`, data CSVs updated
- Quality: Simplify pass hardened fidelity remap with `isclose` for float comparison, added warning on unknown fidelity values, consolidated to single-pass mask. Docs updated across audit report, architecture, competitive landscape index, and README.
- Notes: TODO-24 complete. Next priorities are TODO-25 (R²-based fidelity thresholds) and TODO-26 (CellFlow dose encoding). Both are §1.2 critical MF-GP fixes.

## Iteration 5 (Ralph Pipeline) — 2026-03-17
- Task: TODO-24 — Remap fidelity encodings for MF-GP kernel (§1.2)
- Result: pass
- Changes:
  - Added `FIDELITY_KERNEL_REMAP` / `FIDELITY_KERNEL_UNMAP` constants and `_remap_fidelity()` / `_unmap_fidelity()` helpers to `04_gpbo_loop.py`
  - MF-GP fitting path now remaps fidelity values from {0.0, 0.5, 1.0} → {1/3, 1/2, 2/3} before passing to `SingleTaskMultiFidelityGP`, preventing `LinearTruncatedFidelityKernel` boundary collapse
  - `recommend_next_experiments()` sets fidelity bounds to remapped value (2/3) instead of raw 1.0
  - Unknown fidelity values handled via linear interpolation fallback
- Tests: 586 passing (was 580). 6 new tests in `TestFidelityKernelRemap`.
- Files: `gopro/04_gpbo_loop.py`, `gopro/tests/test_unit.py`

## Post-Ralph Audit — 2026-03-17
- Task: Manual audit of all competitive landscape changes
- Result: 3 bugs found and fixed
- Fixes:
  - **CRITICAL**: TVR cost-scaling inverted (`var / cost` → `var * cost`). Was always selecting expensive model, making `--tvr` useless.
  - **MEDIUM**: `_select_replicate_conditions` now filters `fidelity == 1.0` — was potentially selecting unexecutable virtual conditions.
  - **LOW**: Removed redundant outer `ScaleKernel` on additive kernel (over-parameterized but didn't produce wrong results).
- Validated as correct: Aitchison distance, multiplicative replacement, GP warm-start, adaptive complexity, FBaxis A-P positions, cross-fidelity Spearman, noise estimation
- Tests: 561 → 561 (no new tests, existing tests updated for kernel structure)
- Commit: `db8eda2`

## Iteration 4 — 2026-03-17
- Task: /bug-hunter final sweep (task_plan §1.1)
- Result: pass
- Agents: 4 parallel analysis agents (bug-finder, optimizer, refactorer, test-coverage)
- Findings: 76 total (6 critical, 38 warning, 32 info). 7 fixed this run.
- Fixes applied:
  - A-C-001: KernelSpec namedtuple replaces twin `effective_saasbo`/`effective_kernel_type` variables
  - BF7-W-1: Lengthscale zero guard in importance reporting
  - BF7-C-1: `n_composition_parts > 1` consistency fix
  - OP-A10: Sobol seed=42 for reproducible convergence metrics
  - OP-A3 + RF D-W-009: `_helmert_basis` cached + torch delegates to numpy
  - OP-D2: Removed dead `compute_condition_region_fractions`
- Tests: 580 passing (was 575)
- Remaining criticals: 5 test coverage gaps (not code bugs)
- Reports: `.bug-hunter/SUMMARY.md`, `.bug-hunter/reports/`

## Iteration 3 — 2026-03-17T11:25:08Z
- Task: Data-driven entropy center (task_plan §1.1) + /simplify pass + docs consolidation
- Result: pass
- Commits:
  - `02219d4` [ralph-3] Task 13: Data-driven entropy center — Braun reference mean entropy replaces hardcoded 0.55
  - `4dfdf23` [ralph-simplify] Vectorize entropy loop, extract sigma constant
  - `ffd98d5` [ralph-meta-3] Version history update
  - `578f3f6` Consolidate docs: single task_plan.md, remove 27 stale files
- Files changed: 37 files (+1,438 / -8,171 lines) — major docs consolidation removed 27 stale files, unified into `docs/task_plan.md`
- Quality: Simplify pass vectorized entropy loop and extracted sigma constant. Docs reduced from scattered plans/handoffs/progress across root + docs/plans/ to a single consolidated task_plan.md.
- Notes: Old progress.md, handoff.md, findings.md, version-history.md all deleted in consolidation. This is a fresh progress log. Next iteration should tackle /bug-hunter or the critical TODOs (24-26).
