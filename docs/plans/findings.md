# Findings — Scientific Validation Sweep
> Date: 2026-03-15
> Supersedes: Previous post-audit findings
> Purpose: Validate every modeling choice against published best practices across 4 disciplines

## Executive Summary

After deep literature review across Bayesian optimization, compositional data analysis, neurodevelopmental biology, and perturbation prediction, we identified **8 issues requiring changes** (3 P0, 3 P1, 2 P2) and **7 choices that are validated as correct or acceptable**.

---

## Discipline 1: Bayesian Optimization / GP

### VALIDATED: Matern 5/2 + ARD kernel
- **Status:** CORRECT. Matern 5/2 is the standard choice for BO (BoTorch default). ARD enables per-dimension lengthscale learning, critical for our 24D space.
- **Evidence:** Hvarfner et al. (ICML 2024, arXiv:2402.02229) show vanilla GP + Matern 5/2 + ARD is competitive with elaborate HDBO methods when using proper priors.

### VALIDATED: Zero-variance column dropping
- **Status:** CORRECT. Our `_compute_active_bounds()` drops zero-variance columns, reducing effective dimensionality to ~5-8 active dims. This is sound practice for the curse of dimensionality.

### P0-1: Add dimensionality-scaled log-normal lengthscale prior
- **Current:** BoTorch default priors (no scaling).
- **Issue:** Hvarfner et al. (ICML 2024) show that vanilla GP fails in high-D primarily because of **vanishing gradients** in lengthscale MLE, not kernel choice. A dimensionality-scaled log-normal prior on lengthscales fixes this.
- **Fix:** Add `LogNormalPrior(loc=sqrt(d) * 0.5, scale=1.0)` to the kernel's lengthscale hyperprior. Papenmeier et al. (arXiv:2502.09198, 2025) confirm this finding.
- **Impact:** Better exploration in our ~8D active space.
- **File:** `gopro/04_gpbo_loop.py`, `fit_gp_botorch()` lines 407-416

### P1-1: Improve acquisition function optimization
- **Current:** `num_restarts=5, raw_samples=512` (lines 573-579).
- **Issue:** Hvarfner et al. recommend perturbing the top-5% of initial candidates for better exploitation in high-D. Our `raw_samples=512` is low for 8D.
- **Fix:** Increase `raw_samples=1024`, add `options={"batch_limit": 5}`, perturb top-5% best incumbent points as additional restart candidates.
- **File:** `gopro/04_gpbo_loop.py`, `recommend_next_experiments()` lines 573-579

### P1-2: Consider LassoBO for variable selection (alternative to SAASBO)
- **Current:** SAASBO via NUTS sampling (slow, ~5 min for 48x17 data).
- **Issue:** LassoBO (AISTATS 2025, arXiv:2504.01743) achieves comparable or better variable selection via Lasso-regularized lengthscale estimation, without expensive NUTS sampling. Much faster.
- **Fix:** Implement `--lassobo` flag as alternative to `--saasbo`. LassoBO uses GP kernel lengthscales + Lasso to classify dims as important/unimportant.
- **Impact:** Faster variable selection, especially useful for iterative optimization rounds.
- **File:** `gopro/04_gpbo_loop.py`

### VALIDATED: Multi-fidelity GP setup
- **Status:** ACCEPTABLE. Our fidelity={1.0, 0.5, 0.0} levels for real/CellRank2/CellFlow data are a reasonable starting point.
- **Caveat:** Folch et al. (Nat Comp Sci 2025, arXiv:2410.00544) show MFBO only helps when LF source correlation with HF is above ~0.5. We should **validate** CellRank2/CellFlow correlation against real data before trusting multi-fidelity. If correlation is low, single-fidelity on real data alone may be better.
- **Action:** Add a diagnostic that computes rank correlation between virtual and real predictions for overlapping conditions. Log a warning if correlation < 0.5.

### VALIDATED: qLogEI for single-objective, qLogNEHVI for multi-objective
- **Status:** CORRECT. LogEI provides numerical stability (Ament et al., NeurIPS 2023). qLogNEHVI with data-driven ref_point is the standard multi-objective approach.

---

## Discipline 2: Compositional Data Analysis

### VALIDATED: ILR transform (not CLR)
- **Status:** CORRECT for GP regression. ILR produces full-rank (D-1) coordinates in orthogonal space, while CLR produces singular covariance matrices. For GP fitting, ILR is strictly preferred.
- **Evidence:** Multiple CoDA reviews (2023-2024) confirm ILR is the right choice for regression, PCA, and any method requiring non-singular covariance. CLR is better for visualization/exploration.
- **Note:** For higher-D compositions, ILR outperforms CLR in GP-based models specifically.

### P0-2: Fix pseudo-count value (1e-10 is too small)
- **Current:** `Y_safe = Y + 1e-10` (line 178 of 04_gpbo_loop.py)
- **Issue:** 1e-10 creates extreme log-ratios (log(1e-10) = -23), dominating GP training signal. The CoDA literature recommends:
  - **Multiplicative replacement** (Palarea-Albaladejo & Martin-Fernandez 2015): delta = 0.65 * detection_limit
  - **Bayesian-multiplicative (GBM):** posterior-based imputation
  - **Simple pseudo-count:** 0.5/N or 1/(D*N) where N=sample size, D=parts
- **Fix:** Replace `Y + 1e-10` with multiplicative replacement: for zero entries, impute `delta = 0.5 / n_cells_in_condition`, then rescale row to sum to 1.0. For our ~17 cell types and ~500-5000 cells per condition, this gives pseudo-counts of ~0.001-0.0001, far more reasonable than 1e-10.
- **Impact:** More stable GP fitting, less distortion of ILR coordinates for rare cell types.
- **File:** `gopro/04_gpbo_loop.py`, `ilr_transform()` line 178

### P0-3: Use Aitchison distance instead of cosine similarity for composition comparison
- **Current:** Cosine similarity in `compute_rss()` (line 216-231 of 03_fidelity_scoring.py) and cross-screen QC.
- **Issue:** Cosine similarity does not respect the compositional nature of cell type fractions. It is sensitive to the presence/absence of components and violates sub-compositional dominance. The **Aitchison distance** (Euclidean distance in CLR space) is the mathematically correct metric for comparing compositions.
- **Evidence:** scCODA documentation; Quinn et al. (Bioinformatics 2018); single-cell best practices (sc-best-practices.org) all state compositions should use Aitchison distance.
- **Fix:** Replace cosine similarity with Aitchison distance in `compute_rss()` and `qc_cross_screen.py`. Aitchison distance = Euclidean distance after CLR transform.
- **Impact:** More reliable region matching and cross-screen QC.
- **File:** `gopro/03_fidelity_scoring.py` line 216-231, `gopro/qc_cross_screen.py`

### VALIDATED: Helmert basis for ILR
- **Status:** ACCEPTABLE. The Helmert basis is the standard default ILR basis (used in R `compositions` package). While other bases (e.g., sequential binary partition) can give more interpretable coordinates, the Helmert basis is orthonormal and mathematically correct. Since we don't interpret individual ILR coordinates, Helmert is fine.

### P2-1: Propagate composition uncertainty to GP noise model
- **Current:** Cell type fractions are point estimates with no uncertainty.
- **Issue:** Scanpro (Sci Reports 2024) shows bootstrapping-based uncertainty on cell type proportions can be significant, especially for rare types. Propagating this as heteroscedastic noise in the GP would improve predictions.
- **Fix:** In `02_map_to_hnoca.py`, compute bootstrap confidence intervals on fractions. Pass as observation noise to `SingleTaskGP(..., train_Yvar=...)`.
- **Deferred:** This is a meaningful improvement but lower priority than P0 fixes.

---

## Discipline 3: Neurodevelopmental Biology

### P1-3: Update EC50 values in MORPHOGEN_PATHWAY_MAP
- **Current:** EC50 values in `06_cellflow_virtual.py` lines 430-546 are "approximate midpoints."
- **Issue:** Published pharmacological data shows specific discrepancies:
  - **CHIR99021:** Our EC50=3.0 µM is reasonable for organoid WNT activation (published range 1-8 µM for functional effects, enzymatic IC50 = 6.7 nM for GSK3β). However, the dose-response is highly nonlinear: 1 µM increases NPCs, 10 µM arrests development (Delepine et al., PLOS One 2021). Our Hill=1.5 is reasonable.
  - **BMP4:** Our EC50=0.001 µM (~1 ng/mL) may be too low. Published functional effects in organoids use 5-50 ng/mL (0.4-3.8 µM equivalent at 13 kDa). EC50 should be ~0.001-0.002 µM (10-25 ng/mL). Current value is borderline acceptable.
  - **SHH:** Our EC50=0.005 µM (~100 ng/mL) is within published range.
  - **SAG:** Our EC50=0.5 µM (500 nM) is reasonable; published working range is 50-2000 nM.
- **Fix:** Cross-reference each EC50 against vendor datasheets and Sanchis-Calleja dose-response curves. The most critical update is adding **competence window modeling**: Sanchis-Calleja et al. (Nature Methods 2025) show morphogen effects depend heavily on TIMING, not just dose.
- **File:** `gopro/06_cellflow_virtual.py`, MORPHOGEN_PATHWAY_MAP

### VALIDATED: Off-target cell type list
- **Status:** CORRECT. PSC, MC, EC, Microglia, NC Derivatives are standard off-target types for neural organoids. These are non-neural identities that indicate failed differentiation.

### VALIDATED: Base media composition
- **Status:** ACCEPTABLE. BDNF 20 ng/mL, NT3 20 ng/mL, cAMP 50 µM, Ascorbic acid 200 µM are standard for neural organoid maturation media (based on Amin & Kelley 2024 protocol).

### P2-2: Entropy center of 0.55 is arbitrary
- **Current:** Gaussian penalty centered at 0.55 for normalized entropy (line 395 of 03_fidelity_scoring.py).
- **Issue:** No published evidence for 0.55 as optimal entropy. Fetal brain reference regions have varying entropy levels. The center should be **data-driven** from the Braun fetal brain reference profiles.
- **Fix:** Compute mean normalized entropy across all Braun fetal brain regions and use that as the center. Or remove entropy from composite score (it's only 15% weight).

### Fidelity scoring weights (RSS=0.35, on_target=0.25, off_target=0.25, entropy=0.15)
- **Status:** No published validation for these specific weights. They are reasonable heuristics. Could be improved by learning weights from expert-scored conditions, but not a priority.

---

## Discipline 4: Perturbation Prediction / OT

### VALIDATED: moscot for temporal trajectory
- **Status:** CORRECT. moscot (Nature 2025) is the state-of-the-art for temporal OT. Using it for CellRank2 virtual data is appropriate.

### P1-4: Adjust moscot tau_a/tau_b parameters
- **Current:** `tau_a=0.94, tau_b=0.94` (lines 56-57 of 05_cellrank2_virtual.py).
- **Issue:** moscot documentation recommends `tau_a=tau_b=1.0` (balanced) as default. Values of 0.94 imply mild unbalancedness. For organoid temporal data where cell proliferation/death occurs between timepoints, mild unbalancedness is justified.
- **Assessment:** 0.94 is within the recommended range (0.9-0.99 for mild unbalancedness). This is **borderline acceptable** but should be documented with justification. Consider making it configurable.
- **Fix:** Add docstring explaining why tau=0.94 was chosen. Make tau_a/tau_b CLI-configurable.

### VALIDATED: Heuristic predictor as CellFlow fallback
- **Status:** CORRECT to have a heuristic fallback. Ahlmann-Eltze et al. (Nature Methods 2025) show that **simple linear baselines outperform deep learning foundation models** for perturbation prediction. Our sigmoid dose-response heuristic is more principled than a linear model.
- **Caveat:** When a trained CellFlow model becomes available, we should benchmark it against the heuristic. If the heuristic is competitive, prefer it (cheaper, faster, no GPU).

### VALIDATED: KNN k=10 for cell embedding
- **Status:** ACCEPTABLE. k=10 is standard for single-cell neighbor graphs. Could be tuned but not a priority.

### VALIDATED: Confidence scoring via distance to training data
- **Status:** ACCEPTABLE. Distance-based confidence is a reasonable proxy. More sophisticated approaches (e.g., GP posterior variance) exist but are overkill for the heuristic predictor.

---

## Priority Summary

| ID | Priority | Issue | File | Status |
|----|----------|-------|------|--------|
| P0-1 | P0 | Add dim-scaled log-normal lengthscale prior | 04_gpbo_loop.py | DONE |
| P0-2 | P0 | Fix pseudo-count (1e-10 -> multiplicative replacement) | 04_gpbo_loop.py | DONE |
| P0-3 | P0 | Replace cosine similarity with Aitchison distance | 03_fidelity_scoring.py, qc_cross_screen.py | DONE |
| P1-1 | P1 | Improve acqf optimization (num_restarts 5->10, raw_samples 512->1024) | 04_gpbo_loop.py | DONE |
| P1-2 | P1 | Implement LassoBO as SAASBO alternative | 04_gpbo_loop.py | DEFERRED |
| P1-3 | P1 | Update/validate EC50 values, add competence windows | 06_cellflow_virtual.py | DEFERRED |
| P1-4 | P1 | Document and make moscot tau configurable | 05_cellrank2_virtual.py | DONE |
| P2-1 | P2 | Bootstrap uncertainty -> GP noise model | 02_map_to_hnoca.py | DEFERRED |

## Additional Improvements Implemented

| Feature | File | Description |
|---------|------|-------------|
| Cross-fidelity correlation gate | 04_gpbo_loop.py | `validate_fidelity_correlation()` warns if virtual data correlation < threshold |
| Explicit fidelity costs | config.py | `FIDELITY_COSTS` dict for cost-aware multi-fidelity decisions |
| GP warm-start across rounds | 04_gpbo_loop.py | `save_gp_state()`/`load_gp_state()` with dimension mismatch handling |
| QC duplicate wells | 04_gpbo_loop.py | `n_duplicates` param for within-batch noise estimation |
| Env-configurable moscot params | 05_cellrank2_virtual.py | `GPBO_MOSCOT_TAU_A/B/EPSILON` env vars |

All 487 tests passing.
| P2-2 | P2 | Data-driven entropy center | 03_fidelity_scoring.py | Small |

## Sources

### Bayesian Optimization
- Hvarfner et al. "Vanilla Bayesian Optimization Performs Great in High Dimensions" ICML 2024. arXiv:2402.02229
- Papenmeier et al. "Understanding High-Dimensional Bayesian Optimization" arXiv:2502.09198, 2025
- LassoBO: Hoang et al. "High Dimensional Bayesian Optimization using Lasso Variable Selection" AISTATS 2025. arXiv:2504.01743
- Folch et al. "Best practices for multi-fidelity Bayesian optimization" Nat Comp Sci 2025. arXiv:2410.00544
- BOOST: arXiv:2508.02332, 2025
- BioBO: arXiv:2509.19988, 2025

### Compositional Data Analysis
- Quinn et al. "Understanding sequencing data as compositions" Bioinformatics 34(16), 2018
- Martin-Fernandez et al. "Bayesian-multiplicative treatment of count zeros" Statistical Modelling 15(1), 2015
- scCODA: Buttner et al. Nat Comms 2021
- Review: PMC 11609487, 2024 (CAC/AAC transforms for zero-inflated data)
- sc-best-practices.org compositional analysis chapter

### Neurodevelopmental Biology
- Sanchis-Calleja et al. "Systematic scRNA-seq screens profile neural organoid response to morphogens" Nature Methods 2025
- Amin et al. "Generating human neural diversity with a multiplexed morphogen screen" Cell Stem Cell 2024
- Delepine et al. "GSK3b inhibitor CHIR99021 modulates cerebral organoid development" PLOS One 2021
- BMP4 patterning: Development 146(14), 2019

### Perturbation Prediction / OT
- moscot: Klein et al. Nature 638, 2025
- Ahlmann-Eltze et al. "Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines" Nature Methods 22, 2025
- CellFlow: Klein et al. bioRxiv 2025.04.11.648220
- PerturBench: arXiv:2408.10609, 2024
- STORIES: Nature Methods 2025

---

## Bug Hunter Fixes (2026-03-16)

Verified 8 critical findings from bug-hunter swarm. 3 confirmed, 2 false positives, 3 won't fix.

### Fixed (commit a1078f9)

1. **CRIT-1 — NaN injection via zero row_sums** (`05_cellrank2_virtual.py:699`): `row_sums.replace(0, 1)` prevents NaN propagation into GP training labels when virtual conditions have all-zero push results.

2. **CRIT-2 — Transport matrix dimension mismatch** (`05_cellrank2_virtual.py:544`): Added length check `len(target_dist) != len(target_labels_arr)` in `_project_condition_transport` — falls back to atlas average instead of producing silently wrong fractions.

3. **CRIT-4 — Zero-width fidelity bounds** (`04_gpbo_loop.py:140`): Fidelity column now dropped from `active_cols` when only a single fidelity level exists, preventing zero-variance column from causing NaN in BoTorch Normalize.

### Dismissed

- **CRIT-3** (SAASBO ModelListGP crash): FALSE POSITIVE — multi-output SAASBO always flows through GenericMCObjective scalarization, which supports ModelListGP.
- **CRIT-5** (pandas Index.append deprecated): FALSE POSITIVE — `Index.append()` is not deprecated (only `Series.append()` was).
- **CRIT-6/7** (performance loops): WONT FIX — negligible at current dataset scale (N=48-100 conditions, k=50 vectorized calls).
- **CRIT-8** (god function): WONT FIX — refactoring suggestion, tracked in P3.

## Bug Hunter Fixes (Round 4, 2026-03-16)

### Fixed

1. **C-01 — `--multi-objective` CLI flag silently ignored** (`04_gpbo_loop.py`): Added `multi_objective` parameter to `run_gpbo_loop()` and threaded it through to `recommend_next_experiments(use_multi_objective=...)`. Users passing `--multi-objective` now correctly get `qLogNoisyExpectedHypervolumeImprovement` instead of scalarized `qLogEI`.

2. **C-02 — `--n-duplicates` CLI flag silently ignored** (`04_gpbo_loop.py`): Added `n_duplicates` parameter to `run_gpbo_loop()` and threaded it through to `recommend_next_experiments(n_duplicates=...)`. QC duplicate plate positions for noise estimation now correctly activate when `--n-duplicates > 0`.

### Dismissed (Round 4)

- **C-03** (SAASBO + ModelListGP + qLogEI): FALSE POSITIVE — confirmed by prior round; GenericMCObjective scalarization supports ModelListGP.
- **C-04** (Transport dimension mismatch): FALSE POSITIVE — guard already exists at line 546-552 with fallback to atlas average.
- **C-05** (NaN injection in project_query_forward): FALSE POSITIVE — already fixed in prior audit (line 709: `row_sums.replace(0, 1)`).
- **C-06** (deprecated Index.append): FALSE POSITIVE — `Index.append()` is not deprecated in pandas 2.x.
- **C-07** (Zero-width fidelity bounds): FALSE POSITIVE — already fixed in prior audit (lines 139-152: `nunique() > 1` guard).
- **C-08/C-09** (performance loops): WONT FIX — not correctness bugs, negligible at current scale.
