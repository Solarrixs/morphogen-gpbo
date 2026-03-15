# Literature Intelligence Report: GP-BO Brain Organoid Pipeline

**Search date:** 2026-03-15 | **Coverage:** 2024-2026 | **Sources:** PubMed, bioRxiv, Nature, Cell, arXiv

---

## Priority Implementation Recommendations

| Priority | Paper | Change | File(s) to Modify |
|----------|-------|--------|-------------------|
| **P0** | LassoBO (arXiv 2504.01743) | Replace SAASBO with faster LassoBO for variable selection | `04_gpbo_loop.py` |
| **P0** | Hvarfner vanilla GP-ARD (arXiv 2502.09198) | Add log-normal length scale prior + top-5% perturbation for acquisition | `04_gpbo_loop.py` |
| **P0** | CellFlow published (bioRxiv 2025.04.11.648220) | Update step 06 to match published API; check for pretrained organoid model | `06_cellflow_virtual.py` |
| **P1** | Perturbation benchmarks (Nat Methods 2025) | Add linear baseline virtual data generator alongside CellFlow | `06_cellflow_virtual.py` |
| **P1** | MFBO best practices (Nat Comp Sci 2025) | Validate multi-fidelity vs single-fidelity; tune fidelity weights | `04_gpbo_loop.py` |
| **P1** | Azbukina posterior brain atlas (bioRxiv 2025) | Download as additional GP training data | `00b_download_patterning_screen.py` |
| **P1** | NEST-Score (Cell Reports 2025) | Implement as complementary fidelity metric | `03_fidelity_scoring.py` |
| **P2** | Scanpro bootstrapping (Sci Reports 2024) | Estimate cell type fraction uncertainty for GP noise model | `02_map_to_hnoca.py` |
| **P2** | CLR vs ILR comparison (bioRxiv 2025) | Benchmark CLR transform as alternative | `04_gpbo_loop.py` |
| **P2** | BATCHIE batch design (Nat Comms 2025) | Adopt information-theoretic batch selection | `04_gpbo_loop.py` |
| **P2** | pertpy distances (Nat Methods 2025) | Replace cosine similarity in cross-screen QC | `qc_cross_screen.py` |
| **P3** | STORIES trajectory (Nat Methods 2025) | Evaluate as CellRank2 replacement for virtual data | `05_cellrank2_virtual.py` |
| **P3** | scCODA variable selection (Nat Comms 2021) | Identify morphogen-responsive cell types to focus GP | `03_fidelity_scoring.py` |

---

## 1. Bayesian Optimization for Biological Experiments

### 1A. Cell Culture Media Optimization via BO
- **Hinckley et al.** Nature Communications 16, 6055 (2025)
- https://www.nature.com/articles/s41467-025-61113-5
- BO-based iterative design optimized cytokine-supplemented media in 3-30x fewer experiments than DoE
- **Actionable:** Their batch design (6 per batch, 4 iterations, 24 total) mirrors our plate-map workflow. Transfer learning extension could reuse GP models across cell lines.

### 1B. Best Practices for Multi-Fidelity BO
- **Folch et al.** Nature Computational Science (2025) | arXiv:2410.00544
- Guidelines for when MFBO outperforms single-fidelity BO
- **Actionable:** Validate our fidelity=1.0/0.5/0.0 setup against single-fidelity on real data alone. If CellFlow correlation is below their threshold, increase weight or drop it.

### 1C. Multi-Fidelity Batch BO for Bioprocesses
- arXiv:2508.10970 (2025)
- Scale-as-fidelity concept + custom GP for mixed continuous/categorical variables

### 1D. Evolution-Guided BO with qNEHVI
- npj Computational Materials (2024)
- https://www.nature.com/articles/s41524-024-01274-x
- EGBO adds evolutionary selection alongside qNEHVI for better Pareto front coverage

### 1E. LassoBO: Alternative to SAASBO
- arXiv:2504.01743 (2025)
- **HIGH PRIORITY.** Identifies important variables via GP kernel length scales, constructs multi-subspace search regions. Outperforms SAASBO while being much faster (no NUTS overhead).

### 1F. Vanilla GP-ARD Competitive in High Dimensions
- **Hvarfner et al.** arXiv:2502.09198 (2025)
- Standard GP + Matern-5/2 + ARD is competitive with elaborate methods when using dimensionality-scaled log-normal length scale priors and top-5% perturbation for acquisition optimization.

---

## 2. Brain Organoid Differentiation Protocols

### 2A. Sanchis-Calleja/Azbukina Morphogen Screen (Our Dataset)
- Nature Methods 23, 465-478 (2026)
- Competence window data should constrain GP-BO search space temporally

### 2B. Azbukina Posterior Brain Atlas
- bioRxiv (2025) — https://www.biorxiv.org/content/10.1101/2025.03.20.644368v1
- **HIGH PRIORITY.** Posterior brain conditions expand morphogen-to-cell-type mapping beyond Amin/Kelley

### 2C. NEST-Score for Protocol Benchmarking
- Cell Reports (2025)
- Evaluates neighborhood sample homogeneity; distinguishes reliable vs artifact cell types

### 2D. HNOCA Update
- Nature (2024) — He et al.
- Verify we're using latest HNOCA version and scPoli model

---

## 3. Optimal Transport for Trajectory Inference

### 3A. moscot Published in Nature
- Nature 638 (2025)
- Verify our step 05 uses published API (may differ from preprint)

### 3B. STORIES: Learning Cell Fate Landscapes
- Nature Methods (2025)
- Could replace/augment CellRank2 with continuous trajectory model for higher-quality virtual data

### 3C. Gene Trajectory Inference
- Nature Biotechnology (2025)
- OT distances between gene distributions for identifying morphogen-to-fate gene programs

---

## 4. CellFlow and Perturbation Prediction

### 4A. CellFlow Published
- bioRxiv (2025) — https://www.biorxiv.org/content/10.1101/2025.04.11.648220v1
- **HIGH PRIORITY.** Update step 06 to match published API. Check for pretrained organoid model.

### 4B-C. Perturbation Benchmarks: Simple Baselines Win
- Nature Methods (2025) — 27 methods benchmarked, foundation models don't consistently outperform linear baselines
- **Actionable:** Consider simple linear model as CellFlow alternative for virtual data generation

### 4D. Pertpy Framework
- Nature Methods (2025)
- Modular perturbation analysis; could replace cosine similarity in cross-screen QC

### 4E. BATCHIE: Bayesian Active Learning for Combination Screens
- Nature Communications (2025)
- **HIGH PRIORITY.** Information-theoretic batch selection could improve acquisition function

---

## 5. Compositional Data Analysis

### 5A. CoDA-hd: High-Dimensional Compositional Analysis
- bioRxiv (2025)
- CLR transform preserves all dimensions vs ILR's D-1 reduction. Benchmark CLR vs ILR for GP.

### 5B. scCODA: Bayesian Compositional Model
- Nature Communications (2021)
- Spike-and-slab prior identifies morphogen-responsive cell types; could focus GP targets

### 5C. Scanpro: Proportion Analysis with Bootstrapping
- Scientific Reports (2024)
- Bootstrap uncertainty estimates on cell type fractions → propagate as GP noise

---

## 6. Active Learning for Screening

### 6A. State: Transformer for Perturbation Response
- bioRxiv (2025)
- Pre-trained on 100M+ perturbed cells; potential CellFlow alternative

### 6B. CRISP: Drug Response Prediction
- Nature Computational Science (2025)
- Transfer learning for perturbation responses in unseen cell types
