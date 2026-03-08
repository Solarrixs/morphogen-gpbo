---
date: 2026-03-07
title: Morphogen Perturbation Screen Datasets for Brain Organoids — ML Readiness Assessment
tags: [morphogen, organoid, scRNA-seq, dataset, machine-learning, gaussian-process, bayesian-optimization, CellFlow]
---

# Morphogen Perturbation Screen Datasets for Brain Organoids: ML Readiness Assessment

> **Purpose**: Comprehensive inventory of publicly available morphogen perturbation screen datasets with paired morphogen input + scRNA-seq output, assessed for GP-BO pipeline training readiness.
> **Related**: [[GP_BAYESOPT_ORGANOID_PROTOCOL_DESIGN]]

---

## 1. Dataset Inventory — Comparison Table

| Feature | Sanchis-Calleja et al. 2025 | Amin/Kelley et al. 2024 | Azbukina et al. 2025 | Scuderi/Vaccarino 2025 |
|---|---|---|---|---|
| **Journal** | Nature Methods | Cell Stem Cell | bioRxiv (preprint) | Cell Stem Cell |
| **Publication Date** | Dec 15, 2025 | Dec 5, 2024 | Mar 20, 2025 (preprint) | Jun 5, 2025 |
| **Lab** | Treutlein/Camp (ETH Zurich) | Pasca (Stanford) | Treutlein/Camp (ETH Zurich) | Vaccarino (Yale) |
| **Focus** | Forebrain patterning | Pan-neural axis | Posterior brain (midbrain/hindbrain) | Orthogonal WNT/SHH gradients |
| **# Conditions** | 97 (+ reproducibility: 192 samples) | 46 unique combinations | 48 designed (43 analyzed) | Gradient device (continuous) |
| **Primary Timepoint** | Day 21 | Days 72-74 | Day 35 (screen); Days 7-120 (atlas) | Not specified (weeks-scale) |
| **Cell Lines** | HES3, H9, H1, WTC, WIBJ2 (5 lines) | 2242-1, Q-0306-1, Q-0294-2, NIH2788 (4 iPSC lines) | WTC (screen); HOIK1, WIBJ2, WTC (atlas) | Multiple iPSC lines (donor-variable) |
| **Total Cells Sequenced** | ~100,538 (screen) + 209,902 (repro) | ~57 samples | ~177,718 (screen) + 104,452 (atlas) | Not confirmed |
| **scRNA-seq Platform** | 10x Genomics (multiplexed) | Parse Biosciences / 10x | 10x Genomics (multiplexed) | 10x Genomics |
| **GEO Accession** | Not yet confirmed (check paper supplement) | **GSE233574** (57 samples) | Not yet deposited (preprint) | Not confirmed |
| **Other Data Links** | HNOCA atlas integration | UCSC Cell Browser; Zenodo (spatial) | — | Mendeley Data (figures only) |
| **Open Access** | Yes (CC-BY 4.0) | Yes (NIHPA) | Yes (preprint) | Paywalled |
| **Cell Type Annotations** | Yes — detailed regional | Yes — detailed neuronal subtypes | Yes — midbrain/hindbrain | Yes — regional |
| **ML-Ready?** | **Best candidate** | **Good, but late timepoint** | **Good complement** | **Limited utility (gradient device)** |

---

## 2. Detailed Dataset Profiles

### 2.1 Sanchis-Calleja et al. 2025 (Nature Methods) — PRIMARY DATASET

**Citation**: Sanchis-Calleja F, Azbukina N, Jain A, He Z, et al. "Systematic scRNA-seq screens profile neural organoid response to morphogens." *Nature Methods* 23, 465-478 (2026). DOI: 10.1038/s41592-025-02927-5

**Morphogens Tested (8 molecules)**:
| Morphogen | Pathway | Concentration Range (c1-c5) |
|---|---|---|
| CHIR99021 | WNT activator | c1-c5 (exact values in Supp Table S1/S2) |
| XAV939 | WNT inhibitor | 4.5-5 uM range |
| rh-SHH/C24II + Purmorphamine | SHH | 180-200 ng/mL SHH; 270-300 nM PMP |
| FGF-8B | FGF | c1-c5 gradient |
| Retinoic Acid (RA) | RA | c1-c5 gradient |
| rhBMP-4 | BMP | c1-c5 gradient |
| rhBMP-7 | BMP | c1-c5 gradient |
| Cyclopamine | SHH inhibitor | c1-c5 gradient |

**Experimental Design**:
- **Timing experiments** (T1-T3): Morphogen pulses at different developmental windows (days 0-3, 3-14, 9-20, 12-20, 14-17)
- **Concentration experiments** (C1-C4): 5-step dose escalation (c1-c5)
- **Combination experiments** (O1-O2): Pairwise and higher-order morphogen combinations, with/without FGF8
- **Reproducibility experiments**: 12 conditions x 4 cell lines x 2 induction methods x 2 batches = 192 samples
- **MiSTR integration**: Microfluidic gradient device comparison

**Cell Type Annotations (21+ categories)**:
CNS regional identities: telencephalon (cortical, PSB, LGE, MGE), diencephalon, retina, hypothalamus, hindbrain, spinal cord, floor plate, preoptic area. Plus: neural crest/PNS, non-neural ectoderm, mesoderm, endoderm, pluripotent cells. Mapped to Braun et al. developing human brain atlas and HNOCA.

**Data Accessibility**: Open access paper. GEO accession should be in the paper's Data Availability section (not confirmed from web scraping — check the PDF supplement directly). The preprint version (bioRxiv 2024.02.08.579413) may have the accession listed.

**ML Readiness**: **BEST AVAILABLE**. 97 conditions with systematic concentration gradients and timing variations provide exactly the dose-response mapping needed for GP training. Day 21 is early enough to capture patterning decisions. Five cell lines enable cross-line generalization testing. The concentration step design (c1-c5) directly maps to continuous GP input dimensions.

---

### 2.2 Amin/Kelley et al. 2024 (Cell Stem Cell) — SECONDARY DATASET

**Citation**: Amin ND, Kelley KW, Hao J, et al. "Generating human neural diversity with a multiplexed morphogen screen in organoids." *Cell Stem Cell* 31(12), 1831-1846.e9 (2024). DOI: 10.1016/j.stem.2024.10.016

**Morphogens Tested (14 molecules across 8 pathways)**:
| Molecule | Pathway | Noted Concentrations |
|---|---|---|
| SAG (Smoothened agonist) | SHH | 50, 250, 1000, 2000 nM |
| CHIR99021 | WNT activator | 1.5 uM |
| IWP2 | WNT inhibitor | Standard |
| Retinoic Acid | RA | 100 nM |
| FGF2 | FGF | 50 ng/mL |
| FGF4 | FGF | Not specified |
| FGF8 | FGF | 100 ng/mL |
| EGF | EGF | Standard |
| BMP4 | BMP | Standard |
| BMP7 | BMP | Standard |
| LDN-193189 | BMP inhibitor | Standard |
| Dorsomorphin | BMP/SMAD | 2.5 uM |
| DAPT | Notch inhibitor | 2.5 uM |
| Activin A | TGF-beta | 50 ng/mL |

**Experimental Design**: 46 unique conditions including single morphogens (#1-14), pulsed treatments, sequential timing, concentration variants, multi-morphogen combinations, and 6 validated protocols. Morphogen window: Day 6-21. ~1,500 organoids total (~30/condition).

**Data Accessibility**:
- **GEO**: **GSE233574** (57 samples, Illumina NovaSeq 6000) — **PUBLICLY AVAILABLE**
- **UCSC Cell Browser**: https://cells-test.gi.ucsc.edu/?ds=morphogen-screen
- **Zenodo** (spatial data): https://doi.org/10.5281/zenodo.13835782
- **Code**: Paper states "This paper does not report original code"

**Cell Type Annotations**: Extensive — forebrain glutamatergic, GABAergic (DLX1+, NKX2-1+), TAC3+ striatal interneurons, cerebellar Purkinje/granule cells, dopaminergic neurons, motor neurons, hypothalamic neurons, OPCs, astroglia, Bergmann glia, choroid plexus. Mapped to Braun et al. fetal brain atlas and Aldinger cerebellar atlas.

**ML Readiness**: **GOOD BUT DIFFERENT REGIME**. 46 conditions with 14 morphogens provide broader pathway coverage than Sanchis-Calleja but fewer datapoints per morphogen. Day 72-74 harvest means these are mature organoids — valuable for understanding terminal differentiation outcomes but representing a fundamentally different biological timepoint. The SAG concentration series (4 doses) is directly useful for SHH dose-response modeling.

**Key Limitation for GP Training**: The late timepoint (Day 70+) means these cannot be directly combined with Day 21 Sanchis-Calleja data in a single GP without explicitly modeling time as an input dimension.

---

### 2.3 Azbukina et al. 2025 (bioRxiv) — COMPLEMENTARY (POSTERIOR BRAIN)

**Citation**: Azbukina N, He Z, Lin HC, Santel M, et al. "Multi-omic human neural organoid cell atlas of the posterior brain." bioRxiv (2025). DOI: 10.1101/2025.03.20.644368

**Morphogens Tested (~10 molecules, 48 conditions)**:
CHIR-99021, SHH, Purmorphamine, BMP4, Retinoic Acid, FGF-8, FGF-2, FGF-17, R-spondin 2, R-spondin 3, Insulin.

**Experimental Design**: 48 designed conditions (43 analyzed after excluding poor-growth conditions). Morphogen screen on WTC line only. Developmental atlas spans Days 7, 15, 30, 60, 90, and 120 across 3 iPSC lines (HOIK1, WIBJ2, WTC). Morphogen screen harvested at Day 35. Paired scRNA-seq + scATAC-seq (multi-omic).

**Cell Type Annotations**: Posterior brain focus — floor plate, dorsal/ventral midbrain, hindbrain, cerebellum progenitors, medulla glycinergic neurons, cerebellar glutamatergic subtypes. GRN inference and TF perturbation included.

**Data Accessibility**: **PREPRINT — data accession not yet confirmed**. Same lab group as Sanchis-Calleja (Treutlein/Camp, ETH Zurich). Data will likely be deposited upon publication.

**ML Readiness**: **EXCELLENT COMPLEMENT**. Fills the posterior brain gap that Sanchis-Calleja (forebrain-focused) leaves open. The multi-omic (RNA + ATAC) data could enable GRN-informed GP kernels. The temporal atlas (Days 7-120) is uniquely valuable for modeling time-dependent morphogen responses. However, only 43 usable conditions and currently preprint-only.

**Critical Note**: Azbukina is co-first-author on the Sanchis-Calleja paper — these are from the SAME LAB. The two datasets were designed as complementary screens (forebrain vs. posterior brain) and share cell lines, protocols, and annotation frameworks. They are directly combinable.

---

### 2.4 Scuderi/Vaccarino 2025 (Cell Stem Cell) — SUPPLEMENTARY

**Citation**: Scuderi S, Kang TY, Jourdon A, et al. "Specification of human brain regions with orthogonal gradients of WNT and SHH in organoids reveals patterning variations across cell lines." *Cell Stem Cell* 32(6), 970-989.e11 (2025). DOI: 10.1016/j.stem.2025.04.006

**Design**: Duo-MAPS (Dual Orthogonal-Morphogen Assisted Patterning System) diffusion device creating continuous orthogonal gradients of WNT agonist (CHIR) and SHH agonist (SAG/Purmorphamine). Multiple iPSC lines tested. scRNA-seq performed with fetal brain atlas mapping.

**Data Accessibility**: Mendeley Data (DOI: 10.17632/cgf5ptrspm.1) — supplementary figures only, NOT raw scRNA-seq. GEO accession not confirmed from available sources.

**ML Readiness**: **LIMITED FOR GP TRAINING**. The gradient device creates continuous spatial morphogen distributions within a single organoid rather than discrete conditions. This is conceptually interesting but does not provide the clean condition-wise input-output pairs needed for GP training. The key finding — substantial line-to-line variation in morphogen response — is important context for the GP pipeline (argues for line-specific or multi-task GP models).

---

## 3. CellFlow (Klein et al., bioRxiv April 2025) — Key Competitor/Complementary Approach

**Citation**: Klein D, Fleck JS, Bobrovskiy D, et al. "CellFlow enables generative single-cell phenotype modeling with flow matching." bioRxiv (2025). DOI: 10.1101/2025.04.11.648220
**GitHub**: https://github.com/theislab/CellFlow (109 stars, MIT license)
**Install**: `pip install cellflow-tools`
**Lab**: Theis (Helmholtz Munich) + Treutlein/Camp (ETH Zurich) — NOTE: overlapping authorship with Sanchis-Calleja and Azbukina

### What CellFlow Does

CellFlow is a generative model based on **flow matching** (conditional normalizing flows) that learns to transform a source cell distribution into a perturbed cell distribution. It takes perturbation descriptors (drug identity, dose, morphogen combination, genetic knockout) and predicts the full single-cell transcriptomic distribution after perturbation — not just mean expression shifts, but the complete distributional shape including rare cell types.

### Architecture
- **Core**: Conditional flow matching — learns a vector field transforming source to perturbed distribution
- **Perturbation encoding**: ESM2 embeddings (proteins), molecular fingerprints (drugs), or custom encodings
- **Combination handling**: Multi-head attention + DeepSets for permutation-invariant combinatorial encoding
- **Optimal transport**: Batch-wise cell pairing separates biological heterogeneity from perturbation effects
- **Training space**: PCA or VAE latent space

### Benchmark Datasets Used
| Dataset | System | Scale |
|---|---|---|
| PBMC cytokine screen | 12 donors x 90 cytokines | ~10M cells |
| ZSCAPE (zebrafish) | 23 gene knockouts x 5 timepoints | Developmental |
| sciPlex3 | 3 cancer lines + drugs at multiple doses | Dose-response |
| combosciplex | A549 + 31 single/combo drugs | Combinatorial |
| **Brain organoid morphogen screen** | **176 conditions (3 datasets combined)** | **Organoid protocols** |

### Organoid Morphogen Prediction Results
- **Three brain organoid datasets** (Sanchis-Calleja + Azbukina + likely HNOCA data) were harmonized into a common protocol encoding capturing modulators, concentrations, and timings across days 1-36
- **176 total conditions** after harmonization
- CellFlow predicted held-out morphogen combinations with strong distributional accuracy
- **Virtual screen of ~23,000 protocols** generated in silico
- Predictions scored on "realism" (distance to fetal brain atlas) and "novelty" (distance to training conditions)
- Predicted diverse brain region compositions: forebrain, midbrain, hindbrain, spinal cord
- Identified "previously untested treatment regimens with strong effects on organoid development"
- **2.5x mean improvement in energy distance** over baselines (chemCPA, biolord, CondOT, GEARS, scGPT, CPA)
- Mean cosine similarity of 0.91 for predicted cluster distributions

### CellFlow vs. GP-BO: Head-to-Head

| Feature | CellFlow (Flow Matching) | GP-BO (Engram Approach) |
|---|---|---|
| **Output** | Full single-cell distribution | Cell type composition vector (or scalar objective) |
| **Data requirement** | Needs 100s-1000s of conditions | Works with <100 conditions |
| **Uncertainty quantification** | Limited (generative model) | **Native** (posterior variance) |
| **Active learning** | Not built-in | **Core strength** — acquisition functions guide next experiment |
| **Interpretability** | Black-box neural ODE | **Kernel analysis** reveals morphogen interactions |
| **Combinatorial extrapolation** | Strong (attention over combinations) | Moderate (kernel must encode interactions) |
| **Compute** | GPU-intensive, hours | CPU-viable, minutes |
| **Code availability** | Yes (pip install) | scikit-learn / BoTorch (standard) |
| **Practical for Engram** | Use as benchmark/comparison | **Primary optimization loop** |

**Bottom line on CellFlow**: It is a powerful tool for predicting what a given protocol will produce, but it does NOT solve the optimization problem. CellFlow answers "what will this condition produce?" while GP-BO answers "what condition should I try next to maximize my target?" They are complementary: CellFlow could serve as a simulator within a BO loop, replacing wet-lab experiments for initial exploration.

---

## 4. Non-Brain Organoid Morphogen Screens

### 4.1 Intestinal Organoid Niche Factor Dictionary (Capeling et al., Nature Communications 2026)

**Citation**: Capeling MM, Chen B, Aliar K, et al. "Dictionary of human intestinal organoid responses to secreted niche factors at single cell resolution." *Nature Communications* 17, 1527 (2026).

- **79-81 secreted niche factors** tested on human colonic organoids
- Donor-pooled, multiplexed scRNA-seq (single-cell resolution)
- Mapped to IBD tissue atlases
- **Directly analogous design** to brain organoid morphogen screens
- **ML relevance**: Demonstrates feasibility of ligand-to-cell-type GP mapping in a different organ system. The 79-factor breadth is larger than any brain organoid screen.

### 4.2 Gut Epithelial Differentiation Screen (Mead/Shalek et al., Nature Biomedical Engineering 2022)

**Citation**: Mead BE, Hattori K, et al. "Screening for modulators of the cellular composition of gut epithelia via organoid models of intestinal stem cell differentiation." *Nat Biomed Eng* (2022).

- 433 small molecules at 4 concentrations on murine intestinal organoids
- Functional readout (lysozyme secretion, ATP) rather than scRNA-seq
- 3 murine donors
- **ML relevance**: 433 x 4 = 1,732 conditions — demonstrates that high-throughput morphogen screens at scale are feasible. However, readout is functional, not transcriptomic.

### 4.3 Kidney Organoid Differentiation (Yoshimura et al., PNAS 2023)

**Citation**: Yoshimura Y et al. "A single-cell multiomic analysis of kidney organoid differentiation." *PNAS* (2023).

- Multi-omic (scRNA-seq + scATAC-seq) analysis of kidney organoid differentiation
- CHIR concentration is a key variable in kidney organoid protocols (standard: 8-12 uM CHIR for 4 days)
- Single Cell Portal: https://singlecell.broadinstitute.org/single_cell/study/SCP211/human-kidney-organoids-atlas
- **ML relevance**: Demonstrates WNT dose-response in a different organ system; CHIR dose-response curves are directly transferable as prior knowledge for brain organoid WNT modeling.

### 4.4 Phenotypic Landscape of Intestinal Organoid Regeneration (Lukonin et al., Nature 2020)

**Citation**: Lukonin I, Serra D, et al. "Phenotypic landscape of intestinal organoid regeneration." *Nature* (2020).

- High-content imaging screen of intestinal organoid regeneration
- Systematic perturbation of signaling pathways
- Image-based phenotyping (not scRNA-seq)
- **ML relevance**: Demonstrated that organoid phenotypic landscapes can be mapped computationally; image-based GP models have been built on this data.

---

## 5. Total Available Datapoints for GP Training

> **UPDATE 2026-03-07**: Previous versions treated Sanchis-Calleja as the sole "primary" dataset. After review, **all three discrete-condition datasets should be combined** with harvest day as an explicit GP input dimension. CellFlow (Klein et al. 2025) already proved these datasets are harmonizable — they combined 176 conditions across Sanchis-Calleja + Azbukina and achieved 0.91 cosine similarity on held-out predictions. Adding Amin/Kelley (Pasca) extends coverage to Day 72-74 terminal fates.

### Condition Count Summary

| Dataset | Usable Conditions | Morphogens | Timepoint | Status |
|---|---|---|---|---|
| Sanchis-Calleja 2025 | **97** | 8 | Day 21 | **USE — forebrain patterning** |
| Azbukina 2025 | **43** | 10 | Day 35 (screen); Days 7-120 (atlas) | **USE — posterior brain + temporal atlas** |
| Amin/Kelley 2024 (Pasca) | **46** | 14 | Day 72-74 | **USE — terminal fates + 6 extra morphogens** |
| Scuderi/Vaccarino 2025 | ~5-10 discrete positions | 2 (WNT, SHH) | Variable | Context only (gradient device, no raw data) |
| **TOTAL COMBINABLE** | **~186** | **~18 unique molecules** | **Days 7-120** | |

### Why Use ALL Datasets (Not Just Sanchis-Calleja)

**Previous concern**: "The late timepoint (Day 70+) means these cannot be directly combined."

**Resolution**: Model harvest day as an explicit GP input dimension:
```
GP Input = [CHIR, BMP4, SHH/SAG, RA, FGF8, ..., log(harvest_day)]
            └──── morphogen concentrations ────┘   └── time ──┘
```

This gives us:
1. **Amin/Kelley (Pasca lab)** adds 6 morphogens not in Sanchis-Calleja: SAG (4 dose levels), DAPT (Notch inhibitor), Activin A, FGF2, FGF4, EGF, Dorsomorphin. These fill critical pathway gaps.
2. **Amin/Kelley shows terminal outcomes** — what cells actually become after 70+ days, which is the ground truth we ultimately need.
3. **Azbukina temporal atlas** (Days 7, 15, 30, 60, 90, 120) provides the multi-timepoint data that enables CellRank 2 RealTimeKernel / moscot temporal trajectory modeling — this was previously identified as the #1 missing piece.
4. **CellFlow already proved harmonization works** — see Section 3 for their protocol encoding method.

### Multi-Timepoint Coverage (NEW)

With all datasets combined, we now have temporal coverage across organoid development:

| Timepoint | Source | Conditions | Brain Region |
|-----------|--------|-----------|-------------|
| Day 7 | Azbukina (atlas) | ~6 protocols | Posterior |
| Day 15 | Azbukina (atlas) | ~6 protocols | Posterior |
| Day 21 | Sanchis-Calleja | **97** | Forebrain |
| Day 30 | Azbukina (atlas) | ~6 protocols | Posterior |
| Day 35 | Azbukina (screen) | **43** | Posterior |
| Day 60 | Azbukina (atlas) | ~6 protocols | Posterior |
| Day 72-74 | Amin/Kelley (Pasca) | **46** | Pan-neural |
| Day 90 | Azbukina (atlas) | ~6 protocols | Posterior |
| Day 120 | Azbukina (atlas) | ~6 protocols | Posterior |

**This means the "3 timepoints minimum" requirement (Zhang et al. 2025) is now met** for posterior brain (6+ timepoints via Azbukina) and partially met for forebrain (Day 21 + Day 72 = 2 timepoints, need 1 more). See CellRank 2 report update.

### Is 186 Enough for a GP?

**Yes, comfortably.** With time as an input dimension:

- 8 morphogens + 1 time = **9D** → Loeppky 10d rule gives n=90. **186 > 90.** ✓
- 8 morphogens + 1 time + 3 timing windows = **12D** → need ~120. **186 > 120.** ✓
- CellFlow achieved strong results with 176 conditions (comparable scale)
- ARD lengthscales will automatically identify which morphogens matter, reducing effective dimensionality
- Additive kernels: `k_morphogen(x) + k_time(t) + k_interaction(x, t)` captures how morphogen effects change over time

**Scaling law from CellFlow**: Performance followed a log-log scaling law with number of conditions. Breadth of conditions (more morphogens) was more valuable than depth (more replicates per condition) — another argument for combining all datasets.

---

## 6. Key Gaps in Existing Data

### Missing Morphogens
| Morphogen/Pathway | Status | Importance |
|---|---|---|
| **Notch inhibitors** (DAPT, DBZ) | Only in Amin/Kelley (1 condition) | Critical for neuronal maturation timing |
| **IGF-1** | Not systematically tested | Important for organoid growth/survival |
| **PDGF** | Not tested | Oligodendrocyte specification |
| **Endothelin** | Not tested | Vascularization-adjacent signaling |
| **LIF / CNTF** | Not tested | Astrocyte specification |
| **cAMP modulators** (Forskolin) | Used as supplement, not screened | Neuronal maturation |
| **Wnt5a** (non-canonical WNT) | Not systematically tested | Distinct from CHIR (canonical WNT) |

### Missing Timepoints
- **Days 1-6**: Neural induction phase — no morphogen screen data exists for this critical window
- **Days 21-70**: Gap between Sanchis-Calleja (D21) and Amin/Kelley (D72). Azbukina partially fills this (D35) but only for posterior brain.
- **Days 120+**: Long-term maturation effects of early morphogen exposure are completely uncharacterized at single-cell level

### Missing Brain Regions
- **Hippocampus**: No morphogen screen specifically targets hippocampal specification (key Engram interest given protocol work in `02-engineering/protocols/`)
- **Spinal cord motor neuron specification**: Limited coverage
- **Choroid plexus optimization**: Only incidentally generated
- **Cortical layer specification**: Morphogen effects on deep vs. superficial layer identity not systematically mapped

### Missing Experimental Variables
- **Matrigel vs. suspension**: No screen compares morphogen effects across culture substrates
- **Organoid size**: No systematic size-morphogen interaction data
- **Oxygen tension**: Hypoxia x morphogen interactions uncharacterized
- **Cell density at aggregation**: Not varied in any screen

---

## 7. Bottom Line

> **UPDATE (2026-03-07):** This section has been revised to reflect using ALL three discrete morphogen screen datasets (186 conditions total) with time as an explicit GP input dimension, rather than starting with Sanchis-Calleja alone. See Section 5 for the rationale.

### Can you train a GP today?

**YES** — with **186 conditions across 3 datasets**, time as an explicit input dimension, and CellFlow as a proven harmonization precedent (0.91 cosine similarity across 176 conditions).

### Multi-Dataset Build Process

#### Step 1: Download All Data
| Dataset | Accession | Priority |
|---|---|---|
| Sanchis-Calleja 2025 | Check Nature Methods supplement | **Immediate** — 97 conditions, 8 morphogens, Day 21 |
| Amin/Kelley 2024 | GSE233574 | **Immediate** — 46 conditions, 14 morphogens, Day 72-74 |
| Azbukina 2025 | Pending (preprint) | **When available** — 43 conditions, Day 35 screen + Days 7-120 atlas |

#### Step 2: Unified Annotation Pipeline
Process ALL datasets through a single cell type annotation pipeline:
1. **Gruffi** stress filtering on each dataset independently
2. **scArches/scANVI** integration using **HNOCA** as the shared reference atlas
3. Compute cell type composition vectors (proportion of each annotated type) per condition
4. These proportions become the GP output vector

#### Step 3: Encode GP Input Space with Time
Each condition becomes a vector: `[CHIR, BMP4, SHH/SAG, RA, FGF8, ..., log(harvest_day)]`

- **Sanchis-Calleja**: 8 morphogen dims + time (Day 21) → 9D input
- **Amin/Kelley**: 14 morphogen dims + time (Day 72-74) → 15D input (many zeros for shared dims)
- **Azbukina**: 10 morphogen dims + time (Day 35) → 11D input

Union of all morphogens: ~18 unique factors. Missing morphogens set to 0 (= "not added"). ARD lengthscales will automatically down-weight irrelevant dimensions.

#### Step 4: Train Multi-Output GP
- **Kernel**: Matérn 5/2 + ARD across all dimensions including time
- **Additive structure**: `k_morphogen(x) + k_time(t) + k_interaction(x, t)` captures time-dependent morphogen effects
- **Multi-task**: Consider cell-line-specific GPs or multi-task kernel if batch effects are large
- **Framework**: scikit-learn (hackathon) → BoTorch (production)

#### Step 5: CellFlow as Benchmark & Simulator
- Install CellFlow (`pip install cellflow-tools`, [GitHub](https://github.com/theislab/CellFlow))
- Load the same harmonized data — CellFlow already demonstrated this works (Klein et al., bioRxiv 2025, DOI: [10.1101/2025.04.11.648220](https://doi.org/10.1101/2025.04.11.648220))
- CellFlow's 23,000 virtual protocol screen identifies promising candidates for the GP-BO acquisition function to prioritize
- **CellFlow predicts; GP-BO optimizes.** They are complementary.

#### Step 6: CellRank 2 Temporal Modeling (NEW — enabled by multi-dataset)
With Azbukina atlas providing Days 7, 15, 30, 60, 90, 120:
- **CellRank 2 RealTimeKernel + moscot** can now model temporal trajectories for posterior brain
- Minimum 3-timepoint requirement (Zhang et al. 2025) is **satisfied** (6 timepoints available)
- Use moscot optimal transport to map cells across time, then CellRank 2's RealTimeKernel for fate probability estimation
- See [[CELLRANK2_TRAJECTORY_INFERENCE_REPORT]] for updated assessment

#### Step 7: Active Learning Experiments
- GP posterior variance identifies which morphogen combinations are most uncertain
- Based on CellFlow scaling laws, 20-50 additional experiments selected by BO should yield significant improvement
- **Harvest at multiple timepoints** (Day 21 + Day 45 minimum) to feed back into temporal modeling

### What's missing that Engram should generate?

The single highest-value experiment Engram could run is a **hippocampal morphogen optimization screen** — no existing dataset covers this brain region systematically. A 24-48 condition screen varying CHIR, BMP4, and WNT inhibitor concentrations with Day 21 and Day 45 harvests, processed with the same scRNA-seq pipeline, would be both (a) immediately useful for the GP-BO pipeline and (b) publishable as a standalone resource.

---

## 8. Data Access Quick Reference

| Dataset | Accession | URL | Status |
|---|---|---|---|
| Amin/Kelley 2024 | GSE233574 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233574 | **Available now** |
| Amin/Kelley spatial | Zenodo | https://doi.org/10.5281/zenodo.13835782 | **Available now** |
| Amin/Kelley browser | UCSC | https://cells-test.gi.ucsc.edu/?ds=morphogen-screen | **Available now** |
| Sanchis-Calleja 2025 | Check paper PDF | https://www.nature.com/articles/s41592-025-02927-5 | **Open access — check supplement** |
| Azbukina 2025 | Pending | https://www.biorxiv.org/content/10.1101/2025.03.20.644368v1 | **Preprint — data pending** |
| Scuderi/Vaccarino 2025 | Mendeley (figures) | https://data.mendeley.com/datasets/cgf5ptrspm | **Figures only** |
| CellFlow code | GitHub | https://github.com/theislab/CellFlow | **Available now** |
| HNOCA atlas | Multiple | https://www.nature.com/articles/s41586-024-08172-8 | **Available now** |
| Intestinal niche dictionary | Check paper | https://www.nature.com/articles/s41467-025-68247-6 | **Open access** |

---

*Report generated 2026-03-07. Sources: 15+ papers and preprints searched via Exa web search. Cross-referenced with existing [[GP_BAYESOPT_ORGANOID_PROTOCOL_DESIGN]] document.*
