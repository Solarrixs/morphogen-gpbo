# Morphogen Optimization Pipeline — System Architecture

**Date:** 2026-03-07 | **Author:** Engram (Maxx, Mustafo, Wayne)
**Status:** Design validated through research synthesis of 50+ papers across 5 research agents

---

## 1. One-Liner

**Problem:** Designing brain organoid differentiation protocols requires testing thousands of morphogen combinations across weeks-to-months of culture — brute force is too slow and expensive.

**Solution:** A multi-fidelity Bayesian optimization engine that combines published morphogen screen data (186 conditions), CellRank 2 temporal predictions, CellFlow virtual screening, and user-submitted experiments into an active learning loop that recommends the next best experiment to run.

---

## 2. The Five Datasets

There are two layers: **reference atlases** (ground truth for benchmarking) and **morphogen screens** (training data for the GP).

### Layer 1: Reference Atlases

| Dataset | Data Type | Scale | What It Is | Role in Pipeline |
|---|---|---|---|---|
| **HNOCA** (He et al., Nature 2024) | scRNA-seq | 1.77M cells, 26 protocols, 36 datasets | Integrated atlas of human neural **organoids**. NOT fetal tissue. Maps organoid cells to standardized cell type labels via scArches transfer learning. | **Cell type annotation engine.** All organoid scRNA-seq (published + user-submitted) gets projected onto HNOCA to get standardized cell type labels + composition vectors. |
| **Fetal brain atlases** (Braun/Linnarsson 2023, Zhong/Wang 2018, BrainSTEM) | scRNA-seq (+ spatial for Braun) | 600+ cell states, GW 3-27 | Atlases of actual human fetal brain tissue. The ground truth for what real brain cells look like at single-cell resolution. | **Fidelity scoring.** After HNOCA annotation, organoid cells are mapped onto fetal atlases via BrainSTEM two-tier method to score how faithfully they recapitulate real brain cell types. |

**Key distinction:** HNOCA answers "what cell types does this organoid produce?" Fetal atlases answer "how close are those cells to real human brain cells?" Both are needed.

#### Reference Atlas Details

**HNOCA (Human Neural Organoid Cell Atlas)**
- Paper: He, Dony, Fleck et al., Nature 635, 690-698 (2024). DOI: [10.1038/s41586-024-08172-8](https://doi.org/10.1038/s41586-024-08172-8)
- Data: [CZ CELLxGENE](https://cellxgene.cziscience.com/collections/de379e5f-52d0-498c-9801-0f850823c847)
- Tools: [HNOCA-tools](https://github.com/theislab/neural_organoid_atlas) (Python package for annotation, reference mapping, label transfer)
- Integration method: scPoli (label-aware) for atlas construction; scArches for query mapping
- Covers: 26 differentiation protocols (3 unguided, 23 guided), Days 7-450
- Includes: Reference Similarity Spectrum (RSS) for quantitative fidelity scoring against fetal brain

**Fetal Brain Atlases**

| Atlas | Coverage | Scale | Data Access | Best For |
|---|---|---|---|---|
| Braun/Linnarsson 2023 (Science) | GW 5-14 | ~600 cell states, 12 classes | [CellxGene](https://cellxgene.cziscience.com/collections/4d8fed08-2d6d-4692-b5ea-464f1d072077) | First-trimester cell type/region mapping |
| Zhong/Wang 2018 (Nature) | GW 8-26 (PFC) | 35 subtypes | GSE104276 / GSE119212 | Prefrontal cortex organoid benchmarking |
| BrainSTEM (Science Advances) | GW 3-14 | 679K cells, 39 donors | In paper | Two-tier mapping: whole-brain → region-specific |
| BrainSpan (Allen Institute) | 8 pcw - adult | 16 regions | [brainspan.org](https://www.brainspan.org) | Regional gene expression validation (bulk only) |

### Layer 2: Morphogen Screen Datasets

| Dataset | Conditions | Timepoint | Morphogens | Cells | Platform | Format | Lab |
|---|---|---|---|---|---|---|---|
| **Sanchis-Calleja 2025** (Nature Methods) | 97 | Day 21 | 8 (CHIR, BMP4, SHH, SAG, RA, FGF8, IWP2, SB431542) | ~100,538 | 10x Genomics 3' + cell hashing | h5ad (AnnData) | QuaDBio / Treutlein & Camp (ETH Zurich) |
| **Amin/Kelley 2024** (Cell Stem Cell) | 46 | Day 72-74 | 14 (broader panel incl. Notch inhibitors) | ~36,265 | Parse Biosciences split-pool | **Seurat RDS + MTX** (R) | Pasca (Stanford) |
| **Azbukina 2025** (bioRxiv preprint) | 43 screen + temporal atlas (Days 7, 15, 30, 60, 90, 120) | Day 35 (screen) | 10 | ~104,452 (time course) | Parse Biosciences split-pool | h5ad (AnnData) likely | QuaDBio / Treutlein & Camp (ETH Zurich) |

**Total: 186 discrete morphogen conditions + 6-timepoint temporal atlas**

#### Data Access

| Dataset | Accession | URL | Status |
|---|---|---|---|
| Sanchis-Calleja 2025 | Check Nature Methods supplement | [Paper](https://www.nature.com/articles/s41592-025-02927-5), [GitHub](https://github.com/quadbio/organoid_patterning_screen) | Open access |
| Amin/Kelley 2024 | GSE233574 | [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE233574), [UCSC Browser](https://cells-test.gi.ucsc.edu/?ds=morphogen-screen) | Available now |
| Amin/Kelley spatial | Zenodo | [doi:10.5281/zenodo.13835782](https://doi.org/10.5281/zenodo.13835782) | Available now |
| Azbukina 2025 | Pending | [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.03.20.644368v1) | Preprint — data pending |

#### Compatibility Notes

- **Format mismatch:** Amin/Kelley is in R/Seurat format. Convert to AnnData via `sceasy` or `MuDataSeurat` before pipeline ingestion.
- **Platform mismatch:** Sanchis-Calleja uses 10x Genomics; Amin/Kelley and Azbukina use Parse Biosciences. This is a known batch effect source — scANVI integration handles it.
- **Condition encoding mismatch:** Each dataset encodes morphogen conditions differently (concentration codes c1-c5 vs. human-readable names vs. numeric labels). CellFlow's common protocol encoding scheme resolves this (see Section 5).

---

## 3. How The Datasets Relate

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REFERENCE LAYER                              │
│                                                                     │
│  ┌──────────────┐          ┌─────────────────────────────────┐     │
│  │    HNOCA      │          │       Fetal Brain Atlases       │     │
│  │  1.77M cells  │          │  Braun GW5-14 │ Zhong GW8-26   │     │
│  │  26 protocols │          │  BrainSTEM 679K cells           │     │
│  │              │          │                                  │     │
│  │  Provides:    │          │  Provides:                      │     │
│  │  • Cell type  │          │  • Fidelity scores              │     │
│  │    labels     │──MAP──►  │  • "How real are these cells?"  │     │
│  │  • Composition│          │  • Off-target detection         │     │
│  │    vectors    │          │  • Maturation staging           │     │
│  └──────┬───────┘          └──────────────┬──────────────────┘     │
│         │                                  │                        │
│         │         STANDARDIZED OUTPUT      │                        │
│         └──────────┬───────────────────────┘                        │
│                    │                                                 │
│                    ▼                                                 │
│         Cell type fractions per condition                           │
│         + fidelity scores per cell type                             │
│         = GP training labels (Y vector)                             │
└────────────────────┬────────────────────────────────────────────────┘
                     │
┌────────────────────┼────────────────────────────────────────────────┐
│                    │       TRAINING DATA LAYER                      │
│                    ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              186 Morphogen Screen Conditions                 │   │
│  │                                                              │   │
│  │  Sanchis-Calleja   Amin/Kelley      Azbukina               │   │
│  │  97 cond, Day 21   46 cond, Day 72  43 cond, Day 35        │   │
│  │  8 morphogens      14 morphogens    10 morphogens           │   │
│  │                                     + temporal atlas D7-120  │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                  GP Input (X vector):                               │
│                  [CHIR, BMP4, SHH, SAG, RA, FGF8, ...,            │
│                   log(harvest_day)]                                 │
│                  Union of ~18 morphogens + time                    │
│                  Missing morphogens = 0 ("not added")             │
└─────────────────────────────────────────────────────────────────────┘
```

**The flow:** Morphogen screen scRNA-seq data → Gruffi QC → scArches projection onto HNOCA → cell type labels → BrainSTEM fetal mapping → fidelity scores → cell type fraction vectors (Y) paired with morphogen concentration vectors (X) → GP training set.

---

## 4. Multi-Fidelity Active Reinforcement Learning (ARL) Loop

This is the core innovation: a **multi-fidelity Bayesian optimization loop** that combines real experimental data with virtual predictions from CellRank 2 and CellFlow.

### 4.1 Three Fidelity Levels

| Fidelity | Source | Noise | Cost | Volume |
|---|---|---|---|---|
| **High** (real) | Actual scRNA-seq experiments | σ²_real ≈ 0.01 | ~$600/condition | 186 existing + user experiments |
| **Medium** (CellRank 2 virtual) | Temporal forward prediction from real Day 21 data | σ²_CR2 ≈ 0.05-0.10 | Free (compute only) | 186 × N_timepoints |
| **Low** (CellFlow virtual) | Generative model predictions from protocol encoding | σ²_CF ≈ 0.10-0.20 | Free (compute only) | Up to 23,000 |

### 4.2 How CellRank 2 Generates Virtual Data

CellRank 2's RealTimeKernel + moscot uses optimal transport to model how cells transition between timepoints. The Azbukina temporal atlas (Days 7, 15, 30, 60, 90, 120) provides the transport maps.

**When a user runs a new Day 21 experiment:**
1. Map their Day 21 cells into the shared HNOCA latent space (via scArches)
2. Use the pre-computed moscot transport maps to project cells forward: Day 21 → Day 30 → Day 60 → Day 90
3. At each projected timepoint, compute predicted cell type fractions
4. These become **medium-fidelity training points** for the GP

**Each real experiment generates multiple virtual data points:**
- 1 real experiment at Day 21 → 1 real point + 3 virtual points (Day 30, 60, 90) = 4 effective data points
- 24 real experiments → 24 real + 72 virtual = 96 effective data points
- **3x data amplification per dollar spent**

**Requirements for CellRank 2 temporal prediction:**

| Requirement | Status |
|---|---|
| ≥3 real timepoints for transport map learning | ✅ Azbukina provides 6 (Days 7-120) for posterior brain |
| Shared latent space across timepoints | ✅ scANVI + HNOCA reference |
| AnnData with numeric `time_key` in `obs` | ✅ Standard format |
| PCA embedding in `obsm` | ✅ Standard preprocessing |

**Implementation:**
```python
import moscot as mt
import cellrank as cr

# Learn transport maps from Azbukina temporal atlas
problem = mt.problems.TemporalProblem(adata_temporal)
problem = problem.prepare(time_key="day", joint_attr={"attr": "obsm", "key": "X_pca"})
problem = problem.solve(epsilon=1e-3, tau_a=0.94)

# Convert to CellRank transition matrix
adata_temporal.obs["day"] = adata_temporal.obs["day"].astype("category")
rtk = cr.kernels.RealTimeKernel.from_moscot(problem)
rtk = rtk.compute_transition_matrix(self_transitions="all", conn_weight=0.2, threshold="auto")

# Compute fate probabilities
estimator = cr.estimators.GPCCA(rtk)
estimator.compute_macrostates(n_states=12)
estimator.compute_fate_probabilities()
# → fate_probabilities per cell → aggregate per condition → virtual Y vector
```

### 4.3 How CellFlow Generates Virtual Data

CellFlow (Klein et al., bioRxiv 2025) is a flow-matching generative model trained on all 176 conditions from Sanchis-Calleja + Azbukina. It predicts single-cell distributions from protocol encodings.

- DOI: [10.1101/2025.04.11.648220](https://doi.org/10.1101/2025.04.11.648220)
- GitHub: [github.com/theislab/CellFlow](https://github.com/theislab/CellFlow)
- Install: `pip install cellflow-tools`

**CellFlow's protocol encoding scheme:**
1. **Modulator identity** → molecular fingerprints (small molecules via RDKit) or ESM2 embeddings (proteins)
2. **Concentration** → numeric
3. **Timing window** → which days (1-36) each modulator was applied
4. **Pathway** → which signaling pathway
5. **Base protocol** → one-hot dataset label (accounts for lab-specific confounders)
6. **Combinatorial handling** → multihead attention (permutation-invariant for variable numbers of simultaneous morphogens)

CellFlow screened **~23,000 virtual protocols** and achieved **0.91 cosine similarity** with real cluster distributions.

### 4.4 The Multi-Fidelity GP

The GP must distinguish between fidelity levels. BoTorch's `SingleTaskMultiFidelityGP` handles this natively.

```python
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms import Normalize, Standardize

# X columns: [CHIR, BMP4, SHH, SAG, RA, FGF8, ..., log(day), fidelity]
# fidelity: 1.0 = real, 0.5 = CellRank2 virtual, 0.0 = CellFlow virtual
model = SingleTaskMultiFidelityGP(
    train_X,
    train_Y,
    data_fidelities=[D],  # index of fidelity column
    input_transform=Normalize(d=D+1),
    outcome_transform=Standardize(m=M),
)

# Acquisition function evaluates at highest fidelity
from botorch.acquisition import qLogNoisyExpectedHypervolumeImprovement
acqf = qLogNoisyExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    X_baseline=train_X[train_X[:, -1] == 1.0],  # only real data as baseline
)
```

**How the GP uses multi-fidelity data:**
- Learns correlations between fidelity levels (real vs. CellRank 2 vs. CellFlow)
- Uses cheap virtual data to explore broadly across morphogen space
- Uses expensive real data to confirm and refine near the optimum
- Automatically down-weights virtual data when it conflicts with real observations

### 4.5 The Full ARL Loop

```
Round 0: INITIALIZATION
├── 186 real conditions from 3 published datasets
├── CellRank 2 forward prediction → +186 virtual points at Day 60, +186 at Day 90
├── CellFlow virtual screen → +23,000 virtual points
└── GP trained on: 186 real + 372 CellRank2 + 23K CellFlow (multi-fidelity)
         │
         ▼
    Acquisition function recommends 24 conditions
    (optimizing for LONG-TERM cell type outcomes, not just Day 21)
         │
         ▼
Round 1: USER EXPERIMENT
├── User runs 24 conditions, harvests Day 21 + Day 45
├── scRNA-seq → Gruffi → scArches/HNOCA → cell type fractions
├── 48 new real points (24 at Day 21, 24 at Day 45)
├── CellRank 2 → 48 virtual points (Day 60, Day 90 predictions)
└── GP refitted: 234 real + 420 virtual + 23K CellFlow
         │
         ▼
    Uncertainty collapses around target region
    Acquisition function recommends 16 refinement conditions
         │
         ▼
Round 2: REFINEMENT
├── 16 conditions, harvest Day 21 + Day 45 + Day 90 (validation)
├── Day 90 real data VALIDATES previous CellRank 2 predictions
├── GP recalibrates virtual data noise (σ²_CR2 adjusted)
└── GP refitted: 282 real + 452 virtual + 23K CellFlow
         │
         ▼
Round 3+: CONVERGENCE / EXPANSION
├── Optimum emerging for target brain region
├── Expand to new morphogens, timing windows, or cell lines
└── Each round feeds back, improving all future predictions
```

### 4.6 Risks and Mitigations

| Risk | Description | Mitigation |
|---|---|---|
| **Distribution shift** | Novel protocols push cells outside Azbukina transport map training distribution | Monitor moscot transport cost; flag virtual points where cost exceeds threshold; always validate with real experiments |
| **Compounding errors** | Systematically biased virtual data causes GP to explore wrong region | Multi-fidelity GP naturally down-weights virtual data when conflicting with real data; real data always overrides |
| **Batch effects** | Three datasets from different labs, platforms (10x vs Parse), timepoints | scANVI integration; CellFlow achieved 0.91 cosine similarity harmonizing these same datasets |
| **Forebrain temporal gap** | CellRank 2 temporal prediction works for posterior brain (6 timepoints) but forebrain only has Day 21 + Day 72 (2 timepoints, need 3) | Collect Day 45 for top forebrain conditions in Round 1; closes the gap |
| **Cell type fraction compositionality** | Fractions sum to 1 — violates GP independence assumptions | Use scCODA (Bayesian compositional analysis) or ILR transform before GP training |

---

## 5. User Input System — "Add Your Own Experiment"

### 5.1 What The User Provides

**File 1: Protocol specification (JSON)**
```json
{
  "experiment_id": "ENG-2026-042",
  "lab": "engram",
  "cell_line": "H9",
  "base_media": "mTeSR1",
  "conditions": [
    {
      "well": "A1",
      "CHIR99021_uM": 3.0,
      "BMP4_ng_mL": 10.0,
      "SHH_ng_mL": 0.0,
      "SAG_nM": 500.0,
      "RA_nM": 0.0,
      "FGF8_ng_mL": 0.0,
      "timing_start_day": 0,
      "timing_end_day": 21,
      "harvest_day": 21
    },
    {
      "well": "A2",
      "CHIR99021_uM": 6.0,
      "BMP4_ng_mL": 0.0,
      "SHH_ng_mL": 100.0,
      "SAG_nM": 0.0,
      "RA_nM": 100.0,
      "FGF8_ng_mL": 0.0,
      "timing_start_day": 0,
      "timing_end_day": 16,
      "harvest_day": 21
    }
  ]
}
```

**File 2: scRNA-seq data (AnnData/.h5ad)**

Standard AnnData format:

| Slot | Required Contents |
|---|---|
| `adata.X` | Raw or normalized counts matrix (cells × genes) |
| `adata.obs["condition"]` | Categorical column mapping each cell to a well/condition from the protocol JSON |
| `adata.obs["batch"]` | Optional batch identifier (if multiple sequencing runs) |
| `adata.var["gene_name"]` | Gene symbols (must be human gene nomenclature) |
| `adata.layers["counts"]` | Raw UMI counts (if `X` is normalized) |

### 5.2 Automated Processing Pipeline

```
User uploads protocol.json + experiment.h5ad
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 1: Quality Control      │
    │  • Cell QC (mito%, counts)    │
    │  • Gruffi stress filtering    │
    │  • Doublet removal (Scrublet) │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 2: Integration          │
    │  • scArches projection → HNOCA│
    │  • Label transfer (cell types)│
    │  • No full retraining needed  │
    │    (parameter-efficient)      │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 3: Fidelity Scoring     │
    │  • BrainSTEM Tier 1: whole-   │
    │    brain region assignment     │
    │  • BrainSTEM Tier 2: region-  │
    │    specific subtype mapping    │
    │  • Fidelity scores per type   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 4: Extract GP Labels    │
    │  • Cell type fractions per    │
    │    condition (Y vector)       │
    │  • Fidelity-weighted fractions│
    │  • scCODA compositional       │
    │    analysis (optional)        │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 5: GP Update            │
    │  • Parse protocol.json → X    │
    │  • Append (X_new, Y_new) to   │
    │    training set               │
    │  • Mark as fidelity=1.0 (real)│
    │  • Refit GP                   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 6: Virtual Augmentation │
    │  • CellRank 2 forward predict │
    │    Day 21 → Day 60, 90       │
    │  • Generate virtual (X, Y)    │
    │    at future timepoints       │
    │  • Mark as fidelity=0.5       │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 7: Recommend Next       │
    │  • Run acquisition function   │
    │    (qLogNEHVI for multi-obj)  │
    │  • Output: next 24 conditions │
    │    as plate map CSV           │
    │  • Include uncertainty map    │
    └───────────────────────────────┘
```

### 5.3 Output: Plate Map + Uncertainty Report

The system returns:

**1. Recommended plate map (CSV)**
```csv
well,CHIR99021_uM,BMP4_ng_mL,SHH_ng_mL,SAG_nM,RA_nM,FGF8_ng_mL,harvest_day,predicted_target_fraction,uncertainty
A1,2.5,5.0,0.0,250.0,0.0,0.0,21,0.42,0.08
A2,4.0,0.0,50.0,0.0,100.0,0.0,21,0.38,0.12
...
```

**2. Uncertainty heatmap** — shows where the GP is most uncertain, justifying why these specific conditions were chosen.

**3. Fidelity report** — for the submitted experiment, shows per-condition cell type fractions, fidelity scores, and off-target detection.

### 5.4 Tools & Libraries Required

| Component | Tool | Role | Status |
|---|---|---|---|
| scRNA-seq QC | Scanpy + Scrublet | Cell filtering, doublet removal | Mature, open-source |
| Stress filtering | **Gruffi** | Remove glycolysis/ER stress artifacts | Published, available |
| Atlas integration | **scArches** + scANVI | Project onto HNOCA without retraining | Mature, open-source |
| Cell type annotation | **HNOCA-tools** | Label transfer from HNOCA reference | Published with atlas |
| Fidelity scoring | **BrainSTEM** | Two-tier fetal brain mapping | Published |
| Compositional analysis | **scCODA** | Handle fractions-sum-to-1 constraint | Published |
| Perturbation standardization | **pertpy** (Nature Methods 2025) | Metadata curation against ontologies | Mature, scverse ecosystem |
| GP-BO engine | **BoTorch** + GPyTorch | Multi-fidelity GP, acquisition functions | Industry standard |
| Temporal prediction | **CellRank 2** + **moscot** | RealTimeKernel for virtual data | Published, documented |
| Virtual screening | **CellFlow** | Generative protocol-to-phenotype model | Published, open-source |
| Protocol encoding | CellFlow encoder | Molecular fingerprints (RDKit) + ESM2 | In CellFlow package |

---

## 6. GP-BO Configuration

### 6.1 Input Encoding

**X vector per condition:**
```
[CHIR, BMP4, SHH, SAG, RA, FGF8, IWP2, SB431542, ..., log(harvest_day), fidelity]
```

- Union of all ~18 unique morphogens across 3 datasets
- Missing morphogens set to 0 ("not added to culture")
- Concentrations log-transformed if ranges span orders of magnitude, then normalized to [0, 1]
- Time as explicit dimension: `log(harvest_day)` normalized
- Fidelity indicator: 1.0 (real), 0.5 (CellRank 2), 0.0 (CellFlow)

### 6.2 Output Encoding

**Y vector per condition:**
```
[frac_radial_glia, frac_IPC, frac_excitatory_neuron, frac_inhibitory_neuron,
 frac_astrocyte, frac_OPC, frac_choroid_plexus, frac_off_target, ...]
```

- Cell type fractions from HNOCA annotation
- Raw proportions (sum to 1) — consider ILR transform for GP or use independent GPs per cell type
- Fidelity-weighted: multiply fractions by BrainSTEM fidelity score to penalize low-quality annotations

### 6.3 Kernel

```python
# Additive kernel: captures morphogen effects + time effects + interactions
k = (
    ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_morphogens))  # morphogen effects
    + ScaleKernel(MaternKernel(nu=2.5, active_dims=[time_dim]))    # time effect
    + ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_morphogens+1))  # interactions
)
```

- **Matérn 5/2 + ARD**: Consensus for biology. ARD lengthscales auto-identify which morphogens matter.
- **Additive structure**: Separates morphogen effects from time effects from interactions.
- ARD will automatically down-weight the ~10 morphogen dimensions that are mostly zeros (only present in 1 dataset).

### 6.4 Acquisition Function

| Stage | Acquisition | Why |
|---|---|---|
| Hackathon demo | Expected Improvement (EI) | Simple, well-understood |
| Production (single objective) | Log Expected Improvement (LogEI) | Numerically stable |
| Production (multi-objective) | qLogNEHVI | Maximize target cell fraction + minimize off-target + maximize fidelity |

### 6.5 Expected Experimental Efficiency

```
Round 0: 186 published conditions (free)
         + 372 CellRank 2 virtual + 23K CellFlow virtual
         → GP identifies high-uncertainty target zones

Round 1: 24 new experiments (~$15K, 4-6 weeks organoid culture)
         → 234 real + ~420 virtual
         → Uncertainty collapses in target region

Round 2: 16 refinement experiments (~$10K)
         → 282 real + ~452 virtual
         → Optimum emerging

Round 3: 12 validation experiments (~$8K, harvest at Day 90)
         → 330 real + ~490 virtual
         → Protocol confirmed, CellRank 2 predictions validated

Total: ~52 new experiments, ~$33K, 4-6 months
vs. brute force: 500-6000+ experiments, years, millions
```

---

## 7. CellRank 2 Temporal Prediction — Feasibility by Brain Region

| Brain Region | Available Timepoints | CellRank 2 Temporal Prediction? | What's Needed |
|---|---|---|---|
| **Posterior brain** (midbrain/hindbrain) | Days 7, 15, 30, 35*, 60, 90, 120 | **YES** — 6+ timepoints, well above 3-timepoint minimum | Nothing — ready now |
| **Forebrain (cortical)** | Day 21 (Sanchis-Calleja) + Day 72 (Amin/Kelley) | **Partial** — 2 timepoints, need 1 more | Collect Day 45 for top 10-15 conditions |
| **Hippocampus** | None | **No** — no morphogen screen data exists | Engram's highest-value experiment to run |

*Day 35 from Azbukina morphogen screen (separate from atlas timepoints)

**Key reference:** Zhang et al. 2025 established the mathematical requirement for ≥3 timepoints for optimal transport-based trajectory inference. The Azbukina atlas satisfies this for posterior brain.

---

## 8. What Doesn't Exist Yet (Engram Builds This)

No platform currently handles the full loop:

**Define morphogen conditions → Run organoid experiment → Sequence scRNA-seq → Annotate via HNOCA → Score fidelity via fetal atlas → Generate virtual data via CellRank 2 → Update multi-fidelity GP → Recommend next experiment**

The closest existing tools:
- **Benchling**: Has experiment tracking but no scRNA-seq-to-GP pipeline
- **Self-driving lab platforms** (BayBE, Atlas): Handle BO loops but not single-cell biology
- **CellFlow**: Predicts protocol outcomes but doesn't close the active learning loop
- **GPerturb**: Characterizes perturbation effects with uncertainty but doesn't recommend next experiments

**Engram's product is the glue** — the end-to-end system that connects wet lab protocols to computational predictions to experiment recommendations, with the multi-fidelity virtual data loop as the key differentiator.

---

## 9. Key References

| Paper | Role in Architecture |
|---|---|
| He et al., Nature 2024 (HNOCA) | Reference atlas for cell type annotation |
| Braun et al., Science 2023 (fetal atlas) | Fidelity benchmarking |
| Toh et al., Science Advances (BrainSTEM) | Two-tier mapping method |
| Sanchis-Calleja et al., Nature Methods 2025 | 97-condition morphogen screen |
| Amin & Kelley et al., Cell Stem Cell 2024 | 46-condition Pasca lab screen |
| Azbukina et al., bioRxiv 2025 | 43-condition screen + temporal atlas |
| Klein et al., bioRxiv 2025 (CellFlow) | Virtual protocol screening, dataset harmonization |
| Weiler et al., Nature Methods 2024 (CellRank 2) | Fate mapping framework |
| Klein et al., Nature 2025 (moscot) | Optimal transport for temporal modeling |
| Zhang et al., npj Syst Biol 2025 | 3-timepoint minimum requirement |
| Narayanan et al., Nature Communications 2025 | GP-BO for cell culture media (3-30x efficiency) |
| Tosh et al., Nature Communications 2025 (BATCHIE) | Active learning for combinatorial screens |
| Xing & Yau, Nature Communications 2025 (GPerturb) | GP for single-cell perturbation modeling |
| Cosenza et al., 2022 | Multi-fidelity BO for cell culture |
| Lotfollahi et al., Nature Biotechnology 2021 (scArches) | Transfer learning for atlas mapping |
| Ahlmann-Eltze et al., Nature Methods 2025 | GPs beat deep learning at <200 datapoints |

---

*Architecture document generated 2026-03-07. Based on synthesis of 50+ papers across 5 parallel research agents. Cross-referenced with existing reports in [[GP_BAYESOPT_ORGANOID_PROTOCOL_DESIGN]], [[MORPHOGEN_SCREEN_DATASETS_ML_READINESS]], [[CELLRANK2_TRAJECTORY_INFERENCE_REPORT]], and [[Morphogen Optimization Pipeline- Feasibility Research Report]].*
