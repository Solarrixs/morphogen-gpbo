---
date: 2026-03-07
title: "CellRank 2 & Trajectory Inference for Temporal Gap Filling in Organoid scRNA-seq"
tags: [research, computational-biology, trajectory-inference, cellrank, organoids]
---

# CellRank 2 & Trajectory Inference for Temporal Gap Filling in Organoid scRNA-seq

> **Purpose**: Assess whether CellRank 2 and related trajectory inference tools can predict Day 60-90 cell fates from Day 21 morphogen screen data (Sanchis-Calleja et al. 2025).
> **Date**: 2026-03-07
> **Status**: Research review

---

## Table of Contents

1. [[#1. What Is CellRank 2 and How Does It Work?]]
2. [[#2. Can CellRank 2 Work With Snapshot Data (No Time Series)?]]
3. [[#3. CellRank 2 and Brain Organoid Trajectory Inference]]
4. [[#4. Alternative Tools]]
5. [[#5. Data Requirements — Does It Need Velocity Data?]]
6. [[#6. Scientific Validity of Predicting Day 60-90 from Day 21]]
7. [[#7. Key Limitations and Risks]]
8. [[#8. Bottom Line — Feasibility for Engram]]
9. [[#9. Sources]]

---

## 1. What Is CellRank 2 and How Does It Work?

**CellRank 2** (Weiler, Lange et al., *Nature Methods* 2024; Nature Protocols 2026) is a unified framework for single-cell fate mapping that generalizes the original CellRank beyond RNA velocity.

### Core Architecture

CellRank 2 uses a **modular kernel-estimator design**:

1. **Kernels** compute cell-cell transition probabilities (a transition matrix inducing a Markov chain) from different data modalities
2. **Estimators** (GPCCA or CFLARE) analyze the Markov chain to identify terminal/initial states, fate probabilities, and lineage-correlated genes
3. Multiple kernels can be **combined** (weighted sum) to integrate different data views

### Available Kernels

| Kernel | Input Data | Use Case |
|--------|-----------|----------|
| **VelocityKernel** | RNA velocity (spliced/unspliced) | Classical scVelo-based directionality |
| **ConnectivityKernel** | k-NN graph (gene expression similarity) | Undirected transcriptomic similarity |
| **PseudotimeKernel** | Any pseudotime ordering (e.g., diffusion pseudotime) | Directs transitions along a precomputed pseudotime axis |
| **CytoTRACEKernel** | Gene expression counts (no extra data needed) | Infers developmental potential from gene count diversity; suited for developmental processes |
| **RealTimeKernel** | Experimental time-point annotations | Uses optimal transport (via moscot) to couple cells across real time points |
| **PrecomputedKernel** | Any user-supplied transition matrix | Maximum flexibility; can incorporate lineage tracing, spatial data, or custom models |

### Key Capabilities
- Identifies **terminal and initial states** automatically via GPCCA (Generalized Perron Cluster Cluster Analysis)
- Computes **fate probabilities** — the probability that each cell reaches each terminal state
- Identifies **driver genes** correlated with specific lineage commitments
- Scales to **millions of cells**
- Can integrate downstream of **moscot** (optimal transport across time) or **moslin** (lineage tracing)

### Citation
> Weiler P, Lange M, Klein M, Pe'er D, Theis FJ. CellRank 2: unified fate mapping in multiview single-cell data. *Nature Methods* 21, 1196-1205 (2024). DOI: 10.1038/s41592-024-02303-9

---

## 2. Can CellRank 2 Work With Snapshot Data (No Time Series)?

**Yes, partially — but with critical caveats.**

### What works with a single snapshot

CellRank 2 has three kernels that do NOT require multiple time points:

1. **PseudotimeKernel**: Requires a pre-computed pseudotime ordering (e.g., from diffusion pseudotime, Palantir, or any other method). This can be computed from a single snapshot. The kernel then biases the k-NN transition matrix to flow in the pseudotime direction.

2. **CytoTRACEKernel**: Estimates developmental potential purely from gene expression diversity (number of expressed genes per cell). More differentiated cells tend to express fewer genes. This requires NO additional data — just the expression matrix. However, it assumes a monotonic relationship between gene count diversity and differentiation state, which may not hold for all cell types.

3. **ConnectivityKernel**: Captures transcriptomic similarity but provides no directionality — essentially an undirected graph.

### What does NOT work with a single snapshot

- **RealTimeKernel**: Requires multiple experimental time points. This is the most powerful kernel for temporal analysis but fundamentally requires data from different time points.
- **VelocityKernel**: Requires RNA velocity estimates (spliced/unspliced counts), which can technically be computed from a single snapshot but have serious reliability issues (see Section 5).

### Practical assessment for single-snapshot use

For a single Day 21 snapshot, you could:
1. Compute diffusion pseudotime or Palantir pseudotime
2. Use `PseudotimeKernel` + `ConnectivityKernel` (combined)
3. OR use `CytoTRACEKernel` + `ConnectivityKernel`
4. Run GPCCA to identify terminal states and fate probabilities

**This tells you where cells are heading within the observed manifold** — i.e., which progenitor states are most committed to which fates *within the Day 21 landscape*. It does NOT tell you what cells will look like at Day 60-90.

---

## 3. CellRank 2 and Brain Organoid Trajectory Inference

### Direct precedent: Fleck et al. 2022 — The Treutlein Lab

The most relevant precedent comes from the **Treutlein lab at ETH Zurich** (the same group behind the Sanchis-Calleja morphogen screen). In Fleck et al. (*Nature* 2022), they:

- Collected scRNA-seq and scATAC-seq data over a **dense time course** of brain organoid development (from pluripotency through neurogenesis)
- Used trajectory inference to reconstruct developmental paths
- Developed **Pando** — a framework that incorporates multi-omic data to infer gene regulatory networks (GRNs) governing organoid development
- Used **pooled genetic perturbation** with single-cell readout to validate transcription factor requirements for cell fate decisions

**Key insight**: They required a **dense time course**, not a single snapshot. Their trajectory inference worked because they had cells sampled across development, creating an overlapping continuum of states.

### Zenk et al. 2024 — Epigenomic Trajectories

Zenk, Fleck, Jansen et al. (*Nature Neuroscience* 2024) reconstructed developmental trajectories from pluripotency in human neural organoid systems using single-cell epigenomics (scATAC-seq). This also used a time course design.

### Sanchis-Calleja et al. 2025 — The Morphogen Screen

The Sanchis-Calleja dataset (Nature Methods 2025/2026) screened 97 morphogen conditions with scRNA-seq readout at Day 21. The paper:
- Mapped morphogen timing and concentration to regional cell identities
- Used SCENIC for gene regulatory network analysis (morphoGRN)
- Identified how different morphogen windows push organoids toward different regional fates (hindbrain, retinal, floor plate, etc.)
- Analyzed data at a single time point (Day 21)

The paper itself did NOT use CellRank or trajectory inference to predict later time points. It characterized the Day 21 cell state landscape.

### Caporale et al. 2024 — Multiplexed Longitudinal Organoids

Caporale et al. (*Nature Methods* 2024) developed multiplexing approaches for cortical brain organoids enabling **longitudinal** dissection of developmental traits at single-cell resolution. This is the kind of multi-timepoint data that CellRank's RealTimeKernel would be ideal for — but again, it required collecting data at multiple time points.

---

## 4. Alternative Tools

### moscot (Multi-Omics Single-Cell Optimal Transport)

**Reference**: Klein D, Palla G, Lange M et al. *Nature* 638, 1065-1075 (2025). DOI: 10.1038/s41586-024-08453-2

moscot maps cells across time and space using optimal transport. Key features:
- **TemporalProblem**: Maps cell populations between experimental time points via optimal transport couplings
- Supports multimodal information (gene expression + other modalities)
- Scales to atlases (millions of cells)
- Can compute cell transition probabilities, interpolated distributions, and feature correlations

**Relevance to Engram**: moscot is the backend for CellRank 2's `RealTimeKernel`. It is powerful but **fundamentally requires multiple time points**. With only Day 21 data, moscot has nothing to transport between.

**Potential use case**: If Engram collected even one additional time point (e.g., Day 7 or Day 14, plus Day 21), moscot could model the temporal transitions and extrapolate trends. Two time points is the minimum; three is recommended per the literature (Zhang et al. 2025: "Recovering biomolecular network dynamics from single-cell omics data requires three time points").

### Waddington-OT (WOT)

**Reference**: Schiebinger et al. *Cell* 176(4), 928-943 (2019).

WOT was the pioneering optimal transport method for time-series scRNA-seq. It:
- Requires **multiple time points** (originally validated on 18 time points of reprogramming data)
- Infers ancestor-descendant relationships between cell populations at consecutive time points
- Computes transport maps and fate probabilities

**Limitation for Engram**: Same as moscot — requires multiple time points. WOT cannot work from a single snapshot. moscot supersedes WOT in most respects (better scaling, multimodal support, more principled OT formulation).

### Palantir

**Reference**: Setty et al. *Nature Biotechnology* 37, 451-460 (2019).

Palantir computes pseudotime and differentiation potential from a single snapshot:
- Models differentiation as a stochastic Markov process on a k-NN graph
- Identifies terminal states
- Computes branch probabilities for each cell
- Provides a pseudotime ordering

**Relevance to Engram**: Palantir CAN work from a single snapshot (Day 21 data). It would give you pseudotime ordering and branch probabilities within the observed manifold. The Palantir pseudotime could then be fed into CellRank 2's `PseudotimeKernel` for further analysis. However, it infers trajectories within the existing data — it does not extrapolate to unseen future states.

### scVelo

**Reference**: Bergen et al. *Nature Biotechnology* 38, 1408-1414 (2020).

scVelo computes RNA velocity from spliced/unspliced mRNA ratios using a dynamical model. Key issues:

**Known limitations (especially for organoids):**
- RNA velocity has been shown to produce **unreliable results** in many settings (Bergen et al. 2021, "RNA velocity: Current challenges and future perspectives")
- A recent benchmark of 14 RNA velocity methods across 17 datasets (Luo et al. 2025, bioRxiv) found "no single method exhibited superior performance in all assessments"
- Velocity estimates are particularly problematic for:
  - Cells with **multiple kinetic regimes** (common in brain organoids)
  - **Mature/steady-state cell types** where splicing dynamics are at equilibrium
  - Long-term differentiation processes where the assumption of observing full splicing dynamics breaks down
- Brain organoids at Day 21 likely contain a mix of progenitors and early differentiating cells — the dynamical model assumption that cells are captured during active transcriptional changes may not hold uniformly

**Verdict**: scVelo can be applied to the Day 21 data but velocity estimates should be treated with extreme caution and validated against known biology.

### SCENIC / SCENIC+

**Reference**: Bravo Gonzalez-Blas et al. *Nature Methods* 20, 1355-1367 (2023). Also: Sanchis-Calleja et al. already used SCENIC in their morphoGRN analysis.

SCENIC+ infers enhancer-driven gene regulatory networks from joint scRNA-seq + scATAC-seq data. It is NOT a trajectory inference tool per se, but:
- Identifies transcription factor regulons active in different cell states
- Can reveal regulatory logic underlying fate decisions
- Combined with trajectory inference, can identify which GRN modules are activated along specific developmental paths

**Relevance to Engram**: SCENIC/morphoGRN analysis is complementary to trajectory inference. It can identify the regulatory programs active at Day 21 and, combined with literature knowledge of brain development, help predict which programs will drive cells toward specific fates. But it does not model temporal dynamics directly.

### Other Notable Tools

- **NeuroVelo** (El Kazwini et al., bioRxiv 2023): Uses Neural ODEs for interpretable learning of temporal cellular dynamics. Can identify gene interactions driving temporal dynamics. More suitable for systems where you have some time-series data.
- **VIA** (Stassen et al., *Nature Communications* 2021): Scalable trajectory inference that can work with single snapshots. Outputs pseudotime, cluster-level trajectory graphs, and lineage probabilities.
- **MultistageOT** (Tronstad et al., 2025): Specifically designed for trajectory inference from a SINGLE snapshot using multistage optimal transport. A novel approach but very new and not yet widely validated.
- **DiRL** (Chang et al., bioRxiv 2025): Uses reinforcement learning on single-cell foundation models to learn differentiation trajectories. Experimental, but represents the frontier of in silico differentiation prediction.

---

## 5. Data Requirements — Does CellRank 2 Need Velocity Data?

**No.** This is one of the key advances of CellRank 2 over CellRank 1.

CellRank 1 was tightly coupled to RNA velocity (scVelo). CellRank 2 generalizes beyond velocity with its modular kernel system.

For Engram's Day 21 snapshot data, the following approaches require NO velocity data:

| Approach | What You Need | What It Gives You |
|----------|--------------|-------------------|
| PseudotimeKernel | Pseudotime (computed from expression data) | Directed transition probabilities along pseudotime |
| CytoTRACEKernel | Just the expression matrix | Developmental potential scores; directed transitions |
| ConnectivityKernel | k-NN graph (standard scanpy preprocessing) | Undirected similarity-based transitions |
| PseudotimeKernel + ConnectivityKernel | Pseudotime + k-NN graph | Combined directed/similarity transitions |

RNA velocity (VelocityKernel) is optional and should be treated as one of several possible data views, not a requirement.

---

## 6. Scientific Validity of Predicting Day 60-90 from Day 21

**This is the critical question, and the honest answer is: trajectory inference alone CANNOT reliably predict Day 60-90 phenotypes from Day 21 data.**

### Why not?

1. **Trajectory inference reconstructs, it does not predict**: All trajectory inference methods reconstruct continuous paths through the observed cell state manifold. They order cells that already exist in the data. They do NOT simulate forward in time to predict novel cell states that have not been observed.

2. **The temporal gap is too large**: Day 21 to Day 60-90 represents 40-70 days of brain organoid development. During this period:
   - Neurogenesis ramps up dramatically
   - Gliogenesis begins (astrocytes, oligodendrocytes)
   - Cortical layering occurs
   - Synaptogenesis initiates
   - Entirely new cell types emerge that may have no representation at Day 21
   - Organoid-specific artifacts (metabolic stress, unintended cell fates) accumulate

3. **No trajectory inference tool can extrapolate beyond the observed manifold**: If a cell type does not exist at Day 21 (e.g., mature oligodendrocytes, upper-layer cortical neurons, astrocytes), no computational method can predict its transcriptomic profile from Day 21 data alone. The algorithm has never "seen" these states.

4. **Stochasticity and environmental sensitivity**: Organoid development is highly stochastic. Even genetically identical organoids diverge substantially by Day 60+. Environmental factors (media composition, hypoxia gradients, organoid size) that accumulate over weeks cannot be modeled from a single early snapshot.

5. **The minimum data requirement for temporal prediction**: The review by Zhang et al. (2025) explicitly states that "recovering biomolecular network dynamics from single-cell omics data requires three time points." This is a mathematical argument — with fewer than three time points, the dynamics are underdetermined.

### What CAN you infer from Day 21 data?

Despite the above limitations, Day 21 data is not useless for predicting later fates. Here is what is scientifically defensible:

1. **Progenitor identity mapping**: At Day 21, you can identify which progenitor types are present (forebrain, midbrain, hindbrain, etc.). The regional identity of progenitors at Day 21 strongly constrains (though does not fully determine) their later fates. This is biology, not trajectory inference.

2. **Commitment probability within observed states**: CellRank/Palantir can tell you which cells are more committed vs. more plastic within the Day 21 landscape. Cells with high fate probability toward a specific progenitor identity are likely to remain in that lineage.

3. **Reference atlas comparison**: You can project Day 21 cells onto published brain organoid time-course atlases (e.g., Fleck et al. 2022, Caporale et al. 2024) to see where they map in developmental space and what their expected fates would be based on the atlas trajectories.

4. **Gene regulatory network inference**: SCENIC/Pando-derived GRNs at Day 21 can reveal which regulatory programs are active. If a condition activates GRN modules known to drive cortical neuron production, you can hypothesize (not predict) that cortical neurons will emerge later.

---

## 7. Key Limitations and Risks

### Technical Limitations

| Limitation | Severity | Mitigation |
|-----------|----------|------------|
| No forward temporal prediction from single snapshot | **Critical** | Collect additional time points (Day 45, Day 60) |
| RNA velocity unreliable for organoid data | **High** | Use velocity-free kernels (Pseudotime, CytoTRACE) |
| Day 21 manifold does not contain Day 60-90 cell states | **Critical** | Use reference atlas projection instead |
| Pseudotime is not real time | **High** | Do not equate pseudotime ordering with calendar time |
| Organoid-to-organoid variability | **High** | Use multiplexing (e.g., MiSTR from Sanchis-Calleja) to control batch effects |
| CytoTRACE assumes monotonic gene count decrease with differentiation | **Medium** | Validate against known markers before trusting CytoTRACE scores |

### Conceptual Risks

1. **Overpromising to investors/collaborators**: Claiming that computational methods can predict Day 60-90 phenotypes from Day 21 data alone would be scientifically indefensible. No published paper has demonstrated this capability.

2. **Confusing trajectory reconstruction with temporal prediction**: Trajectory inference tells you about the **structure** of differentiation within your observed data. It does not simulate the **future**.

3. **False confidence from plausible-looking results**: CellRank will always produce terminal states, fate probabilities, and pseudotime orderings — even from data where these concepts are poorly defined. The outputs look convincing but may not reflect biological reality.

---

## 8. Bottom Line — Feasibility for Engram

> **UPDATE (2026-03-07):** Multi-timepoint data NOW EXISTS across combined morphogen screen datasets. The Azbukina 2025 temporal atlas provides Days 7, 15, 30, 60, 90, 120 for posterior brain organoids. Combined with Sanchis-Calleja (Day 21) and Amin/Kelley (Day 72-74), this changes the CellRank 2 assessment fundamentally.

### Can CellRank 2 help fill the Day 21 to Day 60-90 temporal gap?

**Original answer: No — not from Day 21 data alone.**

**REVISED answer: YES — when combining datasets.** The Azbukina temporal atlas provides 6 timepoints (Days 7, 15, 30, 60, 90, 120) for posterior brain organoids, satisfying the 3-timepoint minimum (Zhang et al. 2025) required for CellRank 2's RealTimeKernel + moscot. This changes the feasibility picture:

| Brain Region | Timepoints Available | CellRank 2 Temporal Prediction? |
|---|---|---|
| **Posterior brain** (midbrain/hindbrain) | Days 7, 15, 30, 35*, 60, 90, 120 (Azbukina atlas) | **YES** — 6+ timepoints, well above minimum |
| **Forebrain (cortical)** | Day 21 (Sanchis-Calleja) + Day 72 (Amin/Kelley) | **Partial** — 2 timepoints, need 1 more (e.g., Day 45) |
| **Hippocampus** | None | **No** — no morphogen screen data exists |

*Day 35 from Azbukina morphogen screen (separate from atlas timepoints)

### How to use CellRank 2 RealTimeKernel + moscot

For posterior brain (where multi-timepoint data exists):

1. **Harmonize across timepoints** using scArches/scANVI with HNOCA reference
2. **Run moscot** (Klein et al., Nature 2025) to compute optimal transport maps between consecutive timepoints: Day 7 → 15 → 30 → 60 → 90 → 120
3. **Initialize CellRank 2 RealTimeKernel** with the moscot transport maps
4. **Compute fate probabilities** — now these are TRUE temporal predictions, not pseudotime extrapolations
5. **Use GPCCA estimator** to identify macrostates and absorption probabilities across the full developmental trajectory

**CellFlow validation**: Klein et al. (bioRxiv 2025, DOI: [10.1101/2025.04.11.648220](https://doi.org/10.1101/2025.04.11.648220)) already demonstrated that harmonizing Sanchis-Calleja + Azbukina data works, achieving 0.91 cosine similarity. CellFlow used flow matching on 176 conditions to run a virtual screen of ~23,000 protocols. This validates the cross-dataset harmonization approach.

### What CAN CellRank 2 still do with single-timepoint data?

For forebrain/cortical conditions (Day 21 only from Sanchis-Calleja):

1. **Rank morphogen conditions by developmental trajectory structure**: Apply CytoTRACEKernel or PseudotimeKernel to each condition and characterize which conditions produce the most committed progenitors, the most diverse fate landscapes, or the strongest lineage biases.

2. **Identify terminal progenitor states within Day 21**: GPCCA can identify absorbing states — the most "committed" cell populations at Day 21 — which are strong predictors of later identity.

3. **Prioritize conditions for follow-up**: Use trajectory analysis to select the 10-15 most interesting conditions (out of 97) for longitudinal follow-up to Day 60-90.

### What should Engram actually do?

**Recommended approach** (in order of priority):

1. **Use existing multi-timepoint data for posterior brain temporal modeling NOW**: Apply CellRank 2 RealTimeKernel + moscot to Azbukina temporal atlas (Days 7-120). This is no longer hypothetical — the data exists.

2. **Use CellRank 2 on Day 21 data for forebrain condition prioritization**:
   - CytoTRACEKernel to assess developmental potential across conditions
   - PseudotimeKernel + GPCCA to identify terminal progenitor states
   - Compare fate probability distributions across morphogen conditions

3. **Close the forebrain temporal gap**: Collect Day 45 data for top 10-15 Sanchis-Calleja conditions. Combined with Day 21 + Amin/Kelley Day 72, this gives 3 timepoints for forebrain → enables RealTimeKernel.

4. **Integrate with GRN analysis**: Combine CellRank fate probabilities with the SCENIC/morphoGRN analysis from Sanchis-Calleja to build mechanistic hypotheses about which regulatory programs drive later fates.

5. **Use CellFlow as complementary tool**: CellFlow predicts protocol outcomes (generative model); CellRank 2 models developmental trajectories (fate mapping). Together they provide both "what will this protocol produce?" and "how do cells reach that fate?"

### Summary Table

| Question | Answer |
|----------|--------|
| Can CellRank 2 predict Day 60-90 from Day 21 alone? | **No** (unchanged) |
| Can CellRank 2 predict temporal trajectories with combined datasets? | **YES — for posterior brain** (6 timepoints via Azbukina atlas) |
| Can CellRank 2 characterize Day 21 trajectories? | **Yes** |
| Does CellRank 2 need RNA velocity? | **No** (multiple velocity-free kernels available) |
| Can it work from a single snapshot? | **Partially** (PseudotimeKernel, CytoTRACEKernel) |
| What is the minimum data for temporal prediction? | **3 time points** (mathematical requirement) — **NOW MET for posterior brain** |
| Best tool for Engram's specific situation? | CellRank 2 RealTimeKernel + moscot for posterior brain; CytoTRACEKernel for forebrain condition ranking |
| Should Engram invest in this? | **Yes** — for both temporal prediction (posterior) AND condition prioritization (forebrain) |

---

## 9. Sources

1. Weiler P, Lange M, Klein M, Pe'er D, Theis FJ. **CellRank 2: unified fate mapping in multiview single-cell data.** *Nature Methods* 21, 1196-1205 (2024). https://doi.org/10.1038/s41592-024-02303-9

2. Weiler P, Theis FJ. **CellRank: consistent and data view agnostic fate mapping for single-cell genomics.** *Nature Protocols* (2026). https://doi.org/10.1038/s41596-025-01314-w

3. Klein D, Palla G, Lange M et al. **Mapping cells through time and space with moscot.** *Nature* 638, 1065-1075 (2025). https://doi.org/10.1038/s41586-024-08453-2

4. Sanchis-Calleja F, Azbukina N et al. **Systematic scRNA-seq screens profile neural organoid response to morphogens.** *Nature Methods* 23, 465-478 (2026). https://doi.org/10.1038/s41592-025-02927-5

5. Fleck JS, Jansen SMJ, Wollny D et al. **Inferring and perturbing cell fate regulomes in human brain organoids.** *Nature* 621, 365-372 (2022). https://doi.org/10.1038/s41586-022-05279-8

6. Zenk F, Fleck JS, Jansen SMJ et al. **Single-cell epigenomic reconstruction of developmental trajectories from pluripotency in human neural organoid systems.** *Nature Neuroscience* 27, 1376-1386 (2024). https://doi.org/10.1038/s41593-024-01652-0

7. Bergen V, Lange M, Peidli S, Wolf FA, Theis FJ. **Generalizing RNA velocity to transient cell states through dynamical modeling.** *Nature Biotechnology* 38, 1408-1414 (2020). https://doi.org/10.1038/s41587-020-0591-3

8. Schiebinger G et al. **Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming.** *Cell* 176(4), 928-943 (2019). https://doi.org/10.1016/j.cell.2019.01.006

9. Bravo Gonzalez-Blas C et al. **SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks.** *Nature Methods* 20, 1355-1367 (2023). https://doi.org/10.1038/s41592-023-01938-4

10. Caporale N et al. **Multiplexing cortical brain organoids for the longitudinal dissection of developmental traits at single-cell resolution.** *Nature Methods* (2024). https://doi.org/10.1038/s41592-024-02555-5

11. Zhang Z et al. **Deciphering cell-fate trajectories using spatiotemporal single-cell transcriptomic data.** *npj Systems Biology and Applications* (2025). https://doi.org/10.1038/s41540-025-00624-9

12. Luo Y et al. **Benchmarking RNA velocity methods across 17 independent studies.** bioRxiv (2025). https://doi.org/10.1101/2025.08.02.668272

13. Tronstad M et al. **MultistageOT: Multistage optimal transport infers trajectories from a snapshot of single-cell data.** arXiv (2025). https://arxiv.org/abs/2502.05241

14. Chang KL, Chen H, Liu Z. **Reinforcement learning enables single-cell foundation models to learn cellular differentiation.** bioRxiv (2025).

15. El Kazwini N et al. **NeuroVelo: interpretable learning of temporal cellular dynamics from single-cell data.** bioRxiv (2023). https://doi.org/10.1101/2023.11.17.567500

16. Hutton A, Meyer JG. **Trajectory inference for single cell omics.** arXiv (2025). https://arxiv.org/abs/2502.09354

17. CellRank documentation: https://cellrank.readthedocs.io/
18. moscot documentation: https://moscot.readthedocs.io/
