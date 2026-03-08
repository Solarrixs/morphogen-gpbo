# Fetal Brain Developmental Atlases: Comprehensive Report

> **Date:** 2026-03-07
> **Purpose:** Identify fetal brain scRNA-seq and spatial transcriptomics atlases with multiple gestational timepoints for validating Engram's organoid differentiation trajectories, with emphasis on hippocampus and entorhinal cortex.

---

## Executive Summary

There are now at least 10 major single-cell and spatial transcriptomic atlases of the developing human brain, collectively covering gestational weeks 5 through 41 (first trimester through birth). Several are directly relevant to Engram's goal of mapping morphogen-treated organoid cells against normal fetal brain development. The strongest resources for hippocampal/entorhinal cortex validation are the Zhong/Wang 2020 hippocampal atlas (GW16-27, scRNA-seq) and the Braun/Linnarsson 2023 first-trimester atlas (GW5-14, ~600 cell states). For broader cortical benchmarking, the Wang/Kriegstein 2025 and Keefe/Nowakowski 2025 atlases are state-of-the-art. BrainSpan is useful for bulk gene expression trends but is NOT single-cell resolution and has limited utility for cell-type-level organoid validation.

---

## 1. Atlas-by-Atlas Analysis

### 1.1 Braun/Linnarsson 2023 — First-Trimester Developing Human Brain
- **Paper:** Braun et al., "Comprehensive cell atlas of the first-trimester developing human brain," *Science* 382, eadf1226 (2023)
- **Gestational Weeks:** 5-14 pcw (post-conceptional weeks)
- **Technology:** scRNA-seq + spatial transcriptomics (ISS-based)
- **Cell Count:** ~600 distinct cell states across 12 major classes
- **Brain Regions:** Whole brain — forebrain, midbrain, hindbrain; includes hippocampal anlage and medial pallium at early stages
- **Key Features:**
  - Highest-resolution first-trimester atlas available
  - Detailed differentiation trajectories for forebrain and midbrain
  - Identifies region-specific glioblasts, pre-astrocytes, pre-OPCs
  - Spatial mapping of cell states to anatomical domains at 5 pcw
  - Includes spatial transcriptomics (ISS) to validate spatial positions
- **Data Access:** Data browser at [https://hdca-sweden.scilifelab.se/brain](https://hdca-sweden.scilifelab.se/brain); raw data at ArrayExpress
- **Relevance to Engram:** **HIGH.** The GW5-14 window overlaps with early organoid development (roughly Day 0-50 equivalent). The ~600 cell states provide a fine-grained reference for mapping organoid cells against normal developmental trajectories. Forebrain and medial pallium coverage is directly relevant to hippocampal organoid work.

### 1.2 Zhong/Wang 2020 — Developing Human Hippocampus
- **Paper:** Zhong et al., "Decoding the development of the human hippocampus," *Nature* 577, 531-536 (2020)
- **Gestational Weeks:** GW16, GW18, GW20, GW22, GW25, GW27
- **Technology:** scRNA-seq + ATAC-seq
- **Cell Count:** 30,416 cells from 7 fetal hippocampi
- **Brain Regions:** Hippocampus specifically — CA1, CA3, dentate gyrus
- **Key Features:**
  - 47 cell subtypes with developmental trajectories
  - Migrating paths and lineages of PAX6+ and HOPX+ hippocampal progenitors
  - Regional markers distinguishing CA1, CA3, and dentate gyrus neurons
  - Transcriptional regulatory networks (e.g., PROX1 for dentate gyrus)
  - ATAC-seq chromatin accessibility data paired with scRNA-seq
  - Interactive data browser at [wanglaboratory.org](http://wanglaboratory.org:3838/hipp/)
- **Data Access:** GEO accession **GSE119212**
- **Relevance to Engram:** **CRITICAL.** This is the only dedicated fetal hippocampal scRNA-seq atlas with multiple gestational timepoints. The GW16-27 range corresponds roughly to organoid Day 40-90. The 47 subtypes and CA1/CA3/DG markers are exactly what Engram needs for hippocampal organoid validation. The ATAC-seq data adds regulatory network information for understanding morphogen-responsive transcription factor cascades.

### 1.3 Wang/Kriegstein 2025 — Developing Human Neocortex
- **Paper:** Wang, Li et al., "Molecular and cellular dynamics of the developing human neocortex," *Nature* 647, 169-178 (2025)
- **Gestational Weeks:** First trimester through adolescence (5 main developmental stages)
- **Technology:** Paired snRNA-seq + snATAC-seq (single-nucleus multiome); spatial transcriptomics on subset
- **Cell Count:** 38 neocortical samples (exact cell numbers in hundreds of thousands)
- **Brain Regions:** Prefrontal cortex (PFC) and primary visual cortex (V1)
- **Key Features:**
  - Cell-type-specific, age-specific, and area-specific gene regulatory networks
  - Identified "Tri-IPCs" (tripotential intermediate progenitor cells) for GABAergic neurons, OPCs, and astrocytes
  - Neurogenesis-to-gliogenesis transition mapped
  - Spatial transcriptomics subset for intercellular communication
  - Disease risk map: autism spectrum disorder enriched in second-trimester intratelencephalic neurons
- **Data Access:** Likely GEO/CellxGene (check publication supplementary for accession numbers)
- **Relevance to Engram:** **HIGH.** The multiome (RNA + ATAC) data across development provides gene regulatory network context that is valuable for understanding how morphogen signaling cascades shape cell fate. However, coverage is neocortical (PFC, V1), not hippocampal/entorhinal.

### 1.4 Keefe/Nowakowski 2025 — Lineage-Resolved Developing Human Cortex
- **Paper:** Keefe, Steyert & Nowakowski, "Lineage-resolved atlas of the developing human cortex," *Nature* 647, 194-202 (2025)
- **Gestational Weeks:** Multiple prenatal timepoints (specific GW range in paper)
- **Technology:** scRNA-seq with lineage tracing
- **Cell Count:** Large-scale (exact number in paper)
- **Brain Regions:** Human neocortex
- **Key Features:**
  - True lineage resolution — not just snapshots but actual lineage relationships linking progenitors to progeny
  - Maps differentiation and maturation trajectories
  - Links neural stem cell programs to specific cortical cell types
  - Complements the Wang/Kriegstein atlas with orthogonal lineage data
- **Data Access:** Published as open access; data likely at GEO/CellxGene
- **Relevance to Engram:** **HIGH.** Lineage-resolved data is uniquely valuable for validating that organoid differentiation follows biologically plausible lineage trees. However, focus is neocortical, not hippocampal.

### 1.5 Qian/Walsh 2025 — Spatial Transcriptomics of Human Cortical Development (MERFISH)
- **Paper:** Qian et al., "Spatial transcriptomics reveals human cortical layer and area specification," *Nature* (2025)
- **Gestational Weeks:** GW15-GW34
- **Technology:** MERFISH (multiplexed error-robust FISH) with deep-learning nucleus segmentation
- **Brain Regions:** Multiple cortical areas including cingulate cortex, **hippocampus**, occipital cortex
- **Key Features:**
  - Spatial maps of cell types across cortical areas at multiple gestational ages
  - Layer and area specification resolved spatially
  - **Includes hippocampus** as one of the major cortical areas profiled
  - Deep-learning-based nucleus segmentation for single-cell resolution
- **Data Access:** Open access publication; check supplementary for raw data
- **Relevance to Engram:** **HIGH.** One of the few spatial atlases that explicitly includes hippocampus. The GW15-34 range covers mid-to-late gestation, complementing the Zhong/Wang scRNA-seq atlas. Spatial information provides morphogen gradient context that dissociated scRNA-seq cannot.

### 1.6 Eze/Nowakowski/Kriegstein 2021 — Early Human Brain Development
- **Paper:** Eze et al., "Single-cell atlas of early human brain development highlights heterogeneity of human neuroepithelial cells and early radial glia," *Nature Neuroscience* (2021)
- **Gestational Weeks:** GW5-GW25 (early development focus)
- **Technology:** scRNA-seq
- **Brain Regions:** Whole brain at early stages; cortex focus
- **Key Features:**
  - Heterogeneity of neuroepithelial cells and early radial glia
  - Important for understanding the earliest stages of brain regionalization
  - Defines early progenitor diversity before regional specification
- **Data Access:** GEO (check publication)
- **Relevance to Engram:** **MODERATE-HIGH.** Critical for understanding the earliest organoid stages (Day 0-30 equivalent), when neural induction and initial patterning are occurring. Shows what neuroepithelial cells and radial glia should look like.

### 1.7 Velmeshev/Kriegstein 2023 — Prenatal and Postnatal Human Cortex
- **Paper:** Velmeshev et al., "Single-cell analysis of prenatal and postnatal human cortical development," *Science* 382, eadf0834 (2023)
- **Gestational Weeks:** Prenatal through postnatal (broad developmental span)
- **Technology:** scRNA-seq + SCENIC+ (enhancer gene regulatory networks)
- **Brain Regions:** Cortex
- **Key Features:**
  - Bridges prenatal and postnatal development — one of few atlases spanning birth
  - SCENIC+ enhancer-gene regulatory network analysis
  - Published in same Science issue as Braun/Linnarsson (complementary atlases)
- **Data Access:** CellxGene at [cellxgene.cziscience.com/collections/bacccb91-066d-4453-b70e-59de0b4598cd](https://cellxgene.cziscience.com/collections/bacccb91-066d-4453-b70e-59de0b4598cd)
- **Relevance to Engram:** **MODERATE.** Useful for later-stage organoid validation and for regulatory network context. Cortex-focused rather than hippocampal.

### 1.8 Ramos/Tsankova 2022 — Late Prenatal Human Neurodevelopment
- **Paper:** Ramos et al., "An atlas of late prenatal human neurodevelopment resolved by single-nucleus transcriptomics," *Nature Communications* 13 (2022)
- **Gestational Weeks:** GW17-GW41 (second and third trimesters) + 3 adult controls
- **Technology:** snRNA-seq
- **Cell Count:** >200,000 nuclei from 15 prenatal samples
- **Brain Regions:** Germinal matrix and cortical plate
- **Key Features:**
  - Uniquely covers the third trimester (GW28-41), a period poorly represented in other atlases
  - Focus on prenatal gliogenesis — transient glial intermediate progenitor cells (gIPCs) and nascent astrocytes
  - Lineage trajectory and RNA velocity analysis
  - High temporal resolution across late gestation
- **Data Access:** GEO (check publication supplementary)
- **Relevance to Engram:** **MODERATE.** Important for understanding later organoid maturation stages. Third-trimester gliogenesis data is unique. However, focus is cortical plate/germinal matrix, not specifically hippocampal.

### 1.9 BrainSTEM (Toh/Sun/Ouyang 2025) — Fetal Brain Atlas for Organoid Benchmarking
- **Paper:** Toh et al., "BrainSTEM: A single-cell multiresolution fetal brain atlas reveals transcriptomic fidelity of human midbrain cultures," *Science Advances* 11, eadu7944 (2025)
- **Technology:** Integrated single-cell atlas (meta-atlas from multiple datasets)
- **Brain Regions:** Whole brain, with midbrain subatlas
- **Key Features:**
  - **Explicitly designed for organoid benchmarking** — the two-tier mapping strategy (BrainSTEM) maps organoid cells first to whole-brain regions, then to refined subregion atlases
  - Revealed substantial "off-target" cell populations in published midbrain organoid protocols
  - Demonstrated that many protocols inflate reported mDA yields
  - Framework is generalizable to any brain region, including hippocampus
- **Data Access:** Check publication supplementary
- **Relevance to Engram:** **HIGH (methodologically).** While focused on midbrain, the BrainSTEM two-tier mapping framework is exactly what Engram should adopt for hippocampal organoid validation. The concept of mapping organoid cells to whole-brain first (to detect off-target populations) before refined hippocampal mapping is critical. Engram could construct a similar "BrainSTEM-Hippo" pipeline.

### 1.10 BrainSpan (Allen Institute)
- **Paper:** Consortium publication; data at [brainspan.org](https://www.brainspan.org/)
- **Gestational Weeks/Ages:** 8 pcw through 40 years (full lifespan)
- **Technology:** Bulk RNA-seq, exon microarray, in situ hybridization (ISH), prenatal laser microdissection (LMD) microarray
- **Brain Regions:** 16 brain structures including hippocampus (HIP), mediodorsal nucleus of thalamus, striatum, amygdala, and neocortical areas
- **Key Features:**
  - Developmental Transcriptome: bulk RNA-seq and microarray across full lifespan
  - Prenatal LMD Microarray: laser-microdissected subregions at 15 pcw, 16 pcw, 21 pcw (female/male samples)
  - ISH: in situ hybridization for spatial gene expression
  - Reference Atlases: annotated anatomical atlases at 15 pcw, 21 pcw, 34 years
  - **NOT single-cell resolution** — bulk RNA-seq averages across all cell types in a tissue sample
- **Data Access:** Fully public at [brainspan.org/static/download.html](https://www.brainspan.org/static/download.html); RNA-seq, microarray, and ISH data all downloadable. Also available via Allen Brain Map portal.
- **Relevance to Engram:** **LOW-MODERATE for cell-type validation.** BrainSpan is useful for:
  - Confirming that bulk gene expression profiles of specific morphogen receptors/ligands change across development
  - Checking temporal expression patterns of FGFR, BMPR, WNT pathway genes at bulk level
  - Spatial ISH data for visualizing where specific genes are expressed in prenatal brain sections
  - **NOT useful** for cell-type-level organoid benchmarking — no single-cell resolution
  - Can serve as a "sanity check" reference but should not be the primary validation dataset

### 1.11 BGI Stereo-seq / Spatial Transcriptomics of Fetal Brain
- **Papers:** Multiple BGI/STOmics publications; Zhang et al. 2025 (bioRxiv) — "Spatial Transcriptomics Reveal Developmental Dynamics of the Human Cerebral Cortex and Striatum"
- **Technology:** Stereo-seq (nanoscale spatial transcriptomics)
- **Brain Regions:** Forebrain regions, cortex, striatum; 25 forebrain subregions identified
- **Key Features:**
  - Nanoscale resolution (subcellular)
  - Centimeter-scale field of view (up to 13 cm x 13 cm)
  - Combined with snRNA-seq for cell type annotation
  - Morphogen gradient information through spatial gene expression patterns
  - BGI/Mesoscopic Brain Mapping Consortium contributions
- **Data Access:** STOmics database; check individual publications
- **Relevance to Engram:** **MODERATE-HIGH for morphogen gradients.** Spatial transcriptomics uniquely captures morphogen gradient information that is lost in dissociated scRNA-seq. If hippocampal regions are included in any BGI datasets, the spatial morphogen expression data would be directly useful for validating organoid patterning.

### 1.12 Nano/Bhaduri 2025 — Integrated Meta-Analysis of Cortical Development Atlases
- **Paper:** Nano et al., "Integrated analysis of molecular atlases unveils modules driving developmental cell subtype specification in the human cortex," *Nature Neuroscience* 28, 949-963 (2025)
- **Technology:** Meta-analysis of 7 developmental and 16 adult cortical datasets
- **Key Features:**
  - >500 gene co-expression networks (meta-modules) across cortical development
  - Identifies modules with spatiotemporal expression patterns for cell fate specification
  - Validated in primary human cortical tissues
  - Centers on peak neurogenesis stages
- **Relevance to Engram:** **MODERATE.** The co-expression modules could help interpret morphogen-driven gene expression changes in organoids.

---

## 2. Summary Table

| Atlas | Paper | GW Range | Technology | Cell Count | Hippocampus? | EC? | Data Access | Engram Utility |
|-------|-------|----------|------------|------------|-------------|-----|-------------|----------------|
| Braun/Linnarsson 2023 | Science 382, eadf1226 | GW5-14 | scRNA-seq + spatial (ISS) | ~600 cell states | Early anlage | No | ArrayExpress; hdca-sweden.scilifelab.se | HIGH |
| Zhong/Wang 2020 | Nature 577, 531-536 | GW16-27 | scRNA-seq + ATAC-seq | 30,416 cells | **YES (CA1, CA3, DG)** | No | **GEO: GSE119212** | **CRITICAL** |
| Wang/Kriegstein 2025 | Nature 647, 169-178 | GW~7-adolescence | snRNA-seq + snATAC-seq + spatial | ~38 samples | No (PFC, V1) | No | GEO (check paper) | HIGH |
| Keefe/Nowakowski 2025 | Nature 647, 194-202 | Prenatal | scRNA-seq + lineage | Large | No (neocortex) | No | GEO (check paper) | HIGH |
| Qian/Walsh 2025 | Nature (2025) | GW15-34 | MERFISH | Spatial | **YES** | Likely partial | Check paper | HIGH |
| Eze/Nowakowski 2021 | Nat Neurosci (2021) | GW5-25 | scRNA-seq | Moderate | Early stages | No | GEO | MODERATE-HIGH |
| Velmeshev 2023 | Science 382, eadf0834 | Prenatal-postnatal | scRNA-seq + SCENIC+ | Large | No (cortex) | No | CellxGene | MODERATE |
| Ramos/Tsankova 2022 | Nat Commun 13 (2022) | GW17-41 | snRNA-seq | >200,000 | No (cortical plate) | No | GEO | MODERATE |
| BrainSTEM 2025 | Sci Adv 11, eadu7944 | Multi-stage | Integrated meta-atlas | Meta | Whole-brain | Whole-brain | Check paper | HIGH (method) |
| BrainSpan | Allen Institute | GW8-40 years | Bulk RNA-seq, microarray, ISH | Bulk | Yes (bulk) | Yes (bulk) | brainspan.org | LOW-MOD |
| BGI Stereo-seq | Multiple (2023-2025) | Various | Stereo-seq (spatial) | Spatial | Partial | Unknown | STOmics DB | MOD-HIGH |
| Nano/Bhaduri 2025 | Nat Neurosci 28, 949-963 | Meta-analysis | 23 datasets integrated | Meta | No | No | Check paper | MODERATE |

---

## 3. Gestational Week Coverage vs. Organoid Day Equivalents

Understanding the rough correspondence between organoid culture days and fetal gestational weeks:

| Organoid Day | Approximate GW Equivalent | Best Reference Atlases |
|-------------|---------------------------|----------------------|
| Day 0-14 | ~GW3-5 (neural induction) | Braun/Linnarsson 2023 (earliest timepoints) |
| Day 14-30 | ~GW5-8 (early patterning) | Braun/Linnarsson 2023 |
| Day 30-50 | ~GW8-12 (regionalization) | Braun/Linnarsson 2023 |
| Day 50-70 | ~GW12-18 (neurogenesis peak) | Eze 2021; Zhong/Wang 2020 (from GW16) |
| Day 70-90 | ~GW18-25 (mid-gestation) | Zhong/Wang 2020; Qian/Walsh 2025 |
| Day 90-120 | ~GW25-34 (late neurogenesis, early gliogenesis) | Ramos/Tsankova 2022; Qian/Walsh 2025 |
| Day 120+ | ~GW34+ (gliogenesis, maturation) | Ramos/Tsankova 2022 |

**Note:** These correspondences are approximate. Organoid maturation is generally slower and less organized than in vivo development, and regional identity markers may appear on different timescales.

---

## 4. Morphogen Receptor Expression Data

### Which atlases contain FGFR, BMPR, Frizzled/WNT, and NTRK2/TrkB expression data?

All scRNA-seq atlases contain genome-wide expression data, so in principle **all of them** contain morphogen receptor expression. However, the key question is whether the authors specifically analyzed morphogen receptors and whether the data is accessible for querying:

| Gene Family | Key Genes | Best Atlas for Querying |
|-------------|-----------|----------------------|
| **FGF receptors** | FGFR1, FGFR2, FGFR3 | Braun/Linnarsson (data browser); Zhong/Wang (GEO: GSE119212); BrainSpan (bulk) |
| **BMP receptors** | BMPR1A, BMPR1B, BMPR2, ACVR1 | Same as above — query from count matrices |
| **WNT receptors** | FZD1-10, LRP5, LRP6 | Braun/Linnarsson (spatial data shows WNT signaling gradients); Zhong/Wang |
| **Neurotrophin receptors** | NTRK2 (TrkB), NTRK1, NTRK3, NGFR | All scRNA-seq atlases — query from count matrices |
| **SHH pathway** | PTCH1, SMO, GLI1-3 | Braun/Linnarsson (forebrain ventralization data) |

**Practical recommendation:** Download count matrices from GEO: GSE119212 (hippocampus) and the Braun/Linnarsson data browser to directly query expression of all morphogen receptors across cell types and gestational ages. BrainSpan ISH data can visualize spatial expression of individual genes in tissue sections.

**Relevant related work:** Scuderi et al. (2024, bioRxiv) — "Specification of human regional brain lineages using orthogonal gradients of WNT and SHH in organoids" — directly addresses morphogen gradient effects on brain regionalization in organoids, using WNT and SHH gradients with spatial transcriptomics readout. Directly relevant to Engram's morphogen optimization pipeline.

---

## 5. Entorhinal Cortex Coverage

The entorhinal cortex (EC) is **poorly represented** in existing fetal brain atlases:

- **Zhong/Wang 2020:** Hippocampus but not explicitly entorhinal cortex
- **Qian/Walsh 2025:** Includes hippocampus and multiple cortical areas via MERFISH; entorhinal cortex may be partially covered but is not a primary focus
- **Braun/Linnarsson 2023:** At GW5-14, medial temporal lobe structures are still early in development; some medial pallium coverage
- **BrainSpan:** Includes hippocampus at bulk level, but entorhinal cortex is not a standard dissection target
- **Franjic et al. (2022, Cell):** "Transcriptomic taxonomy and neurogenic trajectories of adult human, macaque, and pig hippocampal and entorhinal cells" — **adult** hippocampal-entorhinal snRNA-seq, not fetal
- **Liu/Bergmann 2021 (Front Neuroanat):** "Development of the Entorhinal Cortex Occurs via Parallel Lamination During Neurogenesis" — anatomical/histological study of EC development, not scRNA-seq
- **Blankvoort et al. 2022 (Front Neural Circuits):** Single-cell transcriptomic and chromatin profiles of entorhinal cortex — **adult mouse**, not human fetal

**Gap assessment:** There is currently **no dedicated fetal human entorhinal cortex scRNA-seq atlas.** The closest resources are:
1. The Zhong/Wang hippocampal atlas (adjacent tissue)
2. The Qian/Walsh MERFISH atlas (may include EC in medial temporal sections)
3. Adult EC atlases that can be used as endpoint references

This represents a genuine gap in the field and an opportunity for Engram — organoid-derived EC-like cells validated against whatever partial reference data exists would be novel.

---

## 6. Spatial Transcriptomics Atlases with Morphogen Gradient Information

| Atlas | Technology | Morphogen Gradient Utility |
|-------|-----------|---------------------------|
| **Qian/Walsh 2025** | MERFISH | Layer-specific and area-specific expression; morphogen receptor spatial patterns across GW15-34 |
| **Braun/Linnarsson 2023** | ISS (in situ sequencing) | Spatial mapping of cell states to anatomical domains; early patterning center signaling |
| **Wang/Kriegstein 2025** | Spatial transcriptomics (subset) | Intercellular communication analysis; ligand-receptor spatial patterns |
| **BGI Stereo-seq** | Stereo-seq (nanoscale) | Highest resolution; potential for subcellular morphogen receptor localization |
| **Scuderi et al. 2024** | Spatial transcriptomics on organoids | Directly shows WNT/SHH gradient effects on regionalization in organoids |

---

## 7. BrainSpan: Detailed Assessment

**What BrainSpan actually contains:**
1. **Developmental Transcriptome:** Bulk RNA-seq (RPKM values) and exon microarray for 16 brain structures across 8 pcw to 40 years. Includes hippocampus. ~524 samples total.
2. **Prenatal LMD Microarray:** Laser-microdissected subregions from 4 brains (15 pcw male, 16 pcw female, 21 pcw female x2). Higher spatial resolution than bulk RNA-seq but still not single-cell.
3. **ISH (In Situ Hybridization):** Spatial gene expression images for selected genes at prenatal stages. Useful for visualizing individual gene expression patterns.
4. **Reference Atlases:** Annotated anatomical atlases at 15 pcw, 21 pcw, and 34 years with hierarchical brain structure ontology.

**What BrainSpan is NOT:**
- Not single-cell resolution
- Not sufficient for cell-type-level organoid benchmarking
- Does not capture cell state heterogeneity within a brain region
- The RNA-seq data averages signal across all cell types

**Where BrainSpan is still useful for Engram:**
- Confirming developmental timing of morphogen receptor expression (bulk level)
- ISH images showing spatial patterns of key genes (e.g., WNT ligands, FGF ligands, BMP antagonists) in prenatal brain sections
- Anatomical ontology for annotating brain regions in organoid mapping
- Supplementary "sanity check" against single-cell data

---

## 8. Bottom Line: Recommendations for Engram

### Tier 1 — Must-Download Datasets
1. **Zhong/Wang 2020 (GEO: GSE119212):** The only fetal hippocampal scRNA-seq atlas. Download raw count matrices. Use for mapping hippocampal organoid cells against CA1/CA3/DG markers across GW16-27. Query FGFR, BMPR, FZD, NTRK2 expression directly.
2. **Braun/Linnarsson 2023:** Use data browser for early developmental stages (GW5-14). Critical for validating early organoid patterning (Day 0-50). The ~600 cell states provide the most granular reference for early brain development.
3. **BrainSTEM framework (Toh et al. 2025):** Adopt the two-tier mapping methodology. Map organoid cells to whole-brain first (detect off-target populations), then to hippocampal subatlas. This is the methodological gold standard for organoid benchmarking.

### Tier 2 — Strongly Recommended
4. **Qian/Walsh 2025 (MERFISH):** Spatial data including hippocampus at GW15-34. Provides spatial morphogen gradient context that no dissociated atlas can.
5. **Wang/Kriegstein 2025:** Multiome (RNA + ATAC) data with gene regulatory networks. Use for understanding morphogen-responsive TF cascades, even though coverage is neocortical.
6. **Keefe/Nowakowski 2025:** Lineage-resolved data for validating that organoid differentiation follows biologically plausible lineage trees.

### Tier 3 — Supplementary
7. **Ramos/Tsankova 2022:** Late prenatal (GW17-41) for later-stage organoid maturation validation.
8. **BrainSpan:** Bulk expression trends and ISH images for specific genes of interest.
9. **Eze/Nowakowski 2021:** Early brain development reference (GW5-25).

### Key Gaps to Be Aware Of
- **No fetal entorhinal cortex scRNA-seq atlas exists.** This is a real gap. Engram may need to use adult EC references (Franjic et al. 2022) as endpoint targets.
- **Organoid-to-fetal mapping is inherently approximate.** Day-to-GW correspondences are rough estimates; organoids mature slower and with less spatial organization.
- **Morphogen receptor expression** must be queried directly from count matrices — no atlas has pre-computed morphogen pathway analyses across all developmental stages.

### Recommended Workflow
1. Download Zhong/Wang GEO: GSE119212 count matrix
2. Download Braun/Linnarsson data from their browser
3. Build integrated reference using Scanpy/Seurat
4. Implement BrainSTEM two-tier mapping:
   - Tier 1: Map organoid cells against whole-brain reference (Braun/Linnarsson + BrainSTEM)
   - Tier 2: Map hippocampal-classified cells against Zhong/Wang hippocampal subatlas
5. Query morphogen receptor expression (FGFR1-3, BMPR1A/1B/2, FZD1-10, NTRK2) across cell types and gestational ages
6. Use Qian/Walsh MERFISH data for spatial validation where available

---

## References

1. Braun E, Danan-Gotthold M, Borm LE, et al. Comprehensive cell atlas of the first-trimester developing human brain. *Science* 382, eadf1226 (2023).
2. Zhong S, Ding W, Sun L, et al. Decoding the development of the human hippocampus. *Nature* 577, 531-536 (2020).
3. Wang L, Wang C, Moriano JA, et al. Molecular and cellular dynamics of the developing human neocortex. *Nature* 647, 169-178 (2025).
4. Keefe MG, Steyert MR, Nowakowski TJ. Lineage-resolved atlas of the developing human cortex. *Nature* 647, 194-202 (2025).
5. Qian X, Coleman K, Jiang S, et al. Spatial transcriptomics reveals human cortical layer and area specification. *Nature* (2025).
6. Eze UC, et al. Single-cell atlas of early human brain development highlights heterogeneity of human neuroepithelial cells and early radial glia. *Nat Neurosci* (2021).
7. Velmeshev D, Perez Y, Yan Z, et al. Single-cell analysis of prenatal and postnatal human cortical development. *Science* 382, eadf0834 (2023).
8. Ramos SI, Mussa ZM, Falk EN, et al. An atlas of late prenatal human neurodevelopment resolved by single-nucleus transcriptomics. *Nat Commun* 13 (2022).
9. Toh HSY, Xu L, Chen C, et al. BrainSTEM: A single-cell multiresolution fetal brain atlas reveals transcriptomic fidelity of human midbrain cultures. *Sci Adv* 11, eadu7944 (2025).
10. Nano PR, Fazzari E, Azizad D, et al. Integrated analysis of molecular atlases unveils modules driving developmental cell subtype specification in the human cortex. *Nat Neurosci* 28, 949-963 (2025).
11. Franjic D, Skarica M, Ma S, et al. Transcriptomic taxonomy and neurogenic trajectories of adult human, macaque, and pig hippocampal and entorhinal cells. *Cell* (2022).
12. Scuderi S, Kang TY, Jourdon A, et al. Specification of human regional brain lineages using orthogonal gradients of WNT and SHH in organoids. *bioRxiv* (2024).
13. BrainSpan Consortium. BrainSpan: Atlas of the Developing Human Brain. brainspan.org.
