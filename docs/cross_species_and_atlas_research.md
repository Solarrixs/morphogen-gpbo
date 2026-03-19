# Cross-Species Conservation & Neurodevelopmental Atlas Landscape

*Deep research report compiled 2026-03-15. Sources: 60+ searches across PubMed, bioRxiv, web.*

---

## Executive Summary

**Should you incorporate mouse data into the GP-BO pipeline?**

**No — not for GP training data.** The dose-response curves between human and mouse organoids differ quantitatively due to species-intrinsic protein stability differences (2-2.5x temporal scaling). Your existing multi-fidelity GP architecture *could* accommodate mouse data at fidelity ~0.1, but there's no equivalent mouse morphogen screen dataset, and inter-individual human iPSC variation is already larger than the signal you'd extract from mouse data.

**Where mouse/cross-species data IS useful:** Informing kernel structure and morphogen interaction priors (which pathways are synergistic/antagonistic), and as qualitative validation that your GP's learned structure is biologically reasonable.

**What you SHOULD do instead:** Integrate more human fetal brain atlases and organoid datasets. There are 10+ high-quality datasets not yet in your pipeline that could dramatically improve fidelity scoring and GP training.

---

## Part 1: Cross-Species Conservation of Brain Cell Types

### The Hierarchy of Conservation

| Level | Conserved? | Evidence |
|-------|-----------|----------|
| Major cell classes (neurons, glia, etc.) | YES — deeply conserved across all mammals | Bakken 2021, Hodge 2019 |
| Subclasses (SST, PVALB, VIP, LAMP5 interneurons) | YES — same marker genes delineate subclasses | Krienen 2020 |
| Fine-grained subtypes | PARTIALLY — few marker genes conserved at finest resolution | Bakken 2021 |
| Cell proportions | NO — significant species differences | Hodge 2019 (20% of expression variance = species) |
| Gene expression within types | NO — receptor/channel families highly divergent | Hodge 2019 (serotonin receptors = 2nd most divergent) |

### Human-Specific / Primate-Enriched Cell Types

These have NO direct mouse counterpart:

1. **Outer radial glia (oRG/bRG)** — Primate-enriched progenitors in expanded OSVZ. Express HOPX, PTPRZ1. Key driver of cortical expansion. Mouse has sparse, non-self-renewing equivalents.
2. **TAC3+ striatal interneurons** — ~30% of striatal interneurons in primates, absent in mouse/ferret (Krienen 2020). Recent work (2025) found an ancestral precursor exists in mouse but is transcriptomically camouflaged.
3. **Von Economo neurons** — Layer 5 spindle neurons in frontoinsula. Found in humans, great apes, cetaceans. Absent in mice.
4. **Expanded L2/L3 intratelencephalic neurons** — Proportionally much more abundant in human cortex than mouse (Bakken 2021).
5. **Accelerated glial divergence** — Microglia, astrocytes, oligodendrocytes show FASTER expression changes between species than neurons (Jorstad 2023).

### Brain Regions: High vs Low Conservation

**HIGH conservation (mouse data potentially informative):**
- Cerebellum — strongest cross-species conservation
- Brainstem / hypothalamus core circuits
- GABAergic interneuron classes (SST, PVALB, VIP, LAMP5)
- Hippocampus basic architecture

**LOW conservation (human-specific, mouse data unreliable):**
- Cerebral cortex, especially association areas — greatest divergence
- Cortical progenitor zones (OSVZ) — primate innovation
- Striatal interneuron composition — 2x interneuron proportion in primates
- Glial cells broadly — accelerated divergence
- Neurotransmitter receptor expression — fundamentally different

### Implications for GP-BO

Your pipeline uses HNOCA annot_level_1/level_2 categories, which correspond to major classes and subclasses — these ARE conserved. But your fidelity scoring against the Braun human fetal reference is the right approach because:
- Cell **proportions** differ between species (your GP optimizes proportions)
- The **output space** (HNOCA labels) is human-specific
- Cortical progenitors (oRG) are a key organoid population with no mouse equivalent

---

## Part 2: Morphogen Response — Does It Transfer Across Species?

### What's Conserved
The core morphogen axes work the same way:
- WNT (CHIR) → caudalization
- SHH (SAG) → ventralization
- BMP → dorsalization
- The same cocktails produce analogous regional identities

### What's NOT Conserved

| Factor | Impact | Source |
|--------|--------|--------|
| **2-2.5x temporal scaling** | Human cells differentiate 2-2.5x slower than mouse — driven by protein stability, NOT morphogen sensitivity | Rayon 2020, Science |
| **FGF8 role diverges** | FGF8 anteriorizes in mouse; SHH has stronger anteriorizing effect in human organoids | Sanchis-Calleja 2026, Bertacchi 2024 |
| **Dose-response thresholds differ** | Same qualitative mapping, different quantitative concentrations needed | Marshall & Mason 2019 |
| **Human iPSC line variation** | Even within human, iPSC lines from different individuals respond differently to WNT/SHH | Scuderi/Vaccarino 2025, Duo-MAPS |
| **No equivalent mouse screen** | No multiplexed morphogen screen with scRNA-seq exists for mouse organoids | — |

### Should You Pool Human + Mouse GP Training Data?

**No.** Arguments against:
1. Harvest day / exposure windows are incompatible (2-2.5x temporal scaling)
2. Concentration thresholds differ quantitatively
3. Output space (HNOCA cell types) is human-specific
4. Human inter-individual variation already exceeds the signal from mouse data
5. No equivalent mouse morphogen screen dataset exists anyway

**If mouse data ever becomes available**, the cleanest path is your existing multi-fidelity framework: assign mouse data fidelity=0.1-0.2 (lower than CellFlow virtual at 0.0). The `SingleTaskMultiFidelityGP` would automatically downweight mouse observations.

---

## Part 3: Available Neurodevelopmental Atlases — What You're Missing

### Currently Used (3 datasets)
| Dataset | Cells | Role in Pipeline |
|---------|-------|-----------------|
| HNOCA (He 2024) | 1.77M | scPoli mapping reference (step 02) |
| Braun fetal brain (2023) | ~1M | Fidelity scoring reference (step 03) |
| Amin/Kelley morphogen screen (2024) | 46+4 conditions | GP training data (steps 02-04) |

### Tier 1 — High-Priority Additions

| # | Dataset | Cells | Accession | Format | Why It Matters |
|---|---------|-------|-----------|--------|---------------|
| 1 | **BrainSTEM** (Ouyang 2025) | 680K fetal | GEO: GSE281535, Zenodo: 13879662 | RDS → h5ad | Two-tier mapping mirrors your pipeline. Midbrain-specific annotations. R package for organoid eval. |
| 2 | **Velmeshev cortical** (2023) | 700K | CellxGene: bacccb91 | h5ad | Extends beyond first trimester. Matches Day 72 organoid timepoints. |
| 3 | **BTS Atlas + CellTypist** (2024) | 393K | Zenodo: 14177002 | h5ad + model | Pre-trained CellTypist model could replace/complement KNN label transfer. |
| 4 | **Brain Cell Atlas** (2024) | 26.3M (2.2M fetal) | Multi-source | h5ad | Consensus annotations across 70 studies. Most robust target compositions. |

### Tier 2 — Moderate Priority

| # | Dataset | Cells | Accession | Why |
|---|---------|-------|-----------|-----|
| 5 | **Bhaduri arealization** (2021) | 700K | CellxGene | Area-specific cortex signatures for region-specific fidelity |
| 6 | **Velasco cortical organoids** (2019) | 166K | GEO: GSE129519 | Reproducibility baseline for cortical organoids |
| 7 | **Uzquiano multi-omic organoid** (2022) | 610K | Broad SCP1756 | Longitudinal trajectories inform CellRank2 |
| 8 | **Fetal Cortex MERFISH** (2025) | 18M | Zenodo: 14422018 | Spatial layer-specific fidelity targets |
| 9 | **Posterior brain organoid** (Azbukina 2025) | 32K | bioRxiv: 2025.03.20.644368 | Hindbrain/cerebellum coverage |

### Tier 3 — Lower Priority

| # | Dataset | Why |
|---|---------|-----|
| 10 | Eze/Bhaduri early brain (2021) | Earliest stages, useful for early organoid benchmarking |
| 11 | Nowakowski cortex (2017) | Historically important but superseded by 2023 atlases |
| 12 | Polioudakis mid-gestation (2019) | Deep cortical characterization but controlled access (dbGaP) |
| 13 | NEST-Score framework (2025) | Protocol evaluation methodology, not direct data |
| 14 | Phenotypic organoid atlas (Wang 2025) | Disease-focused but neurotypical controls usable |

### CellxGene Ready-to-Download Collections

| Collection | CellxGene ID | Description |
|---|---|---|
| Human Brain Cell Atlas v1.0 | 283d65eb | Adult brain, >3M nuclei |
| First-Trimester Brain (Braun) | 4d8fed08 | Fetal 5-14 pcw |
| Prenatal/Postnatal Cortex (Velmeshev) | bacccb91 | 700K nuclei, 106 donors |
| HNOCA (Neural Organoids) | de379e5f | 1.77M cells, 26 protocols |
| Fetal Gene Expression Atlas | c114c20f | Multi-organ including brain |

---

## Part 4: Computational Tools for Cross-Species Integration

If you ever need to align across species:

| Tool | Best For | Handles Non-1:1 Orthologs? |
|------|----------|---------------------------|
| **SATURN** (Rosen 2024) | Best overall across taxonomic distances | YES (protein language model embeddings) |
| **SAMap** (Tarashansky 2021) | Atlas-level comparisons, distant species | YES (reciprocal BLAST) |
| **CAME/CAMEX** (Liu 2023/2026) | Multi-species (>2), interpretable gene modules | YES (heterogeneous GNN) |
| **VoxHunt** (Fleck 2021) | Maps organoid cells to Allen Brain ISH reference | N/A (spatial reference) |
| Scanorama, Harmony, Seurat, scVI | Within-species or closely related species | NO (require 1:1 orthologs) |

**Benchmark recommendation** (Zhong 2025, NAR): SATURN is most robust across all taxonomic distances. SAMap excels for atlas-level comparisons.

---

## Part 5: Transfer Learning Approaches for GP-BO

No paper directly combines multi-species GP with organoid optimization, but relevant adjacent work:

| Approach | Paper | Applicability |
|----------|-------|--------------|
| Multi-fidelity GP for biomanufacturing | Sun 2022 (arXiv:2211.14493) | Cross-cell-line transfer → directly analogous to cross-species |
| BO for cell culture media | Nature Comms 2025 | 3-30x experiment reduction via transfer learning |
| Multi-output GP dose-response | npj Precision Oncology 2024 | MOGP enables transfer with sparse data |
| Modular variational GPs | NeurIPS 2021 (arXiv:2110.13515) | Compose species-specific GP modules |
| Multi-task QSAR with evolutionary distance | J Cheminformatics 2019 | Phylogenetic distance as task-relatedness metric |

---

## Recommendations

### Immediate Actions
1. **Focus on human data** — do NOT invest in mouse data integration
2. **Download BrainSTEM atlas** (Zenodo: 13879662) — highest-value addition for fidelity scoring
3. **Download Velmeshev 2023** from CellxGene — extends temporal coverage
4. **Download BTS CellTypist model** (Zenodo: 14177002) — potential annotation upgrade

### For the Literature Scraping Pipeline
5. **Monitor these sources**: PubMed ("brain organoid scRNA-seq"), bioRxiv (neuroscience), GEO (Homo sapiens + brain), CellxGene collections, Zenodo (brain atlas h5ad)
6. **Extract from papers**: morphogen concentrations → 24D vector, cell type fractions, GEO/Zenodo accessions, harvest timepoints
7. **Cross-reference by**: shared cell types, shared morphogens, comparable datasets, citation network

### For the Knowledge Graph
8. **Seed with the 20 datasets cataloged above** as initial nodes
9. **Link relationships**: which papers use which atlases as references, which protocols target which brain regions
10. **Vector-embed** paper sections for semantic search ("find all protocols producing cerebellar neurons")

---

## Key References (by topic)

### Cross-Species Cell Type Conservation
- Bakken et al. 2021, *Nature* 598:111-119 — BICCN motor cortex cross-species taxonomy
- Hodge et al. 2019, *Nature* 573:61-68 — Conserved cell types with divergent features
- Jorstad et al. 2023, *Science* 382:eade9516 — Accelerated glial divergence across primates
- Krienen et al. 2020, *Nature* 586:262-269 — Primate-specific interneuron innovations
- Tosches et al. 2018, *Science* 360:881-888 — GABAergic diversity predates amniotes

### Species-Specific Tempo
- Rayon et al. 2020, *Science* 369:eaba7667 — Protein stability drives species-specific pace

### Cross-Species Organoid Comparisons
- Marshall & Mason 2019, *Brain Research* 1724:146427 — Mouse vs man organoid review
- Medina-Cano et al. 2025, *Developmental Cell* — Mouse cortical organoid platform
- Lancaster lab 2024, bioRxiv — Mouse brain organoids capture species differences

### Human Morphogen Screens
- Amin & Kelley 2024, *Cell Stem Cell* — Multiplexed morphogen screen (your primary data)
- Sanchis-Calleja/Azbukina 2026, *Nature Methods* 23:465-478 — Systematic patterning screen
- Scuderi/Vaccarino 2025, *Cell Stem Cell* 32:970-989 — WNT/SHH orthogonal gradients (Duo-MAPS)

### Human Fetal Brain Atlases
- Braun et al. 2023, *Science* 382:eadf1226 — First-trimester atlas (used in pipeline)
- Velmeshev et al. 2023, *Science* 382:eadf0834 — Prenatal/postnatal cortex
- Bhaduri et al. 2021, *Nature* 598:200-204 — Cortical arealization
- BrainSTEM 2025, *Science Advances* — Fetal brain + midbrain subatlas
- BTS Atlas 2024, *Exp Mol Med* 56:2271-2282 — Integrative atlas with CellTypist
- Brain Cell Atlas 2024, *Nature Medicine* — 26.3M cell mega-integration

### Brain-Wide Single-Cell Atlases
- Siletti et al. 2023, *Science* 382:eadd7046 — Adult human brain (3M+ nuclei)
- Yao et al. 2023, *Nature* — Whole adult mouse brain (7M cells)
- BICCN Consortium 2021, *Nature* 598:86-102 — Mammalian motor cortex census

### Computational Tools
- Rosen et al. 2024, *Nature Methods* 21:1492-1500 — SATURN
- Tarashansky et al. 2021, *eLife* — SAMap
- Liu et al. 2023, *Genome Research* 33:96-111 — CAME
- Zhong et al. 2025, *NAR* — Cross-species integration benchmark
