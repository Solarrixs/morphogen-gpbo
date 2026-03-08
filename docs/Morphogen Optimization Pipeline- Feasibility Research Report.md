# Morphogen Optimization Pipeline: Feasibility Research Report
**Date:** 2026-03-07 | **For:** Engram Hackathon Team (Maxx, Mustafo, Wayne)
**Sources:** 50+ papers across 5 research agents

---

## Executive Summary

**Is the GP-BO morphogen optimization pipeline feasible?**

**YES — with important scope adjustments.** The core idea is sound and well-supported by literature. But three claims need correction:

| Original Assumption | Reality (Updated 2026-03-07) | Impact |
|---|---|---|
| CellRank 2 can predict Day 60-90 from Day 21 | **YES for posterior brain** — Azbukina atlas provides Days 7-120 (6 timepoints), enabling CellRank 2 RealTimeKernel + moscot. **Still No for forebrain from Day 21 alone** (need Day 45 data). | CellRank 2 temporal prediction is NOW feasible for posterior brain; forebrain needs 1 more timepoint |
| Only 97 datapoints from Sanchis-Calleja | **186 conditions across 3 datasets** (Sanchis-Calleja 97 + Amin/Kelley 46 + Azbukina 43). CellFlow proved cross-dataset harmonization works (0.91 cosine similarity). | Use ALL datasets with time as GP input dimension |
| Organoid-to-fetal atlas mapping is straightforward | **Valid but requires careful filtering** — stress artifacts, off-target cells, maturation ceiling at ~GW16 | Build Gruffi + BrainSTEM two-tier pipeline |
| 97 datapoints is enough for 8D GP | **186 points in 9-12D is comfortable.** Loeppky 10d rule gives n=90-120; 186 exceeds this. ARD handles high dimensionality. | Multi-dataset approach removes the "barely enough" concern |

**Bottom line: The pipeline is feasible and stronger than initially assessed.** 186 conditions across 3 datasets (not just 97). Multi-timepoint data exists for temporal prediction via CellRank 2 + moscot. CellFlow validates cross-dataset harmonization. The GP-BO loop is well-validated in biology (3-30x fewer experiments than DoE). You can build a compelling hackathon demo.

---

## 1. CellRank 2 & Trajectory Inference

### Verdict: NOW capable of temporal prediction for posterior brain (updated)

> **UPDATE (2026-03-07):** Multi-timepoint data now exists across combined datasets. Azbukina 2025 temporal atlas provides Days 7, 15, 30, 60, 90, 120 for posterior brain — satisfying the 3-timepoint minimum for CellRank 2 RealTimeKernel + moscot.

**What CellRank 2 can do with combined multi-timepoint data (posterior brain):**
- **TRUE temporal trajectory modeling** via RealTimeKernel + moscot optimal transport (Days 7 → 15 → 30 → 60 → 90 → 120)
- Compute fate probabilities across the full developmental arc, not just pseudotime
- Model how morphogen conditions at early timepoints influence later cell fate outcomes
- CellFlow (Klein et al., bioRxiv 2025) already validated this cross-dataset harmonization (0.91 cosine similarity across 176 conditions)

**What CellRank 2 can do with Day 21 data alone (forebrain):**
- Rank the 97 morphogen conditions by developmental potential (CytoTRACEKernel)
- Identify terminal progenitor states within Day 21 (GPCCA estimator)
- Compute fate probabilities — which conditions produce the most committed progenitors
- Prioritize the top 10-15 conditions for follow-up experiments

**What CellRank 2 still CANNOT do:**
- Predict forebrain Day 60-90 from Day 21 alone (need ≥1 more timepoint, e.g., Day 45)
- Generate cell types that don't exist in the observed data manifold

**RNA velocity (scVelo) is unreliable** for organoids — a 2025 benchmark of 14 methods across 17 datasets found no method with consistent performance. Brain organoids with mixed kinetic regimes are a known failure mode. But this no longer matters — RealTimeKernel bypasses RNA velocity entirely by using real timepoints.

### Recommended approach for the hackathon
1. **Posterior brain**: Demo CellRank 2 RealTimeKernel + moscot on Azbukina temporal data — show actual temporal fate predictions
2. **Forebrain**: Use CellRank 2 on Day 21 data to **score and rank conditions**
3. Frame the demo as: "For brain regions with multi-timepoint data, we predict temporal trajectories. For regions with single timepoints, we prioritize conditions and design the next experiment."

### Best tools by use case
| Tool | Works from single snapshot? | Best for |
|---|---|---|
| CellRank 2 (RealTimeKernel + moscot) | **No** — but multi-timepoint data NOW EXISTS | **Temporal fate prediction for posterior brain** |
| CellRank 2 (CytoTRACEKernel) | Yes | Developmental potential ranking (forebrain) |
| CellRank 2 (PseudotimeKernel) | Yes | Pseudotime ordering within Day 21 |
| CellFlow (flow matching) | Needs training data | Virtual protocol screening (23K protocols) |
| Palantir | Yes | Branch probabilities, pseudotime |
| scVelo | Technically yes, but unreliable | Avoid for organoids |

### Key references
- Weiler & Lange et al., Nature Methods 2024 (CellRank 2)
- Klein et al., Nature 2025 (moscot)
- Klein et al., bioRxiv 2025 (CellFlow — [10.1101/2025.04.11.648220](https://doi.org/10.1101/2025.04.11.648220), [GitHub](https://github.com/theislab/CellFlow))
- Zhang et al., 2025 ("requires three time points")
- Bergen et al., 2021 ("RNA velocity: Current challenges")

---

## 2. Organoid-to-Fetal Atlas Mapping: The Conflation Question

### Verdict: Valid and standard practice, but NOT naive extrapolation

**Wayne asked: "Are we conflating stuff?"**

**No — but you need careful methodology.** Cross-referencing organoid scRNA-seq with fetal brain atlases is the standard approach in the field (HNOCA, BrainSTEM, 10+ papers validate this). But fidelity varies from "indistinguishable from primary tissue" to "virtually no signal" (Werner & Gillis 2024, meta-analysis of 173 organoid datasets).

### What's real (the good news)
- **Broad cell classes map well:** Radial glia, intermediate progenitors, excitatory/inhibitory neurons — 70-90%+ of organoid cells map to recognized fetal types
- **Directed protocols produce higher-fidelity cells** than unguided protocols
- **HNOCA provides per-cell-type fidelity scores** — quantitative, not hand-wavy
- Velasco et al. 2019: 95% of directed cortical organoids produce cell type diversity comparable to human cerebral cortex

### What's problematic (the bad news)

| Issue | Severity | Mitigation |
|---|---|---|
| **Glycolytic stress artifact** (Bhaduri 2020) | High | **Gruffi algorithm** removes stressed subpopulation computationally |
| **Off-target cells** (BrainSTEM, Toh 2025) | High | **Two-tier mapping**: whole-brain first, then region-specific |
| **Missing microglia/vasculature** | Medium | Acknowledge; these affect niche signaling |
| **Maturation ceiling at ~GW 6-16** | Medium | Only cross-reference within this developmental window |
| **Spatial disorganization** | Medium | Invisible to scRNA-seq; accept limitation |
| **Co-expression network divergence** | Medium | Use Werner & Gillis co-expression metrics, not just markers |

### Required pipeline for defensible results
```
1. Gruffi stress filtering (remove glycolysis/ER stress cells)
   ↓
2. scArches/scANVI integration (batch-corrected mapping)
   ↓
3. Tier 1: Whole-brain atlas mapping (detect off-target populations)
   ↓
4. Tier 2: Region-specific subatlas mapping (hippocampus/EC)
   ↓
5. Report fidelity scores (not just cell type labels)
```

### What this means for the hackathon
- **You CAN map organoid cells onto fetal atlases** — this is exactly what HNOCA/archmap.bio does
- **You MUST filter stressed cells first** (Gruffi) and do global-before-local mapping (BrainSTEM)
- **Frame it honestly:** "Organoid cells recapitulate fetal-like cell identities with quantified fidelity scores, which we use to benchmark morphogen conditions against in vivo development"

### Key references
- He et al., Nature 2024 (HNOCA — 1.77M cells, 26 protocols, fidelity scoring)
- Bhaduri et al., Nature 2020 (stress artifact)
- Vertesy et al., EMBO Journal 2022 (Gruffi — stress is removable)
- Toh et al., Science Advances 2025 (BrainSTEM two-tier mapping)
- Werner & Gillis, PLOS Biology 2024 (meta-analysis, fidelity spectrum)

---

## 3. Fetal Brain Developmental Atlases

### Verdict: Excellent coverage exists, but NO fetal EC atlas

### Tier 1 — Must-Download for Engram

| Atlas | GW Range | Hippocampus? | EC? | Data Access | Key Use |
|---|---|---|---|---|---|
| **Zhong/Wang 2020** | GW16-27 | **YES (CA1/CA3/DG)** | No | **GEO: GSE119212** | Hippocampal organoid validation |
| **Braun/Linnarsson 2023** | GW5-14 | Early anlage | No | hdca-sweden.scilifelab.se | Early patterning (Day 0-50) |
| **BrainSTEM framework** | Multi-stage | Whole-brain | Whole-brain | Check paper | Two-tier mapping methodology |

### Tier 2 — Strongly Recommended

| Atlas | GW Range | Technology | Key Feature |
|---|---|---|---|
| Qian/Walsh 2025 | GW15-34 | MERFISH | Spatial atlas **including hippocampus** |
| Wang/Kriegstein 2025 | GW7-adolescence | Multiome (RNA+ATAC) | Gene regulatory networks across development |
| Keefe/Nowakowski 2025 | Prenatal | scRNA-seq + lineage | True lineage resolution |

### Organoid Day → Gestational Week Equivalents

| Organoid Day | ~GW Equivalent | Best Reference |
|---|---|---|
| Day 0-14 | GW3-5 | Braun/Linnarsson |
| Day 14-30 | GW5-8 | Braun/Linnarsson |
| Day 30-50 | GW8-12 | Braun/Linnarsson |
| Day 50-70 | GW12-18 | Zhong/Wang (from GW16) |
| Day 70-90 | GW18-25 | Zhong/Wang + Qian/Walsh |

### Critical gap: No fetal EC atlas exists
- Zhong/Wang covers hippocampus but not entorhinal cortex
- Qian/Walsh MERFISH may partially include EC (in medial temporal sections)
- Adult EC atlases exist (Franjic 2022) but are not developmental
- **This is a genuine white space** — Engram's EC organoid work would be novel

### BrainSpan is NOT single-cell
BrainSpan = bulk RNA-seq + microarray + ISH. Useful as a sanity check for bulk morphogen receptor expression trends but cannot do cell-type-level organoid validation.

### Morphogen receptor expression
All scRNA-seq atlases contain genome-wide data, so FGFR, BMPR, FZD, NTRK2 can be queried directly from count matrices. **No atlas has pre-computed morphogen pathway analyses** — you must compute these yourself.

---

## 4. Morphogen Screen Datasets & ML Readiness

### Verdict: 97-186 conditions available; sufficient for GP with additive kernels

### Dataset Comparison

| Dataset | Conditions | Morphogens | Timepoint | Cell Lines | Data Access | ML-Ready? |
|---|---|---|---|---|---|---|
| **Sanchis-Calleja 2025** | **97** | 8 | Day 21 | 5 | Open access (check supplement) | **BEST** |
| **Azbukina 2025** | 43 | 10 | Day 35 | 3 | Preprint (pending) | Good complement |
| **Amin/Kelley 2024** | 46 | 14 | Day 72-74 | 4 | **GEO: GSE233574** | Good, late timepoint |
| Scuderi/Vaccarino 2025 | Continuous gradient | 2 | Variable | Multiple | Mendeley (figures only) | Limited |

**Total directly combinable:** ~140 conditions (Sanchis-Calleja + Azbukina, same lab)
**Total with time modeling:** ~186 conditions

### Morphogens tested across datasets
| Morphogen | Sanchis-Calleja | Amin/Kelley | Azbukina |
|---|---|---|---|
| CHIR99021 (WNT) | ✓ (5 doses) | ✓ (1.5 µM) | ✓ |
| SHH/SAG/Purmorphamine | ✓ | ✓ (4 doses) | ✓ |
| BMP4 | ✓ (5 doses) | ✓ | ✓ |
| BMP7 | ✓ (5 doses) | ✓ | — |
| Retinoic Acid | ✓ (5 doses) | ✓ (100 nM) | ✓ |
| FGF8 | ✓ (5 doses) | ✓ (100 ng/mL) | ✓ |
| XAV939 (WNT inhib) | ✓ | — | — |
| Cyclopamine (SHH inhib) | ✓ | — | — |
| DAPT (Notch inhib) | — | ✓ (1 cond) | — |
| Activin A | — | ✓ | — |

### Key gaps in existing data
- **No hippocampal morphogen screen exists** — this is Engram's highest-value experiment to run
- **Days 21-70 temporal gap** — Azbukina partially fills (Day 35) but only for posterior brain
- **Missing morphogens:** Notch inhibitors, IGF-1, PDGF, LIF/CNTF, non-canonical WNT (Wnt5a)

### CellFlow: Competitor or complement?
**CellFlow (Klein et al., bioRxiv April 2025)** used flow matching on 176 harmonized conditions from the same datasets. Results:
- 0.91 cosine similarity for predicted cell type distributions
- Virtual screen of ~23,000 protocols
- 2.5x improvement over baselines (scGPT, Geneformer, GEARS, CPA)

**CellFlow vs GP-BO:**
| | CellFlow | GP-BO |
|---|---|---|
| Answers | "What will this protocol produce?" | "What should I try next?" |
| Uncertainty | Limited | **Native (posterior variance)** |
| Active learning | Not built-in | **Core strength** |
| Data requirement | 100s of conditions | Works with <100 |
| Interpretability | Black-box | **Kernel lengthscales = morphogen importance** |

**They are complementary.** CellFlow predicts; GP-BO optimizes. CellFlow could serve as a simulator within a BO loop.

---

## 5. GP-BO for Biological Optimization

### Verdict: Well-validated, 97 points is sufficient, this is the right approach

### Published precedent

| Paper | System | Result |
|---|---|---|
| **Narayanan 2025** (Nat Commun) | Cell culture media optimization | **3-30x fewer experiments than DoE** |
| **BATCHIE/Tosh 2025** (Nat Commun) | Drug combinations (1.4M space) | **4% of search space** explored, all 10 hits validated |
| **Claes 2024** | Cell therapy manufacturing | Noisy parallel BO validated |
| **Cosenza 2023** | 14-component serum-free media | Multi-objective BO |
| **SAMPLE/Rapp 2024** (Nat Chem Eng) | Protein engineering | **Fully autonomous GP-BO loop**, 4 agents, all improved |
| **GPerturb/Xing 2025** (Nat Commun) | Single-cell perturbation | GP matches GEARS/CPA at <200 points |

### Is 97 datapoints enough for 8D?

**Yes.**
- Loeppky 10d rule: n = 10 × d = 80. You have 97. ✓
- Narayanan 2025: 3x reduction at 4D, **30x at 8-9D** — higher dimensionality makes BO more valuable
- The GP only needs to be "good enough" to guide Round 1; active learning refines from there
- Xu et al. 2024: Standard GPs work well up to 100D with appropriate kernels

### Recommended GP setup

| Choice | Recommendation | Why |
|---|---|---|
| **Kernel** | Matern 5/2 + ARD | Consensus for biology. ARD auto-identifies important morphogens |
| **NOT** | RBF | Too smooth for biological data |
| **Multi-output** | Independent GPs per cell type | Simpler, works with scikit-learn |
| **Framework** | scikit-learn (hackathon) → BoTorch (production) | sklearn for speed, BoTorch for multi-objective |
| **Acquisition** | Expected Improvement | Standard, well-understood |
| **Production acquisition** | qLogNEHVI (BoTorch) | Multi-objective: max target cells + min cost + max reproducibility |

### GP beats foundation models at this scale

Ahlmann-Eltze et al. (Nature Methods 2025): Deep learning models (scGPT, scFoundation, Geneformer, GEARS) **don't beat linear baselines** at <200 datapoints. The GP's calibrated uncertainty is the decisive advantage for active learning.

### Expected experimental efficiency
```
Round 0: 97 published conditions (free) → GP → high uncertainty in target zones
Round 1: 24 new experiments (~$15K) → 121 total → uncertainty collapses
Round 2: 24 more experiments (~$15K) → 145 total → optimum emerging
Round 3: 24 final experiments (~$15K) → 169 total → protocol found

Total: ~72 new experiments, ~$45K, 4-6 months
vs brute force: 500-6000+ experiments, years, millions
```

### Known failure modes & mitigations

| Failure Mode | Mitigation |
|---|---|
| Batch effects between labs | Transfer learning / multi-task GP |
| Cell line variability | Line-specific GPs or multi-task kernel |
| Non-stationarity | Matern 3/2 kernel (less smoothness assumed) |
| Heteroscedastic noise | Heteroscedastic GP variant |
| Boundary effects | Augment with boundary conditions |
| Local optima | Multi-start optimization, diverse initial conditions |

---

## 6. Synthesis: The Complete Pipeline

### What's scientifically sound

```
LAYER 1: Ontology Resolution (Claude LLM)
  "hippocampus CA1" → UBERON ID, markers, developmental position, morphogen regime
  ✅ Straightforward, validated by domain expert

LAYER 2: Atlas Queries
  CellxGene Census → fetal hippocampal gene expression (Zhong/Wang, GSE119212)
  HNOCA → which existing protocols produce hippocampal-like cells?
  BrainSpan → morphogen receptor expression over development (bulk only)
  ✅ All data publicly available, standard bioinformatics

LAYER 3: GP-BO Protocol Optimization
  Input: morphogen concentrations + time (9-12D) → Output: cell type fractions
  186 conditions from 3 datasets (Sanchis-Calleja + Amin/Kelley + Azbukina)
  GP predicts; BO picks next 24 experiments
  ✅ Well-validated, 186 datapoints exceeds Loeppky rule
  ✅ CellFlow validated cross-dataset harmonization (0.91 cosine sim)

LAYER 4: CellRank 2 Temporal Modeling (UPDATED)
  RealTimeKernel + moscot on Azbukina temporal atlas (Days 7-120)
  ✅ 6 timepoints for posterior brain — exceeds 3-timepoint minimum
  ⚠️ Forebrain still needs 1 more timepoint (Day 45)

LAYER 5: Atlas Benchmarking
  Gruffi stress filtering → scArches mapping → BrainSTEM two-tier validation
  ✅ Standard practice, multiple tools available
```

### What needs reframing

| Original Claim | Updated Framing |
|---|---|
| "Only 97 datapoints from one dataset" | "186 conditions across 3 datasets with time as input dimension" |
| "CellRank 2 can't predict temporal trajectories" | "CellRank 2 CAN predict for posterior brain (6 timepoints exist); forebrain needs Day 45" |
| "Map organoid to fetal = truth" | "Map organoid to fetal = quantified approximation with fidelity scores" |

### What's genuinely novel and defensible

1. **GP-BO for morphogen optimization** — no one has done this for brain organoids specifically
2. **Multi-dataset GP with time as input dimension** — combines 3 independent screens into a unified model
3. **Active learning loop** — the data flywheel (each experiment improves the model) is a real competitive advantage
4. **CellRank 2 + moscot temporal prediction** — leveraging existing multi-timepoint data that no one has applied fate mapping to yet
5. **Integrating CellFlow as simulator within BO loop** — novel combination of the two best approaches
6. **Hippocampal morphogen screen** — no one has done a systematic morphogen screen for hippocampal specification

---

## 7. Risks & Open Questions

### High Risk
| Risk | Why It Matters |
|---|---|
| Day 21 composition may not predict Day 60+ phenotype for forebrain | **Partially mitigated:** Posterior brain has Days 7-120 temporal data (Azbukina atlas). Forebrain still gap between Day 21 and Day 72. Collect Day 45 to close. |
| CHIR 0.5-2.0 µM may not be in Sanchis-Calleja data | EC prediction would be extrapolation, not interpolation. But this IS the demo — honest uncertainty. |
| Cross-dataset batch effects | Three datasets from different labs/timepoints/protocols. CellFlow achieved 0.91 cosine similarity harmonizing 176 conditions — promising but monitor carefully. |
| Cell line variability confounds GP | Scuderi/Vaccarino showed substantial line-to-line variation in morphogen response. |

### Medium Risk
| Risk | Mitigation |
|---|---|
| Stressed cells distort atlas mapping | Gruffi filtering |
| Off-target cells inflate on-target estimates | BrainSTEM two-tier mapping |
| GP overfits with 97 points in 8D | Additive kernel, cross-validation, regularization |
| CellFlow already did the virtual screen | Differentiate: GP gives uncertainty + active learning; CellFlow doesn't |

### Low Risk
| Risk | Why It's Manageable |
|---|---|
| Data not downloadable | GSE233574 confirmed available; Sanchis-Calleja is open access |
| APIs fail at hackathon | Cache all responses offline |
| GP implementation is wrong | scikit-learn GP is 10 lines of code, well-tested |

---

## 8. Actionable Recommendations

### For the hackathon (tomorrow)
1. **Demo GP-BO on 186 conditions** — show the multi-dataset GP with time as input dimension, uncertainty collapsing with active learning
2. **Demo CellRank 2 temporal prediction** — apply RealTimeKernel + moscot to Azbukina temporal atlas, show actual fate trajectories for posterior brain
3. **Demo CellRank 2 as condition ranker** — score forebrain conditions by developmental potential
4. **Show CellFlow integration** — reference CellFlow's 23K virtual screen as the simulator within the GP-BO loop
5. **Show atlas mapping** — map example conditions onto HNOCA, display fidelity scores

### For post-hackathon (first experiment)
1. **24-condition hippocampal morphogen plate:** 6 CHIR doses × 4 BMP4 doses, harvest at Day 21 AND Day 45
2. **Day 45 harvest for top 10 Sanchis-Calleja forebrain conditions** — this closes the forebrain temporal gap (Day 21 + Day 45 + Amin/Kelley Day 72 = 3 timepoints → enables RealTimeKernel for forebrain)
3. **scRNA-seq with Gruffi + BrainSTEM pipeline** for quality benchmarking
4. **Map onto Zhong/Wang hippocampal atlas** (GSE119212) for fidelity scoring
5. **Update GP with results** — demonstrate the active learning loop working

### Datasets to download TODAY
1. `GSE233574` — Amin/Kelley morphogen screen, 46 conditions, Day 72-74 (confirmed available)
2. Sanchis-Calleja data — 97 conditions, Day 21 (check Nature Methods supplement for accession)
3. Azbukina 2025 — 43 conditions + temporal atlas Days 7-120 (preprint; check data availability)
4. `GSE119212` — Zhong/Wang fetal hippocampus (confirmed available)
5. HNOCA — via CellxGene/archmap.bio
6. CellFlow — `pip install cellflow-tools` / [GitHub](https://github.com/theislab/CellFlow)

---

## Full Reference List (by agent)

### Agent 1: CellRank 2 & Trajectory Inference (18 sources)
→ Full report: `04-research/literature/CELLRANK2_TRAJECTORY_INFERENCE_REPORT.md`

### Agent 2: Organoid-to-Fetal Mapping (14 sources)
→ Full report: `04-research/organoid-vs-fetal-atlas-cross-referencing.md`

### Agent 3: Fetal Brain Atlases (13 sources)
→ Full report: `04-research/Fetal-Brain-Developmental-Atlases-Report.md`

### Agent 4: Morphogen Datasets (15 sources)
→ Full report: `04-research/literature/MORPHOGEN_SCREEN_DATASETS_ML_READINESS.md`

### Agent 5: GP-BO Precedent (20+ sources)
→ Full report: `04-research/literature/GP_BAYESOPT_ORGANOID_PROTOCOL_DESIGN.md`
