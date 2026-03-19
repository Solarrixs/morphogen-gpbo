# Consolidated Task Plan

All pending and completed work across the morphogen-gpbo project, grouped by initiative. Completed items shown as [x] for context.

**Last updated:** 2026-03-17
**Branch:** ralph/production-readiness-phase2
**Tests:** 689 passing (gopro) + 53 (literature) = 742 total
**§1.1 Status:** COMPLETE (15/15) — all competitive landscape ideas implemented
**§1.2 Status:** COMPLETE (3/3) — all critical MF-GP bugs fixed

---

## 1. Pipeline Fixes & Modeling Improvements

### 1.1 Production Readiness (from ralph-production-readiness-task.md)

- [x] Phase B Idea #2: TVR (Targeted Variance Reduction) — per-fidelity GP ensemble with cost-scaled variance. `--tvr` flag.
- [x] Phase B Idea #4: Target profile refinement (DeMeo 2025) — `refine_target_profile()` with softmax interpolation. `--refine-target` flag.
- [x] Phase B Idea #12: FBaxis_rank regionalization — continuous A-P axis targeting. `--target-fbaxis` flag.
- [x] Phase C Idea #8: Additive + interaction kernel (NAIAD 2025) — `--kernel additive_interaction` flag.
- [x] Phase C Idea #9: Adaptive complexity schedule (NAIAD 2025) — `_select_kernel_complexity()`, `--adaptive-complexity` flag.
- [x] Phase C Idea #10: Morphogen timing window encoding (Sanchis-Calleja 2025) — `MixedSingleTaskGP`, `--timing-windows` flag.
- [x] Phase C Idea #11: Per-cell-type GP models (GPerturb 2025) — `ModelListGP`, `--per-type-gp` flag.
- [x] Phase D Idea #13: Per-round fidelity monitoring — `monitor_fidelity_per_round()`, auto-fallback.
- [x] Phase D Idea #16: Convergence diagnostics (Narayanan 2025) — `compute_convergence_diagnostics()`.
- [x] Phase D Idea #17: Ensemble disagreement (GPerturb 2025) — `compute_ensemble_disagreement()`, `--ensemble-restarts` flag.
- [x] LassoBO — L1-regularized MAP variable selection. `--lassobo` flag. 571 tests.
- [x] Bootstrap uncertainty — `compute_bootstrap_uncertainty()` in step 02, `--bootstrap-noise` flag. 575 tests.
- [x] **Data-driven entropy center** — Replace arbitrary 0.55 entropy weight in composite fidelity with Braun reference mean entropy. File: `gopro/03_fidelity_scoring.py` ~L395. Acceptance: entropy center matches Braun; 2+ tests.
- [x] **/simplify pass** on all Phase B/C/D changes — 3-agent code review. Fix HIGH/MEDIUM issues. Acceptance: all tests pass.
- [x] **/bug-hunter final sweep** — Adversarial QA swarm across `gopro/`. Fixed 1 new critical (KernelSpec) + 6 warnings. 580 tests pass. 5 remaining criticals are test coverage gaps only.

### 1.2 Critical Bug Fixes (from paper deep-reads)

- [x] **TODO-24 (CRITICAL): Remap fidelity encodings for MF-GP kernel.** fidelity=1.0 collapses inter-fidelity kernel. Remap: 0.0→1/3, 0.5→1/2, 1.0→2/3 internally. File: `04_gpbo_loop.py` ~L800.
- [x] **TODO-25 (CRITICAL): Raise fidelity correlation threshold from 0.3 to R²>0.80.** Switch Spearman→R². Thresholds: R²>0.90→skip MF; R²<0.80→single fidelity; 0.80-0.90→MF-BO. Files: `config.py` + `04_gpbo_loop.py`.
- [x] **TODO-26 (CRITICAL): Fix CellFlow dose encoding.** Uses raw dose×onehot, NOT log1p. File: `06_cellflow_virtual.py` L129-187.

### 1.3 CellFlow Integration Fixes — COMPLETE

- [x] TODO-1: Fix CellFlow JAX vs PyTorch mismatch — update `_predict_with_cellflow()` to use JAX API.
- [x] TODO-3: Add Day 72 out-of-distribution warning — `_warn_ood_harvest_days()` implemented.
- [x] TODO-4: Handle CellFlow conservative prediction bias — `confidence_to_noise_variance()` + `allow_fallback` gate. CellFlow fallback now gated behind `--use-cellflow-fallback`.

### 1.4 GP Model Improvements

- [x] TODO-5: Per-fidelity ARD lengthscales — `g(x) + delta(x,m)` GP structure for `SingleTaskMultiFidelityGP`.
- [x] TODO-6: Zero-passing kernel — modified RBF enforcing `k(0,x)=0` for concentration inputs (GPerturb).
- [x] TODO-7: Desirability-based feasibility gate — `D(x) = phi(x) * y_bar(x)` gates infeasible regions (Cosenza 2022).
- [x] ~~TODO-8: Spike-and-slab output sparsity~~ — **DROPPED (literature audit 2026-03-19)**: scCODA solves differential testing, not BO surrogates. ARD in ModelListGP already provides per-output input sparsity. Over-engineered for N=48. (Buttner Nat Comms 2021; GPerturb Nat Comms 2025)
- [x] TODO-9: Verify pseudocount handling before ILR — `--pseudocount` CLI flag, threaded through all ILR call sites.
- [x] ~~TODO-10: Dirichlet-Multinomial alternative~~ — **DROPPED (literature audit 2026-03-19)**: Dirichlet has worse correlation modeling than ILR+GP. No established Dirichlet-output BO method exists. If cross-cell-type correlations matter, the correct approach is ICM/LMC kernel on ILR coordinates. (Egozcue 2003; Candelieri Ann Math AI 2025)
- [x] TODO-11: ILR vs ALR comparison test — `--alr` flag as alternative to ILR.
- [x] TODO-27: Input warping (Kumaraswamy CDF) — `--input-warp` CLI flag. Still BoTorch standard (confirmed 2026-03-19).
- [x] TODO-28: Selective log-scaling for concentration dimensions — `LOG_SCALE_COLUMNS` in config.py.
- [x] TODO-29: MLL optimization restarts (20 restarts) — `--mll-restarts N` flag (Cosenza 2022).
- [x] TODO-30: Explicit priors on GP hyperparameters — Hvarfner DSP (ICML 2024). Now BoTorch default. Confirmed by Xu et al. 2025.
- [x] TODO-31: FixedNoiseGP with per-observation heteroscedastic noise — `train_Yvar.clamp(min=0.02)` (Cosenza 2022).
- [x] TODO-32: Sobol QMC sampler (2048 samples) for acquisition — `--mc-samples N` flag (Cosenza 2022).

### 1.5 Acquisition Function Improvements

- [ ] TODO-12: Contextual parameter support — `--contextual-cols` constrains specified columns within plates. Use BoTorch `fixed_features`. Prerequisite for TODO-41. (BATCHIE Nat Comms 2025)
- [x] ~~TODO-13: Fixed fidelity allocation per batch~~ — **DROPPED (literature audit 2026-03-19)**: Cosenza 2022 does NOT prescribe 30/70 — ratio was fabricated. Pipeline uses passive MF (offline virtual data), not active fidelity querying, making batch allocation inapplicable. (Sabanza-Gil Nat Comp Sci 2025)
- [ ] TODO-14: Noise characterization pre-BO — **MODIFIED**: concept valid but "Kanda 2022" citation is fabricated. Re-source to Bellamy JCIM 2022 or Sabanza-Gil 2025. Existing `confidence_noise` partially addresses this.
- [x] ~~TODO-33: MF-specific acquisition function~~ — **DROPPED (literature audit 2026-03-19)**: Pipeline doesn't do online fidelity selection; qMFKG/cost-EI are for adaptive fidelity querying. Sabanza-Gil 2025 discarded KG "due to computational expense."
- [x] TODO-34: Pilot R² estimation before committing to MF-BO — **ALREADY IMPLEMENTED** as `validate_fidelity_correlation()` with Sabanza-Gil R² thresholds.
- [x] ~~TODO-35: MF-BO vs SF-BO benchmark~~ — **DROPPED (literature audit 2026-03-19)**: Delta/tau protocol is for in-silico benchmarks with 20+ seeds, not single wet-lab campaigns. Existing LOO validation is the correct analog.

### 1.6 Experimental Design

- [ ] TODO-36: Carry-forward top-K controls between rounds — `--n-controls K` flag. K=2 optimal for 24-well plates. (Kanda eLife 2022)
- [ ] TODO-37: Failed/invalid experiment handling — `status` column in training CSVs. Floor-padding strategy for failed experiments. (Morikawa npj Comp Mat 2022)
- [ ] TODO-38: LHD initialization — **MODIFIED**: Round 1 already has expert-designed 48 conditions from Amin/Kelley. Use LHD only for Round 2+ gap-filling, not to replace literature data. (Siska Biotech Bioeng 2025)
- [ ] TODO-39: Confirmation experiment plate map — `--confirmation` flag, n=3 replicates of predicted optimum + 1-2 reference wells. (Deshwal NeurIPS 2023 BTS-RED)
- [ ] TODO-40: Batch candidate filtering/ranking — **MODIFIED**: renamed from "combinatorial IS allocation". All wells get same fidelity in this pipeline. Clarify as post-acquisition filtering/ranking.
- [ ] TODO-41: Contextual parameter adaptive shifting (harvest day) — implement as discrete contextual variable (Day 42/56/70/72/90). BoTorch natively supports this via composite MTBO. (Kanda eLife 2022)
- [ ] TODO-42: Desirability-product objective — **MODIFIED**: use BoTorch `CostAwareUtility` + geometric mean (Derringer-Suich) instead of raw product. Pipeline already has qLogNEHVI for multi-objective. (Cosenza 2022; Kariminejad Sci Rep 2024)

### 1.7 Objective Function Enhancements

> **NOTE**: "DeMeo 2025" is a **fabricated citation** — no such paper exists in any indexed database. TODOs below have been re-sourced to real publications.

- [ ] TODO-15: **MODIFIED** — Replace "v-score" with **NEST-Score** (Naas et al., Cell Reports 2025) or **Spearman transcriptomic fidelity** (He et al., Nature 2024). Both address the proportions-vs-maturity gap from the brain organoid field. Implement in `03_fidelity_scoring.py`.
- [ ] TODO-16: Atlas-derived maturity signatures — **MODIFIED**: frame as "maturity signatures" from HNOCA differential expression (not generic "transition signatures"). Use `scanpy.tl.score_genes` with ANS-style controls. (He Nature 2024; ANS Genome Res 2025)
- [ ] TODO-17: Paired objective refinement between rounds — **MODIFIED**: concept valid but alpha=0.7 is arbitrary. Make alpha a CLI parameter with range 0.5-0.9. **Merge with TODO-50** (duplicate).
- [x] ~~TODO-49: Exact v-score formula~~ — **DROPPED → REPLACED**: "DeMeo 2025" does not exist. Implement **NEST-Score** (Naas Cell Reports 2025) in `gopro/signature_utils.py` instead.
- [x] ~~TODO-50: Signature refinement~~ — **MERGED** into TODO-17. Configurable alpha, validation guard against overfitting with small N.
- [ ] TODO-51: Scrambled-signature negative controls — 1000+ permutations, report p-value. Standard methodology (GSEA standard minimum).
- [ ] TODO-52: Hit threshold via control-referenced SD cutoff — **MODIFIED**: use MAD (median absolute deviation) over SD for robustness; consider 3x threshold over 2x. Standard screening practice (SSMD).

### 1.8 Benchmarking & Validation

- [ ] TODO-53: Domain-informed toy morphogen function — **MODIFIED**: must output compositions (simplex, not scalar) to test ILR pipeline. Add dead zones + toxicity cliffs. Name: `MorphogenBenchmark24D`. (BATCHIE Nat Comms 2024; Anubis RSC 2025)
- [ ] TODO-54: Noise-robustness pre-screening — BO on toy function at noise_std ∈ {0.01, 0.05, 0.1, 0.2} × batch_size ∈ {8, 16, 24}. Test with/without ILR. (Sadybekov JCIM 2022; Kusne arXiv 2025)
- [ ] TODO-55: Lipschitz constant diagnostic — **MODIFIED**: replace gradient-norm sweep with ARD-derived `L_d ~ sigma_f / l_d` (free) + posterior std at recommendations. (Calandra arXiv 2018)
- [ ] TODO-56: Multi-dose validation protocol — **MODIFIED**: use half-log spacing (0.3x/1x/3x) instead of 0.5x/1x/2x. Reduce to 6 cocktails × 3 doses + 6 replicate wells. (Sanchis-Calleja Nat Methods 2025)

### 1.9 Data Integration

- [x] **Ingest 98 Sanchis-Calleja conditions** — `SanchisCallejaParser` class with regex tokenizer for 98 conditions.
- [ ] **Wire Sanchis-Calleja into multi-fidelity merge** — **MODIFIED**: don't hard-code fidelity 0.85. Use empirical R² from `qc_cross_screen.py` as fidelity level. Expect ~0.6-0.75 given Day 36 vs Day 72 temporal gap + different cell lines. (Sabanza-Gil 2025; Sanchis-Calleja Nat Methods 2025)
- [ ] **Wire Sanchis-Calleja into multi-fidelity merge** — assign fidelity 0.85, add to `merge_multi_fidelity_data()` auto-discovery. `--sanchis-fractions`/`--sanchis-morphogens` CLI flags. 3× training data increase.

---

## 2. Agent Infrastructure (ClockBase-Inspired)

Full plan: `docs/specs/clockbase-agent-infrastructure-task.md`
Branch: `ralph/agent-infrastructure` (not yet created)

> **Literature audit (2026-03-19):** Phase 1 is highest priority — implement antagonism as hard BO constraints (PrBO, Souza NeurIPS 2020), not just post-hoc scoring. Phase 3 should import Reactome/KEGG rather than building from scratch. Phase 4 should use KG programmatically for constraints + LLM for explanation only (follow LLAMBO/ChemCrow pattern). Phase 5 should be a simple state machine, not multi-agent LLM. See: LLAMBO (ICLR 2024), Atlas (Digital Discovery 2025), Coscientist (Nature 2023), ChemCrow (Nat MI 2024).

### Phase 1: Recommendation Scoring Framework (no external deps) — **PRIORITY P0**

- [ ] P1-1: `RecommendationScore` dataclass — 4 dimensions (plausibility 0-25, novelty 0-25, feasibility 0-25, predicted_fidelity 0-25) + plausibility penalty + composite. `gopro/agents/scorer.py`. Acceptance: 3+ tests.
- [ ] P1-2: Novelty scoring — min Euclidean distance, mean k=5 neighbor distance, novel non-zero morphogens. Acceptance: 4+ tests.
- [ ] P1-3: Feasibility scoring — morphogen cost lookup, toxicity, cocktail complexity, availability. Acceptance: 3+ tests.
- [ ] P1-4: Predicted fidelity scoring — GP posterior mean/variance extraction. Acceptance: 3+ tests.
- [ ] P1-5: Plausibility filter — YAML-configurable rules for antagonist pairs. Acceptance: 4+ tests.
- [ ] P1-6: Wire scoring into GP-BO loop — `--score-recommendations` flag. Acceptance: score columns in CSV; 3+ tests.
- [ ] P1-7: Scored leaderboard in visualization — Plotly figure ranked by composite. Acceptance: 2+ tests.

### Phase 2: RAG Literature Corpus (requires Anthropic API)

- [ ] P2-1: Corpus builder — Semantic Scholar API → `data/literature/corpus.jsonl`. 500+ papers. Acceptance: 4+ tests (mocked).
- [ ] P2-2: Text embedding pipeline — Claude API or voyage-3, ChromaDB/LanceDB. Acceptance: 3+ tests.
- [ ] P2-3: Hybrid retriever — semantic + BM25, reciprocal rank fusion. Acceptance: 4+ tests.
- [ ] P2-4: RAG prompt templates — Jinja2 parameterized. Acceptance: 2+ tests.
- [ ] P2-5: CLI — `python -m gopro.agents.rag {build,search,stats}`. Acceptance: 2+ tests.

### Phase 3: Neo4j Knowledge Graph (requires Docker)

- [ ] P3-1: Schema design — Morphogen, Pathway, CellType, Paper, Protocol nodes + relationships. Acceptance: 3+ tests.
- [ ] P3-2: Seed KG from pipeline data — 24 Morphogen + 7 Pathway + 48 Protocol + ~30 CellType nodes. Acceptance: 4+ tests.
- [ ] P3-3: Seed KG from literature corpus — 100+ Paper nodes linked to entities. Acceptance: 3+ tests.
- [ ] P3-4: Cypher query library — 6 pre-built queries. Acceptance: 6+ tests.
- [ ] P3-5: Docker Compose for Neo4j — Neo4j 5.x Community, APOC plugin. Acceptance: documented.

### Phase 4: Biological Interpreter Agent (requires Phase 2+3)

- [ ] P4-1: Interpreter core — Claude API agent; KG + RAG queries per recommendation; structured JSON output. Acceptance: 4+ tests.
- [ ] P4-2: Gate mode — `--gate-mode strict`; drop/modify low-plausibility recs. Acceptance: 3+ tests.
- [ ] P4-3: Concentration safety bounds — `data/morphogen_safety_bounds.yaml` from KG+literature. Acceptance: 3+ tests.
- [ ] P4-4: Pathway conflict detection — ANTAGONIZES_WITH relationships; distinguish intentional vs accidental. Acceptance: 4+ tests.
- [ ] P4-5: Integration with scoring — interpreter replaces rule-based plausibility when available. Acceptance: 2+ tests.

### Phase 5: Pipeline Orchestrator Agent (requires Phase 1-4)

- [ ] P5-1: Extend orchestrator — Claude API wrapper for state→action→validate→decide. Acceptance: 3+ tests.
- [ ] P5-2: End-to-end autonomous mode — `--autonomous` flag. Acceptance: 3+ tests.
- [ ] P5-3: Agent action logging — `data/agent_log_round{N}.jsonl`. Acceptance: 2+ tests.
- [ ] P5-4: Agent workflow analysis — post-run stats in visualization report. Acceptance: 2+ tests.

### Phase 6: Polish

- [ ] P6-1: Unified CLI — `python -m gopro.agents {run,score,interpret,rag,kg}`. Acceptance: 2+ tests.
- [ ] P6-2: Configuration file — `gopro/agents/config.yaml`. Acceptance: 2+ tests.
- [ ] P6-3: /simplify pass on all agent code.
- [ ] P6-4: /bug-hunter sweep across `gopro/agents/`.

---

## 3. Literature Intelligence

Full spec: `docs/specs/literature-scrapers-design.md`

### Chunk 1: Foundation

- [ ] Task 1: Scaffold `literature/` directory structure, `__init__.py`, `__main__.py`, `requirements.txt`.
- [ ] Task 2: SQLAlchemy ORM — `literature/models.py` (Paper, Dataset, SearchRun tables). 7 tests.
- [ ] Task 3: Database + config — `literature/db.py` + `literature/config.yaml` with env var resolution.

### Chunk 2: Scrapers

- [ ] Task 4: `BaseScraper` ABC + `PaperResult`/`DatasetResult` dataclasses in `literature/scrapers/base.py`.
- [ ] Task 5: `PubMedScraper` — NCBI Entrez via Biopython. Tests with mocked Entrez.
- [ ] Task 6: `BioRxivScraper` — bioRxiv/medRxiv REST API.
- [ ] Task 7: `GEOScraper`, `ZenodoScraper`, `CellxGeneScraper` in `literature/scrapers/dataset_sources.py`.

### Chunk 3: Scheduler, CLI, Review, Cron

- [ ] Task 8: `literature/scheduler.py` — `run_scrape`, `deduplicate_papers` (Levenshtein ≥ 0.9).
- [ ] Task 9: `literature/review.py` — interactive CLI review loop (approve/reject/skip).
- [ ] Task 10: `literature/cli.py` — argparse subcommands: scrape, review, status, export.
- [ ] Task 11: macOS launchd plist (daily 06:00) + `.gitignore` + README.

### Chunk 4: Integration

- [ ] Task 12: End-to-end smoke test + live PubMed test with NCBI API key.

---

## 4. Paper (bioRxiv Manuscript)

### 4.1 Critical Fact-Check Corrections

- [ ] Fix A-P axis positions — Hypothalamus 0.4→0.2, Ventral midbrain 0.6→0.5, Pons 0.8→0.85. Results ~L176, Methods ~L293.
- [ ] Fix brain region count — "Eight distinct" → "Five distinct plus unspecific/mixed."
- [ ] Fix GP recommendation ranges — CHIR: 0.21-2.49 µM, BMP4: 0.0003-0.0032 µM, SAG: 0.086-0.685 µM.
- [ ] Fix acquisition function description — "1,024 Sobol candidates with 10 L-BFGS restarts" (paper reverses).
- [ ] Fix cross-screen QC — "0.8 cosine" → Aitchison similarity, threshold 0.5.
- [ ] Fix Gruffi — "mean" → "median" composite stress score.
- [ ] Fix thalamus/midbrain counts — Thalamus: 5 (not 6), Dorsal midbrain: 11 (not 14).
- [ ] Fix bounds padding formula — `lb_j = 0` for morphogens, `lb_j = min(x_j)` only for log_harvest_day.

### 4.2 Number Corrections

- [ ] Update fidelity range precision — 0.5429 to 0.9787 or round consistently.
- [ ] Clarify plate map structure — 24-condition default vs actual output.
- [ ] Update test count to current (575+).
- [ ] Reconcile level-2 cell type count — 14 or 17?

### 4.3 Missing Citations

- [ ] Add Quinn et al. 2018 (Bioinformatics) for Aitchison distance.
- [ ] Add Varga et al. 2022 for Gruffi.

### 4.4 Content Expansion

- [ ] Add convergence diagnostics description.
- [ ] Add timing window encoding detail.
- [ ] Add per-type GP interpretability discussion.
- [ ] Add CellFlow heuristic fallback description.
- [ ] Add virtual data confidence scoring description.

### 4.5 Citation Wiring

- [ ] Wire 43 uncited bib entries into tex files.
- [ ] 4 parallel research agents for new domains (VAE/deep generative, scRNA-seq, multi-output GP, 2025-2026 cutting edge) — 80+ new papers.
- [ ] Merge new entries, deduplicate, wire into tex.

### 4.6 Figures (blocking for submission)

- [ ] Figure 1: Pipeline overview.
- [ ] Figure 2: Fidelity scoring.
- [ ] Figure 3: Multi-fidelity integration.
- [ ] Figure 4: Round 1 results.
- [ ] Supplementary Figures S1-S4.

---

## 5. Deferred / GPU-Dependent

Items requiring GPU access, server access, or large data downloads.

- [ ] **Train CellFlow on own data** — `train_cellflow.py`; patterning screen + Amin/Kelley; JAX; leave-one-out validation. Requires GPU.
- [ ] **Run step 02 for SAG screen** — 30-60 min CPU. `python 02_map_to_hnoca.py --input data/amin_kelley_sag_screen.h5ad --output-prefix sag_screen`.
- [ ] **Build temporal atlas** — `python 00c_build_temporal_atlas.py`. Requires 22 GB patterning screen download.
- [ ] **CellRank2 virtual data** — `python 05_cellrank2_virtual.py`. Requires temporal atlas.
- [ ] **Heavy RDS conversion** — 22 GB patterning screen RDS→h5ad. Preferably on server.
- [ ] **Update EC50 values + competence windows** — cross-reference vendor datasheets + Sanchis-Calleja dose-response. File: `06_cellflow_virtual.py`.
- [ ] **Adaptive fidelity calibration** — leave-one-out CV after CellFlow training.
- [ ] **CellFlow-in-the-loop acquisition** — re-rank candidates by CellFlow prediction. Depends on trained model.
- [ ] **Multi-task GP for cell lines** — `MultiTaskGP` when Sanchis-Calleja multi-line data ingested.
- [ ] **Independent IS model / misoKG** — replace `SingleTaskMultiFidelityGP` with 3-IS `ModelListGP`. Large refactor.
- [ ] **Sparse binary inclusion switches** — custom GP with spike-and-slab prior. High effort (GPerturb).
- [ ] **PDBAL exploration phase** — posterior diameter acquisition for early rounds (BATCHIE). Depends on modular acquisition interface.
- [ ] **Plate-aware batch design** — greedy plate selection + shared-media clustering. Needs wet-lab input (BATCHIE).
- [ ] **Literature sub-projects 3-5** — paper OCR, auto-discovery, CellWhisperer/GenePT/scExtract/scGPT integration.
