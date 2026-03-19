# Consolidated Task Plan

All pending and completed work across the morphogen-gpbo project, grouped by initiative. Completed items shown as [x] for context.

**Last updated:** 2026-03-19
**Branch:** ralph/production-readiness-phase2
**Tests:** 866 passing (9 pre-existing failures in gruffi/scgpt/numba)
**§1 Status:** ALL COMPLETE — 21 TODOs implemented, 5 dropped per literature audit, 0 remaining
**Paper location:** `/Users/maxxyung/Projects/engram-projects/latex-papers/morphogen-gbpo/` (see TODO.md there)

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

### 1.4 GP Model Improvements — COMPLETE

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

### 1.5 Acquisition Function Improvements — COMPLETE

- [x] TODO-12: Contextual parameter support — `--contextual-cols` constrains specified columns within plates. Use BoTorch `fixed_features`. Prerequisite for TODO-41. (BATCHIE Nat Comms 2025)
- [x] ~~TODO-13: Fixed fidelity allocation per batch~~ — **DROPPED (literature audit 2026-03-19)**: Cosenza 2022 does NOT prescribe 30/70 — ratio was fabricated. Pipeline uses passive MF (offline virtual data), not active fidelity querying, making batch allocation inapplicable. (Sabanza-Gil Nat Comp Sci 2025)
- [x] TODO-14: Noise characterization pre-BO — noise robustness characterization implemented. Re-sourced from fabricated "Kanda 2022" to Bellamy JCIM 2022 / Sabanza-Gil 2025. Existing `confidence_noise` addresses this.
- [x] ~~TODO-33: MF-specific acquisition function~~ — **DROPPED (literature audit 2026-03-19)**: Pipeline doesn't do online fidelity selection; qMFKG/cost-EI are for adaptive fidelity querying. Sabanza-Gil 2025 discarded KG "due to computational expense."
- [x] TODO-34: Pilot R² estimation before committing to MF-BO — **ALREADY IMPLEMENTED** as `validate_fidelity_correlation()` with Sabanza-Gil R² thresholds.
- [x] ~~TODO-35: MF-BO vs SF-BO benchmark~~ — **DROPPED (literature audit 2026-03-19)**: Delta/tau protocol is for in-silico benchmarks with 20+ seeds, not single wet-lab campaigns. Existing LOO validation is the correct analog.

### 1.6 Experimental Design — COMPLETE

- [x] TODO-36: Carry-forward top-K controls between rounds — `--n-controls K` flag. K=2 optimal for 24-well plates. (Kanda eLife 2022)
- [x] TODO-37: Failed/invalid experiment handling — `status` column in training CSVs. Floor-padding strategy for failed experiments. (Morikawa npj Comp Mat 2022)
- [x] TODO-38: LHD initialization — LHD gap-filling for Round 2+ implemented. Round 1 uses expert-designed 48 conditions from Amin/Kelley. (Siska Biotech Bioeng 2025)
- [x] TODO-39: Confirmation experiment plate map — `--confirmation` flag, n=3 replicates of predicted optimum + 1-2 reference wells. (Deshwal NeurIPS 2023 BTS-RED)
- [x] ~~TODO-40: Batch candidate filtering/ranking~~ — **DROPPED (per audit)**: renamed to batch filtering; not applicable to current pipeline design.
- [x] TODO-41: Contextual parameter adaptive shifting (harvest day) — implemented as discrete contextual variable (Day 42/56/70/72/90). BoTorch natively supports this via composite MTBO. (Kanda eLife 2022)
- [x] TODO-42: Desirability-product objective — cost-aware desirability gate using BoTorch `CostAwareUtility` + geometric mean (Derringer-Suich 1980). (Cosenza 2022; Kariminejad Sci Rep 2024)

### 1.7 Objective Function Enhancements — COMPLETE

> **NOTE**: "DeMeo 2025" is a **fabricated citation** — no such paper exists in any indexed database. TODOs below have been re-sourced to real publications.

- [x] TODO-15: Replaced "v-score" with **NEST-Score** (Naas et al., Cell Reports 2025). Transcriptomic fidelity metric addressing proportions-vs-maturity gap. Implemented in `03_fidelity_scoring.py`.
- [x] TODO-16: Atlas-derived maturity signatures — KNN latent distance maturity proxy from HNOCA. Uses `scanpy.tl.score_genes` with ANS-style controls. (He Nature 2024; ANS Genome Res 2025)
- [x] TODO-17+50: Paired objective / signature refinement between rounds — merged. Configurable alpha CLI parameter (range 0.5-0.9), validation guard against overfitting with small N.
- [x] ~~TODO-49: Exact v-score formula~~ — **DROPPED/REPLACED**: "DeMeo 2025" does not exist. Replaced with **NEST-Score** (Naas Cell Reports 2025) in `gopro/signature_utils.py`.
- [x] TODO-51: Scrambled-signature negative controls — 1000+ permutations, report p-value. Standard methodology (GSEA standard minimum).
- [x] TODO-52: Hit threshold via control-referenced SD cutoff — MAD (median absolute deviation) for robustness; 3x threshold. Standard screening practice (SSMD).

### 1.8 Benchmarking & Validation — COMPLETE

- [x] TODO-53: Domain-informed toy morphogen function — `MorphogenBenchmark24D` outputting compositions (simplex) with dead zones + toxicity cliffs. Tests ILR pipeline end-to-end. (BATCHIE Nat Comms 2024; Anubis RSC 2025)
- [x] TODO-54: Noise-robustness pre-screening — BO on toy function at noise_std in {0.01, 0.05, 0.1, 0.2} x batch_size in {8, 16, 24}. Tested with/without ILR. (Sadybekov JCIM 2022; Kusne arXiv 2025)
- [x] TODO-55: Lipschitz constant diagnostic — ARD-derived `L_d ~ sigma_f / l_d` (free) + posterior std at recommendations. (Calandra arXiv 2018)
- [x] TODO-56: Multi-dose validation protocol — half-log spacing (0.3x/1x/3x). 6 cocktails x 3 doses + 6 replicate wells. (Sanchis-Calleja Nat Methods 2025)

### 1.9 Data Integration — COMPLETE

- [x] **Ingest 98 Sanchis-Calleja conditions** — `SanchisCallejaParser` class with regex tokenizer for 98 conditions.
- [x] **Wire Sanchis-Calleja into multi-fidelity merge** — CLI flags (`--sanchis-fractions`/`--sanchis-morphogens`) + auto-discovery implemented. Fidelity level uses empirical R² from `qc_cross_screen.py` (~0.6-0.75 given Day 36 vs Day 72 temporal gap + different cell lines). (Sabanza-Gil 2025; Sanchis-Calleja Nat Methods 2025)

---

## 2. Agent Infrastructure (ClockBase-Inspired)

Full plan: `docs/specs/clockbase-agent-infrastructure-task.md`
Branch: `ralph/agent-infrastructure` (not yet created)

> **Literature audit (2026-03-19):** Phase 1 is highest priority — implement antagonism as hard BO constraints (PrBO, Souza NeurIPS 2020), not just post-hoc scoring. Phase 3 should import Reactome/KEGG rather than building from scratch. Phase 4 should use KG programmatically for constraints + LLM for explanation only (follow LLAMBO/ChemCrow pattern). Phase 5 should be a simple state machine, not multi-agent LLM. See: LLAMBO (ICLR 2024), Atlas (Digital Discovery 2025), Coscientist (Nature 2023), ChemCrow (Nat MI 2024).

### Phase 1: Recommendation Scoring Framework — SPIKED (26 tests)

- [x] P1-1: `RecommendationScore` dataclass — 4 dimensions (plausibility 0-25, novelty 0-25, feasibility 0-25, predicted_fidelity 0-25) + composite. `gopro/agents/scorer.py`. 26 tests in `test_agents.py`.
- [x] P1-2: Novelty scoring — k-NN distance to training data, normalized by max pairwise distance.
- [x] P1-3: Feasibility scoring — morphogen cost lookup using `MORPHOGEN_COST_PER_UM` from config.
- [x] P1-4: Predicted fidelity scoring — GP posterior mean mapping with low-uncertainty bonus.
- [x] P1-5: Plausibility filter — YAML-configurable antagonist pair rules (`gopro/agents/pathway_rules.yaml`). 7 antagonist pairs (BMP, WNT, SHH pathways).
- [ ] P1-6: Wire scoring into GP-BO loop — `--score-recommendations` flag. **NOT YET WIRED** into `run_gpbo_loop()`.
- [ ] P1-7: Scored leaderboard in visualization — Plotly figure ranked by composite. **NOT YET IMPLEMENTED**.

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

### Chunk 1: Foundation — SPIKED (7 tests)

- [x] Task 1: Scaffold `literature/` directory structure, `__init__.py`, `__main__.py`.
- [x] Task 2: SQLAlchemy ORM — `literature/models.py` (Paper, Dataset, SearchRun tables). 7 tests in `literature/tests/test_models.py`.
- [x] Task 3: Database + config — `literature/db.py` + `literature/config.yaml` with env var resolution.

### Chunk 2: Scrapers

- [x] Task 4: `BaseScraper` ABC + `PaperResult`/`DatasetResult` dataclasses in `literature/scrapers/base.py`. (spiked)
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

> **Paper location:** `/Users/maxxyung/Projects/engram-projects/latex-papers/morphogen-gbpo/`
> **See also:** `TODO.md` in that directory for the full paper-specific checklist.

### 4.1 Critical Fact-Check Corrections — COMPLETE

- [x] Fix A-P axis positions — already correct in current version (verified 2026-03-19)
- [x] Fix brain region count — fixed to "Five distinct plus unspecific/mixed" in figure caption
- [x] Fix GP recommendation ranges — CHIR: 0.21–2.49, BMP4: 0.0003–0.0032, SAG: 0.086–0.685
- [x] Fix acquisition function description — already correct ("1,024 Sobol candidates with 10 L-BFGS restarts")
- [x] Fix cross-screen QC — already correct (Aitchison similarity, threshold 0.5)
- [x] Fix Gruffi — already correct ("median")
- [x] Fix thalamus/midbrain counts — already correct (5 thalamic, 11 dorsal midbrain)
- [x] Fix bounds padding formula — already correct (lb=0 for morphogens)

### 4.2 Number Corrections — COMPLETE

- [x] Update test count to 866 (2 locations updated)
- [x] Fidelity range 0.543 to 0.979 — verified correct
- [x] Level-2 cell type count — internally consistent at 14/17
- [x] Supplementary table values updated to match corrected recommendation ranges

### 4.3 Missing Citations — COMPLETE

- [x] Quinn et al. 2018 — not needed; Aitchison1986 is the canonical reference
- [x] Varga et al. 2022 for Gruffi — already cited as Vertesy2022gruffi
- [x] Fixed wrong Gruffi citation (was He2024hnoca, changed to Vertesy2022gruffi)
- [x] Fixed misattributed Qin2025naiad (removed from N/d schedule, kept for NAIAD kernel)
- [x] Added 9 new bib entries: Jessell2000, Kiecker2005, Lin2023esm2, Landrum2023rdkit, Gardner2018gpytorch, Paszke2019pytorch, Traag2019leiden, RasmussenWilliams2006, Virtanen2020scipy
- [x] Added biological citations for WNT/BMP/SHH patterning mechanisms

### 4.4 Content Expansion — COMPLETE (Phase B)

All 12 new pipeline features described in Methods:
- [x] NEST-Score transcriptomic maturity (new subsection)
- [x] Confidence-based heteroscedastic noise (CellFlow + CellRank2)
- [x] CellRank2 transport quality noise inflation
- [x] CellFlow fallback gating
- [x] Carry-forward controls (Kanda eLife 2022)
- [x] Failed experiment handling (Morikawa 2022)
- [x] LHD gap-filling for Round 2+
- [x] Contextual BO with fixed harvest day
- [x] Cost-aware desirability gate (Derringer-Suich 1980)
- [x] Validation and confirmation plate protocols
- [x] Noise robustness characterization
- [x] Sanchis-Calleja multi-fidelity integration (fidelity 0.7)

### 4.5 Citation Wiring — COMPLETE

- [x] Wired Eze2021 into Introduction
- [x] Wired Lange2022cellrank into multi-fidelity contribution
- [x] Removed unused Quinn2017propr
- [x] All uncited claims now have citations or qualifiers (cost estimates, biological mechanisms, software tools)
- [x] Final state: 45+ citation calls, 45 bib entries, clean correspondence

### 4.6 Figures — NOT STARTED (blocking for submission)

- [ ] Figure 1: Pipeline overview schematic (can make now in BioRender)
- [ ] Figure 2: Fidelity scoring (can generate via `05_visualize.py` on existing data)
- [ ] Figure 4: Round 1 results (can generate via `05_visualize.py` on existing data)
- [ ] Figure 3: Multi-fidelity integration (BLOCKED on pipeline Jobs 3-6)
- [ ] Figure S1: Cell type composition stacked bars (can generate now)
- [ ] Figure S2: GP diagnostics (needs GP model output)
- [ ] Figure S3: Brain region UMAP (can generate from existing mapped h5ad)
- [ ] Figure S4: Algorithm pseudocode (can write now as LaTeX)

### 4.7 Remaining Paper TODOs

- [ ] Update abstract to mention NEST-Score, cost-awareness, recommendation scoring
- [ ] Add Discussion paragraph on production readiness features
- [ ] Add Discussion paragraph on recommendation scoring framework
- [ ] Line ~234: "224 conditions total" — add arithmetic "(46 + 2 + 176)"
- [ ] Line ~336: Clarify MCMC settings as BoTorch SAASBO defaults
- [ ] Verify entropy center "0.55" matches `compute_braun_entropy_center()` output
- [ ] Pre-submission: run `pdflatex` + `bibtex` to verify all citations compile

---

## 5. Pipeline Execution (CPU/GPU Jobs)

> Full spec: `docs/specs/pipeline-execution-spec.md` (7-job plan with dependency graph, validation scripts, troubleshooting)

- [ ] Job 1: Run step 02 for SAG screen (~30-60 min CPU)
- [ ] Job 2: Download patterning screen (22 GB) + RDS→h5ad conversion (requires R 4.2+)
- [ ] Job 3: Build temporal atlas from patterning screen
- [ ] Job 4: Map Sanchis-Calleja to HNOCA
- [ ] Job 5: Generate CellRank2 virtual data (requires Job 3)
- [ ] Job 6: Train CellFlow model (requires GPU + JAX)
- [ ] Job 7: Run full GP-BO Round 1 with all data sources

## 6. Future / Exploratory (low priority)

- [ ] Multi-task GP for cell lines — `MultiTaskGP` when Sanchis-Calleja multi-line data ingested
- [ ] Independent IS model / misoKG — replace `SingleTaskMultiFidelityGP` with 3-IS `ModelListGP`
- [ ] PDBAL exploration phase — posterior diameter acquisition for early rounds (BATCHIE)
- [ ] Plate-aware batch design — greedy plate selection + shared-media clustering
- [ ] CellFlow-in-the-loop acquisition — re-rank candidates by CellFlow prediction
- [ ] Update EC50 values from Sanchis-Calleja dose-response data
