# TODO -- GP-BO Pipeline

## Planned (see docs/plans/task_plan.md)

These items have implementation plans and will be addressed in the production-readiness push:

- [ ] **Phase 1A: Inter-step data validation** -- `gopro/validation.py` with schema checks
- [ ] **Phase 1B: Decompose step 05** -- Break `project_query_forward()` into 6 testable functions
- [ ] **Phase 1C: Importable API surface** -- `from gopro import run_gpbo_loop, score_all_conditions`
- [ ] **Phase 2A: Region targeting system** -- `gopro/region_targets.py`, `--target-region` CLI arg, named profiles
- [ ] **Phase 2B: Dynamic label maps** -- Auto-build label mappings between atlases (fuzzy + synonyms)
- [ ] **Phase 3A: Build temporal atlas** -- Run step 00c to produce `azbukina_temporal_atlas.h5ad`
- [ ] **Phase 3B: CellRank2 virtual data** -- Run step 05 end-to-end, wire transport quality filtering
- [ ] **Phase 3C: CellFlow heuristic improvement** -- Dose-response curves, pathway antagonism, confidence wiring
- [ ] **Phase 4A: Config-driven datasets** -- YAML config per dataset, auto-discovery
- [ ] **Phase 4B: Pipeline orchestrator** -- `python -m gopro.run_pipeline` with dependency tracking
- [ ] **Phase 5A: Test coverage push** -- Target 60%+ on steps 02 and 05, 230+ total tests
- [ ] **Phase 5B: Code quality polish** -- Deduplicate md5_file, fix type annotations, clean dead code

## Deferred (requires GPU or server)

- [ ] **Train CellFlow model** -- Step 06 is currently a heuristic stub. Need to train actual CellFlow model from Sanchis-Calleja/Azbukina patterning screen data (176 conditions). Requires GPU (4090/5080 or cloud).
- [ ] **GPU acceleration for scPoli** -- Step 02 training (500 epochs) is slow on CPU. Add CUDA/MPS device selection.
- [ ] **Heavy RDS conversion** -- Patterning screen RDS files (22GB+) need R/Seurat conversion. Plan code locally, run on server.
- [ ] **Run step 02 for SAG screen** -- SAG data exists on disk but has never been mapped through scPoli. Needs 30-60 min CPU or GPU.

## Deferred (low priority / needs research)

- [ ] **Build literature scraping tool** -- Standalone tool to mine PubMed/bioRxiv for neurodevelopmental atlases and morphogen-to-fate mappings. Output: structured reference profiles and morphogen knowledge base.
- [ ] **Cross-species reference integration** -- Mouse brain atlases (e.g., Allen Brain Atlas scRNA-seq) could provide reference profiles for regions not well-covered in human data.
- [ ] **Validate multi-fidelity GP formulation** -- Fidelity values 1.0/0.5/0.0 are arbitrary; need theoretical justification or empirical calibration.
- [ ] **Integrate disease_atlas.h5ad** -- 2.2GB atlas on disk, never referenced. Pending data exploration results from parallel agent.
- [ ] **Checkpoint/resume mechanism** -- No recovery if a step crashes mid-way. Add intermediate state saving.

## Notes

- All morphogen concentrations are standardized to uM (completed 2026-03-08)
- 194 tests currently passing (`python -m pytest gopro/tests/ -v`)
- 15 critical/major bugs fixed in audit (2026-03-14)
- Implementation plan: `docs/plans/task_plan.md`
- Progress tracking: `docs/plans/progress.md`
