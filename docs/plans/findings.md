# Findings — Post-Audit Codebase Analysis
> Date: 2026-03-15
> Supersedes: Previous bugfix swarm findings (2026-03-08)
> Purpose: Document current state of the GP-BO pipeline after 15-bug audit fix, informing implementation roadmap

## Current State Summary

### What Works (Verified)
- **194 tests passing** across 4 test files (unit, integration, properties, phase4/5)
- **Core GP-BO loop** (step 04): ILR transform, multi-fidelity merge, SAASBO, multi-objective acquisition all implemented and tested
- **Morphogen parser**: 48 conditions parsed (46 primary + 2 SAG), class hierarchy in place
- **Fidelity scoring** (step 03): Two-tier scoring with RSS, entropy, on/off-target fractions
- **Visualization report**: Self-contained HTML with Plotly (6 figure types)
- **Gruffi QC**: Stress pathway filtering implemented and wired into step 02
- **Config centralization**: All paths, constants, logging in `gopro/config.py`

### What Has Never Been Run
1. **Step 02 for SAG screen** -- code exists (`--input`, `--output-prefix` CLI), but SAG data has never been mapped through scPoli
2. **Step 00c build_temporal_atlas** -- code exists but never executed (blocks all CellRank2 work)
3. **Step 05 CellRank2 virtual data** -- code exists but has no temporal atlas to run against
4. **Step 06 CellFlow** -- runs only the heuristic baseline (no trained model exists)

### Architecture Gaps

#### 1. Hardcoded Region System (03_fidelity_scoring.py)
Three hardcoded dicts lock the pipeline to HNOCA/Braun regions:
- `HNOCA_TO_BRAUN_REGION` (line 63): 9 region mappings
- `OFF_TARGET_LEVEL1` (line 43): 5 off-target cell classes
- `build_hnoca_to_braun_label_map()` (line 645): 13 label mappings

**Impact**: Cannot target arbitrary brain regions. Adding a new reference atlas requires editing source code.

#### 2. Config-Driven Dataset Addition Not Supported
Adding a new dataset requires:
- Writing a new parser class in `morphogen_parser.py`
- Running step 02 with manual CLI args
- Manually specifying CSV paths to step 04
- No schema validation between steps

**Desired state**: A YAML/JSON config per dataset that drives the entire pipeline.

#### 3. Step 05 Spaghetti Code (`project_query_forward`)
The `project_query_forward()` function (lines 243-569 of `05_cellrank2_virtual.py`) is 327 lines with:
- 3 nested fallback paths (push API, transport composition, atlas average)
- PCA projection logic mixed with transport logic
- Label harmonization inline
- No separation of concerns

#### 4. No Input/Output Validation Between Steps
- Step 02 outputs `*_mapped.h5ad` but step 03 doesn't validate expected obs columns exist before loading 11GB of Braun data
- Step 04 loads CSVs with no schema checks (column alignment, NaN handling)
- Step 05 assumes `X_pca` and `day` columns exist without pre-flight checks

#### 5. CellFlow is a Heuristic Stub
`06_cellflow_virtual.py` has full protocol encoding (SMILES, pathways) and CellFlow API integration code, but:
- The `_predict_baseline()` heuristic is biologically crude (fixed +0.2 increments)
- No pre-trained CellFlow model exists
- Training requires GPU (deferred)
- Confidence scoring exists but is not wired into `merge_multi_fidelity_data()`

### Test Coverage Gaps
| Module | Approx Coverage | Critical Gap |
|--------|----------------|--------------|
| `02_map_to_hnoca.py` | ~16% | `map_to_hnoca_scpoli()`, `prepare_query_for_scpoli()` untested |
| `05_cellrank2_virtual.py` | ~18% | `project_query_forward()` untested (the spaghetti function) |
| `03_fidelity_scoring.py` | ~60% | `main()` flow untested, edge cases in RSS |
| `04_gpbo_loop.py` | ~75% | SAASBO path, multi-fidelity merge edge cases |
| `06_cellflow_virtual.py` | ~40% | CellFlow model path untested (no model) |

### Generalizability Ratings (from prior audit)
| Script | Rating | Blockers |
|--------|--------|----------|
| `config.py` | MEDIUM | MW table + HNOCA column names dataset-specific |
| `morphogen_parser.py` | LOW | Entirely hardcoded for Amin/Kelley 48 conditions |
| `01_load_and_convert_data.py` | LOW | All paths hardcoded, no CLI args |
| `02_map_to_hnoca.py` | HIGH | Full CLI parameterization |
| `03_fidelity_scoring.py` | MEDIUM | Scoring general, but label maps + main() hardcoded |
| `04_gpbo_loop.py` | HIGH | Fully CLI-driven |
| `05_cellrank2_virtual.py` | MEDIUM-LOW | Timepoints, harvest day, label harmonization hardcoded |
| `06_cellflow_virtual.py` | MEDIUM | Grid general, heuristic predictor biology-specific |
| `visualize_report.py` | LOW | Hardcodes `amin_kelley` filenames |

### Data on Disk (47GB, being explored by parallel agent)
- `disease_atlas.h5ad` (2.2GB) -- never referenced in any code
- Patterning screen RDS files (22GB+) -- downloaded, not converted
- `HumanFetalBrainPool` expression data (39MB) -- unused
- Redundant files (OSMGT tar.gz, duplicate RDS)

### Notebook Friendliness
Currently all pipeline steps are CLI scripts with `if __name__ == "__main__"` blocks. Core logic is already in functions, but:
- `main()` functions in steps 02, 03, 05 do too much (load + process + save)
- No `__init__.py` exports for common functions
- No example notebooks exist

## Key Risk Areas
1. **scPoli training on CPU** (step 02) takes 30-60 min for 500 epochs -- acceptable on 48GB Mac but slow
2. **Braun reference** (11GB) loaded in backed mode -- memory-safe but slow for repeated access
3. **moscot OT** (step 05) is memory-intensive -- temporal atlas size matters
4. **SAASBO NUTS sampling** is CPU-bound and slow (~5 min for 48 conditions x 17 cell types)

## Previous Findings Still Relevant
- `md5_file` function duplicated 3 times across download scripts (DRY violation)
- `LABEL_HARMONIZATION` dict hardcoded inside loop body in step 05
- `visualize_report.py` return type annotation mismatch (5-tuple annotated as 4-tuple)
- Several unused fixtures in `conftest.py`
