# Codebase Audit Findings

## CLAUDE.md Issues

1. **Known Issue #1 is FIXED** — `project_query_forward` transport maps bug (line 418). Code now properly uses per-condition transport with atlas-average fallback.
2. **Known Issue #2 is FIXED** — GP bounds zero-width edge case. `_compute_active_bounds` now has `MIN_BOUND_WIDTH = 1e-6` guard at line 125.
3. **Missing from Critical Files Reference** — `build_virtual_morphogen_matrix` in `05_cellrank2_virtual.py`, several `visualize_report.py` functions.

## Type Annotation / Minor Error Issues

| File | Issue | Severity |
|------|-------|----------|
| `visualize_report.py` | `compute_morphogen_pca_with_recommendations` returns 5-tuple but annotated as 4-tuple | Bug |
| `06_cellflow_virtual.py:201-206` | Uses `importlib` to load step 04 for `MORPHOGEN_COLUMNS` — violates convention | Convention violation |
| `04_gpbo_loop.py` | `merge_multi_fidelity_data` mutates caller's DataFrames without `.copy()` | Convention violation |
| `03_fidelity_scoring.py` | Unused import: `cosine_distance` from scipy | Dead code |
| `morphogen_parser.py` | Handler functions lack `-> None` return annotations | Missing types |
| `morphogen_parser.py` | `CombinedParser.parse()` accesses private `p._parsers` | Fragile coupling |
| `00_zenodo_download.py` | No type annotations on any function | Missing types |
| `00a_download_geo.py` | Partial type annotations, missing return types | Missing types |
| `00b_download_patterning_screen.py` | Partial type annotations, missing return types | Missing types |
| `01_load_and_convert_data.py` | No type annotations, uses `assert` for validation, side effect on import | Multiple |
| `config.py:16-28` | Unreachable `isinstance(X, str)` guards after `Path()` | Dead code |
| `05_cellrank2_virtual.py` | `LABEL_HARMONIZATION` dict hardcoded inside loop body | Should be module-level |
| `00_/00a_/00b_` | `md5_file` function duplicated 3 times | DRY violation |
| `conftest.py` | Fixtures defined but never used by any test | Dead code |

## Generalizability Audit

| Script | Rating | Blockers |
|--------|--------|----------|
| `config.py` | MEDIUM | MW table + HNOCA column names dataset-specific |
| `morphogen_parser.py` | LOW | Entirely hardcoded for Amin/Kelley 48 conditions |
| `00_zenodo_download.py` | MEDIUM | Parameterized by `KNOWN_RECORDS` dict |
| `00a_download_geo.py` | LOW | Hardcoded for GSE233574 |
| `00b_download_patterning_screen.py` | MEDIUM | Parameterizable via `RECORD_ID` |
| `00c_build_temporal_atlas.py` | HIGH | CLI flags + auto-detection |
| `01_load_and_convert_data.py` | LOW | All paths hardcoded, no CLI args |
| `02_map_to_hnoca.py` | HIGH | Full CLI parameterization |
| `03_fidelity_scoring.py` | MEDIUM | Scoring general, but label maps + main() hardcoded |
| `04_gpbo_loop.py` | HIGH | Fully CLI-driven |
| `05_cellrank2_virtual.py` | MEDIUM-LOW | Timepoints, harvest day, label harmonization hardcoded |
| `05_visualize.py` | HIGH | Thin CLI wrapper |
| `06_cellflow_virtual.py` | MEDIUM | Grid general, heuristic predictor biology-specific |
| `qc_cross_screen.py` | HIGH | Fully generic |
| `convert_rds_to_h5ad.py` | MEDIUM-LOW | macOS-only R path |
| `visualize_report.py` | LOW | Hardcodes `amin_kelley` filenames |

## Unnecessary Files

### Definitely Delete
- `/.coverage` — pytest coverage DB (add to `.gitignore`)
- `/handoff.md`, `/questions.md`, `/version-history.md` — empty templates

### Probably Delete (agent artifacts)
- `/ralph-pipeline.sh` (51k), `/ralph-task.md` (24k), `/task_plan.md` (2.2k)
- `/.bug-hunter/` directory

### Should Commit (untracked)
- `gopro/convert_rds_to_h5ad.py`, `gopro/qc_cross_screen.py`
- `docs/plans/2026-03-08-multi-dataset-integration.md`
