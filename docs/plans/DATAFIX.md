# Data & Code Audit — Fix List

---

## CRITICAL Issues

### 1. `morphogen_matrix_amin_kelley.csv` is never generated

Referenced by **4 steps** (04, 05/CellRank2, 05/Visualize, 06) but no script produces it. `morphogen_parser.py` has `build_morphogen_matrix()` but its `__main__` only prints to stdout — never calls `to_csv()`.

**Consumers that will fail:**

| File | Line | How it loads |
|------|------|-------------|
| `gopro/04_gpbo_loop.py` | 669 | `DATA_DIR / "morphogen_matrix_amin_kelley.csv"` |
| `gopro/05_cellrank2_virtual.py` | 564 | `DATA_DIR / "morphogen_matrix_amin_kelley.csv"` |
| `gopro/visualize_report.py` | 776 | `data_dir / "morphogen_matrix_amin_kelley.csv"` |
| `gopro/06_cellflow_virtual.py` | 558 | `DATA_DIR / "morphogen_matrix_amin_kelley.csv"` |

**Fix:** Add `to_csv()` to `morphogen_parser.py`'s `__main__` block (after line 409):

```python
from gopro.config import DATA_DIR

# ... existing __main__ code ...

# Add at the end of the block:
output_path = DATA_DIR / "morphogen_matrix_amin_kelley.csv"
df.to_csv(str(output_path))
logger.info("Saved morphogen matrix to %s", output_path)
```

**Downstream effects:** Step 04 currently falls back to synthetic demo data when file is missing (line 671). After fix, it uses real morphogen concentrations. Steps 05/CellRank2 and 06/CellFlow currently abort. Update `CLAUDE.md` to note `python gopro/morphogen_parser.py` must run before step 04.

---

### 2. RSS fidelity scores are broken — label alignment never applied

`03_fidelity_scoring.py` defines `build_hnoca_to_braun_label_map()` (line 576) and `align_composition_to_braun()` (line 604) but **never calls them** in the scoring path.

**Bug site:** `gopro/03_fidelity_scoring.py:491-492` in `score_all_conditions()`:

```python
level1_fracs = subset[pred_level1].value_counts(normalize=True)
rss_region, rss_score = compute_rss(level1_fracs, braun_profiles)
```

`level1_fracs` has HNOCA labels as index (`"NPC"`, `"IP"`, `"Neuroepithelium"`). `braun_profiles` columns are Braun CellClass labels (`"Radial glia"`, `"Neuronal IPC"`). `compute_rss()` (line 308) unions the label sets — since most don't match, vectors are near-orthogonal, producing near-zero cosine similarity.

`main()` at line 707 builds the label map and logs it, but never passes it to `score_all_conditions()`. The comment at line 713 even says "This is done inside score_all_conditions via compute_rss" — but it isn't.

**Fix (2 parts):**

Part 1 — Add `label_map` parameter to `score_all_conditions()` (line 428) and apply before RSS:

```python
def score_all_conditions(
    query_adata, braun_profiles, condition_key="condition",
    label_map=None,  # ADD THIS
):
    # ... at line 491-492, replace with:
    level1_fracs = subset[pred_level1].value_counts(normalize=True)
    if label_map is not None:
        level1_fracs = align_composition_to_braun(level1_fracs, label_map)
    rss_region, rss_score = compute_rss(level1_fracs, braun_profiles)
```

Part 2 — Pass `label_map` from `main()` at line 720:

```python
report = score_all_conditions(
    query_adata=query,
    braun_profiles=braun_profiles,
    condition_key=condition_key,
    label_map=label_map,  # ADD THIS
)
```

**Label mapping (from `build_hnoca_to_braun_label_map()`, lines 586-601):**

| HNOCA level-1 | Braun CellClass | Notes |
|---|---|---|
| `Neuron` | `Neuron` | Direct match |
| `NPC` | `Radial glia` | Neural progenitors ~ radial glia |
| `IP` | `Neuronal IPC` | Intermediate progenitors |
| `Neuroepithelium` | `Radial glia` | Summed with NPC |
| `Glioblast` | `Glioblast` | Direct match |
| `Astrocyte` | `Glioblast` | Braun doesn't separate |
| `OPC` | `Oligo` | |
| `CP` | `Neuron` | Choroid plexus |
| `NC Derivatives` | `Neural crest` | |
| `MC` | `Fibroblast` | |
| `EC` | `Vascular` | |
| `Microglia` | `Immune` | |
| `PSC` | `Radial glia` | Flagged off-target |

**Downstream effects:** RSS scores change dramatically (currently near-zero for most conditions). `composite_fidelity` shifts (RSS has 35% weight at line 397). `fidelity_report.csv` values change. GP-BO rankings may shift. Visualization report reflects corrected values.

---

### 3. CellFlow baseline cell types don't match real training labels

**Bug site:** `gopro/06_cellflow_virtual.py:376-379` — `_predict_baseline()` hardcodes HNOCA **level-1** labels:

```python
CELL_TYPES = [
    "Neuron", "NPC", "IP", "Neuroepithelium", "Glioblast",
    "Astrocyte", "OPC", "CP", "PSC", "MC",
]
```

Real training data from step 02 uses **level-2** labels via `predicted_annot_level_2` (`gopro/02_map_to_hnoca.py:396`). When `merge_multi_fidelity_data()` (`gopro/04_gpbo_loop.py:407-418`) combines them, columns are disjoint — zero-filled in both directions.

**Fix:** Replace hardcoded `CELL_TYPES` with vocabulary loaded from real training data:

```python
def _predict_baseline(protocols, real_fractions_csv=None):
    if real_fractions_csv is not None and real_fractions_csv.exists():
        real_Y = pd.read_csv(str(real_fractions_csv), index_col=0)
        CELL_TYPES = list(real_Y.columns)
    else:
        # Fallback: approximate level-2 labels
        CELL_TYPES = [...]  # needs actual HNOCA level-2 vocabulary
```

Also refactor the heuristic logic (lines 387-437) to use dict keyed by cell type name instead of integer indices, so it works regardless of which vocabulary is loaded.

Thread `real_fractions_csv` through `predict_cellflow()` (line 277) and `run_virtual_screen()` (line 531).

**Note:** The `_predict_with_cellflow()` path (line 280-354) has the same issue — CellFlow model outputs its own labels that also won't match level-2. Needs a label mapping step after prediction.

**Downstream effects:** GP training corruption eliminated. Merged Y matrix will have aligned columns. Consider persisting the level-2 vocabulary to `config.py` or a canonical file after step 02 runs.

---

### 4. `azbukina_temporal_atlas.h5ad` is never generated

**Consumer:** `gopro/05_cellrank2_virtual.py:562`:
```python
atlas_path = DATA_DIR / "azbukina_temporal_atlas.h5ad"
```

**What `load_temporal_atlas()` expects (lines 60-92):**
- h5ad file with `"day"` column in `.obs` (numeric timepoints)
- `X_pca` in `.obsm` (optional — will compute if missing)
- Cell type labels in one of: `predicted_annot_level_2`, `annot_level_2`, `cell_type`, `CellType`, `celltype`
- Expected timepoints: `[7, 15, 30, 60, 90, 120]` (line 46)

**What step 00b downloads to `data/patterning_screen/OSMGT_processed_files/`:**
- `OSMGT.rds.gz` — likely the full multi-timepoint dataset (R/Seurat format)
- `exp1_processed_8.h5ad.gz` — possibly a subset already in h5ad
- `4_M_vs_sM_21d_clean.rds.gz`, `d9_mistr_cleaned.rds.gz` — additional subsets

**Fix:** Create `gopro/00c_build_temporal_atlas.py` that:
1. Decompresses `exp1_processed_8.h5ad.gz` (or converts `OSMGT.rds.gz` via `rpy2`/`anndata2ri`)
2. Inspects `.obs` columns for timepoint metadata, maps to numeric `"day"` column
3. Verifies/adjusts the `ATLAS_TIMEPOINTS` constant ([7, 15, 30, 60, 90, 120])
4. Saves as `data/azbukina_temporal_atlas.h5ad`

**Blocker:** Exact implementation depends on inspecting actual file contents. First step: decompress and inspect `exp1_processed_8.h5ad.gz` to see what's inside. If it lacks multi-timepoint data, use `rpy2` for `OSMGT.rds.gz` (requires adding `rpy2`, `anndata2ri` to `requirements.txt`).

**Downstream effects:** Step 05 is completely blocked without this file. Step 04 handles the absence gracefully (skips CellRank2 data) but loses the multi-fidelity 3x data amplification benefit.

---

## HIGH Issues

### 5. Hardcoded absolute paths in 4 files

| File | Lines | Hardcoded value |
|------|-------|----------------|
| `gopro/05_cellrank2_virtual.py` | 42-43 | `PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")` + `DATA_DIR` |
| `gopro/06_cellflow_virtual.py` | 40-41 | Same |
| `gopro/05_visualize.py` | 15-16 | Same |
| `gopro/00b_download_patterning_screen.py` | 33-34 | Same, plus `OUTPUT_DIR` |

**`config.py` already provides (line 15):**
```python
PROJECT_DIR = Path(os.environ.get("GPBO_PROJECT_DIR", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = Path(os.environ.get("GPBO_DATA_DIR", str(PROJECT_DIR / "data")))
```

**Fix for each file:**

`05_cellrank2_virtual.py` — replace lines 42-43 with:
```python
from gopro.config import DATA_DIR, get_logger
logger = get_logger(__name__)
```

`06_cellflow_virtual.py` — replace lines 40-41 with:
```python
from gopro.config import DATA_DIR, get_logger
logger = get_logger(__name__)
```

`05_visualize.py` — replace lines 15-16 with:
```python
from gopro.config import DATA_DIR
```
Also remove `sys.path.insert(0, str(PROJECT_DIR))` on line 23.

`00b_download_patterning_screen.py` — replace lines 33-34 with:
```python
from gopro.config import DATA_DIR
OUTPUT_DIR = DATA_DIR / "patterning_screen"
```

---

### 6. Incompatible embedding spaces in CellRank2 projection

**Bug site:** `gopro/05_cellrank2_virtual.py:286-300` in `project_query_forward()`:

```python
source_pca = atlas_adata[source_mask].obsm["X_pca"]     # scanpy PCA space
# ...
if "X_pca" not in query_adata.obsm and "X_scpoli" in query_adata.obsm:
    query_embedding = query_adata.obsm["X_scpoli"]       # scPoli latent space (!)
# ...
min_dim = min(source_pca.shape[1], query_embedding.shape[1])
source_pca = source_pca[:, :min_dim]                      # naive truncation
query_embedding = query_embedding[:, :min_dim]
```

scPoli latent space and scanpy PCA are fundamentally different representations. Dimension 0 in scPoli encodes something unrelated to PCA component 0. Truncation doesn't make them comparable. Nearest-neighbor search (line 304) returns arbitrary matches.

**Fix (preferred):** Re-embed query cells into the atlas PCA space using the atlas's PCA loadings:

```python
# Get atlas PCA loadings (stored by sc.pp.pca in varm["PCs"])
hvg_genes = atlas_adata.var_names[atlas_adata.var["highly_variable"]]
shared_genes = query_adata.var_names.intersection(hvg_genes)

query_for_pca = query_adata[:, shared_genes].copy()
sc.pp.normalize_total(query_for_pca, target_sum=1e4)
sc.pp.log1p(query_for_pca)

# Project using atlas PCA loadings
pca_loadings = atlas_adata.varm["PCs"]
gene_idx = [list(hvg_genes).index(g) for g in shared_genes]
query_expr = query_for_pca.X.toarray() if hasattr(query_for_pca.X, "toarray") else query_for_pca.X
query_embedding = query_expr @ pca_loadings[gene_idx, :]
```

**Alternative (simpler):** Compute joint PCA on concatenated atlas source + query cells:

```python
combined = ad.concat([atlas_adata[source_mask], query_adata], join="inner")
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)
sc.pp.highly_variable_genes(combined, n_top_genes=2000)
sc.pp.pca(combined, n_comps=30)
n_atlas = source_mask.sum()
source_pca = combined.obsm["X_pca"][:n_atlas]
query_embedding = combined.obsm["X_pca"][n_atlas:]
```

Preferred approach (atlas PCA projection) is faster and consistent with moscot transport maps.

**Downstream effects:** Nearest-neighbor mapping becomes meaningful; transport-based fate predictions improve. Log shared gene count and warn if coverage is low.

---

### 7. CellRank2 virtual labels may not align with real labels

**Bug site:** `gopro/05_cellrank2_virtual.py:335-346` in `project_query_forward()`:

```python
target_label_col = None
for candidate in [label_key, "annot_level_2", "cell_type", "CellType", "celltype"]:
    if candidate in target_obs.columns:
        target_label_col = candidate
        break
target_labels = target_obs[target_label_col]
```

**Problem 1:** The Azbukina atlas won't have `predicted_annot_level_2` (that's from step 02's KNN transfer). It may have its own column name — the fallback finds *a* column but doesn't validate *values*.

**Problem 2:** Even if found, Azbukina labels may use different vocabulary (e.g., `"Excitatory neuron"` vs HNOCA's `"Cortical EN"`). When merged in `merge_multi_fidelity_data()`, mismatched names become disjoint columns.

**Fix:** Add label harmonization after column discovery (replace lines 335-346):

```python
target_labels = target_obs[target_label_col].copy()

# Harmonize if using a non-HNOCA column
if target_label_col != label_key:
    logger.info("Using atlas column '%s' (not '%s')", target_label_col, label_key)
    mapped = target_labels.map(LABEL_HARMONIZATION)
    n_unmapped = mapped.isna().sum()
    if n_unmapped > 0:
        unmapped = target_labels[mapped.isna()].unique()
        logger.warning("%d cells (%d labels) unmapped: %s", n_unmapped, len(unmapped), list(unmapped)[:10])
        mapped = mapped.fillna(target_labels)  # keep unmapped as-is
    target_labels = mapped
```

Define `LABEL_HARMONIZATION` dict mapping common alternative labels to HNOCA level-2 names. Consider centralizing this in `config.py` or a shared `label_utils.py` since bugs #2, #3, and #7 all need label mapping.

---

## MEDIUM Issues

### 8. Dead data (generated but never consumed downstream)

| File | Generated at | Recommendation |
|------|-------------|----------------|
| `amin_kelley_sag_screen.h5ad` | `01_load_and_convert_data.py:115-121` | **Keep.** Wire into step 02 for additional training data. Short-term: add comment noting future use. |
| `amin_kelley_fidelity.h5ad` | `03_fidelity_scoring.py:745-746` | **Delete generation code.** `fidelity_report.csv` captures all scores. Cell-level fidelity can be recomputed. |
| `braun_reference_celltype_profiles.csv` | `03_fidelity_scoring.py:661-664` | **Wire in.** Add as Tier-2 sub-score in `score_all_conditions()` for finer-grained fidelity, or wire into visualization. Variable `braun_celltype_profiles` is assigned at line 661 but never referenced. |
| `cellrank2_transport_quality.csv` | `05_cellrank2_virtual.py:634-635` | **Wire in.** Use quality scores to filter/weight virtual data before passing to step 04. |
| `cellflow_screening_report.csv` | `06_cellflow_virtual.py:597` | **Wire in.** Use confidence scores to filter low-confidence predictions in `merge_multi_fidelity_data()`. |
| `patterning_screen/*` (extracted) | `00b_download_patterning_screen.py:212-229` | **Keep.** Blocked by Critical Bug #4 — needs conversion script. |

---

### 9. `neural_organoid_atlas/` location mismatch

**`config.py` default (line 23-26):**
```python
MODEL_DIR = Path(os.environ.get(
    "GPBO_MODEL_DIR",
    str(PROJECT_DIR / "neural_organoid_atlas" / "supplemental_files" / "scpoli_model_params"),
))
```

This resolves to `<project_root>/neural_organoid_atlas/...` but the actual model files live at `<project_root>/data/neural_organoid_atlas/supplemental_files/scpoli_model_params/` (confirmed: `attr.pkl`, `model_params.pt`, `var_names.csv` present there).

Step 01 (`01_load_and_convert_data.py:79`) correctly references `DATA_DIR / "neural_organoid_atlas/..."` — inconsistent with `config.py`.

**Fix:** Change `config.py` line 25 default to:
```python
str(PROJECT_DIR / "data" / "neural_organoid_atlas" / "supplemental_files" / "scpoli_model_params"),
```

Also update `data/README.md` Section 4 and `CLAUDE.md` to say the repo lives under `data/`.

**Priority:** Medium (step 02 fails without `GPBO_MODEL_DIR` env var workaround). **Effort:** Very low (one-line change + docs).

---

### 10. `data/README.md` is incomplete

Missing from the "Pipeline-Generated Files" table (lines 76-86):

| Missing file | Generated by |
|---|---|
| `morphogen_matrix_amin_kelley.csv` | `morphogen_parser.py` |
| `gp_training_regions_amin_kelley.csv` | Step 02 |
| `gp_recommendations_round{N}.csv` | Step 04 |
| `gp_diagnostics_round{N}.csv` | Step 04 |
| `braun_reference_profiles.csv` | Step 03 (cache) |
| `braun_reference_celltype_profiles.csv` | Step 03 (cache) |
| `report_round{N}.html` | Step 05/visualize |
| `cellrank2_virtual_*.csv` | Step 05/CellRank2 |
| `cellflow_virtual_*.csv` | Step 06 |

**Fix:** Add these to the table, grouped by step. **Effort:** Very low (docs only).

---

### 11. GEO raw data has no automated download

Step 01 (`01_load_and_convert_data.py:107-118`) reads 6 uncompressed files:
- `GSE233574_OrganoidScreen_counts.mtx`
- `GSE233574_OrganoidScreen_cellMetaData.csv`
- `GSE233574_OrganoidScreen_geneInfo.csv`
- `GSE233574_Organoid.SAG.secondaryScreen_counts.mtx`
- `GSE233574_Organoid.SAG.secondaryScreen_cellMetaData.csv`
- `GSE233574_Organoid.SAG.secondaryScreen_geneInfo.csv`

No script downloads these from GEO. `data/README.md` says "Download these files from GEO and place them in this data/ directory" — manual only.

**Fix:** Create `gopro/00a_download_geo.py` that downloads from GEO FTP (`https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233574/suppl/`), verifies checksums, and gunzips. Follow the same pattern as `00b_download_patterning_screen.py`.

**Priority:** Medium (blocks new users). **Effort:** Medium (pattern exists in 00b).

---

### 12. Logging inconsistency

| File | `print()` calls to convert | Lines to change |
|------|---------------------------|-----------------|
| `gopro/05_cellrank2_virtual.py` | ~58 | Throughout all functions |
| `gopro/06_cellflow_virtual.py` | ~26 | Throughout all functions |
| **Total** | **~84** | |

**Fix for both files:**
1. Add `from gopro.config import get_logger` and `logger = get_logger(__name__)` (combines with Bug #5 fix)
2. Replace `print(f"ERROR: ...")` → `logger.error(...)`
3. Replace `print(f"  WARNING: ...")` → `logger.warning(...)`
4. Replace all other `print(...)` → `logger.info(...)`
5. Convert f-string formatting to `%s`/`%d` logger-style formatting

**Priority:** Low. **Effort:** Low (mechanical find-and-replace, no logic changes).

---

## What's Actually Fine

- **Fidelity column handling**: Fidelity is correctly added at merge time by `build_training_set()`, not baked into virtual CSVs. `recommend_next_experiments()` correctly fixes fidelity=1.0 for recommendations.
- **Single-fidelity fallback**: When no virtual sources exist, the GP correctly uses `SingleTaskGP` instead of multi-fidelity.
- **`morphogen_parser.py`**: Internally consistent — correctly imports and uses `MORPHOGEN_COLUMNS` from `config.py`.
- **`MORPHOGEN_BOUNDS` keys**: Match `MORPHOGEN_COLUMNS` — both use mixed units reflecting real experimental units (`_ng_mL`, `_nM`, `_uM`).

---

## Suggested Fix Order

1. **#1** — Generate the missing `morphogen_matrix_amin_kelley.csv` (add `to_csv()` to `morphogen_parser.py`)
2. **#2** — Wire up `align_composition_to_braun()` in `score_all_conditions()` before `compute_rss()`
3. **#5** — Replace hardcoded paths with `config.py` imports (4 files)
4. **#9** — Fix `MODEL_DIR` default in `config.py` (one-line)
5. **#3** — Fix CellFlow baseline to use level-2 label vocabulary
6. **#12** — Convert 84 `print()` calls to `logger` (mechanical, do with #5)
7. **#4** — Add conversion script for Azbukina patterning screen → `azbukina_temporal_atlas.h5ad`
8. **#6** — Fix embedding space alignment in CellRank2 projection
9. **#7** — Add label vocabulary validation/harmonization in CellRank2
10. **#8** — Wire in or delete dead data outputs
11. **#10** — Update `data/README.md`
12. **#11** — Add GEO download script
