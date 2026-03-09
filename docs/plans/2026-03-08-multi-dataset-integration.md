# Multi-Dataset Integration: SAG Screen + RDS→h5ad Converter

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate the SAG secondary screen (2 new conditions) into the GP-BO training pipeline, refactor `morphogen_parser.py` into a generic parser class, and write an RDS→h5ad conversion script for future Azbukina data.

**Architecture:** Three independent workstreams: (1) SAG screen integration via modified step 02 + new morphogen parser entries, (2) morphogen parser refactor into a class hierarchy, (3) standalone RDS→h5ad conversion script. The GP loop (step 04) already supports multi-source data via `merge_multi_fidelity_data()` — we just need to produce the right CSVs.

**Tech Stack:** Python 3.14, scanpy, anndata, scArches/scPoli, BoTorch, R 4.2 (via subprocess), Seurat/SeuratDisk.

---

## Key Decisions (from user interview)

- **SAG conditions to add:** Only SAG_50nM and SAG_2uM (2 genuinely new points). SAG_250nM and SAG_1uM are near-duplicates of primary screen CHIR1.5-SAG250 and CHIR1.5-SAG1000 → dropped.
- **SAG morphogen vectors:** CHIR=1.5µM + SAG at dose, all else=0, log_harvest_day=ln(70). Paper confirms CHIR 1.5µM base + days 6-21 window + Day 70 harvest.
- **Fidelity:** All real data = 1.0 (binary: real vs virtual).
- **scPoli mapping:** Sequential fine-tuning (reuse existing primary screen model, don't retrain).
- **QC:** Validate overlapping conditions between screens (cosine similarity check).
- **Data files:** Separate CSVs per dataset, merged at GP time in step 04.
- **Morphogen parser:** Generic parser class (simplest approach).
- **RDS converter:** Uses subprocess to call R at `/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/R`. Auto-installs Seurat/SeuratDisk. Targets `4_M_vs_sM_21d_clean.rds.gz`.
- **Cell filtering:** SAG screen uses `ClusterLabel != 'filtered'` (no `quality` column).
- **Deferred:** Age-adjusted fidelity scoring, Azbukina condition parsing, base protocol encoding.

---

## Task 1: Add SAG Secondary Screen Morphogen Parsing

**Files:**
- Modify: `gopro/morphogen_parser.py`
- Test: `gopro/tests/test_unit.py`

### Step 1: Write failing tests for SAG secondary screen parsing

Add to `gopro/tests/test_unit.py`:

```python
class TestSAGSecondaryScreen:
    """Tests for SAG secondary screen condition parsing."""

    def test_sag_50nm_vector(self):
        """SAG_50nM: CHIR=1.5, SAG=0.05, harvest=Day70."""
        from gopro.morphogen_parser import parse_condition_name
        import math
        vec = parse_condition_name("SAG_50nM")
        assert vec["CHIR99021_uM"] == 1.5
        assert vec["SAG_uM"] == pytest.approx(0.05)
        assert vec["log_harvest_day"] == pytest.approx(math.log(70))
        # All other morphogens should be 0
        for col in MORPHOGEN_COLUMNS:
            if col not in ("CHIR99021_uM", "SAG_uM", "log_harvest_day"):
                assert vec[col] == 0.0, f"{col} should be 0"

    def test_sag_2um_vector(self):
        """SAG_2uM: CHIR=1.5, SAG=2.0, harvest=Day70."""
        from gopro.morphogen_parser import parse_condition_name
        import math
        vec = parse_condition_name("SAG_2uM")
        assert vec["CHIR99021_uM"] == 1.5
        assert vec["SAG_uM"] == pytest.approx(2.0)
        assert vec["log_harvest_day"] == pytest.approx(math.log(70))

    def test_sag_secondary_conditions_list(self):
        """SAG_SECONDARY_CONDITIONS should have exactly 2 entries."""
        from gopro.morphogen_parser import SAG_SECONDARY_CONDITIONS
        assert len(SAG_SECONDARY_CONDITIONS) == 2
        assert "SAG_50nM" in SAG_SECONDARY_CONDITIONS
        assert "SAG_2uM" in SAG_SECONDARY_CONDITIONS

    def test_sag_secondary_build_matrix(self):
        """build_morphogen_matrix works for SAG secondary conditions."""
        from gopro.morphogen_parser import build_morphogen_matrix, SAG_SECONDARY_CONDITIONS
        df = build_morphogen_matrix(SAG_SECONDARY_CONDITIONS)
        assert df.shape == (2, 20)
        assert df.loc["SAG_50nM", "CHIR99021_uM"] == 1.5
        assert df.loc["SAG_2uM", "SAG_uM"] == pytest.approx(2.0)
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest gopro/tests/test_unit.py::TestSAGSecondaryScreen -v`
Expected: FAIL — `ValueError: Unrecognized condition name: 'SAG_50nM'`

### Step 3: Implement SAG secondary screen handlers in morphogen_parser.py

Add after the existing `_sag250` handler (around line 311):

```python
# SAG Secondary Screen conditions (Amin/Kelley 2024, Day 70, CHIR 1.5µM base)
_SAG_SECONDARY_HARVEST_DAY: int = 70
_LOG_SAG_SECONDARY_HARVEST_DAY: float = math.log(_SAG_SECONDARY_HARVEST_DAY)

def _sag_50nm(v: dict[str, float]) -> None:
    """SAG secondary: 50nM SAG + 1.5µM CHIR, Day 70."""
    v["CHIR99021_uM"] = 1.5
    v["SAG_uM"] = nM_to_uM(50.0)  # 50 nM = 0.05 µM
    v["log_harvest_day"] = _LOG_SAG_SECONDARY_HARVEST_DAY

def _sag_2um(v: dict[str, float]) -> None:
    """SAG secondary: 2µM SAG + 1.5µM CHIR, Day 70."""
    v["CHIR99021_uM"] = 1.5
    v["SAG_uM"] = 2.0  # 2000 nM = 2.0 µM
    v["log_harvest_day"] = _LOG_SAG_SECONDARY_HARVEST_DAY
```

Add to `_CONDITION_PARSERS` dict:

```python
    "SAG_50nM":             _sag_50nm,
    "SAG_2uM":              _sag_2um,
```

Update the assert from `== 46` to `== 48`:

```python
assert len(_CONDITION_PARSERS) == 48, (
    f"Expected 48 conditions, got {len(_CONDITION_PARSERS)}"
)
```

Add below `ALL_CONDITIONS`:

```python
SAG_SECONDARY_CONDITIONS: list[str] = ["SAG_50nM", "SAG_2uM"]
"""SAG secondary screen conditions (non-duplicate only)."""
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest gopro/tests/test_unit.py::TestSAGSecondaryScreen -v`
Expected: 4 PASS

### Step 5: Run full test suite to check nothing broke

Run: `python -m pytest gopro/tests/ -v`
Expected: All existing tests pass. Some may need the assert count updated.

### Step 6: Commit

```bash
git add gopro/morphogen_parser.py gopro/tests/test_unit.py
git commit -m "feat: add SAG secondary screen conditions (SAG_50nM, SAG_2uM) to morphogen parser"
```

---

## Task 2: Modify Step 02 for Multi-Dataset scPoli Mapping

**Files:**
- Modify: `gopro/02_map_to_hnoca.py`
- Test: `gopro/tests/test_unit.py`

### Step 1: Write failing test for SAG screen quality filter

The SAG screen has no `quality` column — it uses `ClusterLabel != 'filtered'` instead. The existing `filter_quality_cells()` only handles the `quality` column.

Add to `gopro/tests/test_unit.py`:

```python
class TestSAGScreenFiltering:
    """Tests for SAG screen cell filtering (ClusterLabel-based)."""

    def test_filter_sag_screen_cells(self):
        """filter_quality_cells handles SAG screen ClusterLabel column."""
        import anndata
        obs = pd.DataFrame({
            "ClusterLabel": ["MAF_NKX2-1", "filtered", "LHX8_NKX2-1", "filtered", "c0"],
            "condition": ["SAG_1uM"] * 5,
        })
        adata = anndata.AnnData(
            X=np.zeros((5, 10)),
            obs=obs,
        )
        from gopro.tests.conftest import _step02
        filtered = _step02.filter_quality_cells(adata)
        assert filtered.n_obs == 3  # 2 'filtered' cells removed
```

### Step 2: Run test to verify it fails

Run: `python -m pytest gopro/tests/test_unit.py::TestSAGScreenFiltering -v`
Expected: FAIL — returns 5 cells (no filtering applied because no `quality` column)

### Step 3: Update filter_quality_cells to handle SAG screen

In `gopro/02_map_to_hnoca.py`, modify `filter_quality_cells`:

```python
def filter_quality_cells(adata: sc.AnnData) -> sc.AnnData:
    """Filter to quality cells based on dataset-specific QC annotations.

    Handles two formats:
    - Primary screen: 'quality' column, keep rows where quality == 'keep'
    - SAG screen: 'ClusterLabel' column, drop rows where ClusterLabel == 'filtered'

    Args:
        adata: AnnData with QC annotation columns in obs.

    Returns:
        Filtered AnnData.
    """
    n_before = adata.n_obs
    if "quality" in adata.obs.columns:
        adata = adata[adata.obs["quality"] == "keep"].copy()
        logger.info("Quality filter: %d -> %d cells (%d removed)",
                    n_before, adata.n_obs, n_before - adata.n_obs)
    elif "ClusterLabel" in adata.obs.columns:
        adata = adata[adata.obs["ClusterLabel"] != "filtered"].copy()
        logger.info("ClusterLabel filter: %d -> %d cells (%d 'filtered' removed)",
                    n_before, adata.n_obs, n_before - adata.n_obs)
    return adata
```

### Step 4: Run test to verify it passes

Run: `python -m pytest gopro/tests/test_unit.py::TestSAGScreenFiltering -v`
Expected: PASS

### Step 5: Add --input CLI flag to step 02

Modify the `if __name__ == "__main__"` block in `gopro/02_map_to_hnoca.py` to accept `--input`, `--output-prefix`, and `--sequential` flags. The sequential mode loads the existing scPoli model (already trained on primary screen) and only trains on the new query data.

```python
if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Map query data to HNOCA via scPoli")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input h5ad (default: amin_kelley_2024.h5ad)")
    parser.add_argument("--output-prefix", type=str, default=None,
                        help="Output file prefix (default: amin_kelley)")
    parser.add_argument("--condition-key", type=str, default="condition",
                        help="obs column identifying experimental conditions")
    parser.add_argument("--batch-key", type=str, default="sample",
                        help="obs column identifying batch/sample")
    args = parser.parse_args()

    start = time.time()

    # Resolve paths
    query_path = Path(args.input) if args.input else DATA_DIR / "amin_kelley_2024.h5ad"
    output_prefix = args.output_prefix or "amin_kelley"

    ref_path = DATA_DIR / "hnoca_minimal_for_mapping.h5ad"

    for path, name in [(ref_path, "HNOCA reference"), (query_path, "Query data")]:
        if not path.exists():
            logger.error("%s not found at %s", name, path)
            raise SystemExit(1)

    if not MODEL_DIR.exists():
        logger.error("scPoli model not found at %s", MODEL_DIR)
        raise SystemExit(1)

    # Load data
    logger.info("Loading HNOCA minimal reference...")
    ref = sc.read_h5ad(str(ref_path))
    logger.info("Reference: %s", ref.shape)

    logger.info("Loading query data from %s...", query_path.name)
    query = sc.read_h5ad(str(query_path))
    logger.info("Query: %s", query.shape)

    # Filter to quality cells
    query = filter_quality_cells(query)

    # Prepare query
    query = prepare_query_for_scpoli(query, ref, batch_column=args.batch_key)

    # Map to HNOCA via scPoli
    query_latent, ref_latent = map_to_hnoca_scpoli(
        query, ref, MODEL_DIR,
        n_epochs=500,
        batch_size=1024,
    )

    # Store latent in query
    query.obsm["X_scpoli"] = query_latent

    # Transfer labels
    logger.info("Transferring cell type labels...")
    label_cols = [ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3]
    transferred = transfer_labels_knn(
        ref_latent, query_latent,
        ref.obs, query.obs,
        label_columns=label_cols,
        k=50,
    )

    # Add transferred labels to query
    for col in transferred.columns:
        query.obs[col] = transferred[col].values

    # Compute cell type fractions for GP training
    fractions = compute_cell_type_fractions(
        query.obs,
        condition_key=args.condition_key,
        label_key=f"predicted_{ANNOT_LEVEL_2}",
    )

    # Also compute region fractions
    region_fractions = compute_cell_type_fractions(
        query.obs,
        condition_key=args.condition_key,
        label_key=f"predicted_{ANNOT_REGION}",
    )

    # Save outputs
    logger.info("Saving outputs...")

    fractions.to_csv(str(DATA_DIR / f"gp_training_labels_{output_prefix}.csv"))
    logger.info("Cell type fractions -> data/gp_training_labels_%s.csv", output_prefix)

    region_fractions.to_csv(str(DATA_DIR / f"gp_training_regions_{output_prefix}.csv"))
    logger.info("Region fractions -> data/gp_training_regions_%s.csv", output_prefix)

    output_path = DATA_DIR / f"{output_prefix}_mapped.h5ad"
    query.write(str(output_path), compression="gzip")
    logger.info("Mapped data -> %s", output_path)

    elapsed = time.time() - start
    logger.info("Done in %.1f minutes.", elapsed / 60)

    # Summary
    logger.info("--- MAPPING SUMMARY ---")
    logger.info("Cells mapped: %d", query.n_obs)
    logger.info("Conditions: %d", query.obs[args.condition_key].nunique())
    for label_col in label_cols:
        pred_col = f"predicted_{label_col}"
        if pred_col in query.obs.columns:
            logger.info("%s: %d types", label_col, query.obs[pred_col].nunique())
            top3 = query.obs[pred_col].value_counts().head(3)
            for t, c in top3.items():
                logger.info("  %s: %d cells (%.1f%%)", t, c, c/query.n_obs*100)
```

Usage for SAG screen:
```bash
python gopro/02_map_to_hnoca.py \
    --input data/amin_kelley_sag_screen.h5ad \
    --output-prefix sag_screen \
    --batch-key sample
```

This produces:
- `data/sag_screen_mapped.h5ad`
- `data/gp_training_labels_sag_screen.csv`
- `data/gp_training_regions_sag_screen.csv`

### Step 6: Commit

```bash
git add gopro/02_map_to_hnoca.py gopro/tests/test_unit.py
git commit -m "feat: add --input flag to step 02, support SAG screen ClusterLabel filtering"
```

---

## Task 3: QC Validation for Overlapping Conditions

**Files:**
- Create: `gopro/qc_cross_screen.py`
- Test: `gopro/tests/test_unit.py`

### Step 1: Write failing test for cross-screen QC

```python
class TestCrossScreenQC:
    """Tests for cross-screen condition QC validation."""

    def test_cosine_similarity_identical(self):
        """Identical fractions should have cosine similarity = 1.0."""
        from gopro.qc_cross_screen import compute_cross_screen_similarity
        fracs_a = pd.DataFrame({"NPC": [0.5], "Neuron": [0.5]}, index=["SAG250"])
        fracs_b = pd.DataFrame({"NPC": [0.5], "Neuron": [0.5]}, index=["SAG_250nM"])
        mapping = {"SAG250": "SAG_250nM"}
        result = compute_cross_screen_similarity(fracs_a, fracs_b, mapping)
        assert result["SAG250"]["cosine_similarity"] == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal fractions should have cosine similarity = 0.0."""
        from gopro.qc_cross_screen import compute_cross_screen_similarity
        fracs_a = pd.DataFrame({"NPC": [1.0], "Neuron": [0.0]}, index=["cond_a"])
        fracs_b = pd.DataFrame({"NPC": [0.0], "Neuron": [1.0]}, index=["cond_b"])
        mapping = {"cond_a": "cond_b"}
        result = compute_cross_screen_similarity(fracs_a, fracs_b, mapping)
        assert result["cond_a"]["cosine_similarity"] == pytest.approx(0.0)

    def test_flag_low_similarity(self):
        """Should flag conditions with similarity < threshold."""
        from gopro.qc_cross_screen import validate_cross_screen
        fracs_a = pd.DataFrame({"NPC": [0.9], "Neuron": [0.1]}, index=["cond_a"])
        fracs_b = pd.DataFrame({"NPC": [0.1], "Neuron": [0.9]}, index=["cond_b"])
        mapping = {"cond_a": "cond_b"}
        flagged = validate_cross_screen(fracs_a, fracs_b, mapping, threshold=0.8)
        assert len(flagged) == 1
        assert "cond_a" in flagged
```

### Step 2: Run test to verify it fails

Run: `python -m pytest gopro/tests/test_unit.py::TestCrossScreenQC -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gopro.qc_cross_screen'`

### Step 3: Implement qc_cross_screen.py

Create `gopro/qc_cross_screen.py`:

```python
"""Cross-screen QC validation for overlapping morphogen conditions.

Compares cell type fractions between screens for conditions that share
the same morphogen vector, flagging batch effect concerns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from gopro.config import get_logger

logger = get_logger(__name__)


def compute_cross_screen_similarity(
    fracs_a: pd.DataFrame,
    fracs_b: pd.DataFrame,
    condition_mapping: dict[str, str],
) -> dict[str, dict]:
    """Compute cosine similarity between overlapping conditions across screens.

    Args:
        fracs_a: Cell type fractions from screen A (conditions × cell types).
        fracs_b: Cell type fractions from screen B (conditions × cell types).
        condition_mapping: Dict mapping screen_a condition → screen_b condition.

    Returns:
        Dict mapping screen_a condition → {cosine_similarity, fracs_a, fracs_b}.
    """
    # Align columns (union)
    all_cols = sorted(set(fracs_a.columns) | set(fracs_b.columns))
    for col in all_cols:
        if col not in fracs_a.columns:
            fracs_a[col] = 0.0
        if col not in fracs_b.columns:
            fracs_b[col] = 0.0

    results = {}
    for cond_a, cond_b in condition_mapping.items():
        if cond_a not in fracs_a.index or cond_b not in fracs_b.index:
            logger.warning("Condition not found: %s or %s", cond_a, cond_b)
            continue

        vec_a = fracs_a.loc[cond_a, all_cols].values.reshape(1, -1)
        vec_b = fracs_b.loc[cond_b, all_cols].values.reshape(1, -1)

        sim = float(sklearn_cosine(vec_a, vec_b)[0, 0])
        results[cond_a] = {
            "cosine_similarity": sim,
            "screen_b_condition": cond_b,
        }
        logger.info("QC: %s vs %s — cosine similarity = %.3f", cond_a, cond_b, sim)

    return results


def validate_cross_screen(
    fracs_a: pd.DataFrame,
    fracs_b: pd.DataFrame,
    condition_mapping: dict[str, str],
    threshold: float = 0.8,
) -> list[str]:
    """Validate overlapping conditions and return flagged ones.

    Args:
        fracs_a: Cell type fractions from screen A.
        fracs_b: Cell type fractions from screen B.
        condition_mapping: Dict mapping screen_a → screen_b conditions.
        threshold: Minimum cosine similarity (below = flagged).

    Returns:
        List of screen_a condition names that failed QC.
    """
    similarities = compute_cross_screen_similarity(fracs_a, fracs_b, condition_mapping)

    flagged = []
    for cond, result in similarities.items():
        if result["cosine_similarity"] < threshold:
            logger.warning(
                "QC FLAG: %s vs %s similarity %.3f < %.3f threshold",
                cond, result["screen_b_condition"],
                result["cosine_similarity"], threshold,
            )
            flagged.append(cond)

    if not flagged:
        logger.info("Cross-screen QC passed: all overlapping conditions above %.2f threshold", threshold)

    return flagged
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest gopro/tests/test_unit.py::TestCrossScreenQC -v`
Expected: 3 PASS

### Step 5: Commit

```bash
git add gopro/qc_cross_screen.py gopro/tests/test_unit.py
git commit -m "feat: add cross-screen QC validation for overlapping conditions"
```

---

## Task 4: Modify Step 04 to Auto-Discover SAG Screen Data

**Files:**
- Modify: `gopro/04_gpbo_loop.py`
- Test: `gopro/tests/test_unit.py`

### Step 1: Write failing test for multi-source real data merge

```python
class TestMultiSourceRealData:
    """Tests for merging multiple real-data sources at fidelity=1.0."""

    def test_merge_primary_plus_sag(self):
        """merge_multi_fidelity_data handles two fidelity=1.0 sources."""
        import tempfile
        from pathlib import Path
        from gopro.tests.conftest import _step04

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Primary screen (3 conditions, 4 cell types)
            fracs_1 = pd.DataFrame(
                {"NPC": [0.5, 0.3, 0.2], "Neuron": [0.3, 0.4, 0.5], "IP": [0.1, 0.2, 0.2], "Glia": [0.1, 0.1, 0.1]},
                index=["cond_a", "cond_b", "cond_c"],
            )
            morph_1 = pd.DataFrame(
                {"CHIR99021_uM": [1.5, 0, 3.0], "SAG_uM": [0, 0.25, 0], "log_harvest_day": [4.28, 4.28, 4.28]},
                index=["cond_a", "cond_b", "cond_c"],
            )
            fracs_1.to_csv(tmpdir / "fracs_primary.csv")
            morph_1.to_csv(tmpdir / "morph_primary.csv")

            # SAG screen (2 conditions, 3 cell types — missing 'Glia')
            fracs_2 = pd.DataFrame(
                {"NPC": [0.4, 0.2], "Neuron": [0.4, 0.6], "IP": [0.2, 0.2]},
                index=["SAG_50nM", "SAG_2uM"],
            )
            morph_2 = pd.DataFrame(
                {"CHIR99021_uM": [1.5, 1.5], "SAG_uM": [0.05, 2.0], "log_harvest_day": [4.25, 4.25]},
                index=["SAG_50nM", "SAG_2uM"],
            )
            fracs_2.to_csv(tmpdir / "fracs_sag.csv")
            morph_2.to_csv(tmpdir / "morph_sag.csv")

            sources = [
                (tmpdir / "fracs_primary.csv", tmpdir / "morph_primary.csv", 1.0),
                (tmpdir / "fracs_sag.csv", tmpdir / "morph_sag.csv", 1.0),
            ]
            X, Y = _step04.merge_multi_fidelity_data(sources)

            assert len(X) == 5  # 3 + 2
            assert len(Y) == 5
            assert "Glia" in Y.columns  # aligned from primary
            assert Y.loc["SAG_50nM", "Glia"] == 0.0  # filled with 0
            assert np.allclose(Y.sum(axis=1), 1.0, atol=1e-6)  # re-normalized
```

### Step 2: Run test to verify it passes (it should — existing code handles this)

Run: `python -m pytest gopro/tests/test_unit.py::TestMultiSourceRealData -v`
Expected: PASS (existing `merge_multi_fidelity_data` already supports this)

### Step 3: Update step 04 CLI to auto-discover SAG screen data

Add to the `if __name__ == "__main__"` block in `gopro/04_gpbo_loop.py`, after the virtual sources section:

```python
    # Check for SAG secondary screen real data
    sag_frac = Path(args.sag_fractions) if hasattr(args, 'sag_fractions') and args.sag_fractions else DATA_DIR / "gp_training_labels_sag_screen.csv"
    sag_morph = Path(args.sag_morphogens) if hasattr(args, 'sag_morphogens') and args.sag_morphogens else DATA_DIR / "morphogen_matrix_sag_screen.csv"
    if sag_frac.exists() and sag_morph.exists():
        # Insert SAG as first source alongside primary (both fidelity=1.0)
        all_sources = [
            (fractions_path, morphogen_path, 1.0),
            (sag_frac, sag_morph, 1.0),
        ]
        if virtual_sources:
            all_sources.extend(virtual_sources)
        virtual_sources = all_sources[1:]  # everything after primary = "extra sources"
        logger.info("Including SAG secondary screen real data (fidelity=1.0)")
```

Also add CLI args:

```python
    parser.add_argument("--sag-fractions", type=str, default=None,
                        help="Path to SAG screen fractions CSV (fidelity=1.0)")
    parser.add_argument("--sag-morphogens", type=str, default=None,
                        help="Path to SAG screen morphogens CSV")
```

### Step 4: Commit

```bash
git add gopro/04_gpbo_loop.py gopro/tests/test_unit.py
git commit -m "feat: auto-discover SAG screen data in GP-BO loop"
```

---

## Task 5: Generate SAG Screen Morphogen Matrix CSV

**Files:**
- Modify: `gopro/morphogen_parser.py` (add CLI support for SAG matrix)

### Step 1: Add SAG matrix generation to morphogen_parser.py __main__

Extend the `if __name__ == "__main__"` block to also generate the SAG secondary screen matrix:

```python
    # Also generate SAG secondary screen matrix
    sag_df = build_morphogen_matrix(SAG_SECONDARY_CONDITIONS)
    sag_output_path = DATA_DIR / "morphogen_matrix_sag_screen.csv"
    sag_df.to_csv(str(sag_output_path))
    logger.info("Saved SAG secondary screen matrix to %s", sag_output_path)
    logger.info("SAG matrix:\n%s", sag_df.to_string())
```

### Step 2: Run it

Run: `python gopro/morphogen_parser.py`
Expected: Creates `data/morphogen_matrix_sag_screen.csv` with 2 rows × 20 columns.

### Step 3: Commit

```bash
git add gopro/morphogen_parser.py
git commit -m "feat: generate SAG secondary screen morphogen matrix CSV"
```

---

## Task 6: Refactor Morphogen Parser into Generic Class

**Files:**
- Modify: `gopro/morphogen_parser.py`
- Test: `gopro/tests/test_unit.py`

### Step 1: Write failing test for generic parser interface

```python
class TestMorphogenParserClass:
    """Tests for generic MorphogenParser class."""

    def test_amin_kelley_parser(self):
        """AminKelleyParser should parse all 46 primary conditions."""
        from gopro.morphogen_parser import AminKelleyParser
        parser = AminKelleyParser()
        assert len(parser.conditions) == 46
        vec = parser.parse("CHIR1.5")
        assert vec["CHIR99021_uM"] == 1.5

    def test_sag_secondary_parser(self):
        """SAGSecondaryParser should parse 2 conditions."""
        from gopro.morphogen_parser import SAGSecondaryParser
        parser = SAGSecondaryParser()
        assert len(parser.conditions) == 2
        vec = parser.parse("SAG_2uM")
        assert vec["SAG_uM"] == pytest.approx(2.0)

    def test_build_matrix_from_parser(self):
        """Parser.build_matrix() should return DataFrame."""
        from gopro.morphogen_parser import AminKelleyParser
        parser = AminKelleyParser()
        df = parser.build_matrix()
        assert df.shape == (46, 20)

    def test_combined_parser(self):
        """CombinedParser merges multiple parsers."""
        from gopro.morphogen_parser import AminKelleyParser, SAGSecondaryParser, CombinedParser
        combined = CombinedParser([AminKelleyParser(), SAGSecondaryParser()])
        assert len(combined.conditions) == 48
        df = combined.build_matrix()
        assert df.shape == (48, 20)
```

### Step 2: Run test to verify it fails

Run: `python -m pytest gopro/tests/test_unit.py::TestMorphogenParserClass -v`
Expected: FAIL — `ImportError: cannot import name 'AminKelleyParser'`

### Step 3: Implement the class hierarchy

Refactor `gopro/morphogen_parser.py`. Keep ALL existing functions and the `_CONDITION_PARSERS` dict. Add a class wrapper on top:

```python
class MorphogenParser:
    """Base class for morphogen condition parsers."""

    def __init__(self, parsers: dict[str, Any], harvest_day: int):
        self._parsers = parsers
        self._harvest_day = harvest_day
        self._log_harvest_day = math.log(harvest_day)

    @property
    def conditions(self) -> list[str]:
        return sorted(self._parsers.keys())

    def parse(self, name: str) -> dict[str, float]:
        vec = _zeros()
        vec["log_harvest_day"] = self._log_harvest_day
        if name not in self._parsers:
            raise ValueError(f"Unrecognized condition: {name!r}")
        self._parsers[name](vec)
        return vec

    def build_matrix(self, conditions: list[str] | None = None) -> pd.DataFrame:
        conditions = conditions or self.conditions
        rows = [self.parse(c) for c in conditions]
        return pd.DataFrame(rows, index=conditions, columns=MORPHOGEN_COLUMNS)


class AminKelleyParser(MorphogenParser):
    """Parser for Amin/Kelley 2024 primary screen (46 conditions, Day 72)."""
    def __init__(self):
        # Filter to only primary screen conditions
        primary = {k: v for k, v in _CONDITION_PARSERS.items()
                   if k not in SAG_SECONDARY_CONDITIONS}
        super().__init__(primary, harvest_day=72)


class SAGSecondaryParser(MorphogenParser):
    """Parser for SAG secondary screen (2 conditions, Day 70)."""
    def __init__(self):
        sag = {k: v for k, v in _CONDITION_PARSERS.items()
               if k in SAG_SECONDARY_CONDITIONS}
        super().__init__(sag, harvest_day=70)

    def parse(self, name: str) -> dict[str, float]:
        # SAG secondary conditions set their own harvest day in the handler
        vec = _zeros()
        vec["log_harvest_day"] = self._log_harvest_day
        if name not in self._parsers:
            raise ValueError(f"Unrecognized condition: {name!r}")
        self._parsers[name](vec)
        return vec


class CombinedParser:
    """Combines multiple MorphogenParsers."""

    def __init__(self, parsers: list[MorphogenParser]):
        self._parsers = parsers

    @property
    def conditions(self) -> list[str]:
        all_conds = []
        for p in self._parsers:
            all_conds.extend(p.conditions)
        return sorted(all_conds)

    def parse(self, name: str) -> dict[str, float]:
        for p in self._parsers:
            if name in p._parsers:
                return p.parse(name)
        raise ValueError(f"Unrecognized condition: {name!r}")

    def build_matrix(self, conditions: list[str] | None = None) -> pd.DataFrame:
        conditions = conditions or self.conditions
        rows = [self.parse(c) for c in conditions]
        return pd.DataFrame(rows, index=conditions, columns=MORPHOGEN_COLUMNS)
```

Keep backward-compatible functions (`parse_condition_name`, `build_morphogen_matrix`, `ALL_CONDITIONS`) unchanged — they still work via `_CONDITION_PARSERS`.

### Step 4: Run tests

Run: `python -m pytest gopro/tests/test_unit.py::TestMorphogenParserClass -v`
Expected: 4 PASS

### Step 5: Run full suite

Run: `python -m pytest gopro/tests/ -v`
Expected: All pass

### Step 6: Commit

```bash
git add gopro/morphogen_parser.py gopro/tests/test_unit.py
git commit -m "refactor: add generic MorphogenParser class hierarchy"
```

---

## Task 7: RDS→h5ad Conversion Script

**Files:**
- Create: `gopro/convert_rds_to_h5ad.py`
- Test: (manual — requires R)

### Step 1: Write the conversion script

Create `gopro/convert_rds_to_h5ad.py`:

```python
"""Convert Seurat RDS files to AnnData h5ad format.

Uses R (via subprocess) with Seurat and SeuratDisk packages.
Auto-installs missing R packages on first run.

R location: /Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/bin/Rscript

Usage:
    python gopro/convert_rds_to_h5ad.py data/patterning_screen/OSMGT_processed_files/4_M_vs_sM_21d_clean.rds.gz
    python gopro/convert_rds_to_h5ad.py input.rds.gz --output output.h5ad
    python gopro/convert_rds_to_h5ad.py input.rds.gz --check-only  # inspect without converting
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

# R binary location (not in PATH on this system)
R_FRAMEWORK = Path("/Library/Frameworks/R.framework/Versions")
RSCRIPT = None
for version_dir in sorted(R_FRAMEWORK.glob("*/Resources/bin/Rscript"), reverse=True):
    RSCRIPT = version_dir
    break

if RSCRIPT is None:
    # Fallback: try PATH
    RSCRIPT = Path("Rscript")


def _run_r(script: str, timeout: int = 3600) -> str:
    """Run an R script via Rscript and return stdout.

    Args:
        script: R code to execute.
        timeout: Max runtime in seconds.

    Returns:
        stdout from the R process.

    Raises:
        RuntimeError: If R execution fails.
    """
    result = subprocess.run(
        [str(RSCRIPT), "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr}\n"
            f"STDOUT: {result.stdout}"
        )
    return result.stdout


def ensure_r_packages() -> None:
    """Install required R packages if missing."""
    check_script = """
    packages <- c("Seurat", "SeuratObject", "SeuratDisk")
    missing <- packages[!sapply(packages, requireNamespace, quietly=TRUE)]
    if (length(missing) > 0) {
        cat("MISSING:", paste(missing, collapse=","), "\\n")
    } else {
        cat("ALL_INSTALLED\\n")
    }
    """
    output = _run_r(check_script)

    if "ALL_INSTALLED" in output:
        logger.info("All required R packages are installed.")
        return

    logger.info("Installing missing R packages (this may take several minutes)...")

    install_script = """
    if (!requireNamespace("remotes", quietly=TRUE)) {
        install.packages("remotes", repos="https://cloud.r-project.org")
    }
    if (!requireNamespace("Seurat", quietly=TRUE)) {
        install.packages("Seurat", repos="https://cloud.r-project.org")
    }
    if (!requireNamespace("SeuratDisk", quietly=TRUE)) {
        remotes::install_github("mojaveazure/seurat-disk", quiet=TRUE)
    }
    cat("INSTALL_COMPLETE\\n")
    """
    output = _run_r(install_script, timeout=1800)

    if "INSTALL_COMPLETE" not in output:
        raise RuntimeError(f"R package installation may have failed:\n{output}")

    logger.info("R packages installed successfully.")


def decompress_rds(rds_gz_path: Path) -> Path:
    """Decompress .rds.gz to .rds if needed.

    Args:
        rds_gz_path: Path to gzipped RDS file.

    Returns:
        Path to decompressed RDS file.
    """
    if rds_gz_path.suffix != ".gz":
        return rds_gz_path

    rds_path = rds_gz_path.with_suffix("")
    if rds_path.exists():
        logger.info("Decompressed file exists: %s", rds_path.name)
        return rds_path

    logger.info("Decompressing %s (%.1f GB)...", rds_gz_path.name,
                rds_gz_path.stat().st_size / 1e9)

    with gzip.open(str(rds_gz_path), "rb") as f_in:
        with open(str(rds_path), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info("Decompressed to %s (%.1f GB)", rds_path.name,
                rds_path.stat().st_size / 1e9)
    return rds_path


def inspect_rds(rds_path: Path) -> str:
    """Inspect an RDS file's structure without converting.

    Args:
        rds_path: Path to RDS file.

    Returns:
        String summary of the object's metadata.
    """
    script = f"""
    obj <- readRDS("{rds_path}")
    cat("Class:", class(obj), "\\n")
    if (inherits(obj, "Seurat")) {{
        cat("Cells:", ncol(obj), "\\n")
        cat("Genes:", nrow(obj), "\\n")
        cat("Assays:", paste(names(obj@assays), collapse=", "), "\\n")
        cat("Metadata columns:", paste(colnames(obj@meta.data), collapse=", "), "\\n")
        cat("\\nMetadata head:\\n")
        print(head(obj@meta.data, 3))
        cat("\\nUnique values for key columns:\\n")
        for (col in colnames(obj@meta.data)) {{
            n <- length(unique(obj@meta.data[[col]]))
            if (n < 100) {{
                cat(col, "(", n, "unique):", paste(head(unique(obj@meta.data[[col]]), 20), collapse=", "), "\\n")
            }} else {{
                cat(col, "(", n, "unique)\\n")
            }}
        }}
    }} else {{
        cat("Not a Seurat object. str():\\n")
        str(obj, max.level=2)
    }}
    """
    return _run_r(script, timeout=600)


def convert_rds_to_h5ad(rds_path: Path, h5ad_path: Path) -> Path:
    """Convert a Seurat RDS file to h5ad format.

    Uses SeuratDisk as intermediate (RDS → h5seurat → h5ad).

    Args:
        rds_path: Path to input RDS file.
        h5ad_path: Path for output h5ad file.

    Returns:
        Path to the created h5ad file.
    """
    h5seurat_path = h5ad_path.with_suffix(".h5seurat")

    script = f"""
    library(Seurat)
    library(SeuratDisk)

    cat("Loading RDS:", "{rds_path}", "\\n")
    obj <- readRDS("{rds_path}")
    cat("Loaded:", ncol(obj), "cells x", nrow(obj), "genes\\n")

    # Ensure default assay has counts
    if (!"counts" %in% names(obj@assays[[DefaultAssay(obj)]]@layers) &&
        length(obj@assays[[DefaultAssay(obj)]]@counts) > 0) {{
        cat("Counts matrix found in default assay\\n")
    }}

    # Save as h5seurat (intermediate format)
    cat("Saving h5Seurat to:", "{h5seurat_path}", "\\n")
    SaveH5Seurat(obj, filename = "{h5seurat_path}", overwrite = TRUE)

    # Convert h5seurat to h5ad
    cat("Converting to h5ad:", "{h5ad_path}", "\\n")
    Convert("{h5seurat_path}", dest = "h5ad", overwrite = TRUE)

    cat("CONVERSION_COMPLETE\\n")
    """

    logger.info("Converting %s → %s ...", rds_path.name, h5ad_path.name)
    output = _run_r(script, timeout=3600)

    if "CONVERSION_COMPLETE" not in output:
        raise RuntimeError(f"Conversion may have failed:\n{output}")

    # Clean up intermediate h5seurat
    if h5seurat_path.exists():
        h5seurat_path.unlink()
        logger.info("Cleaned up intermediate %s", h5seurat_path.name)

    if not h5ad_path.exists():
        raise FileNotFoundError(f"Expected output not found: {h5ad_path}")

    logger.info("Conversion complete: %s (%.1f GB)",
                h5ad_path.name, h5ad_path.stat().st_size / 1e9)
    return h5ad_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Seurat RDS files to AnnData h5ad format.",
    )
    parser.add_argument("input", type=str,
                        help="Path to input RDS or RDS.gz file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output h5ad path (default: same name with .h5ad extension)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only inspect the RDS file, don't convert")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip R package installation check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    logger.info("R binary: %s", RSCRIPT)

    # Ensure R packages
    if not args.skip_install:
        ensure_r_packages()

    # Decompress if needed
    rds_path = decompress_rds(input_path)

    # Inspect-only mode
    if args.check_only:
        logger.info("=== Inspecting %s ===", rds_path.name)
        output = inspect_rds(rds_path)
        print(output)
        return

    # Determine output path
    if args.output:
        h5ad_path = Path(args.output)
    else:
        # Strip .rds.gz or .rds, add .h5ad
        stem = input_path.name
        for suffix in [".rds.gz", ".rds"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        h5ad_path = DATA_DIR / f"{stem}.h5ad"

    # Convert
    convert_rds_to_h5ad(rds_path, h5ad_path)


if __name__ == "__main__":
    main()
```

### Step 2: Test with --check-only first

Run: `python gopro/convert_rds_to_h5ad.py data/patterning_screen/OSMGT_processed_files/4_M_vs_sM_21d_clean.rds.gz --check-only`

This will:
1. Auto-install Seurat/SeuratDisk if missing
2. Decompress the .rds.gz (3.1 GB)
3. Print metadata (conditions, cell counts, etc.)
4. NOT convert yet

### Step 3: If inspect looks good, run full conversion

Run: `python gopro/convert_rds_to_h5ad.py data/patterning_screen/OSMGT_processed_files/4_M_vs_sM_21d_clean.rds.gz`

Output: `data/4_M_vs_sM_21d_clean.h5ad`

### Step 4: Commit

```bash
git add gopro/convert_rds_to_h5ad.py
git commit -m "feat: add RDS-to-h5ad conversion script using R subprocess"
```

---

## Task 8: Update CLAUDE.md and Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `data/README.md`

### Step 1: Update CLAUDE.md

Add to the Repository Structure section:
- `gopro/qc_cross_screen.py` — Cross-screen QC validation
- `gopro/convert_rds_to_h5ad.py` — RDS→h5ad conversion (requires R)

Add to Pipeline Steps:
```
python 02_map_to_hnoca.py --input data/amin_kelley_sag_screen.h5ad --output-prefix sag_screen
```

Update Known Issues: remove the "only 46 conditions" limitation.

### Step 2: Update data/README.md

Add SAG secondary screen outputs to pipeline-generated files table.

### Step 3: Commit

```bash
git add CLAUDE.md data/README.md
git commit -m "docs: update for SAG screen integration and RDS converter"
```

---

## Execution Order

Tasks 1, 3, 6, and 7 are independent and can be parallelized. Tasks 2 and 4 depend on Task 1. Task 5 depends on Task 1.

```
Task 1 (SAG parsing) ──→ Task 2 (step 02 mods) ──→ Task 5 (generate CSV)
       │                                                      │
       └──→ Task 4 (step 04 auto-discover) ──────────────────┘
                                                              │
Task 3 (QC validation) ──────────────────────────────────────┘
Task 6 (parser refactor) ─────────── depends on Task 1 ──────┘
Task 7 (RDS converter) ── independent ────────────────────────┘
Task 8 (docs) ── last ───────────────────────────────────────→
```
