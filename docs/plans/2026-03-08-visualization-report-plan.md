# GP-BO Visualization Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `gopro/05_visualize.py` that generates a self-contained interactive HTML report showing GP-BO optimization state across multiple rounds.

**Architecture:** Single Python script reads pipeline output CSVs + h5ad, builds Plotly figures, and writes a standalone HTML file with all charts embedded. Uses Jinja2-style string templating for the HTML wrapper. UMAP computed on-the-fly from morphogen matrix; cell-space UMAP read from h5ad obsm.

**Tech Stack:** plotly, umap-learn, pandas, numpy, anndata, pathlib

---

## Data Reality Check

Before implementing, know what actually exists on disk:

| File | Rows | Key columns |
|------|------|-------------|
| `fidelity_report.csv` | 46 | condition (index), composite_fidelity, rss_score, on_target_fraction, off_target_fraction, normalized_entropy, dominant_region, rss_best_region |
| `morphogen_matrix_amin_kelley.csv` | 46 | condition (index), 20 morphogen columns (CHIR99021_uM through log_harvest_day) |
| `gp_recommendations_round1.csv` | 6 | well (index), morphogen cols, predicted_y{0-3}_mean/std, acquisition_value |
| `gp_diagnostics_round1.csv` | 1 | round, n_training_points, n_morphogens, n_cell_types, target_cell_types — **NO lengthscale columns** (bug in step 04) |
| `gp_training_labels_amin_kelley.csv` | 46 | condition (index), 14 cell type fraction columns |
| `gp_training_regions_amin_kelley.csv` | 46 | condition (index), 10 brain region fraction columns |
| `amin_kelley_mapped.h5ad` | ~cells | obs has condition + cell type labels; obsm may have X_umap |

**Important:** Diagnostics currently lack ARD lengthscales. The morphogen importance chart must handle this gracefully (show "not available" or fall back to variance-based importance from the morphogen matrix).

---

### Task 1: Add plotly dependency and create skeleton script

**Files:**
- Modify: `gopro/requirements.txt`
- Create: `gopro/05_visualize.py`
- Test: `gopro/tests/test_unit.py` (add new test class)

**Step 1: Write the failing test**

Add to `gopro/tests/test_unit.py`:

```python
class TestVisualizationReport:
    """Tests for 05_visualize.py report generation."""

    def test_discover_rounds(self, tmp_path):
        """discover_rounds finds all gp_recommendations_round*.csv files."""
        (tmp_path / "gp_recommendations_round1.csv").write_text("well\nA1\n")
        (tmp_path / "gp_recommendations_round2.csv").write_text("well\nA1\n")
        (tmp_path / "other_file.csv").write_text("x\n1\n")

        from gopro import visualize_report
        rounds = visualize_report.discover_rounds(tmp_path)
        assert rounds == [1, 2]

    def test_discover_rounds_empty(self, tmp_path):
        """discover_rounds returns empty list when no rounds exist."""
        from gopro import visualize_report
        rounds = visualize_report.discover_rounds(tmp_path)
        assert rounds == []
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest gopro/tests/test_unit.py::TestVisualizationReport -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gopro.visualize_report'`

**Step 3: Write minimal implementation**

Create `gopro/05_visualize.py` (importable as `gopro.visualize_report`):

```python
"""Step 05: Generate interactive HTML report for GP-BO optimization state."""

from pathlib import Path
import re

PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"


def discover_rounds(data_dir: Path) -> list[int]:
    """Find all GP-BO round numbers from recommendation CSVs on disk."""
    pattern = re.compile(r"gp_recommendations_round(\d+)\.csv")
    rounds = []
    for f in sorted(data_dir.glob("gp_recommendations_round*.csv")):
        m = pattern.match(f.name)
        if m:
            rounds.append(int(m.group(1)))
    return sorted(rounds)
```

Also add `plotly` to `gopro/requirements.txt`.

Also create `gopro/visualize_report.py` as a re-export module (since the script is `05_visualize.py` which can't be imported directly):

Actually, simpler: name the module `gopro/visualize_report.py` for imports, and have `gopro/05_visualize.py` just call it as `__main__`. But to keep consistency with the existing pipeline pattern (scripts named `0N_*.py`), we'll put all logic in `gopro/visualize_report.py` and have `gopro/05_visualize.py` be a thin CLI wrapper.

**Step 4: Run test to verify it passes**

Run: `python -m pytest gopro/tests/test_unit.py::TestVisualizationReport -v`
Expected: PASS

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/05_visualize.py gopro/requirements.txt gopro/tests/test_unit.py
git commit -m "feat(viz): add skeleton 05_visualize.py with round discovery"
```

---

### Task 2: Data loading functions

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py` (extend TestVisualizationReport)

**Step 1: Write the failing tests**

```python
def test_load_fidelity_report(self, tmp_path):
    """load_fidelity_report reads CSV with condition as index."""
    csv = tmp_path / "fidelity_report.csv"
    csv.write_text("condition,composite_fidelity,rss_score\nLDN,0.90,0.79\nIWP2,0.89,0.77\n")

    from gopro import visualize_report
    df = visualize_report.load_fidelity_report(csv)
    assert df.index.name == "condition"
    assert len(df) == 2
    assert "composite_fidelity" in df.columns

def test_load_recommendations(self, tmp_path):
    """load_recommendations reads round CSV with well as index."""
    csv = tmp_path / "gp_recommendations_round1.csv"
    csv.write_text("well,CHIR99021_uM,acquisition_value\nA1,3.0,-1.3\nA2,1.5,-1.3\n")

    from gopro import visualize_report
    df = visualize_report.load_recommendations(csv)
    assert df.index.name == "well"
    assert len(df) == 2

def test_load_morphogen_matrix(self, tmp_path):
    """load_morphogen_matrix reads morphogen CSV with condition as index."""
    csv = tmp_path / "morphogen_matrix_amin_kelley.csv"
    csv.write_text(",CHIR99021_uM,BMP4_ng_mL\nLDN,0.0,0.0\nCHIR,3.0,0.0\n")

    from gopro import visualize_report
    df = visualize_report.load_morphogen_matrix(csv)
    assert df.index.name is None or len(df) == 2
    assert "CHIR99021_uM" in df.columns

def test_load_cell_type_fractions(self, tmp_path):
    """load_cell_type_fractions reads fractions CSV."""
    csv = tmp_path / "gp_training_labels_amin_kelley.csv"
    csv.write_text("condition,Astrocyte,Neuron\nLDN,0.3,0.7\nCHIR,0.5,0.5\n")

    from gopro import visualize_report
    df = visualize_report.load_cell_type_fractions(csv)
    assert len(df) == 2
    assert abs(df.iloc[0].sum() - 1.0) < 0.01

def test_load_diagnostics_without_lengthscales(self, tmp_path):
    """load_diagnostics handles missing lengthscale columns gracefully."""
    csv = tmp_path / "gp_diagnostics_round1.csv"
    csv.write_text("round,n_training_points,n_morphogens,n_cell_types,target_cell_types\n1,20,5,5,all\n")

    from gopro import visualize_report
    diag = visualize_report.load_diagnostics(csv)
    assert diag["round"] == 1
    assert diag["lengthscales"] is None  # graceful fallback
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest gopro/tests/test_unit.py::TestVisualizationReport -v -k "load"`
Expected: FAIL — functions don't exist

**Step 3: Write minimal implementation**

Add to `gopro/visualize_report.py`:

```python
import pandas as pd


def load_fidelity_report(path: Path) -> pd.DataFrame:
    """Load fidelity_report.csv with condition as index."""
    return pd.read_csv(path, index_col="condition")


def load_recommendations(path: Path) -> pd.DataFrame:
    """Load gp_recommendations_round{N}.csv with well as index."""
    return pd.read_csv(path, index_col="well")


def load_morphogen_matrix(path: Path) -> pd.DataFrame:
    """Load morphogen_matrix_amin_kelley.csv."""
    return pd.read_csv(path, index_col=0)


def load_cell_type_fractions(path: Path) -> pd.DataFrame:
    """Load cell type or region fractions CSV."""
    return pd.read_csv(path, index_col="condition")


def load_diagnostics(path: Path) -> dict:
    """Load GP diagnostics. Returns dict with 'lengthscales' key (None if missing)."""
    df = pd.read_csv(path)
    row = df.iloc[0]
    result = {
        "round": int(row["round"]),
        "n_training_points": int(row["n_training_points"]),
        "n_morphogens": int(row["n_morphogens"]),
        "n_cell_types": int(row["n_cell_types"]),
        "target_cell_types": row["target_cell_types"],
    }
    ls_cols = [c for c in df.columns if c.startswith("lengthscale_")]
    if ls_cols:
        result["lengthscales"] = {c.replace("lengthscale_", ""): float(row[c]) for c in ls_cols}
    else:
        result["lengthscales"] = None
    return result
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest gopro/tests/test_unit.py::TestVisualizationReport -v -k "load"`
Expected: PASS

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add data loading functions with graceful lengthscale fallback"
```

---

### Task 3: Auto-generated text summary

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing test**

```python
def test_generate_summary_text(self):
    """generate_summary produces a human-readable paragraph."""
    import pandas as pd
    from gopro import visualize_report

    fidelity = pd.DataFrame({
        "composite_fidelity": [0.90, 0.85, 0.70],
        "dominant_region": ["Dorsal telencephalon", "Ventral telencephalon", "Midbrain"],
    }, index=pd.Index(["LDN", "IWP2", "CHIR"], name="condition"))

    diagnostics = {"round": 1, "n_training_points": 20, "lengthscales": None}
    n_recommendations = 6

    text = visualize_report.generate_summary_text(
        fidelity_df=fidelity,
        diagnostics=diagnostics,
        n_recommendations=n_recommendations,
    )
    assert "Round 1" in text
    assert "LDN" in text  # best condition
    assert "0.90" in text or "0.9" in text  # best fidelity
    assert "6" in text  # n recommendations
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest gopro/tests/test_unit.py::TestVisualizationReport::test_generate_summary_text -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
def generate_summary_text(
    fidelity_df: pd.DataFrame,
    diagnostics: dict,
    n_recommendations: int,
) -> str:
    """Generate a short text summary of the current GP-BO round."""
    round_num = diagnostics["round"]
    best_idx = fidelity_df["composite_fidelity"].idxmax()
    best_score = fidelity_df.loc[best_idx, "composite_fidelity"]
    best_region = fidelity_df.loc[best_idx, "dominant_region"]

    parts = [f"Round {round_num}: {diagnostics['n_training_points']} conditions evaluated."]
    parts.append(f"Best condition: {best_idx} (fidelity {best_score:.2f}, dominant region: {best_region}).")
    parts.append(f"{n_recommendations} new experiments recommended.")

    if diagnostics.get("lengthscales"):
        ls = diagnostics["lengthscales"]
        top3 = sorted(ls, key=ls.get)[:3]  # shortest lengthscale = most important
        parts.append(f"Top morphogens by importance: {', '.join(top3)}.")

    return " ".join(parts)
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add auto-generated text summary"
```

---

### Task 4: UMAP of morphogen space (Panel A)

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing test**

```python
def test_compute_morphogen_umap(self):
    """compute_morphogen_umap returns 2D coordinates for conditions."""
    import numpy as np
    import pandas as pd
    from gopro import visualize_report

    np.random.seed(42)
    morphogens = pd.DataFrame(
        np.random.rand(20, 5),
        columns=[f"morph_{i}" for i in range(5)],
        index=[f"cond_{i}" for i in range(20)],
    )
    coords = visualize_report.compute_morphogen_umap(morphogens)
    assert coords.shape == (20, 2)
    assert list(coords.columns) == ["UMAP1", "UMAP2"]
    assert coords.index.equals(morphogens.index)

def test_build_morphogen_umap_figure(self):
    """build_morphogen_umap_figure returns a Plotly Figure."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from gopro import visualize_report

    coords = pd.DataFrame(
        {"UMAP1": np.random.rand(10), "UMAP2": np.random.rand(10)},
        index=[f"cond_{i}" for i in range(10)],
    )
    fidelity_scores = pd.Series(np.random.rand(10), index=coords.index)
    rec_coords = pd.DataFrame(
        {"UMAP1": np.random.rand(3), "UMAP2": np.random.rand(3)},
        index=["A1", "A2", "A3"],
    )

    fig = visualize_report.build_morphogen_umap_figure(coords, fidelity_scores, rec_coords)
    assert isinstance(fig, go.Figure)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest gopro/tests/test_unit.py::TestVisualizationReport -v -k "umap"`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
import numpy as np


def compute_morphogen_umap(morphogen_df: pd.DataFrame, n_neighbors: int = 15) -> pd.DataFrame:
    """Compute 2D UMAP embedding of morphogen condition vectors."""
    from umap import UMAP

    # Standardize before UMAP
    X = morphogen_df.values.astype(float)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    n_samples = X_std.shape[0]
    effective_neighbors = min(n_neighbors, n_samples - 1)

    reducer = UMAP(n_components=2, n_neighbors=effective_neighbors, random_state=42)
    embedding = reducer.fit_transform(X_std)

    return pd.DataFrame(
        embedding, columns=["UMAP1", "UMAP2"], index=morphogen_df.index
    )


def build_morphogen_umap_figure(
    condition_coords: pd.DataFrame,
    fidelity_scores: pd.Series,
    recommendation_coords: pd.DataFrame | None = None,
) -> "plotly.graph_objects.Figure":
    """Build Plotly UMAP scatter of morphogen space, colored by predicted fidelity."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Existing conditions
    fig.add_trace(go.Scatter(
        x=condition_coords["UMAP1"],
        y=condition_coords["UMAP2"],
        mode="markers",
        marker=dict(
            size=10,
            color=fidelity_scores.values,
            colorscale="Viridis",
            colorbar=dict(title="Fidelity"),
            line=dict(width=0.5, color="white"),
        ),
        text=condition_coords.index,
        hovertemplate="<b>%{text}</b><br>Fidelity: %{marker.color:.3f}<extra></extra>",
        name="Observed conditions",
    ))

    # Recommended next experiments
    if recommendation_coords is not None and len(recommendation_coords) > 0:
        fig.add_trace(go.Scatter(
            x=recommendation_coords["UMAP1"],
            y=recommendation_coords["UMAP2"],
            mode="markers",
            marker=dict(
                size=14,
                color="red",
                symbol="star",
                line=dict(width=1, color="white"),
            ),
            text=recommendation_coords.index,
            hovertemplate="<b>%{text}</b> (recommended)<extra></extra>",
            name="Recommended",
        ))

    fig.update_layout(
        title="Morphogen Space (UMAP)",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        width=700,
        height=500,
    )
    return fig
```

For projecting recommendations into the same UMAP space, we need a helper that fits UMAP on existing conditions and transforms recommendations:

```python
def compute_morphogen_umap_with_recommendations(
    morphogen_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    morphogen_columns: list[str],
    n_neighbors: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute UMAP for existing conditions, then project recommendations into same space."""
    from umap import UMAP

    shared_cols = [c for c in morphogen_columns if c in morphogen_df.columns and c in recommendation_df.columns]

    X_train = morphogen_df[shared_cols].values.astype(float)
    X_rec = recommendation_df[shared_cols].values.astype(float)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-10
    X_train_std = (X_train - mean) / std
    X_rec_std = (X_rec - mean) / std

    n_samples = X_train_std.shape[0]
    effective_neighbors = min(n_neighbors, n_samples - 1)

    reducer = UMAP(n_components=2, n_neighbors=effective_neighbors, random_state=42)
    train_embedding = reducer.fit_transform(X_train_std)
    rec_embedding = reducer.transform(X_rec_std)

    train_coords = pd.DataFrame(train_embedding, columns=["UMAP1", "UMAP2"], index=morphogen_df.index)
    rec_coords = pd.DataFrame(rec_embedding, columns=["UMAP1", "UMAP2"], index=recommendation_df.index)
    return train_coords, rec_coords
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add morphogen-space UMAP with recommendation projection"
```

---

### Task 5: Cell-space UMAP (Panel B)

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing test**

```python
def test_build_cell_umap_figure(self):
    """build_cell_umap_figure creates scatter from h5ad obsm coordinates."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from gopro import visualize_report

    n_cells = 100
    coords = pd.DataFrame({
        "UMAP1": np.random.rand(n_cells),
        "UMAP2": np.random.rand(n_cells),
    })
    cell_types = pd.Series(
        np.random.choice(["Neuron", "Astrocyte", "NPC"], n_cells),
        name="cell_type"
    )
    conditions = pd.Series(
        np.random.choice(["LDN", "CHIR", "IWP2"], n_cells),
        name="condition"
    )

    fig = visualize_report.build_cell_umap_figure(coords, cell_types, conditions)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # one trace per cell type
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
def build_cell_umap_figure(
    coords: pd.DataFrame,
    cell_types: pd.Series,
    conditions: pd.Series,
) -> "plotly.graph_objects.Figure":
    """Build Plotly UMAP scatter of individual cells colored by cell type."""
    import plotly.graph_objects as go

    fig = go.Figure()

    for ct in sorted(cell_types.unique()):
        mask = cell_types == ct
        fig.add_trace(go.Scatter(
            x=coords.loc[mask, "UMAP1"],
            y=coords.loc[mask, "UMAP2"],
            mode="markers",
            marker=dict(size=3, opacity=0.5),
            text=conditions[mask],
            hovertemplate="<b>%{text}</b><br>Type: " + ct + "<extra></extra>",
            name=ct,
        ))

    fig.update_layout(
        title="Cell Space (UMAP)",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        width=700,
        height=500,
        legend=dict(title="Cell Type"),
    )
    return fig


def extract_cell_umap_from_h5ad(h5ad_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract UMAP coordinates, cell types, and conditions from mapped h5ad."""
    import anndata

    adata = anndata.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs

    # Try common UMAP key names
    umap_key = None
    for key in ["X_umap", "X_UMAP"]:
        if key in adata.obsm:
            umap_key = key
            break

    if umap_key is None:
        raise KeyError(f"No UMAP found in obsm. Available keys: {list(adata.obsm.keys())}")

    coords = pd.DataFrame(
        adata.obsm[umap_key][:, :2],
        columns=["UMAP1", "UMAP2"],
        index=obs.index,
    )

    # Use level 2 annotation (brain regions) as cell type
    ct_col = None
    for col in ["annot_level_2", "snapseed_pca_rss_level_2", "cell_type"]:
        if col in obs.columns:
            ct_col = col
            break
    cell_types = obs[ct_col] if ct_col else pd.Series("Unknown", index=obs.index)

    # Condition column
    cond_col = None
    for col in ["condition", "sample", "batch"]:
        if col in obs.columns:
            cond_col = col
            break
    conditions = obs[cond_col] if cond_col else pd.Series("Unknown", index=obs.index)

    return coords, cell_types, conditions
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add cell-space UMAP panel with cell type coloring"
```

---

### Task 6: Plate map visualization

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing test**

```python
def test_build_plate_map_figure(self):
    """build_plate_map_figure creates a 4x6 grid heatmap."""
    import pandas as pd
    import plotly.graph_objects as go
    from gopro import visualize_report

    wells = [f"{r}{c}" for r in "ABCD" for c in range(1, 7)][:6]
    recs = pd.DataFrame({
        "CHIR99021_uM": [3.0, 1.5, 0.0, 2.0, 1.0, 0.5],
        "acquisition_value": [-1.3, -1.2, -1.5, -1.1, -1.4, -1.0],
    }, index=pd.Index(wells, name="well"))

    fidelity_values = pd.Series([0.8, 0.7, 0.6, 0.9, 0.5, 0.85], index=wells)

    fig = visualize_report.build_plate_map_figure(recs, fidelity_values)
    assert isinstance(fig, go.Figure)
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
def build_plate_map_figure(
    recommendations: pd.DataFrame,
    predicted_fidelity: pd.Series,
) -> "plotly.graph_objects.Figure":
    """Build a 4x6 plate map heatmap colored by predicted fidelity."""
    import plotly.graph_objects as go

    rows = list("ABCD")
    cols = list(range(1, 7))

    # Build grid: 4 rows x 6 cols
    z = np.full((4, 6), np.nan)
    hover_text = [["" for _ in range(6)] for _ in range(4)]

    for well in recommendations.index:
        row_idx = rows.index(well[0])
        col_idx = int(well[1:]) - 1
        if well in predicted_fidelity.index:
            z[row_idx][col_idx] = predicted_fidelity[well]

        # Build hover with morphogen concentrations
        morph_cols = [c for c in recommendations.columns
                      if c not in ("acquisition_value",) and not c.startswith("predicted_")]
        recipe = "<br>".join(
            f"{c}: {recommendations.loc[well, c]:.2f}"
            for c in morph_cols
            if recommendations.loc[well, c] != 0
        )
        hover_text[row_idx][col_idx] = f"<b>{well}</b><br>{recipe}"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(c) for c in cols],
        y=rows,
        colorscale="Viridis",
        colorbar=dict(title="Predicted<br>Fidelity"),
        hovertext=hover_text,
        hovertemplate="%{hovertext}<extra></extra>",
        showscale=True,
    ))

    fig.update_layout(
        title="Recommended Plate Map",
        xaxis_title="Column",
        yaxis_title="Row",
        template="plotly_white",
        width=500,
        height=350,
        yaxis=dict(autorange="reversed"),
    )
    return fig
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add 4x6 plate map heatmap"
```

---

### Task 7: Morphogen importance bar chart

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing tests**

```python
def test_build_importance_figure_with_lengthscales(self):
    """build_importance_figure uses 1/lengthscale when available."""
    import plotly.graph_objects as go
    from gopro import visualize_report

    lengthscales = {"CHIR99021_uM": 0.5, "BMP4_ng_mL": 2.0, "SAG_nM": 1.0}
    fig = visualize_report.build_importance_figure(lengthscales=lengthscales)
    assert isinstance(fig, go.Figure)

def test_build_importance_figure_without_lengthscales(self):
    """build_importance_figure falls back to variance when no lengthscales."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from gopro import visualize_report

    morphogens = pd.DataFrame({
        "CHIR99021_uM": [0.0, 3.0, 1.5],
        "BMP4_ng_mL": [10.0, 10.0, 10.0],  # zero variance
        "SAG_nM": [0.0, 500.0, 1000.0],
    })
    fig = visualize_report.build_importance_figure(lengthscales=None, morphogen_df=morphogens)
    assert isinstance(fig, go.Figure)
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
def build_importance_figure(
    lengthscales: dict | None = None,
    morphogen_df: pd.DataFrame | None = None,
) -> "plotly.graph_objects.Figure":
    """Build horizontal bar chart of morphogen importance.

    Uses 1/lengthscale (ARD kernel) if available, falls back to
    normalized variance across conditions.
    """
    import plotly.graph_objects as go

    if lengthscales:
        importance = {k: 1.0 / v for k, v in lengthscales.items()}
        title_suffix = "(1 / ARD lengthscale)"
    elif morphogen_df is not None:
        variance = morphogen_df.var()
        max_var = variance.max()
        importance = (variance / max_var if max_var > 0 else variance).to_dict()
        title_suffix = "(normalized variance — lengthscales not available)"
    else:
        raise ValueError("Need either lengthscales or morphogen_df")

    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color="steelblue",
    ))

    fig.update_layout(
        title=f"Morphogen Importance {title_suffix}",
        xaxis_title="Importance",
        template="plotly_white",
        width=600,
        height=max(300, len(names) * 25),
        yaxis=dict(autorange="reversed"),
    )
    return fig
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add morphogen importance chart with ARD/variance fallback"
```

---

### Task 8: Best conditions leaderboard table

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing test**

```python
def test_build_leaderboard_figure(self):
    """build_leaderboard_figure creates a Plotly table of top conditions."""
    import pandas as pd
    import plotly.graph_objects as go
    from gopro import visualize_report

    fidelity = pd.DataFrame({
        "composite_fidelity": [0.90, 0.85, 0.70, 0.60, 0.50],
        "rss_score": [0.79, 0.77, 0.65, 0.55, 0.45],
        "on_target_fraction": [0.97, 0.94, 0.80, 0.70, 0.60],
        "off_target_fraction": [0.0, 0.0, 0.05, 0.1, 0.2],
        "dominant_region": ["DT", "DT", "VM", "HY", "MB"],
    }, index=pd.Index(["LDN", "IWP2", "I/Act", "CHIR", "BMP4"], name="condition"))

    fig = visualize_report.build_leaderboard_figure(fidelity, top_n=3)
    assert isinstance(fig, go.Figure)
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
def build_leaderboard_figure(fidelity_df: pd.DataFrame, top_n: int = 10) -> "plotly.graph_objects.Figure":
    """Build a Plotly table showing top conditions by composite fidelity."""
    import plotly.graph_objects as go

    top = fidelity_df.nlargest(top_n, "composite_fidelity")

    display_cols = ["composite_fidelity", "rss_score", "on_target_fraction",
                    "off_target_fraction", "dominant_region"]
    display_cols = [c for c in display_cols if c in top.columns]

    header_labels = ["Condition"] + [c.replace("_", " ").title() for c in display_cols]

    cells_values = [top.index.tolist()]
    for col in display_cols:
        vals = top[col].tolist()
        if top[col].dtype == float:
            vals = [f"{v:.3f}" for v in vals]
        cells_values.append(vals)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=header_labels,
            fill_color="steelblue",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cells_values,
            fill_color="white",
            align="left",
        ),
    ))

    fig.update_layout(
        title=f"Top {top_n} Conditions by Composite Fidelity",
        template="plotly_white",
        width=800,
        height=max(200, 50 + top_n * 30),
    )
    return fig
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add best conditions leaderboard table"
```

---

### Task 9: Cell type composition stacked bar chart

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing test**

```python
def test_build_composition_figure(self):
    """build_composition_figure creates stacked bar chart of cell type fractions."""
    import pandas as pd
    import plotly.graph_objects as go
    from gopro import visualize_report

    fractions = pd.DataFrame({
        "Neuron": [0.6, 0.3, 0.1],
        "Astrocyte": [0.2, 0.4, 0.5],
        "NPC": [0.2, 0.3, 0.4],
    }, index=pd.Index(["LDN", "CHIR", "BMP4"], name="condition"))

    sort_order = ["LDN", "CHIR", "BMP4"]

    fig = visualize_report.build_composition_figure(fractions, sort_order, title="Level 2")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # one bar trace per cell type
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
def build_composition_figure(
    fractions_df: pd.DataFrame,
    sort_order: list[str],
    title: str = "Cell Type Composition",
) -> "plotly.graph_objects.Figure":
    """Build stacked bar chart of cell type fractions per condition."""
    import plotly.graph_objects as go

    ordered = fractions_df.loc[[c for c in sort_order if c in fractions_df.index]]

    fig = go.Figure()
    for col in ordered.columns:
        fig.add_trace(go.Bar(
            x=ordered.index,
            y=ordered[col],
            name=col,
            hovertemplate="%{x}<br>" + col + ": %{y:.2%}<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Condition",
        yaxis_title="Fraction",
        template="plotly_white",
        width=max(600, len(ordered) * 20),
        height=450,
        xaxis=dict(tickangle=-45),
        legend=dict(title="Cell Type"),
    )
    return fig
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add cell type composition stacked bar chart"
```

---

### Task 10: Convergence trend chart

**Files:**
- Modify: `gopro/visualize_report.py`
- Test: `gopro/tests/test_unit.py`

**Step 1: Write the failing test**

```python
def test_build_convergence_figure_single_round(self):
    """build_convergence_figure works with a single data point."""
    import plotly.graph_objects as go
    from gopro import visualize_report

    best_per_round = {0: 0.90}
    fig = visualize_report.build_convergence_figure(best_per_round)
    assert isinstance(fig, go.Figure)

def test_build_convergence_figure_multi_round(self):
    """build_convergence_figure shows trend across rounds."""
    import plotly.graph_objects as go
    from gopro import visualize_report

    best_per_round = {0: 0.70, 1: 0.82, 2: 0.88}
    fig = visualize_report.build_convergence_figure(best_per_round)
    assert isinstance(fig, go.Figure)
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

```python
def build_convergence_figure(best_per_round: dict[int, float]) -> "plotly.graph_objects.Figure":
    """Build line chart of best fidelity score per round."""
    import plotly.graph_objects as go

    rounds = sorted(best_per_round.keys())
    scores = [best_per_round[r] for r in rounds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds,
        y=scores,
        mode="lines+markers",
        marker=dict(size=10, color="steelblue"),
        line=dict(width=2),
        hovertemplate="Round %{x}<br>Best fidelity: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="Optimization Convergence",
        xaxis_title="Round",
        yaxis_title="Best Composite Fidelity",
        template="plotly_white",
        width=500,
        height=350,
        xaxis=dict(dtick=1),
        yaxis=dict(range=[0, 1]),
    )
    return fig
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/tests/test_unit.py
git commit -m "feat(viz): add convergence trend chart"
```

---

### Task 11: HTML assembly and main entrypoint

**Files:**
- Modify: `gopro/visualize_report.py`
- Create: `gopro/05_visualize.py` (CLI wrapper)
- Test: `gopro/tests/test_integration.py` (add integration test)

**Step 1: Write the failing test**

Add to `gopro/tests/test_integration.py`:

```python
class TestVisualizationIntegration:
    """Integration tests for the full report generation pipeline."""

    def test_assemble_html_report(self, tmp_path):
        """assemble_html_report produces a valid HTML file with all sections."""
        import plotly.graph_objects as go
        from gopro import visualize_report

        # Create minimal dummy figures
        dummy_fig = go.Figure(go.Scatter(x=[1], y=[1]))

        sections = {
            "summary_text": "Round 1: test summary.",
            "convergence": dummy_fig,
            "morphogen_umap": dummy_fig,
            "cell_umap": dummy_fig,
            "plate_map": dummy_fig,
            "importance": dummy_fig,
            "leaderboard": dummy_fig,
            "composition_regions": dummy_fig,
            "composition_broad": dummy_fig,
        }

        output_path = tmp_path / "report.html"
        visualize_report.assemble_html_report(sections, output_path)

        assert output_path.exists()
        html = output_path.read_text()
        assert "<html>" in html.lower()
        assert "Round 1" in html
        assert "plotly" in html.lower()
```

**Step 2: Run test to verify it fails**

**Step 3: Write minimal implementation**

Add to `gopro/visualize_report.py`:

```python
def assemble_html_report(sections: dict, output_path: Path) -> None:
    """Assemble all sections into a single self-contained HTML file."""
    import plotly.io as pio

    chart_divs = []
    for key, value in sections.items():
        if key == "summary_text":
            continue
        chart_divs.append(pio.to_html(value, full_html=False, include_plotlyjs=False))

    summary = sections.get("summary_text", "")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GP-BO Optimization Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        .summary {{ background: #f8f9fa; padding: 16px 20px; border-radius: 6px;
                    border-left: 4px solid #2c3e50; margin: 20px 0; font-size: 15px; line-height: 1.6; }}
        .chart-section {{ margin: 30px 0; }}
        .two-panel {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .two-panel > div {{ flex: 1; min-width: 300px; }}
    </style>
</head>
<body>
    <h1>GP-BO Optimization Report</h1>
    <div class="summary">{summary}</div>

    <h2>Convergence</h2>
    <div class="chart-section">{chart_divs[0]}</div>

    <h2>Morphogen Space</h2>
    <div class="two-panel">
        <div>{chart_divs[1]}</div>
        <div>{chart_divs[2]}</div>
    </div>

    <h2>Recommended Plate Map</h2>
    <div class="chart-section">{chart_divs[3]}</div>

    <h2>Morphogen Importance</h2>
    <div class="chart-section">{chart_divs[4]}</div>

    <h2>Best Conditions</h2>
    <div class="chart-section">{chart_divs[5]}</div>

    <h2>Cell Type Composition — Brain Regions</h2>
    <div class="chart-section">{chart_divs[6]}</div>

    <h2>Cell Type Composition — Broad Categories</h2>
    <div class="chart-section">{chart_divs[7]}</div>
</body>
</html>"""

    output_path.write_text(html)


def generate_report(data_dir: Path, output_path: Path | None = None) -> Path:
    """Main entrypoint: load all data, build all figures, write HTML report."""
    rounds = discover_rounds(data_dir)
    current_round = max(rounds) if rounds else 0

    # Load data
    fidelity = load_fidelity_report(data_dir / "fidelity_report.csv")
    morphogens = load_morphogen_matrix(data_dir / "morphogen_matrix_amin_kelley.csv")
    regions = load_cell_type_fractions(data_dir / "gp_training_regions_amin_kelley.csv")
    labels = load_cell_type_fractions(data_dir / "gp_training_labels_amin_kelley.csv")

    # Load recommendations and diagnostics for latest round
    rec_path = data_dir / f"gp_recommendations_round{current_round}.csv"
    recommendations = load_recommendations(rec_path) if rec_path.exists() else None
    diag_path = data_dir / f"gp_diagnostics_round{current_round}.csv"
    diagnostics = load_diagnostics(diag_path) if diag_path.exists() else {
        "round": current_round, "n_training_points": len(fidelity), "lengthscales": None
    }

    # Sort conditions by fidelity
    sort_order = fidelity.sort_values("composite_fidelity", ascending=False).index.tolist()

    # Build figures
    n_recs = len(recommendations) if recommendations is not None else 0

    summary_text = generate_summary_text(fidelity, diagnostics, n_recs)

    # Convergence (for now just round 0 = best from initial data)
    best_per_round = {0: fidelity["composite_fidelity"].max()}
    convergence = build_convergence_figure(best_per_round)

    # Morphogen UMAP
    morphogen_cols = morphogens.columns.tolist()
    if recommendations is not None:
        cond_coords, rec_coords = compute_morphogen_umap_with_recommendations(
            morphogens, recommendations, morphogen_cols
        )
    else:
        cond_coords = compute_morphogen_umap(morphogens)
        rec_coords = None
    morphogen_umap = build_morphogen_umap_figure(cond_coords, fidelity["composite_fidelity"], rec_coords)

    # Cell UMAP
    mapped_path = data_dir / "amin_kelley_mapped.h5ad"
    if mapped_path.exists():
        try:
            cell_coords, cell_types, cell_conditions = extract_cell_umap_from_h5ad(mapped_path)
            cell_umap = build_cell_umap_figure(cell_coords, cell_types, cell_conditions)
        except (KeyError, Exception):
            cell_umap = _placeholder_figure("Cell UMAP not available (no UMAP in h5ad)")
    else:
        cell_umap = _placeholder_figure("Cell UMAP not available (no mapped h5ad)")

    # Plate map
    if recommendations is not None:
        # Use predicted mean of first ILR dim as proxy for fidelity
        pred_cols = [c for c in recommendations.columns if c.startswith("predicted_") and c.endswith("_mean")]
        if "fidelity" in recommendations.columns:
            pred_fidelity = recommendations["fidelity"]
        elif pred_cols:
            pred_fidelity = recommendations[pred_cols[0]]
        else:
            pred_fidelity = pd.Series(0.5, index=recommendations.index)
        plate_map = build_plate_map_figure(recommendations, pred_fidelity)
    else:
        plate_map = _placeholder_figure("No recommendations available")

    # Importance
    importance = build_importance_figure(
        lengthscales=diagnostics.get("lengthscales"),
        morphogen_df=morphogens,
    )

    # Leaderboard
    leaderboard = build_leaderboard_figure(fidelity)

    # Composition
    comp_regions = build_composition_figure(regions, sort_order, title="Brain Region Composition")
    comp_broad = build_composition_figure(labels, sort_order, title="Cell Type Composition (Broad)")

    sections = {
        "summary_text": summary_text,
        "convergence": convergence,
        "morphogen_umap": morphogen_umap,
        "cell_umap": cell_umap,
        "plate_map": plate_map,
        "importance": importance,
        "leaderboard": leaderboard,
        "composition_regions": comp_regions,
        "composition_broad": comp_broad,
    }

    if output_path is None:
        output_path = data_dir / f"report_round{current_round}.html"

    assemble_html_report(sections, output_path)
    return output_path


def _placeholder_figure(message: str) -> "plotly.graph_objects.Figure":
    """Create a placeholder figure with a message."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(template="plotly_white", width=500, height=300)
    return fig
```

Create `gopro/05_visualize.py`:

```python
"""Step 05: Generate interactive HTML report for GP-BO optimization state.

Usage:
    python 05_visualize.py [--data-dir PATH] [--output PATH]
"""

import argparse
from pathlib import Path
from gopro.visualize_report import generate_report

PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"


def main():
    parser = argparse.ArgumentParser(description="Generate GP-BO visualization report")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = generate_report(args.data_dir, args.output)
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test, verify pass**

**Step 5: Commit**

```bash
git add gopro/visualize_report.py gopro/05_visualize.py gopro/tests/test_integration.py
git commit -m "feat(viz): add HTML assembly and main entrypoint for report generation"
```

---

### Task 12: Run full pipeline on real data and verify output

**Files:** None (verification only)

**Step 1: Install plotly**

Run: `pip install plotly && pip freeze | grep plotly`

**Step 2: Run the report generator on real data**

Run: `cd /Users/maxxyung/Projects/morphogen-gpbo && python gopro/05_visualize.py`

Expected: `Report written to /Users/maxxyung/Projects/morphogen-gpbo/data/report_round1.html`

**Step 3: Verify HTML output**

Run: `wc -l data/report_round1.html && head -20 data/report_round1.html`

Expected: HTML file with plotly charts, all 8 sections present.

**Step 4: Open in browser to verify visually**

Run: `open data/report_round1.html`

**Step 5: Run full test suite**

Run: `python -m pytest gopro/tests/ -v`

Expected: All tests pass (existing 65 + new ~15 = ~80 tests)

**Step 6: Commit**

```bash
git add gopro/requirements.txt
git commit -m "feat(viz): add plotly dependency, verify report generation on real data"
```

---

## Summary

| Task | Description | Est. |
|------|-------------|------|
| 1 | Skeleton + round discovery | 5 min |
| 2 | Data loading functions | 5 min |
| 3 | Auto-generated text summary | 3 min |
| 4 | Morphogen-space UMAP | 5 min |
| 5 | Cell-space UMAP | 5 min |
| 6 | Plate map heatmap | 5 min |
| 7 | Morphogen importance bar chart | 5 min |
| 8 | Leaderboard table | 3 min |
| 9 | Cell type composition stacked bar | 3 min |
| 10 | Convergence trend chart | 3 min |
| 11 | HTML assembly + main entrypoint | 5 min |
| 12 | Real data verification | 5 min |
