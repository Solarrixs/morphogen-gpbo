"""
Visualization report generator for GP-BO pipeline.

Produces a self-contained HTML report with embedded Plotly charts showing
optimization state: where we've explored, what's working, and what to try next.

Inputs (from pipeline steps 02-04):
  - data/fidelity_report.csv
  - data/morphogen_matrix_amin_kelley.csv
  - data/gp_recommendations_round{N}.csv
  - data/gp_diagnostics_round{N}.csv
  - data/gp_training_labels_amin_kelley.csv
  - data/gp_training_regions_amin_kelley.csv
  - data/amin_kelley_mapped.h5ad (optional, for cell UMAP)

Output:
  - data/report_round{N}.html
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_white"
VIRIDIS = "Viridis"

# ---------------------------------------------------------------------------
# Round discovery
# ---------------------------------------------------------------------------

def discover_rounds(data_dir: Path) -> list[int]:
    """Find all GP-BO round numbers from recommendation files on disk.

    Args:
        data_dir: Directory containing gp_recommendations_round{N}.csv files.

    Returns:
        Sorted list of round integers found.
    """
    pattern = str(data_dir / "gp_recommendations_round*.csv")
    files = glob.glob(pattern)
    rounds = []
    for f in files:
        m = re.search(r"round(\d+)\.csv$", f)
        if m:
            rounds.append(int(m.group(1)))
    return sorted(rounds)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_fidelity_report(path: Path) -> pd.DataFrame:
    """Load fidelity report CSV, indexed by condition."""
    df = pd.read_csv(str(path), index_col="condition")
    return df


def load_recommendations(path: Path) -> pd.DataFrame:
    """Load GP recommendations CSV, indexed by well."""
    df = pd.read_csv(str(path), index_col="well")
    return df


def load_morphogen_matrix(path: Path) -> pd.DataFrame:
    """Load morphogen concentration matrix CSV, indexed by condition."""
    df = pd.read_csv(str(path), index_col=0)
    return df


def load_cell_type_fractions(path: Path) -> pd.DataFrame:
    """Load cell type fraction CSV, indexed by condition."""
    df = pd.read_csv(str(path), index_col="condition")
    return df


def load_diagnostics(path: Path) -> dict:
    """Load GP diagnostics CSV into a dict.

    Returns dict with keys like 'round', 'n_training_points', and
    'lengthscales' (dict mapping morphogen names to lengthscale values,
    or None if no lengthscale columns present).
    """
    df = pd.read_csv(str(path))
    row = df.iloc[0].to_dict()

    # Extract lengthscale columns if present
    ls_cols = [c for c in df.columns if c.startswith("lengthscale_")]
    if ls_cols:
        lengthscales = {}
        for c in ls_cols:
            morph_name = c.replace("lengthscale_", "")
            lengthscales[morph_name] = float(row[c])
        row["lengthscales"] = lengthscales
    else:
        row["lengthscales"] = None

    return row


# ---------------------------------------------------------------------------
# Auto-generated text summary
# ---------------------------------------------------------------------------

def generate_summary_text(
    fidelity_df: pd.DataFrame,
    diagnostics: dict,
    n_recommendations: int,
) -> str:
    """Generate a plain-text summary of the current optimization state.

    Args:
        fidelity_df: Fidelity report DataFrame indexed by condition.
        diagnostics: Dict from load_diagnostics().
        n_recommendations: Number of recommended next experiments.

    Returns:
        Summary string.
    """
    round_num = int(diagnostics.get("round", 1))
    n_conditions = len(fidelity_df)
    best_idx = fidelity_df["composite_fidelity"].idxmax()
    best_score = fidelity_df.loc[best_idx, "composite_fidelity"]

    parts = [
        f"Round {round_num}: {n_conditions} conditions evaluated.",
        f"Best: {best_idx} (fidelity {best_score:.3f}).",
        f"{n_recommendations} new experiments recommended.",
    ]

    lengthscales = diagnostics.get("lengthscales")
    if lengthscales:
        importance = {k: 1.0 / v for k, v in lengthscales.items() if v > 0}
        top3 = sorted(importance, key=importance.get, reverse=True)[:3]
        top3_str = ", ".join(top3)
        parts.append(f"Top morphogens by GP importance: {top3_str}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Panel A: Morphogen-space UMAP
# ---------------------------------------------------------------------------

def compute_morphogen_umap(morphogen_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 2D UMAP of morphogen concentration vectors.

    Args:
        morphogen_df: DataFrame of shape (N, D) with morphogen concentrations.

    Returns:
        DataFrame with columns ['UMAP1', 'UMAP2'], indexed like input.
    """
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    X = StandardScaler().fit_transform(morphogen_df.values)
    n = len(X)
    n_neighbors = min(15, n - 1)
    coords = UMAP(n_neighbors=n_neighbors, random_state=42).fit_transform(X)
    return pd.DataFrame(
        coords, columns=["UMAP1", "UMAP2"], index=morphogen_df.index
    )


def compute_morphogen_umap_with_recommendations(
    morphogen_df: pd.DataFrame,
    recs_df: pd.DataFrame,
    morphogen_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute UMAP on training data and transform recommendations into same space.

    Args:
        morphogen_df: Training morphogen matrix.
        recs_df: Recommendations DataFrame with morphogen columns.
        morphogen_cols: Columns shared between both DataFrames.

    Returns:
        Tuple of (training_coords, recommendation_coords) DataFrames.
    """
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    scaler = StandardScaler()
    X_train = scaler.fit_transform(morphogen_df[morphogen_cols].values)
    X_rec = scaler.transform(recs_df[morphogen_cols].values)

    n = len(X_train)
    n_neighbors = min(15, n - 1)
    reducer = UMAP(n_neighbors=n_neighbors, random_state=42)
    train_coords = reducer.fit_transform(X_train)
    rec_coords = reducer.transform(X_rec)

    train_df = pd.DataFrame(
        train_coords, columns=["UMAP1", "UMAP2"], index=morphogen_df.index
    )
    rec_df = pd.DataFrame(
        rec_coords, columns=["UMAP1", "UMAP2"], index=recs_df.index
    )
    return train_df, rec_df


def build_morphogen_umap_figure(
    coords: pd.DataFrame,
    fidelity_scores: pd.Series,
    rec_coords: pd.DataFrame | None = None,
) -> go.Figure:
    """Build morphogen-space UMAP scatter plot.

    Args:
        coords: Training condition UMAP coordinates.
        fidelity_scores: Series of composite fidelity per condition.
        rec_coords: Optional recommendation UMAP coordinates (shown as stars).

    Returns:
        Plotly Figure.
    """
    # Align fidelity scores to coords index
    common = coords.index.intersection(fidelity_scores.index)
    coords_aligned = coords.loc[common]
    scores_aligned = fidelity_scores.loc[common]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords_aligned["UMAP1"],
        y=coords_aligned["UMAP2"],
        mode="markers",
        marker=dict(
            size=10,
            color=scores_aligned.values,
            colorscale=VIRIDIS,
            colorbar=dict(title="Fidelity"),
            showscale=True,
        ),
        text=coords_aligned.index,
        hovertemplate="<b>%{text}</b><br>Fidelity: %{marker.color:.3f}<extra></extra>",
        name="Training",
    ))

    if rec_coords is not None and len(rec_coords) > 0:
        fig.add_trace(go.Scatter(
            x=rec_coords["UMAP1"],
            y=rec_coords["UMAP2"],
            mode="markers",
            marker=dict(
                size=14,
                color="red",
                symbol="star",
                line=dict(width=1, color="black"),
            ),
            text=rec_coords.index,
            hovertemplate="<b>%{text}</b> (recommended)<extra></extra>",
            name="Recommended",
        ))

    fig.update_layout(
        title="Morphogen Space (UMAP)",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template=PLOTLY_TEMPLATE,
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Panel B: Cell-space UMAP
# ---------------------------------------------------------------------------

def extract_cell_umap_from_h5ad(
    path: Path,
) -> tuple[np.ndarray, pd.Series, pd.Series] | None:
    """Extract UMAP coordinates, cell types, and conditions from h5ad.

    Args:
        path: Path to mapped h5ad file.

    Returns:
        Tuple of (coords, cell_types, conditions) or None if UMAP not found.
    """
    import anndata

    adata = anndata.read_h5ad(str(path), backed="r")

    # Find UMAP coordinates
    umap_key = None
    if "X_umap" in adata.obsm:
        umap_key = "X_umap"
    elif "X_umap_hnoca" in adata.obsm:
        umap_key = "X_umap_hnoca"
    else:
        for key in adata.obsm:
            if "umap" in key.lower():
                umap_key = key
                break

    if umap_key is None:
        return None

    coords = adata.obsm[umap_key][:]

    # Find cell type column
    ct_col = None
    for candidate in ["predicted_annot_level_2", "annot_level_2", "cell_type"]:
        if candidate in adata.obs.columns:
            ct_col = candidate
            break

    cell_types = adata.obs[ct_col] if ct_col else pd.Series("Unknown", index=adata.obs.index)

    # Find condition column
    cond_col = None
    for candidate in ["condition", "sample", "batch"]:
        if candidate in adata.obs.columns:
            cond_col = candidate
            break

    conditions = adata.obs[cond_col] if cond_col else pd.Series("Unknown", index=adata.obs.index)

    return coords, cell_types.reset_index(drop=True), conditions.reset_index(drop=True)


def build_cell_umap_figure(
    coords: np.ndarray,
    cell_types: pd.Series,
    conditions: pd.Series,
) -> go.Figure:
    """Build cell-space UMAP colored by cell type.

    Args:
        coords: Array of shape (N_cells, 2).
        cell_types: Series of cell type labels per cell.
        conditions: Series of condition names per cell.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    unique_types = sorted(cell_types.unique())
    for ct in unique_types:
        mask = cell_types == ct
        fig.add_trace(go.Scattergl(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            marker=dict(size=2, opacity=0.5),
            name=ct,
            text=conditions[mask],
            hovertemplate="<b>%{text}</b><br>%{fullData.name}<extra></extra>",
        ))

    fig.update_layout(
        title="Cell Space (UMAP)",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template=PLOTLY_TEMPLATE,
        showlegend=True,
        legend=dict(itemsizing="constant"),
    )
    return fig


# ---------------------------------------------------------------------------
# Plate map (4×6 grid)
# ---------------------------------------------------------------------------

def build_plate_map_figure(
    recommendations: pd.DataFrame,
    predicted_fidelity: pd.Series | None = None,
) -> go.Figure:
    """Build 24-well plate map heatmap.

    Args:
        recommendations: DataFrame indexed by well (A1-D6).
        predicted_fidelity: Optional series of predicted fidelity per well.
            If None, uses acquisition_value as color.

    Returns:
        Plotly Figure.
    """
    rows = ["A", "B", "C", "D"]
    cols = list(range(1, 7))

    # Build 4x6 grid
    z = np.full((4, 6), np.nan)
    hover = [[" " for _ in range(6)] for _ in range(4)]
    annotations = []

    for well in recommendations.index:
        m = re.match(r"([A-D])(\d+)", well)
        if not m:
            continue
        r = rows.index(m.group(1))
        c = int(m.group(2)) - 1

        if predicted_fidelity is not None and well in predicted_fidelity.index:
            z[r, c] = predicted_fidelity[well]
        elif "acquisition_value" in recommendations.columns:
            z[r, c] = recommendations.loc[well, "acquisition_value"]

        # Build hover text with non-zero morphogens
        morph_cols = [col for col in recommendations.columns
                      if not col.startswith("predicted_") and col != "acquisition_value"
                      and col != "fidelity"]
        nonzero = []
        for mc in morph_cols:
            val = recommendations.loc[well, mc]
            if abs(val) > 0.001:
                nonzero.append(f"{mc}: {val:.2f}")
        hover[r][c] = f"<b>{well}</b><br>" + "<br>".join(nonzero) if nonzero else f"<b>{well}</b>"

        annotations.append(dict(
            x=c, y=r, text=well, showarrow=False,
            font=dict(color="white", size=12),
        ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(c) for c in cols],
        y=rows,
        colorscale=VIRIDIS,
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(title="Value"),
    ))

    fig.update_layout(
        title="Plate Map — Recommended Experiments",
        xaxis_title="Column",
        yaxis_title="Row",
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Morphogen importance bar chart
# ---------------------------------------------------------------------------

def build_importance_figure(
    lengthscales: dict | None = None,
    morphogen_df: pd.DataFrame | None = None,
) -> go.Figure:
    """Build morphogen importance bar chart.

    Primary: 1/lengthscale from ARD kernel (if available).
    Fallback: normalized variance across conditions.

    Args:
        lengthscales: Dict mapping morphogen names to lengthscale values.
        morphogen_df: Morphogen concentration matrix (fallback).

    Returns:
        Plotly Figure.
    """
    if lengthscales:
        importance = {k: 1.0 / v for k, v in lengthscales.items() if v > 0}
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        title = "Morphogen Importance (1/lengthscale)"
    elif morphogen_df is not None:
        variance = morphogen_df.var()
        if variance.max() > 0:
            variance = variance / variance.max()
        sorted_var = variance.sort_values(ascending=False)
        names = list(sorted_var.index)
        values = list(sorted_var.values)
        title = "Morphogen Variance (normalized, lengthscales unavailable)"
    else:
        return _placeholder_figure("No morphogen importance data available")

    fig = go.Figure(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker_color="steelblue",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        height=max(400, len(names) * 25),
    )
    return fig


# ---------------------------------------------------------------------------
# Leaderboard table
# ---------------------------------------------------------------------------

def build_leaderboard_figure(
    fidelity_df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Build top conditions leaderboard table.

    Args:
        fidelity_df: Fidelity report DataFrame indexed by condition.
        top_n: Number of top conditions to show.

    Returns:
        Plotly Figure with go.Table.
    """
    top = fidelity_df.nlargest(top_n, "composite_fidelity")

    display_cols = ["composite_fidelity", "rss_score", "on_target_fraction",
                    "off_target_fraction"]
    available = [c for c in display_cols if c in top.columns]

    header_vals = ["Condition"] + available
    cell_vals = [top.index.tolist()]
    for col in available:
        cell_vals.append([f"{v:.3f}" for v in top[col]])

    fig = go.Figure(go.Table(
        header=dict(
            values=header_vals,
            fill_color="steelblue",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cell_vals,
            fill_color="white",
            align="left",
        ),
    ))

    fig.update_layout(
        title=f"Top {min(top_n, len(top))} Conditions by Fidelity",
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# Cell type composition stacked bar
# ---------------------------------------------------------------------------

def build_composition_figure(
    fractions_df: pd.DataFrame,
    sort_order: list[str] | None = None,
    title: str = "Cell Type Composition",
) -> go.Figure:
    """Build stacked bar chart of cell type composition per condition.

    Args:
        fractions_df: DataFrame (conditions × cell types) with fractions summing to 1.
        sort_order: Optional list of condition names for x-axis order.
        title: Figure title.

    Returns:
        Plotly Figure.
    """
    if sort_order is not None:
        common = [c for c in sort_order if c in fractions_df.index]
        fractions_df = fractions_df.loc[common]

    fig = go.Figure()
    for col in fractions_df.columns:
        fig.add_trace(go.Bar(
            x=fractions_df.index,
            y=fractions_df[col],
            name=col,
        ))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Condition",
        yaxis_title="Fraction",
        yaxis=dict(range=[0, 1]),
        template=PLOTLY_TEMPLATE,
        xaxis=dict(tickangle=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ---------------------------------------------------------------------------
# Convergence trend chart
# ---------------------------------------------------------------------------

def build_convergence_figure(
    best_per_round: dict[int, float],
) -> go.Figure:
    """Build convergence trend line chart.

    Args:
        best_per_round: Dict mapping round number to best fidelity score.

    Returns:
        Plotly Figure.
    """
    rounds = sorted(best_per_round.keys())
    scores = [best_per_round[r] for r in rounds]

    fig = go.Figure(go.Scatter(
        x=rounds,
        y=scores,
        mode="lines+markers",
        marker=dict(size=10, color="steelblue"),
        line=dict(width=2, color="steelblue"),
    ))

    fig.update_layout(
        title="Best Fidelity by Round",
        xaxis_title="Round",
        yaxis_title="Best Composite Fidelity",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(dtick=1),
        template=PLOTLY_TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# Placeholder for missing data
# ---------------------------------------------------------------------------

def _placeholder_figure(message: str) -> go.Figure:
    """Create a placeholder figure with a message for missing data."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=200,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def assemble_html_report(
    sections: dict[str, go.Figure | str],
    output_path: Path,
) -> Path:
    """Assemble final HTML report from sections.

    Args:
        sections: Ordered dict of section_name -> Figure or HTML string.
        output_path: Where to write the HTML file.

    Returns:
        Path to the written file.
    """
    import plotly.io as pio

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GP-BO Visualization Report</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    color: #333;
  }
  h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
  h2 { color: #555; margin-top: 40px; }
  .summary { background: #f8f9fa; padding: 15px 20px; border-radius: 6px;
             border-left: 4px solid steelblue; margin: 20px 0; font-size: 1.05em; }
  .section { margin-bottom: 40px; }
  .plotly-chart { margin: 10px 0; }
</style>
</head>
<body>
<h1>GP-BO Visualization Report</h1>
""")

    for name, content in sections.items():
        html_parts.append(f'<div class="section">')
        html_parts.append(f'<h2>{name}</h2>')
        if isinstance(content, str):
            html_parts.append(f'<div class="summary">{content}</div>')
        elif isinstance(content, go.Figure):
            div = pio.to_html(content, full_html=False, include_plotlyjs="cdn")
            html_parts.append(f'<div class="plotly-chart">{div}</div>')
        html_parts.append('</div>')

    html_parts.append("</body>\n</html>")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts))
    return output_path


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def generate_report(
    data_dir: Path,
    output_path: Path | None = None,
) -> Path:
    """Generate the full GP-BO visualization report.

    Orchestrates: discover rounds → load data → build figures → assemble HTML.

    Args:
        data_dir: Directory containing pipeline outputs.
        output_path: Override for output HTML path. Default: data_dir/report_round{N}.html.

    Returns:
        Path to the generated HTML report.
    """
    data_dir = Path(data_dir)

    # Discover rounds
    rounds = discover_rounds(data_dir)
    if not rounds:
        raise FileNotFoundError(
            f"No gp_recommendations_round*.csv found in {data_dir}"
        )
    current_round = max(rounds)

    # Load data
    fidelity_df = load_fidelity_report(data_dir / "fidelity_report.csv")
    morphogen_df = load_morphogen_matrix(data_dir / "morphogen_matrix_amin_kelley.csv")
    recs = load_recommendations(data_dir / f"gp_recommendations_round{current_round}.csv")
    diagnostics = load_diagnostics(data_dir / f"gp_diagnostics_round{current_round}.csv")

    # Load cell type fractions
    labels_path = data_dir / "gp_training_labels_amin_kelley.csv"
    labels_df = load_cell_type_fractions(labels_path) if labels_path.exists() else None

    regions_path = data_dir / "gp_training_regions_amin_kelley.csv"
    regions_df = load_cell_type_fractions(regions_path) if regions_path.exists() else None

    sections: dict[str, go.Figure | str] = {}

    # 1. Summary text
    summary = generate_summary_text(fidelity_df, diagnostics, len(recs))
    sections["Summary"] = summary

    # 2. Convergence trend
    best_per_round = {}
    for r in rounds:
        best_per_round[r] = fidelity_df["composite_fidelity"].max()
    sections["Convergence"] = build_convergence_figure(best_per_round)

    # 3. Morphogen-space UMAP
    try:
        shared_cols = [c for c in morphogen_df.columns if c in recs.columns]
        if shared_cols and len(morphogen_df) > 2:
            train_coords, rec_coords = compute_morphogen_umap_with_recommendations(
                morphogen_df, recs, shared_cols
            )
            sections["Morphogen Space UMAP"] = build_morphogen_umap_figure(
                train_coords, fidelity_df["composite_fidelity"], rec_coords
            )
        else:
            sections["Morphogen Space UMAP"] = _placeholder_figure(
                "Insufficient data for morphogen UMAP"
            )
    except Exception as e:
        sections["Morphogen Space UMAP"] = _placeholder_figure(f"UMAP error: {e}")

    # 4. Cell-space UMAP
    h5ad_path = data_dir / "amin_kelley_mapped.h5ad"
    if h5ad_path.exists():
        try:
            result = extract_cell_umap_from_h5ad(h5ad_path)
            if result is not None:
                coords, cell_types, conditions = result
                sections["Cell Space UMAP"] = build_cell_umap_figure(
                    coords, cell_types, conditions
                )
            else:
                sections["Cell Space UMAP"] = _placeholder_figure(
                    "No UMAP coordinates in h5ad file"
                )
        except Exception as e:
            sections["Cell Space UMAP"] = _placeholder_figure(f"h5ad error: {e}")
    else:
        sections["Cell Space UMAP"] = _placeholder_figure(
            "amin_kelley_mapped.h5ad not found"
        )

    # 5. Plate map
    sections["Plate Map"] = build_plate_map_figure(recs)

    # 6. Morphogen importance
    lengthscales = diagnostics.get("lengthscales")
    sections["Morphogen Importance"] = build_importance_figure(
        lengthscales=lengthscales, morphogen_df=morphogen_df
    )

    # 7. Leaderboard
    sections["Leaderboard"] = build_leaderboard_figure(fidelity_df)

    # 8. Cell type composition
    sort_order = fidelity_df.sort_values("composite_fidelity", ascending=False).index.tolist()
    if labels_df is not None:
        sections["Cell Type Composition (Level 2)"] = build_composition_figure(
            labels_df, sort_order=sort_order, title="Cell Type Composition (Level 2)"
        )
    if regions_df is not None:
        sections["Brain Region Composition"] = build_composition_figure(
            regions_df, sort_order=sort_order, title="Brain Region Composition"
        )

    # Assemble HTML
    if output_path is None:
        output_path = data_dir / f"report_round{current_round}.html"

    return assemble_html_report(sections, output_path)
