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

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from gopro.config import (
    FIDELITY_DROP_R2_THRESHOLD,
    FIDELITY_SKIP_R2_THRESHOLD,
    get_logger,
)

logger = get_logger(__name__)


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
    rounds = []
    for f in data_dir.glob("gp_recommendations_round*.csv"):
        m = re.search(r"round(\d+)\.csv$", f.name)
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
        importance = {k: 1.0 / v for k, v in lengthscales.items() if v >= 1e-6}
        top3 = sorted(importance, key=importance.get, reverse=True)[:3]
        top3_str = ", ".join(top3)
        parts.append(f"Top morphogens by GP importance: {top3_str}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Panel A: Morphogen-space PCA (PCA is more appropriate than UMAP for N<100)
# ---------------------------------------------------------------------------

def compute_morphogen_pca_with_recommendations(
    morphogen_df: pd.DataFrame,
    recs_df: pd.DataFrame,
    morphogen_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float, list[str]]:
    """Compute PCA on training data and project recommendations into same space.

    Drops zero-variance columns before fitting. Returns PC coordinates plus
    the PCA loadings and variance explained for axis annotation.

    Args:
        morphogen_df: Training morphogen matrix.
        recs_df: Recommendations DataFrame with morphogen columns.
        morphogen_cols: Columns to use from both DataFrames.

    Returns:
        Tuple of (training_coords, rec_coords, loadings, variance_explained_pct, active_cols).
        loadings has shape (n_components, n_features); active_cols lists the non-zero-variance
        column names used.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Drop zero-variance columns (e.g., SHH=0, XAV939=0, constant log_harvest_day)
    train_vals = morphogen_df[morphogen_cols].values
    col_var = train_vals.var(axis=0)
    active_mask = col_var > 1e-10
    active_cols = [c for c, m in zip(morphogen_cols, active_mask) if m]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(morphogen_df[active_cols].values)
    X_rec = scaler.transform(recs_df[active_cols].values)

    pca = PCA(n_components=2, random_state=42)
    train_coords = pca.fit_transform(X_train)
    rec_coords = pca.transform(X_rec)

    var_explained = pca.explained_variance_ratio_.sum() * 100

    train_df = pd.DataFrame(
        train_coords, columns=["PC1", "PC2"], index=morphogen_df.index
    )
    rec_df = pd.DataFrame(
        rec_coords, columns=["PC1", "PC2"], index=recs_df.index
    )
    return train_df, rec_df, pca.components_, var_explained, active_cols


def build_morphogen_pca_figure(
    coords: pd.DataFrame,
    fidelity_scores: pd.Series,
    rec_coords: pd.DataFrame | None = None,
    rec_morphogens: pd.DataFrame | None = None,
    loadings: np.ndarray | None = None,
    active_cols: list[str] | None = None,
    var_explained: float = 0.0,
) -> go.Figure:
    """Build morphogen-space PCA scatter plot with optional loading arrows.

    Args:
        coords: Training condition PCA coordinates (PC1, PC2).
        fidelity_scores: Series of composite fidelity per condition.
        rec_coords: Optional recommendation PCA coordinates (shown as stars).
        loadings: PCA loadings array of shape (2, n_features).
        active_cols: Feature names corresponding to loadings columns.
        var_explained: Total variance explained by PC1+PC2 (percentage).

    Returns:
        Plotly Figure.
    """
    common = coords.index.intersection(fidelity_scores.index)
    coords_aligned = coords.loc[common]
    scores_aligned = fidelity_scores.loc[common]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords_aligned["PC1"],
        y=coords_aligned["PC2"],
        mode="markers+text",
        marker=dict(
            size=10,
            color=scores_aligned.values,
            colorscale=VIRIDIS,
            colorbar=dict(title="Fidelity"),
            showscale=True,
        ),
        text=coords_aligned.index,
        textposition="top center",
        textfont=dict(size=7),
        hovertemplate="<b>%{text}</b><br>Fidelity: %{marker.color:.3f}<extra></extra>",
        name="Training",
    ))

    if rec_coords is not None and len(rec_coords) > 0:
        # Build informative hover text showing key morphogen concentrations
        rec_hover = []
        for well in rec_coords.index:
            parts = [f"<b>{well}</b> (recommended)"]
            if rec_morphogens is not None and well in rec_morphogens.index:
                row = rec_morphogens.loc[well]
                # Show non-zero morphogens sorted by concentration
                _skip = {"fidelity", "log_harvest_day", "acquisition_value"}
                nonzero = [(col, row[col]) for col in rec_morphogens.columns
                           if abs(row[col]) > 0.001
                           and col not in _skip
                           and not col.startswith("predicted_")]
                nonzero.sort(key=lambda x: x[1], reverse=True)
                for col_name, val in nonzero[:8]:  # top 8 morphogens
                    label = col_name.replace("_uM", "").replace("_", " ")
                    parts.append(f"{label}: {val:.1f} \u00b5M")
            rec_hover.append("<br>".join(parts))

        fig.add_trace(go.Scatter(
            x=rec_coords["PC1"],
            y=rec_coords["PC2"],
            mode="markers",
            marker=dict(
                size=14,
                color="red",
                symbol="star",
                line=dict(width=1, color="black"),
            ),
            text=rec_hover,
            hovertemplate="%{text}<extra></extra>",
            name="Recommended",
        ))

    # Add top loading arrows as a biplot overlay
    if loadings is not None and active_cols is not None:
        importance = np.sqrt(loadings[0] ** 2 + loadings[1] ** 2)
        top_k = min(5, len(active_cols))
        top_idx = importance.argsort()[-top_k:][::-1]
        # Scale arrows to ~30% of plot range
        scale = max(
            coords_aligned["PC1"].abs().max(),
            coords_aligned["PC2"].abs().max(),
        ) * 0.6
        for idx in top_idx:
            dx = loadings[0, idx] * scale
            dy = loadings[1, idx] * scale
            fig.add_annotation(
                x=dx, y=dy, ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2, arrowsize=1.5, arrowwidth=1.5,
                arrowcolor="rgba(100,100,100,0.7)",
            )
            fig.add_annotation(
                x=dx * 1.12, y=dy * 1.12,
                text=active_cols[idx].replace("_", " "),
                showarrow=False,
                font=dict(size=9, color="gray"),
            )

    fig.update_layout(
        title=f"Morphogen Space (PCA, {var_explained:.0f}% variance explained)",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
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
        # Compute UMAP from scPoli latent space if available
        latent_key = None
        for key in adata.obsm:
            if "scpoli" in key.lower():
                latent_key = key
                break
        if latent_key is None:
            return None

        import scanpy as sc
        # Load into memory for UMAP computation
        adata = adata.to_memory()
        sc.pp.neighbors(adata, use_rep=latent_key, n_neighbors=15)
        sc.tl.umap(adata, random_state=42)
        umap_key = "X_umap"

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
    import plotly.express as px

    fig = go.Figure()

    unique_types = sorted(cell_types.unique())
    # Use a large qualitative palette to avoid color repetition with 14+ cell types
    palette = (
        px.colors.qualitative.Dark24
        if len(unique_types) > 10
        else px.colors.qualitative.Plotly
    )
    for i, ct in enumerate(unique_types):
        mask = cell_types == ct
        fig.add_trace(go.Scattergl(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            marker=dict(size=2, opacity=0.5, color=palette[i % len(palette)]),
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

    # Determine color values and normalize acquisition values to 0-1
    use_acq = predicted_fidelity is None and "acquisition_value" in recommendations.columns
    if use_acq:
        acq_vals = recommendations["acquisition_value"].values
        acq_min, acq_max = acq_vals.min(), acq_vals.max()
        acq_range = acq_max - acq_min if acq_max > acq_min else 1.0

    for well in recommendations.index:
        m = re.match(r"([A-D])(\d+)", well)
        if not m:
            continue
        r = rows.index(m.group(1))
        c = int(m.group(2)) - 1

        if predicted_fidelity is not None and well in predicted_fidelity.index:
            z[r, c] = predicted_fidelity[well]
        elif use_acq:
            # Normalize to 0-1 for interpretability
            raw = recommendations.loc[well, "acquisition_value"]
            z[r, c] = (raw - acq_min) / acq_range

        # Build hover text with non-zero morphogens
        morph_cols = [col for col in recommendations.columns
                      if not col.startswith("predicted_")
                      and col not in ("acquisition_value", "fidelity", "log_harvest_day")]
        nonzero = []
        for mc in morph_cols:
            val = recommendations.loc[well, mc]
            if abs(val) > 0.001:
                label = mc.replace("_uM", "").replace("_", " ")
                nonzero.append(f"{label}: {val:.1f} \u00b5M")
        hover[r][c] = f"<b>{well}</b><br>" + "<br>".join(nonzero) if nonzero else f"<b>{well}</b>"

        annotations.append(dict(
            x=c, y=r, text=well, showarrow=False,
            font=dict(color="white", size=12),
        ))

    colorbar_title = "Predicted Fidelity" if predicted_fidelity is not None else "Acquisition Score"
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[str(c) for c in cols],
        y=rows,
        colorscale=VIRIDIS,
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        colorbar=dict(title=colorbar_title),
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
    # Non-morphogen columns to exclude from importance chart
    _EXCLUDE_COLS = {"fidelity", "log_harvest_day"}

    if lengthscales:
        importance = {
            k: 1.0 / v
            for k, v in lengthscales.items()
            if v >= 1e-6 and k not in _EXCLUDE_COLS
        }
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_items]
        values = [x[1] for x in sorted_items]
        title = "Morphogen Importance (1/lengthscale, min across output tasks)"
    elif morphogen_df is not None:
        # Use coefficient of variation (std/mean) to measure diversity per morphogen,
        # which is scale-invariant (nM vs uM doesn't matter).
        # Drop zero-variance and all-zero columns.
        nonzero_cols = morphogen_df.columns[morphogen_df.var() > 1e-10]
        if len(nonzero_cols) == 0:
            return _placeholder_figure("All morphogen columns have zero variance")
        # For each morphogen: fraction of conditions that use it (non-zero),
        # weighted by the coefficient of variation among users
        usage = {}
        for col in nonzero_cols:
            vals = morphogen_df[col]
            n_used = (vals.abs() > 1e-6).sum()
            if n_used >= 2:
                # CV among conditions that actually use this morphogen
                used_vals = vals[vals.abs() > 1e-6]
                cv = used_vals.std() / used_vals.mean() if used_vals.mean() > 0 else 0
                # Combine usage fraction and CV into an importance proxy
                usage[col] = (n_used / len(vals)) * (1 + cv)
            else:
                usage[col] = n_used / len(vals)
        importance = pd.Series(usage)
        if importance.max() > 0:
            importance = importance / importance.max()
        sorted_imp = importance.sort_values(ascending=False)
        names = list(sorted_imp.index)
        values = list(sorted_imp.values)
        title = "Morphogen Exploration Diversity (lengthscales unavailable)"
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


def build_fidelity_trend_figure(
    monitoring_df: pd.DataFrame,
) -> go.Figure:
    """Build fidelity correlation trend line chart across rounds.

    Args:
        monitoring_df: DataFrame with columns ``round``, ``fidelity_label``,
            ``overall_correlation``, ``recommendation``.

    Returns:
        Plotly Figure with one trace per fidelity source.
    """
    fig = go.Figure()

    colors = {"CellRank2": "darkorange", "CellFlow": "mediumpurple"}
    for label in monitoring_df["fidelity_label"].unique():
        sub = monitoring_df[monitoring_df["fidelity_label"] == label].sort_values("round")
        color = colors.get(label, "gray")
        fig.add_trace(go.Scatter(
            x=sub["round"].tolist(),
            y=sub["overall_correlation"].tolist(),
            mode="lines+markers",
            name=label,
            marker=dict(size=8, color=color),
            line=dict(width=2, color=color),
        ))

    # Threshold lines
    fig.add_hline(y=FIDELITY_SKIP_R2_THRESHOLD, line_dash="dash", line_color="green",
                  annotation_text=f"Skip MF-BO (R²>{FIDELITY_SKIP_R2_THRESHOLD})")
    fig.add_hline(y=FIDELITY_DROP_R2_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"Drop source (R²<{FIDELITY_DROP_R2_THRESHOLD})")

    fig.update_layout(
        title="Cross-Fidelity Correlation by Round",
        xaxis_title="Round",
        yaxis_title="R² (Coefficient of Determination)",
        yaxis=dict(range=[-0.1, 1.05]),
        xaxis=dict(dtick=1),
        template=PLOTLY_TEMPLATE,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )
    return fig


# ---------------------------------------------------------------------------
# Convergence diagnostics chart
# ---------------------------------------------------------------------------

def build_convergence_diagnostics_figure(
    conv_df: pd.DataFrame,
) -> go.Figure:
    """Build convergence diagnostics multi-panel figure.

    Plots three convergence metrics across rounds on a shared x-axis:
    mean posterior std, max acquisition value, and recommendation spread.

    Args:
        conv_df: DataFrame with columns ``round``, ``mean_posterior_std``,
            ``max_acquisition_value``, ``recommendation_spread``.

    Returns:
        Plotly Figure with three y-axes.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Mean Posterior Std",
            "Max Acquisition Value",
            "Recommendation Spread",
        ),
        vertical_spacing=0.08,
    )

    conv_df = conv_df.sort_values("round")
    rounds = conv_df["round"].tolist()

    metrics = [
        ("mean_posterior_std", "steelblue", "Posterior Std"),
        ("max_acquisition_value", "darkorange", "Max Acq Value"),
        ("recommendation_spread", "mediumseagreen", "Rec Spread"),
    ]
    for row_idx, (col, color, name) in enumerate(metrics, start=1):
        if col in conv_df.columns:
            fig.add_trace(go.Scatter(
                x=rounds,
                y=conv_df[col].tolist(),
                mode="lines+markers",
                marker=dict(size=8, color=color),
                line=dict(width=2, color=color),
                name=name,
            ), row=row_idx, col=1)

    fig.update_layout(
        title="Convergence Diagnostics",
        height=600,
        showlegend=False,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_xaxes(title_text="Round", dtick=1, row=3, col=1)

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
    sections: dict[str, tuple[str, go.Figure | str]],
    output_path: Path,
) -> Path:
    """Assemble final HTML report from sections.

    Args:
        sections: Ordered dict of section_name -> (description, Figure_or_HTML).
            description is a short explanation shown below the heading.
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
  .description { color: #666; font-size: 0.95em; margin: 4px 0 12px 0; line-height: 1.5; }
  .summary { background: #f8f9fa; padding: 15px 20px; border-radius: 6px;
             border-left: 4px solid steelblue; margin: 20px 0; font-size: 1.05em; }
  .section { margin-bottom: 40px; }
  .plotly-chart { margin: 10px 0; }
</style>
</head>
<body>
<h1>GP-BO Visualization Report</h1>
""")

    for name, (description, content) in sections.items():
        html_parts.append(f'<div class="section">')
        html_parts.append(f'<h2>{name}</h2>')
        if description:
            html_parts.append(f'<p class="description">{description}</p>')
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
    output_prefix: str = "amin_kelley",
) -> Path:
    """Generate the full GP-BO visualization report.

    Orchestrates: discover rounds → load data → build figures → assemble HTML.

    Args:
        data_dir: Directory containing pipeline outputs.
        output_path: Override for output HTML path. Default: data_dir/report_round{N}.html.
        output_prefix: Dataset prefix for CSV filenames (default: "amin_kelley").

    Returns:
        Path to the generated HTML report.
    """
    data_dir = Path(data_dir)
    logger.info("Generating report from %s", data_dir)

    # Discover rounds
    rounds = discover_rounds(data_dir)
    if not rounds:
        raise FileNotFoundError(
            f"No gp_recommendations_round*.csv found in {data_dir}"
        )
    current_round = max(rounds)
    logger.info("Found %d round(s), latest is round %d", len(rounds), current_round)

    # Load data
    logger.info("Loading pipeline data files")
    fidelity_df = load_fidelity_report(data_dir / "fidelity_report.csv")
    morphogen_df = load_morphogen_matrix(data_dir / f"morphogen_matrix_{output_prefix}.csv")
    recs = load_recommendations(data_dir / f"gp_recommendations_round{current_round}.csv")
    diagnostics = load_diagnostics(data_dir / f"gp_diagnostics_round{current_round}.csv")

    # Load cell type fractions
    labels_path = data_dir / f"gp_training_labels_{output_prefix}.csv"
    labels_df = load_cell_type_fractions(labels_path) if labels_path.exists() else None

    regions_path = data_dir / f"gp_training_regions_{output_prefix}.csv"
    regions_df = load_cell_type_fractions(regions_path) if regions_path.exists() else None

    # Sections: dict of name -> (description, content)
    sections: dict[str, tuple[str, go.Figure | str]] = {}

    # 1. Summary text
    logger.info("Building section: Summary")
    summary = generate_summary_text(fidelity_df, diagnostics, len(recs))
    sections["Summary"] = (
        "",
        summary,
    )

    # 2. Convergence trend — cumulative best up to each round
    logger.info("Building section: Convergence")
    best_per_round = {}
    if "round" in fidelity_df.columns:
        cumulative_best = -np.inf
        for r in rounds:
            round_mask = fidelity_df["round"] <= r
            if round_mask.any():
                cumulative_best = fidelity_df.loc[round_mask, "composite_fidelity"].max()
            best_per_round[r] = cumulative_best
    else:
        # Single-round fallback: all data belongs to latest round
        for r in rounds:
            best_per_round[r] = fidelity_df["composite_fidelity"].max()
    sections["Convergence"] = (
        "Best composite fidelity score achieved across optimization rounds. "
        "Each point shows the highest fidelity seen up to that round. "
        "With only round 1 data, this shows a single point; the trend emerges as more rounds are run.",
        build_convergence_figure(best_per_round),
    )

    # 2b. Fidelity monitoring trend (if multi-fidelity history exists)
    fidelity_monitor_path = data_dir / "fidelity_monitoring.csv"
    if fidelity_monitor_path.exists():
        logger.info("Building section: Fidelity Monitoring")
        try:
            monitor_df = pd.read_csv(fidelity_monitor_path)
            if len(monitor_df) > 0:
                sections["Fidelity Monitoring"] = (
                    "Cross-fidelity correlation between real and virtual data sources "
                    "across optimization rounds. Sustained decline triggers auto-fallback "
                    "to single-fidelity GP. Green dashed line: correlation too high for "
                    "MF-BO benefit. Red dashed line: correlation too low — source dropped.",
                    build_fidelity_trend_figure(monitor_df),
                )
        except Exception as e:
            logger.warning("Fidelity monitoring section failed: %s", e, exc_info=True)

    # 3. Morphogen-space PCA — use all 20 training dimensions,
    #    zero-pad recommendations for missing columns
    logger.info("Building section: Morphogen Space PCA")
    try:
        all_morph_cols = list(morphogen_df.columns)
        if all_morph_cols and len(morphogen_df) > 2:
            # Build a rec frame with all 20 morphogen cols, zero for missing ones
            recs_full = pd.DataFrame(0.0, index=recs.index, columns=all_morph_cols)
            shared_cols = [c for c in all_morph_cols if c in recs.columns]
            recs_full[shared_cols] = recs[shared_cols]

            train_coords, rec_coords, loadings, var_pct, active_cols = (
                compute_morphogen_pca_with_recommendations(
                    morphogen_df, recs_full, all_morph_cols
                )
            )
            sections["Morphogen Space"] = (
                "PCA projection of the 20-dimensional morphogen concentration space "
                "(zero-variance columns excluded). Each dot is a tested condition, colored by "
                "composite fidelity. Red stars are GP-recommended next experiments. "
                "Gray arrows show the top 5 morphogen loadings driving separation along PC1/PC2. "
                "Note: GP bounds extend beyond the training data range, so recommendations "
                "may appear outside the training cluster — this is expected exploration behavior.",
                build_morphogen_pca_figure(
                    train_coords, fidelity_df["composite_fidelity"], rec_coords,
                    rec_morphogens=recs,
                    loadings=loadings, active_cols=active_cols,
                    var_explained=var_pct,
                ),
            )
        else:
            sections["Morphogen Space"] = (
                "", _placeholder_figure("Insufficient data for morphogen PCA"),
            )
    except Exception as e:
        logger.warning("Morphogen Space PCA failed: %s", e)
        sections["Morphogen Space"] = (
            "", _placeholder_figure(f"PCA error: {e}"),
        )

    # 4. Cell-space UMAP
    logger.info("Building section: Cell Space UMAP")
    h5ad_path = data_dir / f"{output_prefix}_mapped.h5ad"
    cell_umap_desc = (
        "UMAP of individual cells from the scRNA-seq data, colored by predicted cell type "
        "(transferred from the HNOCA reference atlas via scPoli/KNN). "
        "Computed from the 30-dimensional scPoli latent space."
    )
    if h5ad_path.exists():
        try:
            result = extract_cell_umap_from_h5ad(h5ad_path)
            if result is not None:
                coords, cell_types, conditions = result
                sections["Cell Space UMAP"] = (
                    cell_umap_desc,
                    build_cell_umap_figure(coords, cell_types, conditions),
                )
            else:
                sections["Cell Space UMAP"] = (
                    cell_umap_desc,
                    _placeholder_figure("No UMAP coordinates in h5ad file"),
                )
        except Exception as e:
            logger.warning("Cell Space UMAP failed: %s", e)
            sections["Cell Space UMAP"] = (
                cell_umap_desc,
                _placeholder_figure(f"h5ad error: {e}"),
            )
    else:
        sections["Cell Space UMAP"] = (
            cell_umap_desc,
            _placeholder_figure(f"{output_prefix}_mapped.h5ad not found"),
        )

    # 5. Plate map
    logger.info("Building section: Plate Map")
    sections["Plate Map"] = (
        "24-well plate layout for the next recommended experiment. "
        "Hover over each well to see the exact morphogen concentrations. "
        "Color indicates predicted fidelity or acquisition function value.",
        build_plate_map_figure(recs),
    )

    # 6. Morphogen importance
    logger.info("Building section: Morphogen Importance")
    lengthscales = diagnostics.get("lengthscales")
    if lengthscales:
        imp_desc = (
            "Morphogen importance derived from the GP ARD kernel (1/lengthscale). "
            "A shorter lengthscale means the GP output changes more rapidly along that "
            "dimension — i.e., that morphogen has more influence on cell type composition."
        )
    else:
        imp_desc = (
            "Morphogen exploration diversity (GP lengthscales unavailable). "
            "Shows how broadly each morphogen was varied across conditions, "
            "combining usage frequency and concentration variation. "
            "Not a measure of biological importance — re-run step 04 to get GP-learned importance."
        )
    sections["Morphogen Importance"] = (
        imp_desc,
        build_importance_figure(lengthscales=lengthscales, morphogen_df=morphogen_df),
    )

    # 7. Leaderboard
    logger.info("Building section: Leaderboard")
    sections["Leaderboard"] = (
        "Top conditions ranked by composite fidelity. Composite fidelity combines: "
        "<b>on_target_fraction</b> (fraction of cells assigned to the dominant brain region), "
        "<b>off_target_fraction</b> (fraction in undesired regions — lower is better), and "
        "<b>rss_score</b> (cosine similarity of cell-class composition vs. Braun fetal brain reference).",
        build_leaderboard_figure(fidelity_df),
    )

    # 8. Cell type composition
    logger.info("Building section: Cell Type Composition")
    sort_order = fidelity_df.sort_values("composite_fidelity", ascending=False).index.tolist()
    if labels_df is not None:
        sections["Cell Type Composition (Level 2)"] = (
            "Stacked bar chart showing the fraction of each cell type per condition, "
            "from HNOCA level-2 annotations (e.g., Dorsal Telencephalic Neuron, Ventral Telencephalic NPC). "
            "Conditions are sorted left-to-right by decreasing composite fidelity.",
            build_composition_figure(
                labels_df, sort_order=sort_order, title="Cell Type Composition (Level 2)"
            ),
        )
    if regions_df is not None:
        sections["Brain Region Composition"] = (
            "Brain region assignment per condition, based on HNOCA region annotations. "
            "Many conditions with minimal morphogen input (e.g., LDN, IWP2, DAPT, FGF4, FGF8) show "
            "~95-100% dorsal telencephalon — this is biologically expected as the default neural "
            "differentiation trajectory trends cortical. Conditions with CHIR+SAG or RA drive "
            "non-telencephalic fates (midbrain, cerebellum, hypothalamus).",
            build_composition_figure(
                regions_df, sort_order=sort_order, title="Brain Region Composition"
            ),
        )

    # Assemble HTML
    if output_path is None:
        output_path = data_dir / f"report_round{current_round}.html"

    result = assemble_html_report(sections, output_path)
    logger.info("Report written to %s", result)
    return result
