"""
Export publication-quality matplotlib figures for the morphogen GP-BO paper.

Reads the same pipeline data files as visualize_report.py and produces
static PDF figures suitable for LaTeX inclusion.

Usage:
    cd /Users/maxxyung/Projects/engram-projects/morphogen-gpbo
    PYTHONPATH=. .venv/bin/python gopro/export_paper_figures.py

Output:
    paper_figures/fig2-fidelity-scores.pdf
    paper_figures/fig4a-morphogen-pca.pdf
    paper_figures/fig4b-morphogen-importance.pdf
    paper_figures/fig4e-leaderboard.pdf
    paper_figures/fig-supp-s1-composition.pdf
    paper_figures/fig-supp-s3-umap.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Style setup — matches zHenriquesLab template (Helvetica sans / Times body)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # TrueType — editable in Illustrator
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Column width for bioRxiv 2-column: ~88mm; full width: ~180mm
COL_WIDTH = 3.46  # inches (88mm)
FULL_WIDTH = 7.09  # inches (180mm)

# ---------------------------------------------------------------------------
# Data loading (mirrors visualize_report.py)
# ---------------------------------------------------------------------------

def load_data(data_dir: Path, prefix: str = "amin_kelley"):
    """Load all pipeline data files."""
    fidelity = pd.read_csv(data_dir / "fidelity_report.csv", index_col=0)
    morphogens = pd.read_csv(data_dir / f"morphogen_matrix_{prefix}.csv", index_col=0)

    # Find latest round
    rec_files = sorted(data_dir.glob("gp_recommendations_round*.csv"))
    if not rec_files:
        raise FileNotFoundError("No recommendation files found")
    latest = rec_files[-1]
    round_num = int(latest.stem.split("round")[-1])
    recs = pd.read_csv(latest, index_col=0)
    diag = pd.read_csv(data_dir / f"gp_diagnostics_round{round_num}.csv", index_col=0)

    labels_path = data_dir / f"gp_training_labels_{prefix}.csv"
    labels = pd.read_csv(labels_path, index_col=0) if labels_path.exists() else None

    return fidelity, morphogens, recs, diag, labels, round_num


# ---------------------------------------------------------------------------
# Fig 2: Fidelity scores bar chart
# ---------------------------------------------------------------------------

def fig2_fidelity(fidelity_df: pd.DataFrame, out: Path):
    """Composite fidelity scores colored by dominant brain region."""
    df = fidelity_df.sort_values("composite_fidelity", ascending=True)

    # Color by brain region
    region_col = None
    for c in ["dominant_region", "best_region", "region"]:
        if c in df.columns:
            region_col = c
            break

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.5))

    if region_col:
        regions = df[region_col].fillna("Unspecific")
        unique_regions = sorted(regions.unique())
        cmap = plt.cm.get_cmap("tab10", len(unique_regions))
        region_colors = {r: cmap(i) for i, r in enumerate(unique_regions)}
        colors = [region_colors[r] for r in regions]
        bars = ax.barh(range(len(df)), df["composite_fidelity"], color=colors)
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=region_colors[r], label=r) for r in unique_regions]
        ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=True,
                  title="Brain region", title_fontsize=7)
    else:
        ax.barh(range(len(df)), df["composite_fidelity"], color="steelblue")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=5.5)
    ax.set_xlabel("Composite fidelity")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 4a: Morphogen-space PCA
# ---------------------------------------------------------------------------

def fig4a_pca(morphogen_df: pd.DataFrame, fidelity_df: pd.DataFrame,
              recs: pd.DataFrame, out: Path):
    """PCA of morphogen space with recommendations as red stars."""
    # Drop zero-variance columns
    cols = morphogen_df.columns.tolist()
    vals = morphogen_df[cols].values
    col_var = vals.var(axis=0)
    active_mask = col_var > 1e-10
    active_cols = [c for c, m in zip(cols, active_mask) if m]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(morphogen_df[active_cols].values)

    # Zero-pad recs for missing columns
    recs_aligned = pd.DataFrame(0.0, index=recs.index, columns=cols)
    shared = [c for c in cols if c in recs.columns]
    recs_aligned[shared] = recs[shared]
    X_rec = scaler.transform(recs_aligned[active_cols].values)

    pca = PCA(n_components=2, random_state=42)
    train_pc = pca.fit_transform(X_train)
    rec_pc = pca.transform(X_rec)
    var_exp = pca.explained_variance_ratio_.sum() * 100

    # Align fidelity scores
    common = morphogen_df.index.intersection(fidelity_df.index)
    fid_vals = fidelity_df.loc[common, "composite_fidelity"]
    train_idx = [list(morphogen_df.index).index(c) for c in common]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.55, FULL_WIDTH * 0.45))

    # Training points
    sc = ax.scatter(
        train_pc[train_idx, 0], train_pc[train_idx, 1],
        c=fid_vals.values, cmap="viridis", s=30, edgecolors="white",
        linewidths=0.3, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label("Fidelity", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Recommendations
    ax.scatter(
        rec_pc[:, 0], rec_pc[:, 1],
        c="red", s=60, marker="*", edgecolors="black", linewidths=0.4,
        zorder=4, label="Recommended",
    )

    # Loading arrows (top 5)
    loadings = pca.components_
    importance = np.sqrt(loadings[0] ** 2 + loadings[1] ** 2)
    top_k = min(5, len(active_cols))
    top_idx = importance.argsort()[-top_k:][::-1]
    scale = max(abs(train_pc[:, 0]).max(), abs(train_pc[:, 1]).max()) * 0.6
    for idx in top_idx:
        dx = loadings[0, idx] * scale
        dy = loadings[1, idx] * scale
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        )
        ax.text(dx * 1.12, dy * 1.12, active_cols[idx].replace("_", " "),
                fontsize=5.5, color="gray", ha="center", va="center")

    # Condition labels for outliers only
    for i, cond in enumerate(common):
        pc1, pc2 = train_pc[train_idx[i], 0], train_pc[train_idx[i], 1]
        if abs(pc1) > 3 or abs(pc2) > 3:
            ax.text(pc1, pc2 + 0.3, cond, fontsize=4, ha="center", va="bottom",
                    color="dimgray")

    ax.set_xlabel(f"PC 1")
    ax.set_ylabel(f"PC 2")
    ax.set_title("Round 1: Morphogen Space and GP-BO Recommendations")
    ax.legend(fontsize=6, loc="lower left")

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 4b: Morphogen importance
# ---------------------------------------------------------------------------

def fig4b_importance(diagnostics: pd.DataFrame, morphogen_df: pd.DataFrame,
                     out: Path):
    """Horizontal bar chart of morphogen importance."""
    # Lengthscales are columns like "lengthscale_CHIR99021_uM"
    ls_cols = [c for c in diagnostics.columns if "lengthscale" in c.lower()]

    if ls_cols:
        # Use last row (latest round)
        row = diagnostics.iloc[-1]
        names = [c.replace("lengthscale_", "").replace("_uM", "").replace("_", " ")
                 for c in ls_cols]
        values = 1.0 / row[ls_cols].values.astype(float)
        title = "Morphogen Importance (1/lengthscale)"
    else:
        # Fallback: coefficient of variation
        nonzero = morphogen_df.columns[morphogen_df.var() > 1e-10]
        cv = {}
        for col in nonzero:
            vals = morphogen_df[col]
            used = vals[vals.abs() > 1e-6]
            if len(used) >= 2 and used.mean() > 0:
                cv[col] = used.std() / used.mean()
        s = pd.Series(cv).sort_values(ascending=True)
        names = [n.replace("_", " ") for n in s.index]
        values = s.values
        title = "Morphogen Exploration Diversity"

    # Sort ascending for horizontal bar
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = values[order]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, max(2.5, len(names) * 0.22)))
    ax.barh(range(len(names)), values, color="steelblue", height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Importance")
    ax.set_title(title)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig 4e: Leaderboard table
# ---------------------------------------------------------------------------

def fig4e_leaderboard(fidelity_df: pd.DataFrame, out: Path, top_n: int = 10):
    """Top conditions table rendered as a matplotlib figure."""
    top = fidelity_df.nlargest(top_n, "composite_fidelity")
    display_cols = ["composite_fidelity", "rss_score", "on_target_fraction",
                    "off_target_fraction"]
    available = [c for c in display_cols if c in top.columns]
    col_labels = ["Condition"] + [c.replace("_", " ").title() for c in available]

    cell_text = []
    for cond in top.index:
        row = [cond] + [f"{top.loc[cond, c]:.3f}" for c in available]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 0.3 * (top_n + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1, 1.3)
    # Header styling
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("steelblue")
        cell.set_text_props(color="white", fontweight="bold")

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig S1: Cell type composition stacked bars
# ---------------------------------------------------------------------------

def figs1_composition(labels_df: pd.DataFrame, fidelity_df: pd.DataFrame,
                      out: Path):
    """Stacked bar chart of cell type fractions per condition."""
    # Sort by fidelity
    common = labels_df.index.intersection(fidelity_df.index)
    order = fidelity_df.loc[common].sort_values("composite_fidelity", ascending=False).index
    df = labels_df.loc[order]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4))

    n_types = df.shape[1]
    cmap = plt.cm.get_cmap("tab20", n_types)
    bottom = np.zeros(len(df))
    x = np.arange(len(df))

    for i, col in enumerate(df.columns):
        vals = df[col].values.astype(float)
        ax.bar(x, vals, bottom=bottom, color=cmap(i), label=col, width=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=60, ha="right", fontsize=4.5)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cell Type Composition per Condition (ordered by fidelity)")
    ax.legend(fontsize=5, ncol=3, loc="upper right", bbox_to_anchor=(1.0, -0.15),
              frameon=False)

    fig.subplots_adjust(bottom=0.3)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig S3: Cell-space UMAP
# ---------------------------------------------------------------------------

def figs3_umap(h5ad_path: Path, out: Path):
    """UMAP of individual cells colored by predicted cell type."""
    import anndata

    adata = anndata.read_h5ad(str(h5ad_path), backed="r")

    # Find UMAP coordinates
    umap_key = None
    for key in ["X_umap", "X_umap_hnoca"]:
        if key in adata.obsm:
            umap_key = key
            break
    if umap_key is None:
        for key in adata.obsm:
            if "umap" in key.lower():
                umap_key = key
                break
    if umap_key is None:
        print(f"  WARNING: No UMAP found in {h5ad_path}, skipping Fig S3")
        return

    coords = adata.obsm[umap_key][:]

    # Cell type labels
    ct_col = None
    for c in ["predicted_annot_level_2", "annot_level_2", "cell_type"]:
        if c in adata.obs.columns:
            ct_col = c
            break
    cell_types = adata.obs[ct_col].values if ct_col else np.full(len(adata), "Unknown")

    unique_types = sorted(set(cell_types))
    n_types = len(unique_types)
    cmap = plt.cm.get_cmap("tab20", max(n_types, 20))
    type_colors = {t: cmap(i) for i, t in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, FULL_WIDTH * 0.5))

    for i, ct in enumerate(unique_types):
        mask = cell_types == ct
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[type_colors[ct]], s=0.5, alpha=0.4, label=ct, rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Cell Space (UMAP)")
    ax.legend(fontsize=5, markerscale=6, loc="center left", bbox_to_anchor=(1, 0.5),
              frameon=False, scatterpoints=1)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Fig S2: GP diagnostics (lengthscale + posterior uncertainty)
# ---------------------------------------------------------------------------

def figs2_gp_diagnostics(diagnostics: pd.DataFrame, morphogen_df: pd.DataFrame,
                         out: Path):
    """GP diagnostics: ARD lengthscales with importance ranking."""
    ls_cols = [c for c in diagnostics.columns if "lengthscale" in c.lower()]
    if not ls_cols:
        print(f"  WARNING: No lengthscale columns in diagnostics, skipping Fig S2")
        return

    row = diagnostics.iloc[-1]
    names_raw = [c.replace("lengthscale_", "") for c in ls_cols]
    ls_values = row[ls_cols].values.astype(float)

    # Sort by lengthscale ascending (most important first)
    order = np.argsort(ls_values)
    names = [names_raw[i].replace("_uM", "").replace("_", " ") for i in order]
    ls_sorted = ls_values[order]
    imp_sorted = 1.0 / ls_sorted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5))

    # Left: lengthscales
    colors = ["#2166ac" if ls < 1.0 else "#92c5de" if ls < 2.0 else "#d1e5f0"
              for ls in ls_sorted]
    ax1.barh(range(len(names)), ls_sorted, color=colors, height=0.7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("ARD Lengthscale")
    ax1.set_title("GP Lengthscales (shorter = more important)")
    ax1.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Right: importance (1/lengthscale)
    ax2.barh(range(len(names)), imp_sorted, color=colors, height=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=6)
    ax2.set_xlabel("Importance (1/lengthscale)")
    ax2.set_title("Morphogen Importance Ranking")

    # Annotate key stats
    n_train = int(row.get("n_training_points", 0))
    n_dims = int(row.get("n_morphogens", 0))
    post_std = row.get("mean_posterior_std", float("nan"))
    if n_train > 0:
        ax2.text(0.95, 0.05, f"N={n_train}, d={n_dims}\n$\\sigma_{{post}}$={post_std:.2f}",
                 transform=ax2.transAxes, fontsize=6, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path("data")
    out_dir = Path("paper_figures")
    out_dir.mkdir(exist_ok=True)
    prefix = "amin_kelley"

    print("Loading data...")
    fidelity, morphogens, recs, diag, labels, round_num = load_data(data_dir, prefix)
    print(f"  Round {round_num}: {len(fidelity)} conditions, {len(recs)} recommendations")

    print("\nGenerating figures:")

    print("  Fig 2: Fidelity scores")
    fig2_fidelity(fidelity, out_dir / "fig2-fidelity-scores.pdf")

    print("  Fig 4a: Morphogen PCA")
    fig4a_pca(morphogens, fidelity, recs, out_dir / "fig4a-morphogen-pca.pdf")

    print("  Fig 4b: Morphogen importance")
    fig4b_importance(diag, morphogens, out_dir / "fig4b-morphogen-importance.pdf")

    print("  Fig 4e: Leaderboard")
    fig4e_leaderboard(fidelity, out_dir / "fig4e-leaderboard.pdf")

    if labels is not None:
        print("  Fig S1: Cell type composition")
        figs1_composition(labels, fidelity, out_dir / "fig-supp-s1-composition.pdf")
    else:
        print("  Fig S1: SKIPPED (no labels file)")

    print("  Fig S2: GP diagnostics")
    figs2_gp_diagnostics(diag, morphogens, out_dir / "fig-supp-s2-gp-diagnostics.pdf")

    h5ad_path = data_dir / f"{prefix}_mapped.h5ad"
    if h5ad_path.exists():
        print("  Fig S3: Cell UMAP")
        figs3_umap(h5ad_path, out_dir / "fig-supp-s3-umap.pdf")
    else:
        print(f"  Fig S3: SKIPPED ({h5ad_path} not found)")

    print(f"\nDone. Figures in {out_dir.resolve()}")
    print(f"\nCopy to paper:")
    print(f"  cp {out_dir}/*.pdf /Users/maxxyung/Projects/engram-projects/latex-papers/morphogen-gbpo/paper/figures/")


if __name__ == "__main__":
    main()
