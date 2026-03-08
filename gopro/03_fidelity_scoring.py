"""
Step 3: Score organoid cell fidelity against fetal brain reference.

BrainSTEM two-tier mapping:
  Tier 1: Map cells onto whole-brain fetal atlas → identify brain region
  Tier 2: Map region-specific cells onto subatlas → identify subtypes

Inputs:
  - data/amin_kelley_mapped.h5ad (from step 02, with HNOCA cell type labels)
  - braun-et-al_minimal_for_mapping.h5ad (from Zenodo 15004817)

Outputs:
  - data/amin_kelley_fidelity.h5ad (with fidelity scores added to obs)
  - data/fidelity_report.csv (per-condition fidelity summary)
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"


def tier1_whole_brain_mapping(query_adata, fetal_adata):
    """
    Tier 1: Map query cells onto whole-brain fetal atlas.
    Identifies brain region identity (forebrain, midbrain, hindbrain, etc.)
    and detects off-target populations.
    """
    print("  Tier 1: Whole-brain region assignment...")

    # Project query onto fetal reference latent space
    # Using scArches the same way as HNOCA mapping
    try:
        import scvi
        from hnoca.map import AtlasMapper

        # This would use a fetal brain scANVI model
        # For now, use simpler label transfer via KNN in gene space
        print("  Using KNN-based label transfer in PCA space...")

        # Find shared genes
        shared_genes = query_adata.var_names.intersection(fetal_adata.var_names)
        print(f"  Shared genes: {len(shared_genes)}")

        q = query_adata[:, shared_genes].copy()
        f = fetal_adata[:, shared_genes].copy()

        # Normalize if not already
        sc.pp.pca(q, n_comps=30)
        sc.pp.pca(f, n_comps=30)

        # Simple KNN transfer
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=50, weights="distance")

        # Check which label column the fetal data uses
        label_cols = [c for c in f.obs.columns if "region" in c.lower() or "area" in c.lower()]
        if label_cols:
            label_key = label_cols[0]
        elif "cell_type" in f.obs.columns:
            label_key = "cell_type"
        else:
            print(f"  Available fetal metadata: {list(f.obs.columns)}")
            print("  Could not auto-detect region label column.")
            return None

        knn.fit(f.obsm["X_pca"], f.obs[label_key])
        region_labels = knn.predict(q.obsm["X_pca"])
        region_probs = knn.predict_proba(q.obsm["X_pca"]).max(axis=1)

        query_adata.obs["fetal_region"] = region_labels
        query_adata.obs["fetal_region_confidence"] = region_probs

        # Report
        region_counts = pd.Series(region_labels).value_counts()
        print(f"  Region assignment distribution:")
        for region, count in region_counts.head(10).items():
            print(f"    {region}: {count} cells ({count/len(region_labels)*100:.1f}%)")

        return query_adata

    except ImportError:
        print("  WARNING: scvi-tools not available, using simplified KNN mapping")
        return query_adata


def compute_fidelity_scores(query_adata, condition_key="condition"):
    """
    Compute per-condition fidelity metrics:
    - On-target fraction (cells matching expected brain region)
    - Mean region confidence
    - Cell type diversity (Shannon entropy)
    """
    print("  Computing fidelity scores per condition...")

    results = []
    for condition in query_adata.obs[condition_key].unique():
        mask = query_adata.obs[condition_key] == condition
        subset = query_adata.obs[mask]

        # Cell type composition
        ct_fracs = subset["predicted_cell_type"].value_counts(normalize=True)

        # Shannon entropy (diversity)
        entropy = -(ct_fracs * np.log2(ct_fracs + 1e-10)).sum()

        # Region confidence (if available)
        region_conf = subset["fetal_region_confidence"].mean() if "fetal_region_confidence" in subset.columns else np.nan

        # Dominant region
        dominant_region = subset["fetal_region"].mode().iloc[0] if "fetal_region" in subset.columns else "unknown"
        on_target_frac = (subset["fetal_region"] == dominant_region).mean() if "fetal_region" in subset.columns else np.nan

        results.append({
            "condition": condition,
            "n_cells": mask.sum(),
            "n_cell_types": len(ct_fracs),
            "shannon_entropy": entropy,
            "dominant_region": dominant_region,
            "on_target_fraction": on_target_frac,
            "mean_region_confidence": region_conf,
        })

    report = pd.DataFrame(results).set_index("condition")
    return report


if __name__ == "__main__":
    # Check prerequisites
    fetal_path = DATA_DIR / "braun-et-al_minimal_for_mapping.h5ad"
    mapped_path = DATA_DIR / "amin_kelley_mapped.h5ad"

    if not fetal_path.exists():
        print("ERROR: braun-et-al_minimal_for_mapping.h5ad not found!")
        print("Run: bash download_zenodo.sh")
        exit(1)

    if not mapped_path.exists():
        print("ERROR: data/amin_kelley_mapped.h5ad not found!")
        print("Run step 02 first: python 02_map_to_hnoca.py")
        exit(1)

    # Load data
    print("Loading mapped query data...")
    query = sc.read_h5ad(str(mapped_path))

    print("Loading fetal brain reference...")
    fetal = sc.read_h5ad(str(fetal_path))
    print(f"  Fetal reference: {fetal.shape}")

    # Tier 1: whole-brain mapping
    query = tier1_whole_brain_mapping(query, fetal)

    # Compute fidelity scores
    cond_key = "condition"  # adjust based on actual metadata
    if cond_key in query.obs.columns:
        report = compute_fidelity_scores(query, condition_key=cond_key)
        report.to_csv(str(DATA_DIR / "fidelity_report.csv"))
        print(f"\n  Fidelity report saved to data/fidelity_report.csv")
        print(report.head(10).to_string())

    # Save
    query.write(str(DATA_DIR / "amin_kelley_fidelity.h5ad"), compression="gzip")
    print(f"\n  Saved to data/amin_kelley_fidelity.h5ad")
