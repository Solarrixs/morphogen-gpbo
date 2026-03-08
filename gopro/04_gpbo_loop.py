"""
Step 4: GP-BO Active Reinforcement Learning Loop.

This is the core optimization engine:
  1. Loads GP training data (X=morphogen conditions, Y=cell type fractions)
  2. Applies ILR transform to compositional Y data
  3. Fits a multi-fidelity GP using BoTorch with Matérn 5/2 + ARD kernel
  4. Runs multi-objective acquisition function to recommend next experiments
  5. Outputs plate map CSV for wet lab execution

Inputs:
  - data/gp_training_labels_amin_kelley.csv (from step 02)
  - data/morphogen_matrix_amin_kelley.csv (from morphogen_parser)
  - data/fidelity_report.csv (from step 03, optional)

Outputs:
  - data/gp_recommendations_round{N}.csv (next experiment plate map)
  - data/gp_model_diagnostics.csv (kernel parameters, lengthscales)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"

# BoTorch requires float64, which MPS doesn't support. Use CPU for GP fitting.
# MPS can be used for neural network components (CellFlow, scPoli) separately.
DEVICE = torch.device("cpu")
DTYPE = torch.double

# ==============================================================================
# Morphogen encoding: union of all morphogens across datasets
# Missing morphogens = 0 ("not added to culture")
# ==============================================================================
MORPHOGEN_COLUMNS = [
    "CHIR99021_uM",       # WNT agonist
    "BMP4_ng_mL",         # BMP signaling
    "BMP7_ng_mL",         # BMP signaling
    "SHH_ng_mL",          # Sonic hedgehog
    "SAG_nM",             # Smoothened agonist (SHH pathway)
    "RA_nM",              # Retinoic acid
    "FGF8_ng_mL",         # Fibroblast growth factor
    "FGF2_ng_mL",         # Fibroblast growth factor
    "FGF4_ng_mL",         # Fibroblast growth factor
    "IWP2_uM",            # WNT inhibitor
    "XAV939_uM",          # WNT inhibitor (tankyrase)
    "SB431542_uM",        # TGF-beta inhibitor
    "LDN193189_nM",       # BMP inhibitor
    "DAPT_uM",            # Notch inhibitor
    "EGF_ng_mL",          # Epidermal growth factor
    "ActivinA_ng_mL",     # TGF-beta / Activin
    "purmorphamine_uM",   # SHH agonist
    "cyclopamine_uM",     # SHH antagonist
    "Dorsomorphin_uM",    # BMP inhibitor (small molecule)
    "log_harvest_day",    # Time dimension
]

# Realistic morphogen bounds per dimension (for acquisition function search)
MORPHOGEN_BOUNDS = {
    "CHIR99021_uM":     (0.0, 12.0),
    "BMP4_ng_mL":       (0.0, 50.0),
    "BMP7_ng_mL":       (0.0, 50.0),
    "SHH_ng_mL":        (0.0, 500.0),
    "SAG_nM":           (0.0, 2000.0),
    "RA_nM":            (0.0, 1000.0),
    "FGF8_ng_mL":       (0.0, 200.0),
    "FGF2_ng_mL":       (0.0, 100.0),
    "FGF4_ng_mL":       (0.0, 200.0),
    "IWP2_uM":          (0.0, 10.0),
    "XAV939_uM":        (0.0, 10.0),
    "SB431542_uM":      (0.0, 20.0),
    "LDN193189_nM":     (0.0, 500.0),
    "DAPT_uM":          (0.0, 10.0),
    "EGF_ng_mL":        (0.0, 50.0),
    "ActivinA_ng_mL":   (0.0, 100.0),
    "purmorphamine_uM": (0.0, 2.0),
    "cyclopamine_uM":   (0.0, 10.0),
    "Dorsomorphin_uM":  (0.0, 5.0),
    "log_harvest_day":  (np.log(7), np.log(120)),  # Day 7 to Day 120
}


def _helmert_basis(D: int) -> np.ndarray:
    """Construct the Helmert ILR basis matrix.

    Args:
        D: Number of composition parts.

    Returns:
        Array of shape (D, D-1) — the ILR contrast matrix.
    """
    V = np.zeros((D, D - 1))
    for j in range(D - 1):
        V[:j + 1, j] = -1.0 / (j + 1)
        V[j + 1, j] = 1.0
        V[:, j] *= np.sqrt((j + 1) / (j + 2))
    return V


def ilr_transform(Y: np.ndarray) -> np.ndarray:
    """Isometric log-ratio transform for compositional data.

    Transforms D-part compositions to (D-1)-dimensional real space,
    removing the sum-to-one constraint that violates GP assumptions.

    Args:
        Y: Array of shape (N, D) with rows summing to 1.

    Returns:
        Array of shape (N, D-1) in ILR space.
    """
    D = Y.shape[1]
    Y_safe = Y + 1e-10
    Y_safe = Y_safe / Y_safe.sum(axis=1, keepdims=True)
    log_Y = np.log(Y_safe)

    V = _helmert_basis(D)
    return log_Y @ V


def ilr_inverse(Z: np.ndarray, D: int) -> np.ndarray:
    """Inverse ILR transform: (D-1) real coordinates → D-part composition.

    Args:
        Z: Array of shape (N, D-1) in ILR space.
        D: Number of composition parts.

    Returns:
        Array of shape (N, D) with rows summing to 1.
    """
    V = _helmert_basis(D)
    log_x = Z @ V.T

    # Closure (softmax) to get compositions
    log_x -= log_x.max(axis=1, keepdims=True)
    x = np.exp(log_x)
    return x / x.sum(axis=1, keepdims=True)


def build_training_set(
    fractions_csv: Path,
    morphogen_csv: Path,
    fidelity: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build GP training matrices from cell type fractions + morphogen concentrations.

    Args:
        fractions_csv: Path to cell type fractions CSV (conditions × cell types).
        morphogen_csv: Path to morphogen concentration matrix CSV.
        fidelity: Fidelity level (1.0=real, 0.5=CellRank2, 0.0=CellFlow).

    Returns:
        Tuple of (X, Y) DataFrames.
        X: (N_conditions, D_morphogens + 1_time + 1_fidelity)
        Y: (N_conditions, M_cell_types)
    """
    Y = pd.read_csv(str(fractions_csv), index_col=0)
    X = pd.read_csv(str(morphogen_csv), index_col=0)

    # Align indices
    common = Y.index.intersection(X.index)
    if len(common) < len(Y):
        print(f"  WARNING: {len(Y) - len(common)} conditions in Y not found in X")
    X = X.loc[common]
    Y = Y.loc[common]

    # Add fidelity column
    X["fidelity"] = fidelity

    print(f"  X (morphogens): {X.shape}")
    print(f"  Y (cell type fractions): {Y.shape}")

    return X, Y


def fit_gp_botorch(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    target_cell_types: Optional[list[str]] = None,
    use_ilr: bool = True,
) -> tuple:
    """Fit a multi-fidelity GP using BoTorch.

    Uses SingleTaskMultiFidelityGP with Matérn 5/2 + ARD kernel.

    Args:
        X: Morphogen concentration matrix with fidelity column.
        Y: Cell type fraction matrix.
        target_cell_types: List of cell types to optimize. If None, uses all.
        use_ilr: Whether to apply ILR transform to Y.

    Returns:
        Tuple of (model, train_X_tensor, train_Y_tensor, cell_type_columns).
    """
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll

    # Select target cell types
    if target_cell_types:
        Y_selected = Y[target_cell_types]
    else:
        Y_selected = Y
    cell_type_cols = list(Y_selected.columns)

    # Prepare tensors
    train_X = torch.tensor(X.values, dtype=DTYPE, device=DEVICE)
    Y_values = Y_selected.values

    if use_ilr and Y_values.shape[1] > 1:
        print("  Applying ILR transform to compositional Y data...")
        Y_values = ilr_transform(Y_values)
        print(f"  ILR-transformed Y shape: {Y_values.shape}")

    train_Y = torch.tensor(Y_values, dtype=DTYPE, device=DEVICE)

    # Fit GP
    print(f"\n  Fitting BoTorch GP...")
    print(f"  X: {train_X.shape}, Y: {train_Y.shape}")

    # Check if we have fidelity column
    has_fidelity = "fidelity" in X.columns
    fidelity_idx = list(X.columns).index("fidelity") if has_fidelity else None

    if has_fidelity and X["fidelity"].nunique() > 1:
        from botorch.models import SingleTaskMultiFidelityGP
        print("  Using multi-fidelity GP (fidelity column detected)")
        model = SingleTaskMultiFidelityGP(
            train_X,
            train_Y,
            data_fidelities=[fidelity_idx],
            input_transform=Normalize(d=train_X.shape[1]),
            outcome_transform=Standardize(m=train_Y.shape[1]),
        )
    else:
        print("  Using single-task GP (single fidelity level)")
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=train_X.shape[1]),
            outcome_transform=Standardize(m=train_Y.shape[1]),
        )

    # Optimize hyperparameters
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Report kernel parameters
    print("\n  GP Kernel Parameters:")
    if hasattr(model.covar_module, 'base_kernel'):
        lengthscales = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        importance = 1.0 / lengthscales
        morph_importance = pd.Series(importance[:len(X.columns)], index=X.columns)
        morph_importance = morph_importance.sort_values(ascending=False)
        print("  Morphogen importance (1/lengthscale):")
        for morph, imp in morph_importance.head(8).items():
            print(f"    {morph}: {imp:.4f}")

    return model, train_X, train_Y, cell_type_cols


def recommend_next_experiments(
    model,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: dict[str, tuple[float, float]],
    columns: list[str],
    n_recommendations: int = 24,
    use_multi_objective: bool = False,
    ref_point: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Use acquisition function to recommend next experiments.

    Args:
        model: Fitted BoTorch GP model.
        train_X: Training X tensor.
        train_Y: Training Y tensor.
        bounds: Dict mapping column names to (lower, upper) bounds.
        columns: Column names for X.
        n_recommendations: Number of conditions to recommend.
        use_multi_objective: Whether to use multi-objective optimization.
        ref_point: Reference point for hypervolume calculation.

    Returns:
        DataFrame with recommended morphogen concentrations and predictions.
    """
    from botorch.optim import optimize_acqf
    from botorch.acquisition import qLogExpectedImprovement

    # Build bounds tensor
    lower = []
    upper = []
    for col in columns:
        if col in bounds:
            lo, hi = bounds[col]
        elif col == "fidelity":
            lo, hi = 1.0, 1.0  # always recommend at highest fidelity
        else:
            lo, hi = 0.0, 1.0
        lower.append(lo)
        upper.append(hi)

    bounds_tensor = torch.tensor(
        [lower, upper], dtype=DTYPE, device=DEVICE
    )

    if use_multi_objective and train_Y.shape[1] > 1:
        from botorch.acquisition.multi_objective import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        if ref_point is None:
            ref_point = [0.0] * train_Y.shape[1]
        ref_point_tensor = torch.tensor(ref_point, dtype=DTYPE, device=DEVICE)

        acqf = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point_tensor,
            X_baseline=train_X,
        )
    else:
        # For multi-output models, use a scalarization posterior transform
        if train_Y.shape[1] > 1:
            from botorch.acquisition.objective import GenericMCObjective
            from botorch.utils.transforms import normalize, standardize

            # Default: maximize mean of all outputs (equal weight scalarization)
            weights = torch.ones(train_Y.shape[1], dtype=DTYPE, device=DEVICE)
            weights = weights / weights.sum()

            objective = GenericMCObjective(
                lambda samples, X=None: (samples * weights).sum(dim=-1)
            )
            acqf = qLogExpectedImprovement(
                model=model,
                best_f=(train_Y @ weights.unsqueeze(1)).max(),
                objective=objective,
            )
        else:
            acqf = qLogExpectedImprovement(
                model=model,
                best_f=train_Y.max(),
            )

    # Optimize acquisition function
    print(f"\n  Optimizing acquisition function for {n_recommendations} candidates...")
    candidates, acq_values = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor,
        q=n_recommendations,
        num_restarts=10,
        raw_samples=2048,
    )

    # Get predictions at recommended points
    with torch.no_grad():
        posterior = model.posterior(candidates)
        pred_mean = posterior.mean.cpu().numpy()
        pred_var = posterior.variance.cpu().numpy()

    # Build output DataFrame
    recommendations = pd.DataFrame(
        candidates.cpu().numpy(),
        columns=columns,
    )

    # Add predictions
    for i in range(pred_mean.shape[1]):
        recommendations[f"predicted_y{i}_mean"] = pred_mean[:, i]
        recommendations[f"predicted_y{i}_std"] = np.sqrt(pred_var[:, i])

    recommendations["acquisition_value"] = acq_values.detach().cpu().numpy() if acq_values.dim() > 0 else acq_values.item()

    # Add well labels (A1-D6 for 24-well plate)
    wells = [f"{chr(65 + i // 6)}{i % 6 + 1}" for i in range(n_recommendations)]
    recommendations.index = wells[:len(recommendations)]
    recommendations.index.name = "well"

    return recommendations


def run_gpbo_loop(
    fractions_csv: Path,
    morphogen_csv: Path,
    target_cell_types: Optional[list[str]] = None,
    n_recommendations: int = 24,
    round_num: int = 1,
    use_ilr: bool = True,
) -> pd.DataFrame:
    """Run one iteration of the GP-BO loop.

    Args:
        fractions_csv: Path to cell type fractions CSV.
        morphogen_csv: Path to morphogen matrix CSV.
        target_cell_types: Cell types to optimize for. None = all.
        n_recommendations: Number of experiments to recommend.
        round_num: Current optimization round number.
        use_ilr: Whether to apply ILR transform.

    Returns:
        DataFrame of recommended next experiments.
    """
    print("=" * 60)
    print(f"GP-BO ROUND {round_num}")
    print("=" * 60)

    # Build training set
    print("\nBuilding training set...")
    X, Y = build_training_set(fractions_csv, morphogen_csv)

    # Fit GP
    model, train_X, train_Y, cell_type_cols = fit_gp_botorch(
        X, Y,
        target_cell_types=target_cell_types,
        use_ilr=use_ilr,
    )

    # Recommend next experiments
    recommendations = recommend_next_experiments(
        model, train_X, train_Y,
        bounds=MORPHOGEN_BOUNDS,
        columns=list(X.columns),
        n_recommendations=n_recommendations,
    )

    # Save outputs
    output_path = DATA_DIR / f"gp_recommendations_round{round_num}.csv"
    recommendations.to_csv(str(output_path))
    print(f"\n  Plate map saved to {output_path}")

    # Save model diagnostics
    diagnostics = {
        "round": round_num,
        "n_training_points": len(X),
        "n_morphogens": X.shape[1],
        "n_cell_types": Y.shape[1],
        "target_cell_types": str(target_cell_types or "all"),
    }
    if hasattr(model.covar_module, 'base_kernel'):
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        for i, col in enumerate(X.columns):
            if i < len(ls):
                diagnostics[f"lengthscale_{col}"] = ls[i]

    diag_df = pd.DataFrame([diagnostics])
    diag_path = DATA_DIR / f"gp_diagnostics_round{round_num}.csv"
    diag_df.to_csv(str(diag_path), index=False)
    print(f"  Diagnostics saved to {diag_path}")

    return recommendations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GP-BO loop for morphogen optimization")
    parser.add_argument("--fractions", type=str, default=None,
                        help="Path to cell type fractions CSV")
    parser.add_argument("--morphogens", type=str, default=None,
                        help="Path to morphogen concentration matrix CSV")
    parser.add_argument("--target-cell-types", nargs="+", default=None,
                        help="Cell types to optimize for (default: all)")
    parser.add_argument("--n-recommendations", type=int, default=24,
                        help="Number of experiments to recommend (default: 24)")
    parser.add_argument("--round", type=int, default=1,
                        help="Optimization round number (default: 1)")
    parser.add_argument("--no-ilr", action="store_true",
                        help="Disable ILR transform")
    parser.add_argument("--multi-objective", action="store_true",
                        help="Use multi-objective acquisition (qLogNEHVI)")
    args = parser.parse_args()

    fractions_path = Path(args.fractions) if args.fractions else DATA_DIR / "gp_training_labels_amin_kelley.csv"
    morphogen_path = Path(args.morphogens) if args.morphogens else DATA_DIR / "morphogen_matrix_amin_kelley.csv"

    if not fractions_path.exists() or not morphogen_path.exists():
        print("Training data not found. Running with synthetic data for demo...")

        np.random.seed(42)
        n_conditions = 46
        n_cell_types = 8

        Y_demo = pd.DataFrame(
            np.random.dirichlet(np.ones(n_cell_types), size=n_conditions),
            columns=[f"celltype_{i}" for i in range(n_cell_types)],
            index=[f"cond_{i}" for i in range(n_conditions)],
        )

        X_demo = pd.DataFrame(
            np.zeros((n_conditions, len(MORPHOGEN_COLUMNS))),
            columns=MORPHOGEN_COLUMNS,
            index=Y_demo.index,
        )
        for col in MORPHOGEN_COLUMNS:
            lo, hi = MORPHOGEN_BOUNDS.get(col, (0, 1))
            X_demo[col] = np.random.uniform(lo, hi, size=n_conditions)

        fractions_path = DATA_DIR / "gp_training_labels_demo.csv"
        morphogen_path = DATA_DIR / "morphogen_matrix_demo.csv"
        DATA_DIR.mkdir(exist_ok=True)
        Y_demo.to_csv(str(fractions_path))
        X_demo.to_csv(str(morphogen_path))
        print(f"  Synthetic data saved.")

    # Run GP-BO loop
    recs = run_gpbo_loop(
        fractions_csv=fractions_path,
        morphogen_csv=morphogen_path,
        target_cell_types=args.target_cell_types,
        n_recommendations=args.n_recommendations,
        round_num=args.round,
        use_ilr=not args.no_ilr,
    )

    print("\n" + "=" * 60)
    print("NEXT EXPERIMENT RECOMMENDATIONS")
    print("=" * 60)
    morph_cols = [c for c in MORPHOGEN_COLUMNS if c in recs.columns]
    nonzero_cols = [c for c in morph_cols if recs[c].abs().sum() > 0.01]
    pred_cols = [c for c in recs.columns if "predicted" in c or "acquisition" in c]
    print(recs[nonzero_cols + pred_cols].to_string())
    print("\n  Give this plate map to the wet lab team!")
