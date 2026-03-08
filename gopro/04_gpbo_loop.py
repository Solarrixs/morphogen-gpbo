"""
Step 4: GP-BO Active Reinforcement Learning Loop.

This is the core optimization engine:
  1. Loads GP training data (X=morphogen conditions, Y=cell type fractions)
  2. Fits a multi-output GP with Matérn 5/2 + ARD kernel
  3. Runs acquisition function to recommend next experiments
  4. Outputs plate map CSV for wet lab execution

Inputs:
  - data/gp_training_labels_amin_kelley.csv (from step 02)
  - Protocol JSON files (morphogen concentrations per condition)

Outputs:
  - data/gp_recommendations_round{N}.csv (next experiment plate map)
  - data/gp_uncertainty_map.csv (posterior variance across morphogen space)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"

# ==============================================================================
# Morphogen encoding: union of all morphogens across 3 datasets
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
    "IWP2_uM",            # WNT inhibitor
    "XAV939_uM",          # WNT inhibitor (tankyrase)
    "SB431542_uM",        # TGF-beta inhibitor
    "LDN193189_nM",       # BMP inhibitor
    "DAPT_uM",            # Notch inhibitor
    "EGF_ng_mL",          # Epidermal growth factor
    "purmorphamine_uM",   # SHH agonist
    "cyclopamine_uM",     # SHH antagonist
    "log_harvest_day",    # Time dimension
]


def build_training_set(fractions_csv, protocol_csv=None):
    """
    Build GP training matrices from cell type fractions + protocol info.

    X: (N_conditions, D_morphogens + 1_time)
    Y: (N_conditions, M_cell_types)
    """
    Y = pd.read_csv(fractions_csv, index_col=0)
    print(f"  Y (cell type fractions): {Y.shape}")

    # TODO: Parse actual morphogen concentrations from protocol metadata
    # For now, create placeholder X from condition names
    # In production, this comes from the protocol JSON
    print("  WARNING: Using placeholder X values. Replace with actual protocol parsing.")
    X = pd.DataFrame(
        np.zeros((len(Y), len(MORPHOGEN_COLUMNS))),
        index=Y.index,
        columns=MORPHOGEN_COLUMNS,
    )

    return X, Y


def fit_gp_sklearn(X, Y, target_cell_type=None):
    """
    Fit a GP using scikit-learn (hackathon/prototype version).
    For production, switch to BoTorch (see fit_gp_botorch below).
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

    # Matérn 5/2 + ARD (each morphogen gets its own lengthscale)
    kernel = (
        ConstantKernel(1.0)
        * Matern(
            nu=2.5,
            length_scale=np.ones(X.shape[1]),  # ARD: one per dimension
            length_scale_bounds=(1e-3, 1e3),
        )
        + WhiteKernel(noise_level=0.01)
    )

    if target_cell_type and target_cell_type in Y.columns:
        y = Y[target_cell_type].values
    else:
        # Default: use first cell type column
        y = Y.iloc[:, 0].values
        target_cell_type = Y.columns[0]

    print(f"\n  Fitting GP for target: {target_cell_type}")
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X.values, y)

    print(f"  Kernel: {gp.kernel_}")
    print(f"  Log-marginal-likelihood: {gp.log_marginal_likelihood_value_:.2f}")

    # ARD lengthscales → morphogen importance
    if hasattr(gp.kernel_, 'k1') and hasattr(gp.kernel_.k1, 'k2'):
        lengthscales = gp.kernel_.k1.k2.length_scale
        importance = pd.Series(1.0 / lengthscales, index=MORPHOGEN_COLUMNS)
        importance = importance.sort_values(ascending=False)
        print(f"\n  Morphogen importance (1/lengthscale):")
        for morph, imp in importance.head(5).items():
            print(f"    {morph}: {imp:.4f}")

    return gp


def recommend_next_experiments(gp, X_train, n_recommendations=24, strategy="ucb"):
    """
    Use acquisition function to recommend next experiments.

    Strategies:
    - "ucb": Upper Confidence Bound (exploration-exploitation tradeoff)
    - "ei": Expected Improvement (exploitation-focused)
    """
    from scipy.optimize import differential_evolution

    # Generate candidate grid (Latin Hypercube Sampling)
    from scipy.stats import qmc
    n_candidates = 10000
    sampler = qmc.LatinHypercube(d=X_train.shape[1])
    candidates = sampler.random(n=n_candidates)

    # Scale to realistic morphogen ranges
    # TODO: Set proper bounds per morphogen
    lower = np.zeros(X_train.shape[1])
    upper = np.ones(X_train.shape[1]) * 10  # placeholder
    candidates = qmc.scale(candidates, lower, upper)

    # Predict mean + uncertainty at all candidates
    mu, sigma = gp.predict(candidates, return_std=True)

    # UCB acquisition: a(x) = mu(x) + kappa * sigma(x)
    kappa = 2.0  # exploration weight
    if strategy == "ucb":
        acquisition = mu + kappa * sigma
    elif strategy == "ei":
        from scipy.stats import norm
        best_y = gp.predict(X_train.values).max()
        z = (mu - best_y) / (sigma + 1e-10)
        acquisition = (mu - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
    else:
        acquisition = sigma  # pure exploration

    # Select top N candidates
    top_idx = np.argsort(acquisition)[-n_recommendations:][::-1]

    recommendations = pd.DataFrame(
        candidates[top_idx],
        columns=MORPHOGEN_COLUMNS,
    )
    recommendations["predicted_mean"] = mu[top_idx]
    recommendations["predicted_uncertainty"] = sigma[top_idx]
    recommendations["acquisition_value"] = acquisition[top_idx]

    # Add well labels
    wells = [f"{chr(65 + i // 6)}{i % 6 + 1}" for i in range(n_recommendations)]
    recommendations.index = wells[:len(recommendations)]
    recommendations.index.name = "well"

    return recommendations


if __name__ == "__main__":
    fractions_path = DATA_DIR / "gp_training_labels_amin_kelley.csv"

    if not fractions_path.exists():
        print("ERROR: GP training labels not found!")
        print("Run steps 01-02 first.")
        print("\nFor demo purposes, generating synthetic training data...")

        # Demo: create synthetic data to test the GP pipeline
        np.random.seed(42)
        n_conditions = 46
        n_cell_types = 8

        Y_demo = pd.DataFrame(
            np.random.dirichlet(np.ones(n_cell_types), size=n_conditions),
            columns=[f"celltype_{i}" for i in range(n_cell_types)],
            index=[f"cond_{i}" for i in range(n_conditions)],
        )

        X_demo = pd.DataFrame(
            np.random.rand(n_conditions, len(MORPHOGEN_COLUMNS)) * 5,
            columns=MORPHOGEN_COLUMNS,
            index=Y_demo.index,
        )

        fractions_path = DATA_DIR / "gp_training_labels_demo.csv"
        DATA_DIR.mkdir(exist_ok=True)
        Y_demo.to_csv(fractions_path)
        X_demo.to_csv(DATA_DIR / "gp_training_X_demo.csv")
        print(f"  Demo data saved to {fractions_path}")

        X, Y = X_demo, Y_demo
    else:
        X, Y = build_training_set(fractions_path)

    # Fit GP
    gp = fit_gp_sklearn(X, Y, target_cell_type=Y.columns[0])

    # Recommend next experiments
    print("\n" + "="*60)
    print("NEXT EXPERIMENT RECOMMENDATIONS")
    print("="*60)
    recs = recommend_next_experiments(gp, X, n_recommendations=24)
    print(recs[["predicted_mean", "predicted_uncertainty", "acquisition_value"]].to_string())

    # Save
    output_path = DATA_DIR / "gp_recommendations_round1.csv"
    recs.to_csv(output_path)
    print(f"\n  Plate map saved to {output_path}")
    print("  Give this to the wet lab team!")
