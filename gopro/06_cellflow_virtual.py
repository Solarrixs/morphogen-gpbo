"""
Step 6: CellFlow Virtual Protocol Screening.

Uses CellFlow (Klein et al., bioRxiv 2025) — a flow-matching generative model
trained on 176 conditions from Sanchis-Calleja + Azbukina — to predict
single-cell distributions from novel protocol encodings.

CellFlow encodes protocols using:
  1. Molecular fingerprints (small molecules via RDKit)
  2. ESM2 protein embeddings (growth factors)
  3. Concentration, timing window, pathway annotations
  4. Base protocol / dataset label

This enables virtual screening of ~23,000 novel morphogen combinations,
generating low-fidelity (0.0) training points for the multi-fidelity GP.

Inputs:
  - Pre-trained CellFlow model (or train from data)
  - Protocol specifications (morphogen combinations to screen)

Outputs:
  - data/cellflow_virtual_fractions.csv (predicted cell type fractions)
  - data/cellflow_virtual_morphogens.csv (morphogen concentration vectors)
  - data/cellflow_screening_report.csv (quality metrics per prediction)
"""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from gopro.config import (
    PROTEIN_MW_KDA as MW,
    nM_to_uM,
    ng_mL_to_uM,
)

PROJECT_DIR = Path("/Users/maxxyung/Projects/morphogen-gpbo")
DATA_DIR = PROJECT_DIR / "data"

# Low fidelity level for CellFlow virtual data
FIDELITY_LEVEL = 0.0

# Morphogen identity mapping for protocol encoding
# Maps morphogen column names to (molecule_type, canonical_name) pairs
MORPHOGEN_IDENTITIES: dict[str, tuple[str, str]] = {
    "CHIR99021_uM": ("small_molecule", "CHIR99021"),
    "BMP4_uM": ("protein", "BMP4"),
    "BMP7_uM": ("protein", "BMP7"),
    "SHH_uM": ("protein", "SHH"),
    "SAG_uM": ("small_molecule", "SAG"),
    "RA_uM": ("small_molecule", "retinoic_acid"),
    "FGF8_uM": ("protein", "FGF8"),
    "FGF2_uM": ("protein", "FGF2"),
    "FGF4_uM": ("protein", "FGF4"),
    "IWP2_uM": ("small_molecule", "IWP2"),
    "XAV939_uM": ("small_molecule", "XAV939"),
    "SB431542_uM": ("small_molecule", "SB431542"),
    "LDN193189_uM": ("small_molecule", "LDN193189"),
    "DAPT_uM": ("small_molecule", "DAPT"),
    "EGF_uM": ("protein", "EGF"),
    "ActivinA_uM": ("protein", "ActivinA"),
    "purmorphamine_uM": ("small_molecule", "purmorphamine"),
    "cyclopamine_uM": ("small_molecule", "cyclopamine"),
    "Dorsomorphin_uM": ("small_molecule", "Dorsomorphin"),
}

# Signaling pathway annotations for morphogens
MORPHOGEN_PATHWAYS: dict[str, str] = {
    "CHIR99021_uM": "WNT",
    "BMP4_uM": "BMP",
    "BMP7_uM": "BMP",
    "SHH_uM": "SHH",
    "SAG_uM": "SHH",
    "RA_uM": "RA",
    "FGF8_uM": "FGF",
    "FGF2_uM": "FGF",
    "FGF4_uM": "FGF",
    "IWP2_uM": "WNT",
    "XAV939_uM": "WNT",
    "SB431542_uM": "TGFb",
    "LDN193189_uM": "BMP",
    "DAPT_uM": "Notch",
    "EGF_uM": "EGF",
    "ActivinA_uM": "TGFb",
    "purmorphamine_uM": "SHH",
    "cyclopamine_uM": "SHH",
    "Dorsomorphin_uM": "BMP",
}

# SMILES strings for small molecule morphogens (for RDKit fingerprints)
MORPHOGEN_SMILES: dict[str, str] = {
    "CHIR99021": "C1=CC(=CC=C1C2=CN=C(N=C2N)NC3=CC(=C(C=C3)Cl)Cl)Cl",
    "SAG": "C1CCC(CC1)CNC(=O)C2=CC3=CC=CC=C3N2CC4=CC=C(C=C4)Cl",
    "retinoic_acid": "CC1=C(/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C)C(=CC=C1)C",
    "IWP2": "C1=CC=C(C=C1)CSC2=NN=C3N2CC(=O)N4CCCC5=CC=CC=C54",
    "XAV939": "C1=CC(=CC=C1C2=CSC3=CC(=CC=C32)O)C(F)(F)F",
    "SB431542": "CC1=NC2=CC=C(C=C2N1)C(=O)NC3=CC=CC(=C3)C4=CC=CC=N4",
    "LDN193189": "C1=CC2=C(N=C1C3=CC=NC4=CC=CC=C43)C=CC(=C2)C5=CC=CC=C5",
    "DAPT": "CC(C(=O)NC(CC1=CC=CC=C1)C(=O)OC(C)C)NC(=O)CC2=CC(=CC=C2)F",
    "purmorphamine": "C1=CC=C(C=C1)CNC2=NC=C3C(=N2)N=CN3CCCC4=CC=CC=C4",
    "cyclopamine": "CC1(CCC2C1CCC3C2CC=C4CC(CCC34C)NC5CC6C(C(O5)CO)OC7C6OC(C7O)CO)O",
    "Dorsomorphin": "C1=CC(=CC=C1C2=CC(=NC=C2)OCC3=CC=C(C=C3)C4=NN=CC=C4)N",
}


def encode_protocol_cellflow(
    morphogen_vec: dict[str, float],
    timing_start: int = 0,
    timing_end: int = 21,
    base_protocol: str = "standard",
) -> dict:
    """Encode a morphogen protocol in CellFlow format.

    CellFlow uses a multi-part protocol encoding:
    - Modulator identity (RDKit fingerprints or ESM2 embeddings)
    - Concentration (numeric)
    - Timing window (which days each modulator was applied)
    - Pathway annotation
    - Base protocol label

    Args:
        morphogen_vec: Dict mapping morphogen columns to concentrations.
        timing_start: First day of morphogen application.
        timing_end: Last day of morphogen application.
        base_protocol: Base protocol identifier.

    Returns:
        Dict with CellFlow protocol encoding fields.
    """
    modulators = []

    for col, conc in morphogen_vec.items():
        if col == "log_harvest_day" or conc <= 0:
            continue

        if col not in MORPHOGEN_IDENTITIES:
            continue

        mol_type, canonical_name = MORPHOGEN_IDENTITIES[col]
        pathway = MORPHOGEN_PATHWAYS.get(col, "unknown")

        modulator = {
            "name": canonical_name,
            "type": mol_type,
            "concentration": conc,
            "concentration_unit": col.split("_")[-1],
            "timing_start": timing_start,
            "timing_end": timing_end,
            "pathway": pathway,
        }

        # Add SMILES for small molecules (for RDKit fingerprint)
        if mol_type == "small_molecule" and canonical_name in MORPHOGEN_SMILES:
            modulator["smiles"] = MORPHOGEN_SMILES[canonical_name]

        modulators.append(modulator)

    return {
        "modulators": modulators,
        "base_protocol": base_protocol,
        "harvest_day": int(round(math.exp(
            morphogen_vec.get("log_harvest_day", math.log(21))
        ))),
    }


def generate_virtual_screen_grid(
    morphogen_ranges: dict[str, list[float]],
    harvest_days: list[int] = None,
    max_combinations: int = 5000,
) -> pd.DataFrame:
    """Generate a grid of morphogen combinations for virtual screening.

    Creates a combinatorial grid across specified morphogen concentration
    levels, with optional harvest day variation.

    Args:
        morphogen_ranges: Dict mapping morphogen column names to lists
            of concentration values to screen.
        harvest_days: List of harvest days to include.
        max_combinations: Maximum number of combinations to generate.

    Returns:
        DataFrame with morphogen concentration vectors.
    """
    if harvest_days is None:
        harvest_days = [21]

    # Import morphogen columns from step 04
    from importlib import util
    spec = util.spec_from_file_location(
        "step04", str(Path(__file__).parent / "04_gpbo_loop.py")
    )
    step04 = util.module_from_spec(spec)
    spec.loader.exec_module(step04)
    MORPHOGEN_COLUMNS = step04.MORPHOGEN_COLUMNS

    # Build the grid
    grid_dims = {}
    for col in MORPHOGEN_COLUMNS:
        if col == "log_harvest_day":
            grid_dims[col] = [math.log(d) for d in harvest_days]
        elif col in morphogen_ranges:
            grid_dims[col] = morphogen_ranges[col]
        else:
            grid_dims[col] = [0.0]  # Not varied

    # Generate combinations
    keys = list(grid_dims.keys())
    value_lists = [grid_dims[k] for k in keys]

    # Calculate total combinations
    total = 1
    for v in value_lists:
        total *= len(v)

    if total > max_combinations:
        print(f"  Grid would produce {total:,} combinations, "
              f"sampling {max_combinations:,} randomly")
        # Random sampling instead of full grid
        rows = []
        rng = np.random.RandomState(42)
        for _ in range(max_combinations):
            row = {}
            for key, values in zip(keys, value_lists):
                row[key] = rng.choice(values)
            rows.append(row)
        grid = pd.DataFrame(rows)
    else:
        combos = list(itertools.product(*value_lists))
        grid = pd.DataFrame(combos, columns=keys)

    # Name the conditions
    grid.index = [f"virtual_{i:05d}" for i in range(len(grid))]
    grid.index.name = None

    print(f"  Generated {len(grid):,} virtual protocol combinations")
    return grid


def predict_cellflow(
    protocols: pd.DataFrame,
    model_path: Optional[Path] = None,
    batch_size: int = 256,
    n_cells_per_condition: int = 500,
) -> pd.DataFrame:
    """Run CellFlow predictions for a batch of protocols.

    Uses the CellFlow generative model to predict single-cell
    distributions, then aggregates to cell type fractions.

    Args:
        protocols: DataFrame of morphogen concentration vectors.
        model_path: Path to pre-trained CellFlow model.
        batch_size: Number of protocols to predict at once.
        n_cells_per_condition: Number of virtual cells to generate.

    Returns:
        DataFrame with predicted cell type fractions per protocol.
    """
    try:
        import cellflow
        HAS_CELLFLOW = True
    except ImportError:
        HAS_CELLFLOW = False

    if HAS_CELLFLOW and model_path is not None and model_path.exists():
        return _predict_with_cellflow(
            protocols, model_path, batch_size, n_cells_per_condition
        )
    else:
        print("  CellFlow not available or no model found. "
              "Using baseline prediction.")
        return _predict_baseline(protocols)


def _predict_with_cellflow(
    protocols: pd.DataFrame,
    model_path: Path,
    batch_size: int,
    n_cells_per_condition: int,
) -> pd.DataFrame:
    """Run predictions using actual CellFlow model.

    Args:
        protocols: Morphogen concentration vectors.
        model_path: Path to trained CellFlow model directory.
        batch_size: Batch size for prediction.
        n_cells_per_condition: Cells to sample per condition.

    Returns:
        Predicted cell type fractions.
    """
    import cellflow
    import torch

    print(f"  Loading CellFlow model from {model_path}...")
    model = cellflow.CellFlowModel.load(str(model_path))

    all_fractions = []
    n_batches = (len(protocols) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(protocols))
        batch = protocols.iloc[start:end]

        if batch_idx % 10 == 0:
            print(f"  Predicting batch {batch_idx + 1}/{n_batches} "
                  f"({start}-{end})...")

        # Encode protocols
        encodings = []
        for _, row in batch.iterrows():
            enc = encode_protocol_cellflow(row.to_dict())
            encodings.append(enc)

        # Generate virtual cells
        with torch.no_grad():
            predictions = model.predict(
                encodings,
                n_cells=n_cells_per_condition,
            )

        # Aggregate to cell type fractions
        for i, (cond_name, pred) in enumerate(
            zip(batch.index, predictions)
        ):
            if hasattr(pred, "obs") and "cell_type" in pred.obs.columns:
                fracs = pred.obs["cell_type"].value_counts(normalize=True)
                frac_dict = fracs.to_dict()
            else:
                # Cluster predicted cells to get fractions
                import scanpy as sc
                sc.pp.pca(pred, n_comps=20)
                sc.pp.neighbors(pred, n_neighbors=15)
                sc.tl.leiden(pred, resolution=0.5)
                fracs = pred.obs["leiden"].value_counts(normalize=True)
                frac_dict = {f"cluster_{k}": v for k, v in fracs.items()}

            frac_dict["condition"] = cond_name
            all_fractions.append(frac_dict)

    result = pd.DataFrame(all_fractions).set_index("condition")
    result = result.fillna(0.0)

    # Normalize rows
    row_sums = result.sum(axis=1)
    result = result.div(row_sums.replace(0, 1), axis=0)

    return result


def _predict_baseline(
    protocols: pd.DataFrame,
) -> pd.DataFrame:
    """Baseline prediction when CellFlow is not available.

    Uses a heuristic model based on known morphogen-to-fate mappings
    from literature. This is a placeholder that produces reasonable
    but approximate cell type fractions.

    Args:
        protocols: Morphogen concentration vectors.

    Returns:
        Predicted cell type fractions (heuristic).
    """
    print("  Using heuristic baseline predictor...")

    # Known morphogen-to-fate associations from literature
    # These are rough approximations based on Amin/Kelley + Sanchis-Calleja results
    CELL_TYPES = [
        "Neuron", "NPC", "IP", "Neuroepithelium", "Glioblast",
        "Astrocyte", "OPC", "CP", "PSC", "MC",
    ]

    results = []
    for cond_name, row in protocols.iterrows():
        # Start with a base composition
        fracs = np.ones(len(CELL_TYPES)) * 0.05

        # WNT signaling (CHIR) -> caudalizes; IWP2 -> anteriorizes
        chir = row.get("CHIR99021_uM", 0)
        iwp2 = row.get("IWP2_uM", 0)
        if chir > 2.0:
            fracs[0] += 0.2  # More neurons with high CHIR
            fracs[1] -= 0.02
        if iwp2 > 0:
            fracs[1] += 0.15  # More NPC with WNT inhibition
            fracs[3] += 0.1   # More neuroepithelium

        # SHH pathway -> ventral fates (all values now in µM)
        shh = row.get("SHH_uM", 0) + row.get("SAG_uM", 0)
        if shh > 0:
            fracs[0] += 0.1 * min(shh / 0.05, 1)  # Ventral neurons
            fracs[2] += 0.05  # Intermediate progenitors

        # BMP -> dorsal / choroid plexus (all values now in µM)
        bmp = row.get("BMP4_uM", 0) + row.get("BMP7_uM", 0)
        if bmp > 0:
            fracs[7] += 0.15 * min(bmp / 0.003, 1)  # Choroid plexus
            fracs[9] += 0.05  # Some mesenchymal

        # FGF -> proliferation / gliogenesis (all values now in µM)
        fgf = (row.get("FGF2_uM", 0) + row.get("FGF4_uM", 0)
               + row.get("FGF8_uM", 0))
        if fgf > 0:
            fracs[4] += 0.1 * min(fgf / 0.005, 1)  # Glioblast
            fracs[1] += 0.05  # NPC maintenance

        # LDN/SB -> dual SMAD inhibition -> neural induction (all in µM)
        ldn = row.get("LDN193189_uM", 0)
        sb = row.get("SB431542_uM", 0)
        if ldn > 0 or sb > 0:
            fracs[3] += 0.15  # Neuroepithelium
            fracs[8] -= 0.03  # Less PSC
            fracs[9] -= 0.03  # Less MC

        # RA -> caudal / hindbrain (all values now in µM)
        ra = row.get("RA_uM", 0)
        if ra > 0:
            fracs[0] += 0.1 * min(ra / 0.5, 1)  # Neurons
            fracs[2] += 0.05  # IP

        # DAPT -> Notch inhibition -> neuronal differentiation
        dapt = row.get("DAPT_uM", 0)
        if dapt > 0:
            fracs[0] += 0.15  # Neurons
            fracs[1] -= 0.05  # Fewer NPCs

        # Ensure non-negative and normalize
        fracs = np.maximum(fracs, 0.01)
        fracs = fracs / fracs.sum()

        results.append(dict(zip(CELL_TYPES, fracs)))

    result = pd.DataFrame(results, index=protocols.index)
    return result


def compute_prediction_confidence(
    predictions: pd.DataFrame,
    training_morphogens: pd.DataFrame,
    n_neighbors: int = 5,
) -> pd.Series:
    """Estimate confidence of CellFlow predictions.

    Uses distance to nearest training points as a proxy for
    prediction reliability. Points far from training data are
    extrapolations and should be down-weighted.

    Args:
        predictions: Virtual protocol morphogen vectors.
        training_morphogens: Real training morphogen vectors.
        n_neighbors: Number of nearest training points to consider.

    Returns:
        Series of confidence scores in [0, 1] per virtual condition.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    # Exclude non-morphogen columns
    morph_cols = [c for c in predictions.columns
                  if c in training_morphogens.columns and c != "log_harvest_day"]

    if not morph_cols:
        return pd.Series(0.5, index=predictions.index)

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(training_morphogens[morph_cols].values)
    X_pred = scaler.transform(predictions[morph_cols].values)

    # Find distances to nearest training points
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(X_train)))
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_pred)

    # Convert mean distance to confidence (sigmoid-like transform)
    mean_dist = distances.mean(axis=1)
    if len(X_train) >= 2:
        median_train_dist = np.median(
            NearestNeighbors(n_neighbors=2).fit(X_train).kneighbors(X_train)[0][:, 1]
        )
    else:
        # Fallback for single training point: use mean distance as scale
        median_train_dist = mean_dist.mean() if mean_dist.mean() > 0 else 1.0

    # Confidence decays with distance from training data
    confidence = np.exp(-mean_dist / (2 * median_train_dist + 1e-10))
    confidence = np.clip(confidence, 0.01, 1.0)

    return pd.Series(confidence, index=predictions.index, name="confidence")


def run_virtual_screen(
    morphogen_ranges: dict[str, list[float]],
    real_morphogen_csv: Path,
    harvest_days: list[int] = None,
    max_combinations: int = 5000,
    model_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full CellFlow virtual screen.

    Args:
        morphogen_ranges: Morphogen concentration ranges to screen.
        real_morphogen_csv: Path to real training morphogen CSV.
        harvest_days: Harvest days to include.
        max_combinations: Max number of virtual protocols.
        model_path: Path to CellFlow model (optional).

    Returns:
        Tuple of (virtual_morphogens, virtual_fractions, quality_report).
    """
    # Generate protocol grid
    protocols = generate_virtual_screen_grid(
        morphogen_ranges,
        harvest_days=harvest_days,
        max_combinations=max_combinations,
    )

    # Run predictions
    predictions = predict_cellflow(
        protocols,
        model_path=model_path,
    )

    # Compute confidence scores
    real_morphogens = pd.read_csv(str(real_morphogen_csv), index_col=0)
    confidence = compute_prediction_confidence(protocols, real_morphogens)

    # Build quality report
    quality_report = pd.DataFrame({
        "condition": protocols.index,
        "confidence": confidence.values,
        "n_nonzero_morphogens": (protocols.drop(columns=["log_harvest_day"],
                                                 errors="ignore") > 0).sum(axis=1).values,
        "fidelity": FIDELITY_LEVEL,
    })

    return protocols, predictions, quality_report


def main() -> None:
    """Run the full CellFlow virtual screening pipeline."""
    import time
    start = time.time()

    print("=" * 60)
    print("PHASE 5: CellFlow Virtual Protocol Screening")
    print("=" * 60)

    morphogen_path = DATA_DIR / "morphogen_matrix_amin_kelley.csv"
    if not morphogen_path.exists():
        print(f"ERROR: Morphogen matrix not found at {morphogen_path}")
        return

    # Define screening ranges for key morphogens
    # Based on literature ranges from Sanchis-Calleja + Amin/Kelley
    morphogen_ranges = {
        "CHIR99021_uM": [0.0, 0.5, 1.0, 1.5, 3.0, 6.0, 9.0],
        "BMP4_uM": [0.0, ng_mL_to_uM(5.0, MW["BMP4"]), ng_mL_to_uM(10.0, MW["BMP4"]), ng_mL_to_uM(25.0, MW["BMP4"])],
        "SHH_uM": [0.0, ng_mL_to_uM(50.0, MW["SHH"]), ng_mL_to_uM(100.0, MW["SHH"]), ng_mL_to_uM(200.0, MW["SHH"])],
        "SAG_uM": [0.0, nM_to_uM(50.0), nM_to_uM(250.0), nM_to_uM(500.0), nM_to_uM(1000.0)],
        "RA_uM": [0.0, nM_to_uM(10.0), nM_to_uM(50.0), nM_to_uM(100.0), nM_to_uM(500.0)],
        "FGF8_uM": [0.0, ng_mL_to_uM(50.0, MW["FGF8"]), ng_mL_to_uM(100.0, MW["FGF8"])],
        "IWP2_uM": [0.0, 2.5, 5.0],
        "SB431542_uM": [0.0, 5.0, 10.0],
        "LDN193189_uM": [0.0, nM_to_uM(50.0), nM_to_uM(100.0), nM_to_uM(200.0)],
    }

    print("\n  Screening ranges:")
    for morph, vals in morphogen_ranges.items():
        print(f"    {morph}: {vals}")

    # Run virtual screen
    virtual_X, virtual_Y, quality = run_virtual_screen(
        morphogen_ranges=morphogen_ranges,
        real_morphogen_csv=morphogen_path,
        harvest_days=[21, 45, 72],
        max_combinations=5000,
        model_path=DATA_DIR / "cellflow_model",
    )

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving virtual screening data")
    print("=" * 60)

    virtual_Y.to_csv(str(DATA_DIR / "cellflow_virtual_fractions.csv"))
    virtual_X.to_csv(str(DATA_DIR / "cellflow_virtual_morphogens.csv"))
    quality.to_csv(str(DATA_DIR / "cellflow_screening_report.csv"), index=False)

    print(f"  Virtual fractions  -> data/cellflow_virtual_fractions.csv")
    print(f"  Virtual morphogens -> data/cellflow_virtual_morphogens.csv")
    print(f"  Screening report   -> data/cellflow_screening_report.csv")

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("CELLFLOW VIRTUAL SCREEN SUMMARY")
    print("=" * 60)
    print(f"  Protocols screened:  {len(virtual_X):,}")
    print(f"  Cell types predicted: {virtual_Y.shape[1]}")
    print(f"  Mean confidence:     {quality['confidence'].mean():.3f}")
    print(f"  Fidelity level:      {FIDELITY_LEVEL}")
    print(f"  Time elapsed:        {elapsed:.1f}s")


if __name__ == "__main__":
    main()
