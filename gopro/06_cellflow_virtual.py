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
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from gopro.config import (
    DATA_DIR,
    PROTEIN_MW_KDA as MW,
    get_logger,
    nM_to_uM,
    ng_mL_to_uM,
)
from gopro.morphogen_parser import _BASE_MEDIA

logger = get_logger(__name__)

# Low fidelity level for CellFlow virtual data
FIDELITY_LEVEL = 0.0

# Candidate paths for pre-trained CellFlow model
CELLFLOW_MODEL_PATHS = [
    DATA_DIR / "cellflow_model",
    DATA_DIR / "cellflow_model.pt",
    DATA_DIR / "patterning_screen" / "cellflow_model",
]

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
    "BDNF_uM": ("protein", "BDNF"),
    "NT3_uM": ("protein", "NT3"),
    "cAMP_uM": ("small_molecule", "dibutyryl_cAMP"),
    "AscorbicAcid_uM": ("small_molecule", "ascorbic_acid_2_phosphate"),
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
    "BDNF_uM": "neurotrophin",
    "NT3_uM": "neurotrophin",
    "cAMP_uM": "unknown",
    "AscorbicAcid_uM": "unknown",
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
            "concentration": math.log1p(conc),
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

    from gopro.config import MORPHOGEN_COLUMNS

    # Build the grid
    grid_dims = {}
    for col in MORPHOGEN_COLUMNS:
        if col == "log_harvest_day":
            grid_dims[col] = [math.log(d) for d in harvest_days]
        elif col in morphogen_ranges:
            grid_dims[col] = morphogen_ranges[col]
        else:
            grid_dims[col] = [_BASE_MEDIA.get(col, 0.0)]  # Use base media defaults

    # Generate combinations
    keys = list(grid_dims.keys())
    value_lists = [grid_dims[k] for k in keys]

    # Calculate total combinations
    total = 1
    for v in value_lists:
        total *= len(v)

    if total > max_combinations:
        logger.info("Grid would produce %s combinations, sampling %s randomly",
                     f"{total:,}", f"{max_combinations:,}")
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

    logger.info("Generated %s virtual protocol combinations", f"{len(grid):,}")
    return grid


def discover_cellflow_model() -> Optional[Path]:
    """Search candidate paths for a pre-trained CellFlow model.

    Checks ``CELLFLOW_MODEL_PATHS`` in order and returns the first
    path that exists, or ``None`` if no model is found.

    Returns:
        Path to CellFlow model directory/file, or None.
    """
    for candidate in CELLFLOW_MODEL_PATHS:
        if candidate.exists():
            logger.info("Found pre-trained CellFlow model at %s", candidate)
            return candidate
    return None


def predict_cellflow(
    protocols: pd.DataFrame,
    model_path: Optional[Path] = None,
    batch_size: int = 256,
    n_cells_per_condition: int = 500,
    real_fractions_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Run CellFlow predictions for a batch of protocols.

    Uses the CellFlow generative model to predict single-cell
    distributions, then aggregates to cell type fractions.

    If ``model_path`` is None, automatically searches
    ``CELLFLOW_MODEL_PATHS`` for a pre-trained model.

    Args:
        protocols: DataFrame of morphogen concentration vectors.
        model_path: Path to pre-trained CellFlow model.
        batch_size: Number of protocols to predict at once.
        n_cells_per_condition: Number of virtual cells to generate.
        real_fractions_csv: Path to real training fractions CSV for
            cell type vocabulary alignment (used by heuristic fallback).

    Returns:
        DataFrame with predicted cell type fractions per protocol.
    """
    # Auto-discover model if not explicitly provided
    if model_path is None:
        model_path = discover_cellflow_model()

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
        if not HAS_CELLFLOW:
            logger.info("CellFlow package not installed. Falling back to heuristic predictor.")
        elif model_path is None:
            logger.info(
                "No pre-trained CellFlow model found. Searched: %s. "
                "Falling back to heuristic predictor.",
                [str(p) for p in CELLFLOW_MODEL_PATHS],
            )
        else:
            logger.info("CellFlow model path does not exist: %s. Falling back to heuristic.", model_path)
        return _predict_baseline(protocols, real_fractions_csv=real_fractions_csv)


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

    logger.info("Loading CellFlow model from %s...", model_path)
    model = cellflow.CellFlowModel.load(str(model_path))

    all_fractions = []
    n_batches = (len(protocols) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(protocols))
        batch = protocols.iloc[start:end]

        if batch_idx % 10 == 0:
            logger.info("Predicting batch %d/%d (%d-%d)...",
                        batch_idx + 1, n_batches, start, end)

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


def sigmoid_response(concentration: float, ec50: float, hill_coeff: float = 1.0) -> float:
    """Compute sigmoid (Hill) dose-response curve.

    Models the saturating effect of a morphogen at increasing concentration.
    Returns a value in [0, 1] where 0 = no effect, 1 = maximal effect.

    Args:
        concentration: Morphogen concentration (same units as ec50).
        ec50: Half-maximal effective concentration.
        hill_coeff: Hill coefficient controlling steepness (default 1.0).

    Returns:
        Fractional response in [0, 1].
    """
    if concentration <= 0 or ec50 <= 0:
        return 0.0
    return float(
        concentration ** hill_coeff
        / (ec50 ** hill_coeff + concentration ** hill_coeff)
    )


# Morphogen-pathway lookup table mapping each morphogen to its signaling
# pathway, agonist/antagonist direction, EC50 (µM), Hill coefficient,
# and expected effects on cell type composition.
# EC50 values are approximate midpoints of published dose-response ranges.
MORPHOGEN_PATHWAY_MAP: dict[str, dict] = {
    "CHIR99021_uM": {
        "pathway": "WNT", "direction": "agonist", "ec50": 3.0, "hill": 1.5,
        "effects": {
            "Neuron": +0.15, "NPC": -0.05, "IP": +0.05,
            "Neuroepithelium": -0.10, "Glioblast": +0.03,
        },
    },
    "IWP2_uM": {
        "pathway": "WNT", "direction": "antagonist", "ec50": 2.5, "hill": 1.0,
        "effects": {
            "NPC": +0.15, "Neuroepithelium": +0.10, "Neuron": -0.05,
        },
    },
    "XAV939_uM": {
        "pathway": "WNT", "direction": "antagonist", "ec50": 5.0, "hill": 1.0,
        "effects": {
            "NPC": +0.10, "Neuroepithelium": +0.08, "Neuron": -0.03,
        },
    },
    "BMP4_uM": {
        "pathway": "BMP", "direction": "agonist", "ec50": 0.001, "hill": 1.2,
        "effects": {
            "CP": +0.15, "MC": +0.05, "Neuroepithelium": -0.05, "NPC": -0.05,
        },
    },
    "BMP7_uM": {
        "pathway": "BMP", "direction": "agonist", "ec50": 0.002, "hill": 1.0,
        "effects": {
            "CP": +0.10, "MC": +0.03, "NPC": -0.03,
        },
    },
    "LDN193189_uM": {
        "pathway": "BMP", "direction": "antagonist", "ec50": 0.1, "hill": 1.5,
        "effects": {
            "Neuroepithelium": +0.12, "NPC": +0.05, "PSC": -0.03, "MC": -0.05,
        },
    },
    "Dorsomorphin_uM": {
        "pathway": "BMP", "direction": "antagonist", "ec50": 2.0, "hill": 1.0,
        "effects": {
            "Neuroepithelium": +0.08, "MC": -0.03,
        },
    },
    "SHH_uM": {
        "pathway": "SHH", "direction": "agonist", "ec50": 0.005, "hill": 1.2,
        "effects": {
            "Neuron": +0.10, "IP": +0.05, "NPC": +0.03,
        },
    },
    "SAG_uM": {
        "pathway": "SHH", "direction": "agonist", "ec50": 0.5, "hill": 1.0,
        "effects": {
            "Neuron": +0.10, "IP": +0.05, "NPC": +0.03,
        },
    },
    "purmorphamine_uM": {
        "pathway": "SHH", "direction": "agonist", "ec50": 1.0, "hill": 1.0,
        "effects": {
            "Neuron": +0.08, "IP": +0.04,
        },
    },
    "cyclopamine_uM": {
        "pathway": "SHH", "direction": "antagonist", "ec50": 5.0, "hill": 1.0,
        "effects": {
            "Neuron": -0.05, "IP": -0.03, "NPC": +0.03,
        },
    },
    "RA_uM": {
        "pathway": "RA", "direction": "agonist", "ec50": 0.1, "hill": 1.0,
        "effects": {
            "Neuron": +0.10, "IP": +0.05, "NPC": -0.03,
        },
    },
    "FGF2_uM": {
        "pathway": "FGF", "direction": "agonist", "ec50": 0.001, "hill": 1.0,
        "effects": {
            "NPC": +0.05, "Glioblast": +0.08, "Astrocyte": +0.03,
        },
    },
    "FGF4_uM": {
        "pathway": "FGF", "direction": "agonist", "ec50": 0.002, "hill": 1.0,
        "effects": {
            "NPC": +0.05, "Glioblast": +0.06,
        },
    },
    "FGF8_uM": {
        "pathway": "FGF", "direction": "agonist", "ec50": 0.003, "hill": 1.2,
        "effects": {
            "Glioblast": +0.08, "NPC": +0.05, "Neuron": +0.03,
        },
    },
    "SB431542_uM": {
        "pathway": "TGFb", "direction": "antagonist", "ec50": 5.0, "hill": 1.5,
        "effects": {
            "Neuroepithelium": +0.12, "NPC": +0.05, "PSC": -0.03, "MC": -0.03,
        },
    },
    "ActivinA_uM": {
        "pathway": "TGFb", "direction": "agonist", "ec50": 0.001, "hill": 1.0,
        "effects": {
            "PSC": +0.05, "MC": +0.03, "Neuroepithelium": -0.05,
        },
    },
    "DAPT_uM": {
        "pathway": "Notch", "direction": "antagonist", "ec50": 5.0, "hill": 1.0,
        "effects": {
            "Neuron": +0.15, "NPC": -0.08, "IP": +0.05,
        },
    },
    "EGF_uM": {
        "pathway": "EGF", "direction": "agonist", "ec50": 0.003, "hill": 1.0,
        "effects": {
            "NPC": +0.05, "Glioblast": +0.05, "OPC": +0.03,
        },
    },
}


def _compute_pathway_antagonism(
    row: pd.Series,
) -> dict[str, float]:
    """Compute net agonist/antagonist balance per pathway.

    When both an agonist and antagonist for the same pathway are present,
    their effects partially cancel. Returns a per-pathway scaling factor
    in [0, 1] where 0 = fully cancelled, 1 = no antagonism.

    Args:
        row: A single protocol's morphogen concentrations.

    Returns:
        Dict mapping pathway name to net scaling factor.
    """
    # Accumulate agonist and antagonist sigmoid responses per pathway
    pathway_agonist: dict[str, float] = {}
    pathway_antagonist: dict[str, float] = {}

    for morph_col, info in MORPHOGEN_PATHWAY_MAP.items():
        conc = float(row.get(morph_col, 0.0))
        if conc <= 0:
            continue
        response = sigmoid_response(conc, info["ec50"], info.get("hill", 1.0))
        pathway = info["pathway"]
        if info["direction"] == "agonist":
            pathway_agonist[pathway] = pathway_agonist.get(pathway, 0.0) + response
        else:
            pathway_antagonist[pathway] = pathway_antagonist.get(pathway, 0.0) + response

    # Compute net scaling: agonist signal is reduced by antagonist presence
    all_pathways = set(pathway_agonist.keys()) | set(pathway_antagonist.keys())
    scaling = {}
    for pw in all_pathways:
        ago = pathway_agonist.get(pw, 0.0)
        ant = pathway_antagonist.get(pw, 0.0)
        if ago + ant > 0:
            # Net effect: agonist fraction minus antagonist cancellation
            # Both agonists and antagonists are kept but with reduced magnitude
            scaling[pw] = max(0.0, (ago - ant) / (ago + ant))
        else:
            scaling[pw] = 1.0
    return scaling


def _load_dirichlet_prior(
    real_fractions_csv: Optional[Path],
    cell_types: list[str],
) -> np.ndarray:
    """Load a Dirichlet prior from real training data.

    If real training fractions are available, the mean composition across
    all conditions is used as the prior (more informative than uniform).
    Otherwise falls back to a uniform prior.

    Args:
        real_fractions_csv: Path to real training fractions CSV.
        cell_types: List of cell type names to align to.

    Returns:
        1-D array of shape (len(cell_types),) summing to 1.0.
    """
    if real_fractions_csv is not None and real_fractions_csv.exists():
        real_Y = pd.read_csv(str(real_fractions_csv), index_col=0)
        # Compute mean composition across all real conditions
        prior = np.zeros(len(cell_types))
        for i, ct in enumerate(cell_types):
            if ct in real_Y.columns:
                prior[i] = float(real_Y[ct].mean())
            else:
                prior[i] = 0.01  # small pseudo-count for unseen types
        # Ensure positive and normalized
        prior = np.maximum(prior, 0.01)
        prior = prior / prior.sum()
        return prior
    else:
        # Uniform Dirichlet prior
        return np.ones(len(cell_types)) / len(cell_types)


def _predict_baseline(
    protocols: pd.DataFrame,
    real_fractions_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Baseline prediction when CellFlow is not available.

    Uses a heuristic model based on:
    1. Dirichlet prior from real training data (or uniform fallback)
    2. Morphogen-pathway lookup table with sigmoid dose-response curves
    3. Pathway antagonism (agonist + inhibitor partially cancel)

    Args:
        protocols: Morphogen concentration vectors.
        real_fractions_csv: Path to real training fractions CSV. If provided,
            cell type vocabulary and Dirichlet prior are loaded from it.

    Returns:
        Predicted cell type fractions (heuristic).
    """
    logger.info("Using heuristic baseline predictor (sigmoid dose-response)...")

    # Load cell type vocabulary from real training data if available
    if real_fractions_csv is not None and real_fractions_csv.exists():
        real_Y = pd.read_csv(str(real_fractions_csv), index_col=0)
        CELL_TYPES = list(real_Y.columns)
        logger.info("Loaded %d cell types from real training data", len(CELL_TYPES))
    else:
        # Fallback: HNOCA level-1 labels
        CELL_TYPES = [
            "Neuron", "NPC", "IP", "Neuroepithelium", "Glioblast",
            "Astrocyte", "OPC", "CP", "PSC", "MC",
        ]
        logger.warning("No real training data — using level-1 fallback cell types")

    # Load Dirichlet prior
    prior = _load_dirichlet_prior(real_fractions_csv, CELL_TYPES)

    results = []
    for cond_name, row in protocols.iterrows():
        # Start with prior composition (data-driven or uniform)
        frac_arr = prior.copy()

        # Compute pathway antagonism scaling
        pw_scaling = _compute_pathway_antagonism(row)

        # Apply morphogen effects via sigmoid dose-response
        for morph_col, info in MORPHOGEN_PATHWAY_MAP.items():
            conc = float(row.get(morph_col, 0.0))
            if conc <= 0:
                continue

            response = sigmoid_response(conc, info["ec50"], info.get("hill", 1.0))

            # Scale by pathway antagonism
            pathway = info["pathway"]
            pw_scale = pw_scaling.get(pathway, 1.0)

            # For antagonists, the antagonism scaling represents how much
            # the antagonist "wins" — invert for antagonist-specific effects
            if info["direction"] == "antagonist":
                effective_response = response * (1.0 - pw_scale)
            else:
                effective_response = response * pw_scale

            # Apply per-cell-type effects
            for ct, effect_magnitude in info["effects"].items():
                if ct in CELL_TYPES:
                    idx = CELL_TYPES.index(ct)
                    frac_arr[idx] += effect_magnitude * effective_response

        # Ensure all values positive (floor at 0.01)
        frac_arr = np.maximum(frac_arr, 0.01)
        # Normalize to sum to 1
        frac_arr = frac_arr / frac_arr.sum()
        results.append(dict(zip(CELL_TYPES, frac_arr)))

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
    real_fractions_csv: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full CellFlow virtual screen.

    Args:
        morphogen_ranges: Morphogen concentration ranges to screen.
        real_morphogen_csv: Path to real training morphogen CSV.
        harvest_days: Harvest days to include.
        max_combinations: Max number of virtual protocols.
        model_path: Path to CellFlow model (optional).
        real_fractions_csv: Path to real training fractions CSV for
            cell type vocabulary alignment.

    Returns:
        Tuple of (virtual_morphogens, virtual_fractions, quality_report).
    """
    # Validate real fractions CSV if provided
    if real_fractions_csv is not None and real_fractions_csv.exists():
        from gopro.validation import validate_training_csvs
        validate_training_csvs(real_fractions_csv, real_morphogen_csv)

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
        real_fractions_csv=real_fractions_csv,
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

    logger.info("=" * 60)
    logger.info("PHASE 5: CellFlow Virtual Protocol Screening")
    logger.info("=" * 60)

    morphogen_path = DATA_DIR / "morphogen_matrix_amin_kelley.csv"
    if not morphogen_path.exists():
        logger.error("Morphogen matrix not found at %s", morphogen_path)
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

    logger.info("Screening ranges:")
    for morph, vals in morphogen_ranges.items():
        logger.info("  %s: %s", morph, vals)

    # Run virtual screen
    virtual_X, virtual_Y, quality = run_virtual_screen(
        morphogen_ranges=morphogen_ranges,
        real_morphogen_csv=morphogen_path,
        harvest_days=[21, 45, 72],
        max_combinations=5000,
        model_path=DATA_DIR / "cellflow_model",
        real_fractions_csv=DATA_DIR / "gp_training_labels_amin_kelley.csv",
    )

    # Save outputs
    logger.info("Saving virtual screening data")

    virtual_Y.to_csv(str(DATA_DIR / "cellflow_virtual_fractions.csv"))
    virtual_X.to_csv(str(DATA_DIR / "cellflow_virtual_morphogens.csv"))
    # TODO: Wire confidence scores into merge_multi_fidelity_data() to filter low-confidence predictions
    quality.to_csv(str(DATA_DIR / "cellflow_screening_report.csv"), index=False)

    logger.info("Virtual fractions  -> data/cellflow_virtual_fractions.csv")
    logger.info("Virtual morphogens -> data/cellflow_virtual_morphogens.csv")
    logger.info("Screening report   -> data/cellflow_screening_report.csv")

    # Summary
    elapsed = time.time() - start
    logger.info("CELLFLOW VIRTUAL SCREEN SUMMARY")
    logger.info("Protocols screened:  %s", f"{len(virtual_X):,}")
    logger.info("Cell types predicted: %d", virtual_Y.shape[1])
    logger.info("Mean confidence:     %.3f", quality["confidence"].mean())
    logger.info("Fidelity level:      %s", FIDELITY_LEVEL)
    logger.info("Time elapsed:        %.1fs", elapsed)


if __name__ == "__main__":
    main()
