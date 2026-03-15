"""Centralized configuration for the GP-BO pipeline.

This module provides all shared constants, paths, and logging utilities
used across the pipeline. Import from here instead of defining locally.

IMPORTANT: This module must NOT import from any other gopro module
to avoid circular imports.
"""

import logging
import os
from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(os.environ.get("GPBO_PROJECT_DIR", str(Path(__file__).resolve().parent.parent)))
DATA_DIR = Path(os.environ.get("GPBO_DATA_DIR", str(PROJECT_DIR / "data")))
MODEL_DIR = Path(os.environ.get(
    "GPBO_MODEL_DIR",
    str(PROJECT_DIR / "data" / "neural_organoid_atlas" / "supplemental_files" / "scpoli_model_params"),
))

# --- Morphogen columns (canonical ordering, all concentrations in µM) ---
MORPHOGEN_COLUMNS: list[str] = [
    "CHIR99021_uM",       # 0 - WNT agonist
    "BMP4_uM",            # 1 - BMP signaling
    "BMP7_uM",            # 2 - BMP signaling
    "SHH_uM",             # 3 - Sonic hedgehog
    "SAG_uM",             # 4 - Smoothened agonist
    "RA_uM",              # 5 - Retinoic acid
    "FGF8_uM",            # 6 - FGF8
    "FGF2_uM",            # 7 - FGF2
    "FGF4_uM",            # 8 - FGF4
    "IWP2_uM",            # 9 - WNT inhibitor
    "XAV939_uM",          # 10 - WNT inhibitor
    "SB431542_uM",        # 11 - TGF-beta inhibitor
    "LDN193189_uM",       # 12 - BMP inhibitor
    "DAPT_uM",            # 13 - Notch inhibitor
    "EGF_uM",             # 14 - EGF
    "ActivinA_uM",        # 15 - Activin
    "Dorsomorphin_uM",    # 16 - BMP inhibitor
    "purmorphamine_uM",   # 17 - SHH agonist
    "cyclopamine_uM",     # 18 - SHH antagonist
    "log_harvest_day",    # 19 - Time dimension
    "BDNF_uM",            # 20 - Brain-derived neurotrophic factor
    "NT3_uM",             # 21 - Neurotrophin-3
    "cAMP_uM",            # 22 - Dibutyryl-cAMP
    "AscorbicAcid_uM",   # 23 - L-Ascorbic acid 2-phosphate
]

# --- Molecular weights (kDa) for recombinant protein morphogens ---
# Used to convert ng/mL → µM: µM = (ng/mL) / (MW_kDa × 1000)
PROTEIN_MW_KDA: dict[str, float] = {
    "BMP4":     13.0,   # Mature monomer (recombinant human BMP4)
    "BMP7":     15.7,   # Mature monomer
    "SHH":      19.6,   # N-terminal signaling domain
    "FGF2":     17.2,   # 154-aa isoform (bFGF)
    "FGF4":     19.2,
    "FGF8":     22.5,   # FGF8b isoform
    "EGF":       6.2,
    "ActivinA": 26.0,   # Homodimer
    "BDNF":     13.5,   # Mature monomer (recombinant human BDNF)
    "NT3":      13.6,   # Mature monomer (neurotrophin-3)
}


def ng_mL_to_uM(ng_per_mL: float, mw_kda: float) -> float:
    """Convert ng/mL to µM using molecular weight.

    Formula: µM = (ng/mL) / (MW_Da) = (ng/mL) / (MW_kDa × 1000)
    """
    return ng_per_mL / (mw_kda * 1000.0)


def nM_to_uM(nM: float) -> float:
    """Convert nM to µM."""
    return nM / 1000.0

# --- HNOCA annotation column names ---
ANNOT_LEVEL_1 = "annot_level_1"
ANNOT_LEVEL_2 = "annot_level_2"
ANNOT_REGION = "annot_region_rev2"
ANNOT_LEVEL_3 = "annot_level_3_rev2"

# --- Logging ---
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        Configured ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    level = os.environ.get("GPBO_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger

# --- File hashing ---

def md5_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute MD5 hash of a file without loading it all into memory.

    Args:
        path: Path to the file.
        chunk_size: Read buffer size in bytes (default 8 MiB).

    Returns:
        Hex-encoded MD5 digest string.
    """
    import hashlib
    from pathlib import Path as _Path

    path = _Path(path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Gruffi stress-filtering defaults ---
GRUFFI_DEFAULT_THRESHOLD = 0.15
GRUFFI_DEFAULT_RESOLUTION = 2.0
GRUFFI_MIN_CELLS_PER_CONDITION = 50
