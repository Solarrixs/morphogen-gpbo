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
if isinstance(PROJECT_DIR, str):
    PROJECT_DIR = Path(PROJECT_DIR)

DATA_DIR = Path(os.environ.get("GPBO_DATA_DIR", str(PROJECT_DIR / "data")))
if isinstance(DATA_DIR, str):
    DATA_DIR = Path(DATA_DIR)

MODEL_DIR = Path(os.environ.get(
    "GPBO_MODEL_DIR",
    str(PROJECT_DIR / "neural_organoid_atlas" / "supplemental_files" / "scpoli_model_params"),
))
if isinstance(MODEL_DIR, str):
    MODEL_DIR = Path(MODEL_DIR)

# --- Morphogen columns (canonical ordering) ---
MORPHOGEN_COLUMNS: list[str] = [
    "CHIR99021_uM",       # 0 - WNT agonist
    "BMP4_ng_mL",         # 1 - BMP signaling
    "BMP7_ng_mL",         # 2 - BMP signaling
    "SHH_ng_mL",          # 3 - Sonic hedgehog
    "SAG_nM",             # 4 - Smoothened agonist
    "RA_nM",              # 5 - Retinoic acid
    "FGF8_ng_mL",         # 6 - FGF8
    "FGF2_ng_mL",         # 7 - FGF2
    "FGF4_ng_mL",         # 8 - FGF4
    "IWP2_uM",            # 9 - WNT inhibitor
    "XAV939_uM",          # 10 - WNT inhibitor
    "SB431542_uM",        # 11 - TGF-beta inhibitor
    "LDN193189_nM",       # 12 - BMP inhibitor
    "DAPT_uM",            # 13 - Notch inhibitor
    "EGF_ng_mL",          # 14 - EGF
    "ActivinA_ng_mL",     # 15 - Activin
    "Dorsomorphin_uM",    # 16 - BMP inhibitor
    "purmorphamine_uM",   # 17 - SHH agonist
    "cyclopamine_uM",     # 18 - SHH antagonist
    "log_harvest_day",    # 19 - Time dimension
]

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
