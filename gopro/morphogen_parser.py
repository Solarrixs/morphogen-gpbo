"""Morphogen concentration parser for Amin/Kelley GSE233574 condition names.

Parses the 46 unique condition names from the Amin & Kelley (2024, Cell Stem Cell)
morphogen screen into numeric concentration vectors suitable for GP-BO.

Each condition name encodes which morphogens were applied, at what concentration,
and (sometimes) during which temporal window. This module decodes those strings
into a DataFrame of morphogen concentrations aligned with MORPHOGEN_COLUMNS.

Usage:
    python gopro/morphogen_parser.py          # Print full morphogen matrix
    from gopro.morphogen_parser import build_morphogen_matrix
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from gopro.config import MORPHOGEN_COLUMNS, get_logger

logger = get_logger(__name__)

# ==============================================================================
# Default concentrations (from Amin & Kelley 2024)
# ==============================================================================
_DEFAULTS: dict[str, float] = {
    "CHIR99021_uM": 1.5,
    "BMP4_ng_mL": 10.0,
    "BMP7_ng_mL": 25.0,
    "SAG_nM": 50.0,       # Lowest SAG tier; explicit concentrations override
    "RA_nM": 100.0,
    "FGF2_ng_mL": 20.0,
    "FGF4_ng_mL": 100.0,
    "FGF8_ng_mL": 100.0,
    "IWP2_uM": 5.0,
    "LDN193189_nM": 100.0,
    "DAPT_uM": 2.5,
    "Dorsomorphin_uM": 2.5,
    "EGF_ng_mL": 20.0,
    "SB431542_uM": 10.0,
    "ActivinA_ng_mL": 50.0,
}

# Harvest day for all Amin/Kelley conditions
_HARVEST_DAY: int = 72
_LOG_HARVEST_DAY: float = math.log(_HARVEST_DAY)

# ==============================================================================
# Timing helpers
# ==============================================================================
# Standard protocol window is days 6-21 (15 days).
_STANDARD_WINDOW: tuple[int, int] = (6, 21)
_STANDARD_DURATION: float = float(_STANDARD_WINDOW[1] - _STANDARD_WINDOW[0])  # 15


def _time_fraction(start: int, end: int) -> float:
    """Return the fraction of the standard window covered by [start, end].

    Args:
        start: First day of morphogen application.
        end: Last day of morphogen application.

    Returns:
        Fraction in [0, 1] representing temporal coverage.
    """
    return (end - start) / _STANDARD_DURATION


# ==============================================================================
# Per-condition parser
# ==============================================================================

def _zeros() -> dict[str, float]:
    """Return a zeroed-out morphogen vector."""
    return {col: 0.0 for col in MORPHOGEN_COLUMNS}


def parse_condition_name(name: str) -> dict[str, float]:
    """Parse a single Amin/Kelley condition name into morphogen concentrations.

    Args:
        name: Condition name string exactly as it appears in the metadata CSV.

    Returns:
        Dictionary mapping each entry in MORPHOGEN_COLUMNS to a numeric
        concentration value. Morphogens not present in the condition are 0.

    Raises:
        ValueError: If the condition name is not recognized.
    """
    vec = _zeros()
    vec["log_harvest_day"] = _LOG_HARVEST_DAY

    # Dispatch to the explicit lookup table
    if name not in _CONDITION_PARSERS:
        raise ValueError(f"Unrecognized condition name: {name!r}")

    _CONDITION_PARSERS[name](vec)
    return vec


# ---------------------------------------------------------------------------
# Individual condition handlers
# ---------------------------------------------------------------------------
# Each handler mutates the pre-zeroed vector *vec* in place.

def _bmp4_chir(v: dict[str, float]) -> None:
    v["BMP4_ng_mL"] = 10.0
    v["CHIR99021_uM"] = 1.5

def _bmp4_chir_d11_16(v: dict[str, float]) -> None:
    v["BMP4_ng_mL"] = 10.0
    v["CHIR99021_uM"] = 1.5 * _time_fraction(11, 16)

def _bmp4_sag(v: dict[str, float]) -> None:
    v["BMP4_ng_mL"] = 10.0
    v["SAG_nM"] = 50.0

def _bmp7(v: dict[str, float]) -> None:
    v["BMP7_ng_mL"] = 25.0

def _bmp7_chir(v: dict[str, float]) -> None:
    v["BMP7_ng_mL"] = 25.0
    v["CHIR99021_uM"] = 1.5

def _bmp7_sag(v: dict[str, float]) -> None:
    v["BMP7_ng_mL"] = 25.0
    v["SAG_nM"] = 50.0

def _c_l_s_fgf8(v: dict[str, float]) -> None:
    """C/L/S/FGF8 = CHIR + LDN + SB431542 + FGF8."""
    v["CHIR99021_uM"] = 1.5
    v["LDN193189_nM"] = 100.0
    v["SB431542_uM"] = 10.0
    v["FGF8_ng_mL"] = 100.0

def _c_s_bmp7_d(v: dict[str, float]) -> None:
    """C/S/BMP7/D = CHIR + SB431542 + BMP7 + DAPT."""
    v["CHIR99021_uM"] = 1.5
    v["SB431542_uM"] = 10.0
    v["BMP7_ng_mL"] = 25.0
    v["DAPT_uM"] = 2.5

def _c_s_d_fgf4(v: dict[str, float]) -> None:
    """C/S/D/FGF4 = CHIR + SB431542 + DAPT + FGF4."""
    v["CHIR99021_uM"] = 1.5
    v["SB431542_uM"] = 10.0
    v["DAPT_uM"] = 2.5
    v["FGF4_ng_mL"] = 100.0

def _c_s_r_e_fgf2_d(v: dict[str, float]) -> None:
    """C/S/R/E/FGF2/D = CHIR + SB431542 + RA + EGF + FGF2 + DAPT."""
    v["CHIR99021_uM"] = 1.5
    v["SB431542_uM"] = 10.0
    v["RA_nM"] = 100.0
    v["EGF_ng_mL"] = 20.0
    v["FGF2_ng_mL"] = 20.0
    v["DAPT_uM"] = 2.5

def _chir_sag_fgf4(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5
    v["SAG_nM"] = 50.0
    v["FGF4_ng_mL"] = 100.0

def _chir_sag_fgf8(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5
    v["SAG_nM"] = 50.0
    v["FGF8_ng_mL"] = 100.0

def _chir_switch_iwp2(v: dict[str, float]) -> None:
    """CHIR first half, then IWP2 second half of standard window."""
    v["CHIR99021_uM"] = 1.5 * _time_fraction(6, 13)
    v["IWP2_uM"] = 5.0 * _time_fraction(13, 21)

def _chir_d11_16(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5 * _time_fraction(11, 16)

def _chir_d16_21(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5 * _time_fraction(16, 21)

def _chir_d6_11(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5 * _time_fraction(6, 11)

def _chir_sag_d16_21(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5 * _time_fraction(16, 21)
    v["SAG_nM"] = 50.0 * _time_fraction(16, 21)

def _chir_sag_ldn(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5
    v["SAG_nM"] = 50.0
    v["LDN193189_nM"] = 100.0

def _chir_sagd10_21(v: dict[str, float]) -> None:
    """CHIR full window, SAG days 10-21 only."""
    v["CHIR99021_uM"] = 1.5
    v["SAG_nM"] = 50.0 * _time_fraction(10, 21)

def _chir1_5_sag1000(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5
    v["SAG_nM"] = 1000.0

def _chir1_5_sag250(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5
    v["SAG_nM"] = 250.0

def _chir1_5(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 1.5

def _chir3_sag1000(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 3.0
    v["SAG_nM"] = 1000.0

def _chir3_sag250(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 3.0
    v["SAG_nM"] = 250.0

def _chir3(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 3.0

def _dapt(v: dict[str, float]) -> None:
    v["DAPT_uM"] = 2.5

def _fgf20_egf(v: dict[str, float]) -> None:
    """FGF-20/EGF: FGF2 at 20 ng/mL + EGF."""
    v["FGF2_ng_mL"] = 20.0
    v["EGF_ng_mL"] = 20.0

def _fgf2_20(v: dict[str, float]) -> None:
    v["FGF2_ng_mL"] = 20.0

def _fgf2_50(v: dict[str, float]) -> None:
    v["FGF2_ng_mL"] = 50.0

def _fgf4(v: dict[str, float]) -> None:
    v["FGF4_ng_mL"] = 100.0

def _fgf8(v: dict[str, float]) -> None:
    v["FGF8_ng_mL"] = 100.0

def _i_activin_dapt_sr11(v: dict[str, float]) -> None:
    """I/Activin/DAPT/SR11 = IWP2 + Activin A + DAPT + SR11237 (retinoid)."""
    v["IWP2_uM"] = 5.0
    v["ActivinA_ng_mL"] = 50.0
    v["DAPT_uM"] = 2.5
    v["RA_nM"] = 100.0  # SR11237 treated as RA equivalent

def _iwp2(v: dict[str, float]) -> None:
    v["IWP2_uM"] = 5.0

def _iwp2_switch_chir(v: dict[str, float]) -> None:
    """IWP2 first half, then CHIR second half of standard window."""
    v["IWP2_uM"] = 5.0 * _time_fraction(6, 13)
    v["CHIR99021_uM"] = 1.5 * _time_fraction(13, 21)

def _iwp2_sag(v: dict[str, float]) -> None:
    v["IWP2_uM"] = 5.0
    v["SAG_nM"] = 50.0

def _ldn(v: dict[str, float]) -> None:
    v["LDN193189_nM"] = 100.0

def _ra10(v: dict[str, float]) -> None:
    v["RA_nM"] = 10.0

def _ra100(v: dict[str, float]) -> None:
    v["RA_nM"] = 100.0

def _s_i_e_fgf2(v: dict[str, float]) -> None:
    """S/I/E/FGF2 = SB431542 + IWP2 + EGF + FGF2."""
    v["SB431542_uM"] = 10.0
    v["IWP2_uM"] = 5.0
    v["EGF_ng_mL"] = 20.0
    v["FGF2_ng_mL"] = 20.0

def _sag_chir_d16_21(v: dict[str, float]) -> None:
    """SAG full window, CHIR days 16-21 only."""
    v["SAG_nM"] = 50.0
    v["CHIR99021_uM"] = 1.5 * _time_fraction(16, 21)

def _sag_chird10_21(v: dict[str, float]) -> None:
    """SAG full window, CHIR days 10-21 only."""
    v["SAG_nM"] = 50.0
    v["CHIR99021_uM"] = 1.5 * _time_fraction(10, 21)

def _sag_d11_16(v: dict[str, float]) -> None:
    v["SAG_nM"] = 50.0 * _time_fraction(11, 16)

def _sag_d16_21(v: dict[str, float]) -> None:
    v["SAG_nM"] = 50.0 * _time_fraction(16, 21)

def _sag_d6_11(v: dict[str, float]) -> None:
    v["SAG_nM"] = 50.0 * _time_fraction(6, 11)

def _sag1000(v: dict[str, float]) -> None:
    v["SAG_nM"] = 1000.0

def _sag250(v: dict[str, float]) -> None:
    v["SAG_nM"] = 250.0


# ==============================================================================
# Dispatch table: condition name -> handler
# ==============================================================================
_CONDITION_PARSERS: dict[str, Any] = {
    "BMP4 CHIR":            _bmp4_chir,
    "BMP4 CHIR d11-16":     _bmp4_chir_d11_16,
    "BMP4 SAG":             _bmp4_sag,
    "BMP7":                 _bmp7,
    "BMP7 CHIR":            _bmp7_chir,
    "BMP7 SAG":             _bmp7_sag,
    "C/L/S/FGF8":           _c_l_s_fgf8,
    "C/S/BMP7/D":           _c_s_bmp7_d,
    "C/S/D/FGF4":           _c_s_d_fgf4,
    "C/S/R/E/FGF2/D":       _c_s_r_e_fgf2_d,
    "CHIR SAG FGF4":        _chir_sag_fgf4,
    "CHIR SAG FGF8":        _chir_sag_fgf8,
    "CHIR switch IWP2":     _chir_switch_iwp2,
    "CHIR-d11-16":          _chir_d11_16,
    "CHIR-d16-21":          _chir_d16_21,
    "CHIR-d6-11":           _chir_d6_11,
    "CHIR-SAG-d16-21":      _chir_sag_d16_21,
    "CHIR-SAG-LDN":         _chir_sag_ldn,
    "CHIR-SAGd10-21":       _chir_sagd10_21,
    "CHIR1.5-SAG1000":      _chir1_5_sag1000,
    "CHIR1.5-SAG250":       _chir1_5_sag250,
    "CHIR1.5":              _chir1_5,
    "CHIR3-SAG1000":        _chir3_sag1000,
    "CHIR3-SAG250":         _chir3_sag250,
    "CHIR3":                _chir3,
    "DAPT":                 _dapt,
    "FGF-20/EGF":           _fgf20_egf,
    "FGF2-20":              _fgf2_20,
    "FGF2-50":              _fgf2_50,
    "FGF4":                 _fgf4,
    "FGF8":                 _fgf8,
    "I/Activin/DAPT/SR11":  _i_activin_dapt_sr11,
    "IWP2":                 _iwp2,
    "IWP2 switch CHIR":     _iwp2_switch_chir,
    "IWP2-SAG":             _iwp2_sag,
    "LDN":                  _ldn,
    "RA10":                 _ra10,
    "RA100":                _ra100,
    "S/I/E/FGF2":           _s_i_e_fgf2,
    "SAG-CHIR-d16-21":      _sag_chir_d16_21,
    "SAG-CHIRd10-21":       _sag_chird10_21,
    "SAG-d11-16":           _sag_d11_16,
    "SAG-d16-21":           _sag_d16_21,
    "SAG-d6-11":            _sag_d6_11,
    "SAG1000":              _sag1000,
    "SAG250":               _sag250,
}

# Sanity check: we must have exactly 46 conditions
assert len(_CONDITION_PARSERS) == 46, (
    f"Expected 46 conditions, got {len(_CONDITION_PARSERS)}"
)


# ==============================================================================
# Public API
# ==============================================================================

def build_morphogen_matrix(conditions: list[str]) -> pd.DataFrame:
    """Build a morphogen concentration matrix for a list of condition names.

    Args:
        conditions: List of condition name strings from the Amin/Kelley dataset.

    Returns:
        DataFrame of shape (len(conditions), len(MORPHOGEN_COLUMNS)) with
        condition names as the index and morphogen concentrations as columns.
        Morphogens not present in a condition are 0.0.
    """
    rows = [parse_condition_name(c) for c in conditions]
    df = pd.DataFrame(rows, index=conditions, columns=MORPHOGEN_COLUMNS)
    return df


ALL_CONDITIONS: list[str] = sorted(_CONDITION_PARSERS.keys())
"""All 46 condition names in alphabetical order."""


# ==============================================================================
# Main: print the full matrix for inspection
# ==============================================================================

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    df = build_morphogen_matrix(ALL_CONDITIONS)

    logger.info("Morphogen matrix: %d conditions x %d columns", df.shape[0], df.shape[1])
    logger.info("Columns: %s", list(df.columns))

    # Show non-zero entries per condition for readability
    for cond in ALL_CONDITIONS:
        row = df.loc[cond]
        nonzero = row[row > 0]
        parts = [f"{col}={val:.2f}" for col, val in nonzero.items()]
        logger.info("  %s -> %s", cond, ', '.join(parts))

    logger.info("Full matrix:\n%s", df.to_string())
