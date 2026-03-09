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

from gopro.config import (
    MORPHOGEN_COLUMNS,
    PROTEIN_MW_KDA,
    nM_to_uM,
    ng_mL_to_uM,
    get_logger,
)

logger = get_logger(__name__)

# ==============================================================================
# Default concentrations (from Amin & Kelley 2024) — all in µM
# ==============================================================================
_DEFAULTS: dict[str, float] = {
    "CHIR99021_uM":    1.5,
    "BMP4_uM":         ng_mL_to_uM(10.0, PROTEIN_MW_KDA["BMP4"]),      # 10 ng/mL
    "BMP7_uM":         ng_mL_to_uM(25.0, PROTEIN_MW_KDA["BMP7"]),      # 25 ng/mL
    "SAG_uM":          nM_to_uM(50.0),                                   # 50 nM
    "RA_uM":           nM_to_uM(100.0),                                  # 100 nM
    "FGF2_uM":         ng_mL_to_uM(20.0, PROTEIN_MW_KDA["FGF2"]),      # 20 ng/mL
    "FGF4_uM":         ng_mL_to_uM(100.0, PROTEIN_MW_KDA["FGF4"]),     # 100 ng/mL
    "FGF8_uM":         ng_mL_to_uM(100.0, PROTEIN_MW_KDA["FGF8"]),     # 100 ng/mL
    "IWP2_uM":         5.0,
    "LDN193189_uM":    nM_to_uM(100.0),                                  # 100 nM
    "DAPT_uM":         2.5,
    "Dorsomorphin_uM": 2.5,
    "EGF_uM":          ng_mL_to_uM(20.0, PROTEIN_MW_KDA["EGF"]),       # 20 ng/mL
    "SB431542_uM":     10.0,
    "ActivinA_uM":     ng_mL_to_uM(50.0, PROTEIN_MW_KDA["ActivinA"]),  # 50 ng/mL
}

# Shorthand aliases for frequently hardcoded default concentrations.
# Handlers should use these instead of repeating magic numbers.
_CHIR = _DEFAULTS["CHIR99021_uM"]     # 1.5 µM
_SB = _DEFAULTS["SB431542_uM"]        # 10.0 µM
_DAPT = _DEFAULTS["DAPT_uM"]          # 2.5 µM
_IWP2 = _DEFAULTS["IWP2_uM"]          # 5.0 µM

# Base media morphogens — constant across all conditions (from day 21 onwards)
# These are not varied in the screen but are present in every well.
_BASE_MEDIA: dict[str, float] = {
    "BDNF_uM":          ng_mL_to_uM(20.0, PROTEIN_MW_KDA["BDNF"]),   # 20 ng/mL
    "NT3_uM":           ng_mL_to_uM(20.0, PROTEIN_MW_KDA["NT3"]),    # 20 ng/mL
    "cAMP_uM":          50.0,                                          # 50 µM (dibutyryl-cAMP)
    "AscorbicAcid_uM": 200.0,                                         # 200 µM (L-ascorbic acid 2-phosphate)
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
    """Return a morphogen vector with base media defaults set."""
    vec = {col: 0.0 for col in MORPHOGEN_COLUMNS}
    # Set base media morphogens (constant across all conditions)
    vec.update(_BASE_MEDIA)
    return vec


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
# All concentration values are in µM. Protein morphogens use _DEFAULTS (which
# were computed via ng_mL_to_uM at module load). Non-default concentrations
# use nM_to_uM() or ng_mL_to_uM() inline.

def _bmp4_chir(v: dict[str, float]) -> None:
    v["BMP4_uM"] = _DEFAULTS["BMP4_uM"]
    v["CHIR99021_uM"] = _CHIR

def _bmp4_chir_d11_16(v: dict[str, float]) -> None:
    v["BMP4_uM"] = _DEFAULTS["BMP4_uM"]
    v["CHIR99021_uM"] = _CHIR * _time_fraction(11, 16)

def _bmp4_sag(v: dict[str, float]) -> None:
    v["BMP4_uM"] = _DEFAULTS["BMP4_uM"]
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]

def _bmp7(v: dict[str, float]) -> None:
    v["BMP7_uM"] = _DEFAULTS["BMP7_uM"]

def _bmp7_chir(v: dict[str, float]) -> None:
    v["BMP7_uM"] = _DEFAULTS["BMP7_uM"]
    v["CHIR99021_uM"] = _CHIR

def _bmp7_sag(v: dict[str, float]) -> None:
    v["BMP7_uM"] = _DEFAULTS["BMP7_uM"]
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]

def _c_l_s_fgf8(v: dict[str, float]) -> None:
    """C/L/S/FGF8 = CHIR + LDN + SB431542 + FGF8."""
    v["CHIR99021_uM"] = _CHIR
    v["LDN193189_uM"] = _DEFAULTS["LDN193189_uM"]
    v["SB431542_uM"] = _SB
    v["FGF8_uM"] = _DEFAULTS["FGF8_uM"]

def _c_s_bmp7_d(v: dict[str, float]) -> None:
    """C/S/BMP7/D = CHIR + SB431542 + BMP7 + DAPT."""
    v["CHIR99021_uM"] = _CHIR
    v["SB431542_uM"] = _SB
    v["BMP7_uM"] = _DEFAULTS["BMP7_uM"]
    v["DAPT_uM"] = _DAPT

def _c_s_d_fgf4(v: dict[str, float]) -> None:
    """C/S/D/FGF4 = CHIR + SB431542 + DAPT + FGF4."""
    v["CHIR99021_uM"] = _CHIR
    v["SB431542_uM"] = _SB
    v["DAPT_uM"] = _DAPT
    v["FGF4_uM"] = _DEFAULTS["FGF4_uM"]

def _c_s_r_e_fgf2_d(v: dict[str, float]) -> None:
    """C/S/R/E/FGF2/D = CHIR + SB431542 + RA + EGF + FGF2 + DAPT."""
    v["CHIR99021_uM"] = _CHIR
    v["SB431542_uM"] = _SB
    v["RA_uM"] = _DEFAULTS["RA_uM"]
    v["EGF_uM"] = _DEFAULTS["EGF_uM"]
    v["FGF2_uM"] = _DEFAULTS["FGF2_uM"]
    v["DAPT_uM"] = _DAPT

def _chir_sag_fgf4(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]
    v["FGF4_uM"] = _DEFAULTS["FGF4_uM"]

def _chir_sag_fgf8(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]
    v["FGF8_uM"] = _DEFAULTS["FGF8_uM"]

def _chir_switch_iwp2(v: dict[str, float]) -> None:
    """CHIR first half, then IWP2 second half of standard window."""
    v["CHIR99021_uM"] = _CHIR * _time_fraction(6, 13)
    v["IWP2_uM"] = _IWP2 * _time_fraction(13, 21)

def _chir_d11_16(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR * _time_fraction(11, 16)

def _chir_d16_21(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR * _time_fraction(16, 21)

def _chir_d6_11(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR * _time_fraction(6, 11)

def _chir_sag_d16_21(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR * _time_fraction(16, 21)
    v["SAG_uM"] = _DEFAULTS["SAG_uM"] * _time_fraction(16, 21)

def _chir_sag_ldn(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]
    v["LDN193189_uM"] = _DEFAULTS["LDN193189_uM"]

def _chir_sagd10_21(v: dict[str, float]) -> None:
    """CHIR full window, SAG days 10-21 only."""
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = _DEFAULTS["SAG_uM"] * _time_fraction(10, 21)

def _chir1_5_sag1000(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = nM_to_uM(1000.0)  # 1000 nM = 1.0 µM

def _chir1_5_sag250(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = nM_to_uM(250.0)  # 250 nM = 0.25 µM

def _chir1_5(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = _CHIR

def _chir3_sag1000(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 3.0
    v["SAG_uM"] = nM_to_uM(1000.0)  # 1000 nM = 1.0 µM

def _chir3_sag250(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 3.0
    v["SAG_uM"] = nM_to_uM(250.0)  # 250 nM = 0.25 µM

def _chir3(v: dict[str, float]) -> None:
    v["CHIR99021_uM"] = 3.0

def _dapt(v: dict[str, float]) -> None:
    v["DAPT_uM"] = _DAPT

def _fgf20_egf(v: dict[str, float]) -> None:
    """FGF-20/EGF: FGF2 at 20 ng/mL + EGF."""
    v["FGF2_uM"] = _DEFAULTS["FGF2_uM"]
    v["EGF_uM"] = _DEFAULTS["EGF_uM"]

def _fgf2_20(v: dict[str, float]) -> None:
    v["FGF2_uM"] = _DEFAULTS["FGF2_uM"]  # 20 ng/mL

def _fgf2_50(v: dict[str, float]) -> None:
    v["FGF2_uM"] = ng_mL_to_uM(50.0, PROTEIN_MW_KDA["FGF2"])  # 50 ng/mL

def _fgf4(v: dict[str, float]) -> None:
    v["FGF4_uM"] = _DEFAULTS["FGF4_uM"]

def _fgf8(v: dict[str, float]) -> None:
    v["FGF8_uM"] = _DEFAULTS["FGF8_uM"]

def _i_activin_dapt_sr11(v: dict[str, float]) -> None:
    """I/Activin/DAPT/SR11 = IWP2 + Activin A + DAPT + SR11237 (retinoid)."""
    v["IWP2_uM"] = _IWP2
    v["ActivinA_uM"] = _DEFAULTS["ActivinA_uM"]
    v["DAPT_uM"] = _DAPT
    v["RA_uM"] = _DEFAULTS["RA_uM"]  # SR11237 treated as RA equivalent

def _iwp2(v: dict[str, float]) -> None:
    v["IWP2_uM"] = _IWP2

def _iwp2_switch_chir(v: dict[str, float]) -> None:
    """IWP2 first half, then CHIR second half of standard window."""
    v["IWP2_uM"] = _IWP2 * _time_fraction(6, 13)
    v["CHIR99021_uM"] = _CHIR * _time_fraction(13, 21)

def _iwp2_sag(v: dict[str, float]) -> None:
    v["IWP2_uM"] = _IWP2
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]

def _ldn(v: dict[str, float]) -> None:
    v["LDN193189_uM"] = _DEFAULTS["LDN193189_uM"]

def _ra10(v: dict[str, float]) -> None:
    v["RA_uM"] = nM_to_uM(10.0)  # 10 nM = 0.01 µM

def _ra100(v: dict[str, float]) -> None:
    v["RA_uM"] = _DEFAULTS["RA_uM"]  # 100 nM = 0.1 µM

def _s_i_e_fgf2(v: dict[str, float]) -> None:
    """S/I/E/FGF2 = SB431542 + IWP2 + EGF + FGF2."""
    v["SB431542_uM"] = _SB
    v["IWP2_uM"] = _IWP2
    v["EGF_uM"] = _DEFAULTS["EGF_uM"]
    v["FGF2_uM"] = _DEFAULTS["FGF2_uM"]

def _sag_chir_d16_21(v: dict[str, float]) -> None:
    """SAG full window, CHIR days 16-21 only."""
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]
    v["CHIR99021_uM"] = _CHIR * _time_fraction(16, 21)

def _sag_chird10_21(v: dict[str, float]) -> None:
    """SAG full window, CHIR days 10-21 only."""
    v["SAG_uM"] = _DEFAULTS["SAG_uM"]
    v["CHIR99021_uM"] = _CHIR * _time_fraction(10, 21)

def _sag_d11_16(v: dict[str, float]) -> None:
    v["SAG_uM"] = _DEFAULTS["SAG_uM"] * _time_fraction(11, 16)

def _sag_d16_21(v: dict[str, float]) -> None:
    v["SAG_uM"] = _DEFAULTS["SAG_uM"] * _time_fraction(16, 21)

def _sag_d6_11(v: dict[str, float]) -> None:
    v["SAG_uM"] = _DEFAULTS["SAG_uM"] * _time_fraction(6, 11)

def _sag1000(v: dict[str, float]) -> None:
    v["SAG_uM"] = nM_to_uM(1000.0)  # 1000 nM = 1.0 µM

def _sag250(v: dict[str, float]) -> None:
    v["SAG_uM"] = nM_to_uM(250.0)  # 250 nM = 0.25 µM


# SAG Secondary Screen conditions (Amin/Kelley 2024, Day 70, CHIR 1.5µM base)
_SAG_SECONDARY_HARVEST_DAY: int = 70
_LOG_SAG_SECONDARY_HARVEST_DAY: float = math.log(_SAG_SECONDARY_HARVEST_DAY)

def _sag_50nm(v: dict[str, float]) -> None:
    """SAG secondary: 50nM SAG + 1.5µM CHIR, Day 70."""
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = nM_to_uM(50.0)  # 50 nM = 0.05 µM
    v["log_harvest_day"] = _LOG_SAG_SECONDARY_HARVEST_DAY

def _sag_2um(v: dict[str, float]) -> None:
    """SAG secondary: 2µM SAG + 1.5µM CHIR, Day 70."""
    v["CHIR99021_uM"] = _CHIR
    v["SAG_uM"] = 2.0  # 2000 nM = 2.0 µM
    v["log_harvest_day"] = _LOG_SAG_SECONDARY_HARVEST_DAY


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
    # SAG secondary screen (Day 70, non-duplicate conditions only)
    "SAG_50nM":             _sag_50nm,
    "SAG_2uM":              _sag_2um,
}

# Sanity check: 46 primary + 2 SAG secondary = 48 conditions
assert len(_CONDITION_PARSERS) == 48, (
    f"Expected 48 conditions, got {len(_CONDITION_PARSERS)}"
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
"""All 48 condition names in alphabetical order (46 primary + 2 SAG secondary)."""

SAG_SECONDARY_CONDITIONS: list[str] = ["SAG_50nM", "SAG_2uM"]
"""SAG secondary screen conditions (non-duplicate only)."""


# ==============================================================================
# Generic parser class hierarchy
# ==============================================================================

class MorphogenParser:
    """Base class for morphogen condition parsers."""

    def __init__(self, parsers: dict[str, Any], harvest_day: int):
        self._parsers = parsers
        self._harvest_day = harvest_day
        self._log_harvest_day = math.log(harvest_day)

    @property
    def conditions(self) -> list[str]:
        return sorted(self._parsers.keys())

    def parse(self, name: str) -> dict[str, float]:
        vec = _zeros()
        vec["log_harvest_day"] = self._log_harvest_day
        if name not in self._parsers:
            raise ValueError(f"Unrecognized condition: {name!r}")
        self._parsers[name](vec)
        return vec

    def build_matrix(self, conditions: list[str] | None = None) -> pd.DataFrame:
        conditions = conditions or self.conditions
        rows = [self.parse(c) for c in conditions]
        return pd.DataFrame(rows, index=conditions, columns=MORPHOGEN_COLUMNS)


class AminKelleyParser(MorphogenParser):
    """Parser for Amin/Kelley 2024 primary screen (46 conditions, Day 72)."""
    def __init__(self):
        primary = {k: v for k, v in _CONDITION_PARSERS.items()
                   if k not in SAG_SECONDARY_CONDITIONS}
        super().__init__(primary, harvest_day=72)


class SAGSecondaryParser(MorphogenParser):
    """Parser for SAG secondary screen (2 conditions, Day 70)."""
    def __init__(self):
        sag = {k: v for k, v in _CONDITION_PARSERS.items()
               if k in SAG_SECONDARY_CONDITIONS}
        super().__init__(sag, harvest_day=70)


class CombinedParser:
    """Combines multiple MorphogenParsers.

    Delegates ``parse`` to the first sub-parser that recognizes the condition.
    Inherits ``build_matrix`` logic identical to ``MorphogenParser``.
    """

    def __init__(self, parsers: list[MorphogenParser]):
        self._sub_parsers = parsers

    @property
    def conditions(self) -> list[str]:
        all_conds = []
        for p in self._sub_parsers:
            all_conds.extend(p.conditions)
        return sorted(all_conds)

    def parse(self, name: str) -> dict[str, float]:
        for p in self._sub_parsers:
            if name in p.conditions:
                return p.parse(name)
        raise ValueError(f"Unrecognized condition: {name!r}")

    def build_matrix(self, conditions: list[str] | None = None) -> pd.DataFrame:
        # Same as MorphogenParser.build_matrix — kept for duck-type compatibility
        conditions = conditions or self.conditions
        rows = [self.parse(c) for c in conditions]
        return pd.DataFrame(rows, index=conditions, columns=MORPHOGEN_COLUMNS)


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

    # Save morphogen matrix for downstream pipeline steps (04, 05, 06)
    from gopro.config import DATA_DIR
    output_path = DATA_DIR / "morphogen_matrix_amin_kelley.csv"
    # Only include primary screen conditions (exclude SAG secondary)
    primary_conditions = [c for c in ALL_CONDITIONS if c not in SAG_SECONDARY_CONDITIONS]
    primary_df = build_morphogen_matrix(primary_conditions)
    primary_df.to_csv(str(output_path))
    logger.info("Saved primary screen matrix to %s (%d conditions)", output_path, len(primary_df))

    # Also generate SAG secondary screen matrix
    sag_df = build_morphogen_matrix(SAG_SECONDARY_CONDITIONS)
    sag_output_path = DATA_DIR / "morphogen_matrix_sag_screen.csv"
    sag_df.to_csv(str(sag_output_path))
    logger.info("Saved SAG secondary screen matrix to %s", sag_output_path)
    logger.info("SAG matrix:\n%s", sag_df.to_string())
