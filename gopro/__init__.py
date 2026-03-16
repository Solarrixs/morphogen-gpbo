"""GP-BO pipeline for brain organoid morphogen protocol optimization.

Direct re-exports from non-numeric modules are always available.
Functions from numeric-prefixed pipeline steps are lazily loaded
on first access to avoid importing heavy dependencies (torch, scanpy)
at ``import gopro`` time.

Usage::

    from gopro import MORPHOGEN_COLUMNS, run_gpbo_loop, ilr_transform
"""

import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Direct re-exports (lightweight, no heavy deps)
# ---------------------------------------------------------------------------
from gopro.config import (
    MORPHOGEN_COLUMNS,
    DATA_DIR,
    PROJECT_DIR,
    MODEL_DIR,
    PROTEIN_MW_KDA,
    get_logger,
    ng_mL_to_uM,
    nM_to_uM,
    ANNOT_LEVEL_1,
    ANNOT_LEVEL_2,
    ANNOT_REGION,
    ANNOT_LEVEL_3,
)
from gopro.morphogen_parser import (
    build_morphogen_matrix,
    parse_condition_name,
    ALL_CONDITIONS,
)
from gopro.datasets import (
    DatasetConfig,
    FilterCriteria,
    load_dataset_registry,
    get_dataset,
    get_real_datasets,
    get_virtual_datasets,
    collect_fidelity_sources,
)
# region_targets constants are lazily loaded below to avoid pulling in anndata

# ---------------------------------------------------------------------------
# Lazy imports for numeric-prefixed pipeline modules
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, str] = {
    # 04_gpbo_loop.py
    "run_gpbo_loop": "04_gpbo_loop.py",
    "build_training_set": "04_gpbo_loop.py",
    "fit_gp_botorch": "04_gpbo_loop.py",
    "save_gp_state": "04_gpbo_loop.py",
    "load_gp_state": "04_gpbo_loop.py",
    "ilr_transform": "04_gpbo_loop.py",
    "ilr_inverse": "04_gpbo_loop.py",
    "merge_multi_fidelity_data": "04_gpbo_loop.py",
    "refine_target_profile": "04_gpbo_loop.py",
    # 03_fidelity_scoring.py
    "score_all_conditions": "03_fidelity_scoring.py",
    "compute_composite_fidelity": "03_fidelity_scoring.py",
    "extract_braun_region_profiles": "03_fidelity_scoring.py",
    "run_fidelity_scoring": "03_fidelity_scoring.py",
    # 02_map_to_hnoca.py
    "filter_quality_cells": "02_map_to_hnoca.py",
    "compute_cell_type_fractions": "02_map_to_hnoca.py",
    "run_mapping_pipeline": "02_map_to_hnoca.py",
    # 05_cellrank2_virtual.py
    "generate_virtual_training_data": "05_cellrank2_virtual.py",
    # validation.py (non-numeric, but keep lazy to avoid circular deps)
    "validate_training_csvs": "validation.py",
    "validate_mapped_adata": "validation.py",
    "validate_temporal_atlas": "validation.py",
    "validate_fidelity_report": "validation.py",
    # region_targets.py (lazy to avoid anndata import at gopro load time)
    "NAMED_REGION_PROFILES": "region_targets.py",
    "OFF_TARGET_LEVEL1": "region_targets.py",
    "HNOCA_TO_BRAUN_REGION": "region_targets.py",
    "build_hnoca_to_braun_label_map": "region_targets.py",
    "list_named_profiles": "region_targets.py",
    "build_custom_target": "region_targets.py",
    "discover_available_regions": "region_targets.py",
    "load_region_profile": "region_targets.py",
    "load_target_profile_csv": "region_targets.py",
}

_GOPRO_DIR = Path(__file__).parent
_MODULE_CACHE: dict[str, object] = {}


def _load_cached(filename: str):
    """Load a module by filename, caching the result."""
    if filename not in _MODULE_CACHE:
        filepath = _GOPRO_DIR / filename
        spec = importlib.util.spec_from_file_location(
            f"gopro._{filename.replace('.py', '')}", str(filepath)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE[filename] = module
    return _MODULE_CACHE[filename]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = _load_cached(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'gopro' has no attribute {name!r}")
