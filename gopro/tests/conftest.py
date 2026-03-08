"""Pytest configuration and shared fixtures."""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
import pytest

# Add gopro directory to Python path
GOPRO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(GOPRO_DIR))
sys.path.insert(0, str(GOPRO_DIR.parent))


def _import_pipeline_module(name: str):
    """Import a pipeline module by filename (handles numeric prefixes).

    Args:
        name: Module filename without .py (e.g., '01_load_and_convert_data').

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(
        name, str(GOPRO_DIR / f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Pre-import pipeline modules and make them available as fixtures
# We do this at module level so tests can import them
_step01 = _import_pipeline_module("01_load_and_convert_data")
_step02 = _import_pipeline_module("02_map_to_hnoca")
_step04 = _import_pipeline_module("04_gpbo_loop")
_step05 = _import_pipeline_module("05_cellrank2_virtual")
_step06 = _import_pipeline_module("06_cellflow_virtual")


@pytest.fixture
def step01():
    return _step01


@pytest.fixture
def step02():
    return _step02


@pytest.fixture
def step04():
    return _step04


@pytest.fixture
def step05():
    return _step05


@pytest.fixture
def step06():
    return _step06
