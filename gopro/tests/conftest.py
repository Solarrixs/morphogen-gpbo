"""Pytest configuration and shared fixtures."""

import sys
import importlib.util
from pathlib import Path

# Add gopro directory and tests directory to Python path
GOPRO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(GOPRO_DIR))
sys.path.insert(0, str(GOPRO_DIR.parent))
sys.path.insert(0, str(Path(__file__).parent))


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
