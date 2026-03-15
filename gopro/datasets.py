"""Config-driven dataset registry for the GP-BO pipeline.

Loads dataset definitions from ``datasets.yaml`` so that new datasets
can be added without code changes.  Each entry describes a dataset's
source, file layout, quality-filter criteria, fidelity level, and
morphogen parser class.

Usage::

    from gopro.datasets import load_dataset_registry, get_dataset, get_real_datasets

    registry = load_dataset_registry()          # all datasets
    ds = get_dataset("amin_kelley")             # single dataset
    real = get_real_datasets()                   # fidelity == 1.0, enabled
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from gopro.config import DATA_DIR, get_logger

logger = get_logger(__name__)

_YAML_PATH = Path(__file__).parent / "datasets.yaml"

# Module-level cache (populated on first call to load_dataset_registry)
_REGISTRY_CACHE: dict[str, DatasetConfig] | None = None


@dataclass(frozen=True)
class FilterCriteria:
    """Quality-filter specification for a dataset."""

    column: str
    keep_value: Optional[str] = None
    exclude_value: Optional[str] = None

    def __post_init__(self):
        if self.keep_value is None and self.exclude_value is None:
            raise ValueError(
                "FilterCriteria must specify either keep_value or exclude_value"
            )


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration for a single dataset."""

    name: str
    description: str
    source: str  # "geo", "zenodo", "local"
    source_id: Optional[str]
    input_file: Optional[str]
    condition_key: str
    batch_key: Optional[str]
    filter_criteria: Optional[FilterCriteria]
    fidelity: float
    harvest_day: Optional[int]
    morphogen_parser_class: Optional[str]
    fractions_file: Optional[str]
    morphogens_file: Optional[str]
    enabled: bool = True

    # -- derived helpers --

    @property
    def input_path(self) -> Optional[Path]:
        """Absolute path to the input h5ad file, or None."""
        if self.input_file is None:
            return None
        return DATA_DIR / self.input_file

    @property
    def fractions_path(self) -> Optional[Path]:
        """Absolute path to the fractions CSV, or None."""
        if self.fractions_file is None:
            return None
        return DATA_DIR / self.fractions_file

    @property
    def morphogens_path(self) -> Optional[Path]:
        """Absolute path to the morphogen matrix CSV, or None."""
        if self.morphogens_file is None:
            return None
        return DATA_DIR / self.morphogens_file

    def has_training_data(self) -> bool:
        """Return True if both fractions and morphogens CSVs exist on disk."""
        fp = self.fractions_path
        mp = self.morphogens_path
        return fp is not None and mp is not None and fp.exists() and mp.exists()

    def as_fidelity_source(self) -> tuple[Path, Path, float] | None:
        """Return a (fractions_path, morphogens_path, fidelity) tuple for GP-BO.

        Returns None if the required files don't exist.
        """
        fp = self.fractions_path
        mp = self.morphogens_path
        if fp is None or mp is None:
            return None
        if not fp.exists() or not mp.exists():
            return None
        return (fp, mp, self.fidelity)


def _parse_filter_criteria(raw: Any) -> Optional[FilterCriteria]:
    """Parse the filter_criteria field from YAML."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"filter_criteria must be a dict, got {type(raw)}")
    return FilterCriteria(
        column=raw["column"],
        keep_value=raw.get("keep_value"),
        exclude_value=raw.get("exclude_value"),
    )


def _parse_dataset(name: str, raw: dict[str, Any]) -> DatasetConfig:
    """Parse a single dataset entry from the YAML dict."""
    return DatasetConfig(
        name=name,
        description=raw.get("description", ""),
        source=raw["source"],
        source_id=raw.get("source_id"),
        input_file=raw.get("input_file"),
        condition_key=raw.get("condition_key", "condition"),
        batch_key=raw.get("batch_key"),
        filter_criteria=_parse_filter_criteria(raw.get("filter_criteria")),
        fidelity=float(raw.get("fidelity", 1.0)),
        harvest_day=raw.get("harvest_day"),
        morphogen_parser_class=raw.get("morphogen_parser_class"),
        fractions_file=raw.get("fractions_file"),
        morphogens_file=raw.get("morphogens_file"),
        enabled=raw.get("enabled", True),
    )


def load_dataset_registry(
    yaml_path: Path | str | None = None,
) -> dict[str, DatasetConfig]:
    """Load and validate the dataset registry from YAML.

    Args:
        yaml_path: Path to the YAML file.  Defaults to
            ``gopro/datasets.yaml`` next to this module.

    Returns:
        Dict mapping dataset name to ``DatasetConfig``.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: On invalid YAML structure.
    """
    global _REGISTRY_CACHE

    path = Path(yaml_path) if yaml_path else _YAML_PATH

    if _REGISTRY_CACHE is not None and yaml_path is None:
        return _REGISTRY_CACHE

    if not path.exists():
        raise FileNotFoundError(f"Dataset registry YAML not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "datasets" not in raw:
        raise ValueError("YAML must contain a top-level 'datasets' key")

    datasets_raw = raw["datasets"]
    if not isinstance(datasets_raw, dict):
        raise ValueError("'datasets' must be a mapping of name → config")

    registry: dict[str, DatasetConfig] = {}
    for name, entry in datasets_raw.items():
        ds = _parse_dataset(name, entry)
        registry[name] = ds

    logger.info(
        "Loaded %d datasets from %s (%d enabled)",
        len(registry),
        path.name,
        sum(1 for d in registry.values() if d.enabled),
    )

    if yaml_path is None:
        _REGISTRY_CACHE = registry

    return registry


def get_dataset(name: str, yaml_path: Path | str | None = None) -> DatasetConfig:
    """Look up a single dataset by name.

    Raises:
        KeyError: If the dataset name is not found.
    """
    registry = load_dataset_registry(yaml_path)
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"Dataset '{name}' not found. Available: {available}"
        )
    return registry[name]


def get_real_datasets(yaml_path: Path | str | None = None) -> list[DatasetConfig]:
    """Return all enabled datasets with fidelity == 1.0."""
    registry = load_dataset_registry(yaml_path)
    return [
        ds for ds in registry.values()
        if ds.enabled and ds.fidelity == 1.0
    ]


def get_virtual_datasets(yaml_path: Path | str | None = None) -> list[DatasetConfig]:
    """Return all enabled datasets with fidelity < 1.0."""
    registry = load_dataset_registry(yaml_path)
    return [
        ds for ds in registry.values()
        if ds.enabled and ds.fidelity < 1.0
    ]


def collect_fidelity_sources(
    yaml_path: Path | str | None = None,
) -> list[tuple[Path, Path, float]]:
    """Collect all enabled datasets that have training data on disk.

    Returns a list of (fractions_path, morphogens_path, fidelity) tuples
    ready to pass to ``merge_multi_fidelity_data()``.
    """
    registry = load_dataset_registry(yaml_path)
    sources = []
    for ds in registry.values():
        if not ds.enabled:
            continue
        src = ds.as_fidelity_source()
        if src is not None:
            sources.append(src)
    # Sort by fidelity descending so real data appears first
    sources.sort(key=lambda t: -t[2])
    return sources


def invalidate_cache() -> None:
    """Clear the module-level registry cache (useful in tests)."""
    global _REGISTRY_CACHE
    _REGISTRY_CACHE = None
