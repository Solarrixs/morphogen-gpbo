"""Tests for config-driven dataset registry (Phase 4A)."""

import textwrap

import pytest

from gopro.datasets import (
    DatasetConfig,
    FilterCriteria,
    collect_fidelity_sources,
    get_dataset,
    get_real_datasets,
    get_virtual_datasets,
    invalidate_cache,
    load_dataset_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure each test starts with a clean registry cache."""
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture
def minimal_yaml(tmp_path):
    """Create a minimal valid datasets YAML."""
    content = textwrap.dedent("""\
        datasets:
          test_primary:
            description: "Test primary dataset"
            source: geo
            source_id: GSE000001
            input_file: test_primary.h5ad
            condition_key: condition
            batch_key: sample
            filter_criteria:
              column: quality
              keep_value: "keep"
            fidelity: 1.0
            harvest_day: 72
            morphogen_parser_class: AminKelleyParser
            fractions_file: fractions_primary.csv
            morphogens_file: morphogens_primary.csv
            enabled: true

          test_virtual:
            description: "Test virtual dataset"
            source: local
            input_file: null
            condition_key: condition
            filter_criteria: null
            fidelity: 0.5
            fractions_file: fractions_virtual.csv
            morphogens_file: morphogens_virtual.csv
            enabled: true

          test_disabled:
            description: "A disabled dataset"
            source: local
            condition_key: condition
            filter_criteria: null
            fidelity: 1.0
            fractions_file: fractions_disabled.csv
            morphogens_file: morphogens_disabled.csv
            enabled: false
    """)
    path = tmp_path / "datasets.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def yaml_with_data(tmp_path):
    """Create a YAML + actual CSV files on disk for fidelity-source tests."""
    import pandas as pd

    content = textwrap.dedent("""\
        datasets:
          real_ds:
            description: "Real dataset with files"
            source: local
            condition_key: condition
            filter_criteria: null
            fidelity: 1.0
            fractions_file: frac_real.csv
            morphogens_file: morph_real.csv
            enabled: true

          virtual_ds:
            description: "Virtual dataset with files"
            source: local
            condition_key: condition
            filter_criteria: null
            fidelity: 0.5
            fractions_file: frac_virtual.csv
            morphogens_file: morph_virtual.csv
            enabled: true

          missing_ds:
            description: "Dataset without files"
            source: local
            condition_key: condition
            filter_criteria: null
            fidelity: 1.0
            fractions_file: frac_missing.csv
            morphogens_file: morph_missing.csv
            enabled: true
    """)
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(content)

    # Create CSV files for real_ds and virtual_ds (referenced as DATA_DIR-relative)
    # We'll need to monkey-patch DATA_DIR for these tests
    return yaml_path, tmp_path


# ---------------------------------------------------------------------------
# Tests: load_dataset_registry
# ---------------------------------------------------------------------------

class TestLoadDatasetRegistry:

    def test_loads_default_yaml(self):
        """Loading the actual project YAML succeeds."""
        registry = load_dataset_registry()
        assert isinstance(registry, dict)
        assert len(registry) > 0

    def test_loads_custom_yaml(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        assert len(registry) == 3
        assert "test_primary" in registry
        assert "test_virtual" in registry
        assert "test_disabled" in registry

    def test_dataset_types(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        primary = registry["test_primary"]
        assert isinstance(primary, DatasetConfig)
        assert primary.name == "test_primary"
        assert primary.source == "geo"
        assert primary.fidelity == 1.0
        assert primary.harvest_day == 72
        assert primary.enabled is True

    def test_filter_criteria_keep(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        fc = registry["test_primary"].filter_criteria
        assert isinstance(fc, FilterCriteria)
        assert fc.column == "quality"
        assert fc.keep_value == "keep"
        assert fc.exclude_value is None

    def test_null_filter_criteria(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        assert registry["test_virtual"].filter_criteria is None

    def test_missing_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset_registry(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_structure_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("just_a_string")
        with pytest.raises(ValueError, match="top-level 'datasets' key"):
            load_dataset_registry(path)

    def test_missing_datasets_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("other_key: value")
        with pytest.raises(ValueError, match="top-level 'datasets' key"):
            load_dataset_registry(path)

    def test_datasets_not_mapping_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("datasets:\n  - item1\n  - item2")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_dataset_registry(path)

    def test_caching(self, minimal_yaml):
        """Second call returns same dict object (cached)."""
        reg1 = load_dataset_registry(minimal_yaml)
        # Clear cache and load again with explicit path (no caching for non-default)
        invalidate_cache()
        reg2 = load_dataset_registry(minimal_yaml)
        assert reg1 == reg2


# ---------------------------------------------------------------------------
# Tests: DatasetConfig
# ---------------------------------------------------------------------------

class TestDatasetConfig:

    def test_path_properties(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        ds = registry["test_primary"]
        assert ds.input_path is not None
        assert ds.input_path.name == "test_primary.h5ad"
        assert ds.fractions_path is not None
        assert ds.morphogens_path is not None

    def test_null_paths(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        ds = registry["test_virtual"]
        assert ds.input_path is None  # input_file is null

    def test_has_training_data_false(self, minimal_yaml):
        """No files on disk, so has_training_data should be False."""
        registry = load_dataset_registry(minimal_yaml)
        ds = registry["test_primary"]
        assert ds.has_training_data() is False

    def test_as_fidelity_source_none_when_no_files(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        ds = registry["test_primary"]
        assert ds.as_fidelity_source() is None

    def test_frozen(self, minimal_yaml):
        registry = load_dataset_registry(minimal_yaml)
        ds = registry["test_primary"]
        with pytest.raises(AttributeError):
            ds.name = "changed"


# ---------------------------------------------------------------------------
# Tests: FilterCriteria
# ---------------------------------------------------------------------------

class TestFilterCriteria:

    def test_keep_value(self):
        fc = FilterCriteria(column="quality", keep_value="keep")
        assert fc.column == "quality"
        assert fc.keep_value == "keep"

    def test_exclude_value(self):
        fc = FilterCriteria(column="ClusterLabel", exclude_value="filtered")
        assert fc.column == "ClusterLabel"
        assert fc.exclude_value == "filtered"

    def test_must_have_one_criterion(self):
        with pytest.raises(ValueError, match="keep_value or exclude_value"):
            FilterCriteria(column="x")

    def test_frozen(self):
        fc = FilterCriteria(column="quality", keep_value="keep")
        with pytest.raises(AttributeError):
            fc.column = "changed"


# ---------------------------------------------------------------------------
# Tests: get_dataset
# ---------------------------------------------------------------------------

class TestGetDataset:

    def test_existing_dataset(self, minimal_yaml):
        ds = get_dataset("test_primary", minimal_yaml)
        assert ds.name == "test_primary"

    def test_missing_dataset_raises(self, minimal_yaml):
        with pytest.raises(KeyError, match="not_here"):
            get_dataset("not_here", minimal_yaml)


# ---------------------------------------------------------------------------
# Tests: get_real_datasets / get_virtual_datasets
# ---------------------------------------------------------------------------

class TestFilterHelpers:

    def test_get_real_datasets(self, minimal_yaml):
        real = get_real_datasets(minimal_yaml)
        assert all(d.fidelity == 1.0 for d in real)
        assert all(d.enabled for d in real)
        names = [d.name for d in real]
        assert "test_primary" in names
        assert "test_disabled" not in names  # disabled

    def test_get_virtual_datasets(self, minimal_yaml):
        virtual = get_virtual_datasets(minimal_yaml)
        assert all(d.fidelity < 1.0 for d in virtual)
        assert all(d.enabled for d in virtual)
        names = [d.name for d in virtual]
        assert "test_virtual" in names

    def test_collect_fidelity_sources_empty_when_no_files(self, minimal_yaml):
        """No CSV files on disk, so collect returns empty list."""
        sources = collect_fidelity_sources(minimal_yaml)
        assert sources == []


# ---------------------------------------------------------------------------
# Tests: real YAML file
# ---------------------------------------------------------------------------

class TestRealYaml:
    """Validate the actual gopro/datasets.yaml ships correctly."""

    def test_has_amin_kelley(self):
        registry = load_dataset_registry()
        assert "amin_kelley" in registry
        ds = registry["amin_kelley"]
        assert ds.fidelity == 1.0
        assert ds.source == "geo"
        assert ds.condition_key == "condition"
        assert ds.morphogen_parser_class == "AminKelleyParser"

    def test_has_sag_screen(self):
        registry = load_dataset_registry()
        assert "sag_screen" in registry
        ds = registry["sag_screen"]
        assert ds.fidelity == 1.0
        assert ds.filter_criteria is not None
        assert ds.filter_criteria.exclude_value == "filtered"

    def test_has_virtual_datasets(self):
        registry = load_dataset_registry()
        virtual = get_virtual_datasets()
        names = [d.name for d in virtual]
        assert "cellrank2_virtual" in names or "cellflow_virtual" in names

    def test_all_enabled_have_required_fields(self):
        registry = load_dataset_registry()
        for name, ds in registry.items():
            if not ds.enabled:
                continue
            assert ds.condition_key, f"{name}: condition_key is required"
            assert ds.source in ("geo", "zenodo", "local"), (
                f"{name}: unknown source '{ds.source}'"
            )
            assert 0.0 <= ds.fidelity <= 1.0, (
                f"{name}: fidelity must be in [0, 1]"
            )

    def test_exclude_filter_criteria(self):
        """SAG screen uses exclude-style filter."""
        ds = get_dataset("sag_screen")
        fc = ds.filter_criteria
        assert fc.column == "ClusterLabel"
        assert fc.exclude_value == "filtered"

    def test_keep_filter_criteria(self):
        """Primary screen uses keep-style filter."""
        ds = get_dataset("amin_kelley")
        fc = ds.filter_criteria
        assert fc.column == "quality"
        assert fc.keep_value == "keep"
