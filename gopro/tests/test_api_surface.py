"""Tests for the importable API surface (Track 1C)."""

import pytest


class TestDirectImports:
    """Test direct re-exports from gopro.__init__."""

    def test_import_morphogen_columns(self):
        from gopro import MORPHOGEN_COLUMNS
        assert isinstance(MORPHOGEN_COLUMNS, list)
        assert len(MORPHOGEN_COLUMNS) == 24

    def test_import_config_functions(self):
        from gopro import get_logger, ng_mL_to_uM, nM_to_uM
        assert callable(get_logger)
        assert callable(ng_mL_to_uM)
        assert ng_mL_to_uM(1000.0, 1.0) == pytest.approx(1.0)
        assert nM_to_uM(1000.0) == pytest.approx(1.0)

    def test_import_parser_functions(self):
        from gopro import build_morphogen_matrix, parse_condition_name
        assert callable(build_morphogen_matrix)
        assert callable(parse_condition_name)


class TestLazyImports:
    """Test lazy imports from numeric-prefixed modules."""

    def test_import_ilr_transform(self):
        from gopro import ilr_transform
        assert callable(ilr_transform)

    def test_import_run_gpbo_loop(self):
        from gopro import run_gpbo_loop
        assert callable(run_gpbo_loop)

    def test_import_score_all_conditions(self):
        from gopro import score_all_conditions
        assert callable(score_all_conditions)

    def test_import_filter_quality_cells(self):
        from gopro import filter_quality_cells
        assert callable(filter_quality_cells)


class TestImportErrors:
    """Test error handling for nonexistent attributes."""

    def test_nonexistent_attr_raises(self):
        import gopro
        with pytest.raises(AttributeError, match="no attribute"):
            _ = gopro.nonexistent_function_xyz
