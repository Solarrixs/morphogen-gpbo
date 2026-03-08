"""Unit tests for GP-BO pipeline components."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import importlib.util

# Import pipeline modules using spec loader (handles numeric prefixes)
GOPRO_DIR = Path(__file__).parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(GOPRO_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


step01 = _load("01_load_and_convert_data")
step02 = _load("02_map_to_hnoca")
step04 = _load("04_gpbo_loop")


class TestILRTransform:
    """Tests for ILR (isometric log-ratio) transform."""

    def test_ilr_reduces_dimension(self):
        Y = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        Z = step04.ilr_transform(Y)
        assert Z.shape == (2, 2)

    def test_ilr_inverse_roundtrip(self):
        Y = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.33, 0.33, 0.34]])
        Z = step04.ilr_transform(Y)
        Y_recovered = step04.ilr_inverse(Z, D=3)
        np.testing.assert_allclose(Y, Y_recovered, atol=1e-3)

    def test_ilr_handles_near_zero(self):
        Y = np.array([[0.99, 0.005, 0.005]])
        Z = step04.ilr_transform(Y)
        assert np.all(np.isfinite(Z))

    def test_ilr_handles_uniform(self):
        Y = np.array([[0.25, 0.25, 0.25, 0.25]])
        Z = step04.ilr_transform(Y)
        np.testing.assert_allclose(Z, np.zeros((1, 3)), atol=0.1)

    def test_ilr_single_row(self):
        Y = np.array([[0.7, 0.2, 0.1]])
        Z = step04.ilr_transform(Y)
        assert Z.shape == (1, 2)
        assert np.all(np.isfinite(Z))


class TestMorphogenBounds:
    """Tests for morphogen bounds configuration."""

    def test_all_columns_have_bounds(self):
        for col in step04.MORPHOGEN_COLUMNS:
            assert col in step04.MORPHOGEN_BOUNDS, f"Missing bounds for {col}"

    def test_bounds_are_valid(self):
        for col, (lo, hi) in step04.MORPHOGEN_BOUNDS.items():
            assert lo < hi, f"Invalid bounds for {col}: ({lo}, {hi})"
            assert lo >= 0, f"Negative lower bound for {col}: {lo}"

    def test_harvest_day_bounds(self):
        lo, hi = step04.MORPHOGEN_BOUNDS["log_harvest_day"]
        assert np.exp(lo) == pytest.approx(7, abs=1)
        assert np.exp(hi) == pytest.approx(120, abs=1)


class TestBuildTrainingSet:
    """Tests for build_training_set function."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        Y = pd.DataFrame(
            {"ct_A": [0.6, 0.3], "ct_B": [0.4, 0.7]},
            index=["cond_1", "cond_2"],
        )
        X = pd.DataFrame(
            {"CHIR99021_uM": [1.5, 3.0], "SAG_nM": [250, 1000]},
            index=["cond_1", "cond_2"],
        )
        y_path = tmp_path / "Y.csv"
        x_path = tmp_path / "X.csv"
        Y.to_csv(y_path)
        X.to_csv(x_path)
        return x_path, y_path

    def test_loads_and_aligns(self, sample_data):
        x_path, y_path = sample_data
        X, Y = step04.build_training_set(y_path, x_path)
        assert len(X) == len(Y) == 2
        assert "fidelity" in X.columns

    def test_fidelity_column_added(self, sample_data):
        x_path, y_path = sample_data
        X, Y = step04.build_training_set(y_path, x_path, fidelity=0.5)
        assert (X["fidelity"] == 0.5).all()

    def test_mismatched_indices(self, tmp_path):
        Y = pd.DataFrame({"ct": [0.5, 0.5]}, index=["cond_1", "cond_3"])
        X = pd.DataFrame({"morph": [1.0, 2.0]}, index=["cond_1", "cond_2"])
        Y.to_csv(tmp_path / "Y.csv")
        X.to_csv(tmp_path / "X.csv")
        X_out, Y_out = step04.build_training_set(
            tmp_path / "Y.csv", tmp_path / "X.csv"
        )
        assert len(X_out) == 1


class TestComputeCellTypeFractions:
    """Tests for compute_cell_type_fractions function."""

    def test_fractions_sum_to_one(self):
        obs = pd.DataFrame({
            "condition": ["A", "A", "A", "B", "B"],
            "predicted_annot_level_2": ["Neuron", "NPC", "Neuron", "Glia", "Glia"],
            "quality": ["keep"] * 5,
        })
        fracs = step02.compute_cell_type_fractions(
            obs, "condition", "predicted_annot_level_2"
        )
        np.testing.assert_allclose(fracs.sum(axis=1), 1.0, atol=1e-6)

    def test_correct_fractions(self):
        obs = pd.DataFrame({
            "condition": ["A", "A", "A", "A"],
            "predicted_annot_level_2": ["Neuron", "Neuron", "NPC", "NPC"],
            "quality": ["keep"] * 4,
        })
        fracs = step02.compute_cell_type_fractions(
            obs, "condition", "predicted_annot_level_2"
        )
        assert fracs.loc["A", "Neuron"] == pytest.approx(0.5)
        assert fracs.loc["A", "NPC"] == pytest.approx(0.5)

    def test_single_cell_type(self):
        obs = pd.DataFrame({
            "condition": ["A", "A", "A"],
            "label": ["X", "X", "X"],
            "quality": ["keep"] * 3,
        })
        fracs = step02.compute_cell_type_fractions(obs, "condition", "label")
        assert fracs.loc["A", "X"] == pytest.approx(1.0)


class TestFilterQualityCells:
    """Tests for filter_quality_cells function."""

    def test_filters_correctly(self):
        import anndata
        obs = pd.DataFrame({
            "quality": ["keep", "low-quality", "stress", "keep"],
        })
        adata = anndata.AnnData(X=np.zeros((4, 2)), obs=obs)
        filtered = step02.filter_quality_cells(adata)
        assert filtered.n_obs == 2

    def test_no_quality_column(self):
        import anndata
        adata = anndata.AnnData(X=np.zeros((3, 2)))
        filtered = step02.filter_quality_cells(adata)
        assert filtered.n_obs == 3


class TestStep01ConvertGeoToAnndata:
    """Tests for step 01 data conversion."""

    def test_creates_anndata(self, tmp_path):
        from scipy.io import mmwrite
        from scipy import sparse

        n_cells, n_genes = 10, 5
        counts = sparse.random(n_genes, n_cells, density=0.3, format="coo")
        mmwrite(str(tmp_path / "counts.mtx"), counts)

        meta = pd.DataFrame(
            {"condition": ["A"] * 5 + ["B"] * 5},
            index=[f"cell_{i}" for i in range(n_cells)],
        )
        meta.to_csv(tmp_path / "meta.csv")

        genes = pd.DataFrame(
            {"gene_name": [f"gene_{i}" for i in range(n_genes)]},
            index=[f"ENSG{i:011d}" for i in range(n_genes)],
        )
        genes.to_csv(tmp_path / "genes.csv")

        output = tmp_path / "test.h5ad"
        adata = step01.convert_geo_to_anndata(
            tmp_path / "counts.mtx",
            tmp_path / "meta.csv",
            tmp_path / "genes.csv",
            output,
            "test",
        )

        assert adata.shape == (n_cells, n_genes)
        assert output.exists()
        assert "counts" in adata.layers

    def test_verify_references(self):
        result = step01.verify_references()
        assert isinstance(result, bool)


class TestMorphogenColumns:
    """Tests for morphogen column consistency."""

    def test_columns_are_unique(self):
        assert len(step04.MORPHOGEN_COLUMNS) == len(set(step04.MORPHOGEN_COLUMNS))

    def test_columns_are_strings(self):
        for col in step04.MORPHOGEN_COLUMNS:
            assert isinstance(col, str)
            assert len(col) > 0
