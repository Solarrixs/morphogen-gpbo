"""Unit tests for GP-BO pipeline components."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import importlib.util
import plotly.graph_objects as go

# Import pipeline modules using spec loader (handles numeric prefixes)
GOPRO_DIR = Path(__file__).parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(GOPRO_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


step01 = _load("01_load_and_convert_data")
step02 = _load("02_map_to_hnoca")
step03 = _load("03_fidelity_scoring")
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
            {"CHIR99021_uM": [1.5, 3.0], "SAG_uM": [0.25, 1.0]},
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


class TestFidelityScoring:
    """Tests for step 03 fidelity scoring functions."""

    def test_cosine_similarity_identical(self):
        a = np.array([0.5, 0.3, 0.2])
        assert step03.cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert step03.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.5, 0.3])
        b = np.array([0.0, 0.0])
        assert step03.cosine_similarity(a, b) == 0.0

    def test_shannon_entropy_uniform(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert step03.shannon_entropy(p) == pytest.approx(2.0)

    def test_shannon_entropy_deterministic(self):
        p = np.array([1.0, 0.0, 0.0])
        assert step03.shannon_entropy(p) == pytest.approx(0.0)

    def test_normalized_entropy_bounds(self):
        p = np.array([0.5, 0.3, 0.2])
        h = step03.normalized_entropy(p)
        assert 0.0 <= h <= 1.0

    def test_normalized_entropy_uniform_is_one(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert step03.normalized_entropy(p) == pytest.approx(1.0)

    def test_compute_rss(self):
        condition_vec = pd.Series({"Neuron": 0.6, "Radial glia": 0.3, "Oligo": 0.1})
        ref_profiles = pd.DataFrame({
            "Neuron": [0.5, 0.1],
            "Radial glia": [0.3, 0.8],
            "Oligo": [0.2, 0.1],
        }, index=["Cortex", "Thalamus"])
        region, score = step03.compute_rss(condition_vec, ref_profiles)
        assert region == "Cortex"  # closer to Cortex
        assert 0.0 <= score <= 1.0

    def test_compute_off_target_fraction(self):
        obs = pd.DataFrame({
            "predicted_annot_level_1": ["Neuron", "NPC", "PSC", "MC", "Neuron"],
        })
        frac = step03.compute_off_target_fraction(obs)
        assert frac == pytest.approx(0.4)  # 2/5

    def test_compute_on_target_fraction(self):
        obs = pd.DataFrame({
            "predicted_annot_region_rev2": ["Cortex", "Cortex", "Cortex", "Thalamus"],
        })
        region, frac = step03.compute_on_target_fraction(obs)
        assert region == "Cortex"
        assert frac == pytest.approx(0.75)

    def test_composite_fidelity_bounds(self):
        score = step03.compute_composite_fidelity(
            rss_score=0.8, on_target_frac=0.6,
            off_target_frac=0.1, norm_entropy=0.5,
        )
        assert 0.0 <= score <= 1.0

    def test_composite_fidelity_perfect(self):
        score = step03.compute_composite_fidelity(
            rss_score=1.0, on_target_frac=1.0,
            off_target_frac=0.0, norm_entropy=0.55,
        )
        assert score > 0.9

    def test_composite_fidelity_handles_nan(self):
        score = step03.compute_composite_fidelity(
            rss_score=np.nan, on_target_frac=np.nan,
            off_target_frac=np.nan, norm_entropy=np.nan,
        )
        assert np.isfinite(score)

    def test_hnoca_to_braun_label_map(self):
        label_map = step03.build_hnoca_to_braun_label_map()
        assert "Neuron" in label_map
        assert label_map["Neuron"] == "Neuron"
        assert "PSC" in label_map

    def test_align_composition_to_braun(self):
        fracs = pd.Series({"Neuron": 0.5, "NPC": 0.3, "IP": 0.2})
        label_map = step03.build_hnoca_to_braun_label_map()
        aligned = step03.align_composition_to_braun(fracs, label_map)
        assert aligned.sum() == pytest.approx(1.0)
        # NPC and IP both map to Radial glia and Neuronal IPC
        assert "Neuron" in aligned.index


class TestVisualizationReport:
    """Tests for visualization report module."""

    @pytest.fixture
    def sample_fidelity(self):
        return pd.DataFrame({
            "composite_fidelity": [0.85, 0.72, 0.91],
            "rss_score": [0.80, 0.65, 0.88],
            "on_target_fraction": [0.9, 0.7, 0.95],
            "off_target_fraction": [0.05, 0.1, 0.02],
        }, index=pd.Index(["cond_A", "cond_B", "cond_C"], name="condition"))

    @pytest.fixture
    def sample_morphogens(self):
        np.random.seed(42)
        return pd.DataFrame(
            np.random.rand(10, 5),
            columns=["CHIR99021_uM", "BMP4_uM", "SAG_uM", "RA_uM", "log_harvest_day"],
            index=[f"cond_{i}" for i in range(10)],
        )

    @pytest.fixture
    def sample_recs(self):
        return pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0],
            "BMP4_uM": [0.000769, 0.001538],
            "SAG_uM": [0.25, 0.5],
            "RA_uM": [0, 0.1],
            "log_harvest_day": [4.0, 4.2],
            "predicted_y0_mean": [0.5, 0.6],
            "predicted_y0_std": [0.1, 0.2],
            "acquisition_value": [-1.3, -1.5],
        }, index=pd.Index(["A1", "A2"], name="well"))

    def test_discover_rounds(self, tmp_path):
        from gopro.visualize_report import discover_rounds
        (tmp_path / "gp_recommendations_round1.csv").touch()
        (tmp_path / "gp_recommendations_round3.csv").touch()
        (tmp_path / "gp_recommendations_round2.csv").touch()
        assert discover_rounds(tmp_path) == [1, 2, 3]

    def test_discover_rounds_empty(self, tmp_path):
        from gopro.visualize_report import discover_rounds
        assert discover_rounds(tmp_path) == []

    def test_load_fidelity_report(self, tmp_path, sample_fidelity):
        from gopro.visualize_report import load_fidelity_report
        path = tmp_path / "fidelity_report.csv"
        sample_fidelity.to_csv(path)
        df = load_fidelity_report(path)
        assert df.shape == (3, 4)
        assert df.index.name == "condition"

    def test_load_recommendations(self, tmp_path, sample_recs):
        from gopro.visualize_report import load_recommendations
        path = tmp_path / "recs.csv"
        sample_recs.to_csv(path)
        df = load_recommendations(path)
        assert len(df) == 2
        assert df.index.name == "well"

    def test_load_morphogen_matrix(self, tmp_path, sample_morphogens):
        from gopro.visualize_report import load_morphogen_matrix
        path = tmp_path / "morph.csv"
        sample_morphogens.to_csv(path)
        df = load_morphogen_matrix(path)
        assert df.shape == (10, 5)

    def test_load_cell_type_fractions(self, tmp_path):
        from gopro.visualize_report import load_cell_type_fractions
        fracs = pd.DataFrame(
            {"Neuron": [0.6, 0.3], "NPC": [0.4, 0.7]},
            index=pd.Index(["cond_A", "cond_B"], name="condition"),
        )
        path = tmp_path / "fracs.csv"
        fracs.to_csv(path)
        df = load_cell_type_fractions(path)
        assert df.shape == (2, 2)

    def test_load_diagnostics_no_lengthscales(self, tmp_path):
        from gopro.visualize_report import load_diagnostics
        diag = pd.DataFrame([{"round": 1, "n_training_points": 20}])
        path = tmp_path / "diag.csv"
        diag.to_csv(path, index=False)
        d = load_diagnostics(path)
        assert d["round"] == 1
        assert d["lengthscales"] is None

    def test_load_diagnostics_with_lengthscales(self, tmp_path):
        from gopro.visualize_report import load_diagnostics
        diag = pd.DataFrame([{
            "round": 1, "n_training_points": 20,
            "lengthscale_CHIR99021_uM": 0.5, "lengthscale_SAG_uM": 1.2,
        }])
        path = tmp_path / "diag.csv"
        diag.to_csv(path, index=False)
        d = load_diagnostics(path)
        assert d["lengthscales"]["CHIR99021_uM"] == 0.5

    def test_generate_summary_text(self, sample_fidelity):
        from gopro.visualize_report import generate_summary_text
        diag = {"round": 1, "lengthscales": None}
        text = generate_summary_text(sample_fidelity, diag, 6)
        assert "Round 1" in text
        assert "cond_C" in text
        assert "0.910" in text
        assert "6 new experiments" in text

    def test_generate_summary_with_lengthscales(self, sample_fidelity):
        from gopro.visualize_report import generate_summary_text
        diag = {"round": 1, "lengthscales": {"CHIR": 0.1, "SAG": 0.5, "BMP": 0.3, "RA": 2.0}}
        text = generate_summary_text(sample_fidelity, diag, 6)
        assert "CHIR" in text

    def test_build_morphogen_pca_figure(self):
        from gopro.visualize_report import build_morphogen_pca_figure
        coords = pd.DataFrame(
            {"PC1": [1, 2, 3], "PC2": [4, 5, 6]},
            index=["A", "B", "C"],
        )
        fidelity = pd.Series([0.5, 0.7, 0.9], index=["A", "B", "C"])
        fig = build_morphogen_pca_figure(coords, fidelity)
        assert isinstance(fig, go.Figure)

    def test_compute_morphogen_pca_shape(self, sample_morphogens):
        from gopro.visualize_report import compute_morphogen_pca_with_recommendations
        recs = pd.DataFrame(
            np.random.rand(3, sample_morphogens.shape[1]),
            columns=sample_morphogens.columns,
            index=["R1", "R2", "R3"],
        )
        train_df, rec_df, loadings, var_pct, active_cols = (
            compute_morphogen_pca_with_recommendations(
                sample_morphogens, recs, list(sample_morphogens.columns)
            )
        )
        assert train_df.shape == (10, 2)
        assert rec_df.shape == (3, 2)
        assert list(train_df.columns) == ["PC1", "PC2"]
        assert var_pct > 0

    def test_build_plate_map_figure(self, sample_recs):
        from gopro.visualize_report import build_plate_map_figure
        fig = build_plate_map_figure(sample_recs)
        assert isinstance(fig, go.Figure)

    def test_build_importance_with_lengthscales(self):
        from gopro.visualize_report import build_importance_figure
        ls = {"CHIR": 0.5, "SAG": 2.0, "BMP": 1.0}
        fig = build_importance_figure(lengthscales=ls)
        assert isinstance(fig, go.Figure)

    def test_build_importance_fallback_variance(self, sample_morphogens):
        from gopro.visualize_report import build_importance_figure
        fig = build_importance_figure(morphogen_df=sample_morphogens)
        assert isinstance(fig, go.Figure)

    def test_build_leaderboard_figure(self, sample_fidelity):
        from gopro.visualize_report import build_leaderboard_figure
        fig = build_leaderboard_figure(sample_fidelity, top_n=2)
        assert isinstance(fig, go.Figure)

    def test_build_composition_figure(self):
        from gopro.visualize_report import build_composition_figure
        fracs = pd.DataFrame(
            {"Neuron": [0.6, 0.3], "NPC": [0.4, 0.7]},
            index=["cond_A", "cond_B"],
        )
        fig = build_composition_figure(fracs)
        assert isinstance(fig, go.Figure)

    def test_build_convergence_figure(self):
        from gopro.visualize_report import build_convergence_figure
        fig = build_convergence_figure({1: 0.7})
        assert isinstance(fig, go.Figure)

    def test_build_convergence_multi_round(self):
        from gopro.visualize_report import build_convergence_figure
        fig = build_convergence_figure({1: 0.7, 2: 0.8, 3: 0.85})
        assert isinstance(fig, go.Figure)

    def test_build_cell_umap_figure(self):
        from gopro.visualize_report import build_cell_umap_figure
        coords = np.random.rand(50, 2)
        cell_types = pd.Series(["A"] * 25 + ["B"] * 25)
        conditions = pd.Series(["cond1"] * 25 + ["cond2"] * 25)
        fig = build_cell_umap_figure(coords, cell_types, conditions)
        assert isinstance(fig, go.Figure)


class TestMorphogenColumns:
    """Tests for morphogen column consistency."""

    def test_columns_are_unique(self):
        assert len(step04.MORPHOGEN_COLUMNS) == len(set(step04.MORPHOGEN_COLUMNS))

    def test_columns_are_strings(self):
        for col in step04.MORPHOGEN_COLUMNS:
            assert isinstance(col, str)
            assert len(col) > 0


class TestConfig:
    """Tests for centralized config module."""

    def test_config_project_dir_fallback(self):
        """PROJECT_DIR resolves without env var."""
        from gopro.config import PROJECT_DIR
        assert isinstance(PROJECT_DIR, Path)
        assert PROJECT_DIR.exists()

    def test_config_project_dir_env_override(self, monkeypatch, tmp_path):
        """GPBO_PROJECT_DIR env var overrides PROJECT_DIR."""
        monkeypatch.setenv("GPBO_PROJECT_DIR", str(tmp_path))
        # Need to reimport to pick up the env var
        import importlib
        import gopro.config
        importlib.reload(gopro.config)
        assert gopro.config.PROJECT_DIR == tmp_path
        # Restore
        monkeypatch.delenv("GPBO_PROJECT_DIR")
        importlib.reload(gopro.config)

    def test_config_data_dir_env_override(self, monkeypatch, tmp_path):
        """GPBO_DATA_DIR env var overrides DATA_DIR."""
        data_dir = tmp_path / "custom_data"
        monkeypatch.setenv("GPBO_DATA_DIR", str(data_dir))
        import importlib
        import gopro.config
        importlib.reload(gopro.config)
        assert gopro.config.DATA_DIR == data_dir
        monkeypatch.delenv("GPBO_DATA_DIR")
        importlib.reload(gopro.config)

    def test_morphogen_columns_length(self):
        """MORPHOGEN_COLUMNS has exactly 20 entries."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert len(MORPHOGEN_COLUMNS) == 20

    def test_morphogen_columns_canonical_order(self):
        """Indices 16-18 are Dorsomorphin, purmorphamine, cyclopamine."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert MORPHOGEN_COLUMNS[16] == "Dorsomorphin_uM"
        assert MORPHOGEN_COLUMNS[17] == "purmorphamine_uM"
        assert MORPHOGEN_COLUMNS[18] == "cyclopamine_uM"

    def test_morphogen_columns_unique(self):
        """No duplicate entries in MORPHOGEN_COLUMNS."""
        from gopro.config import MORPHOGEN_COLUMNS
        assert len(MORPHOGEN_COLUMNS) == len(set(MORPHOGEN_COLUMNS))

    def test_annot_level_values(self):
        """All 4 ANNOT constants have expected values."""
        from gopro.config import ANNOT_LEVEL_1, ANNOT_LEVEL_2, ANNOT_REGION, ANNOT_LEVEL_3
        assert ANNOT_LEVEL_1 == "annot_level_1"
        assert ANNOT_LEVEL_2 == "annot_level_2"
        assert ANNOT_REGION == "annot_region_rev2"
        assert ANNOT_LEVEL_3 == "annot_level_3_rev2"

    def test_get_logger_returns_logger(self):
        """get_logger returns a logging.Logger instance."""
        import logging
        from gopro.config import get_logger
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_module"

    def test_get_logger_respects_env_level(self, monkeypatch):
        """GPBO_LOG_LEVEL env var controls logger level."""
        import logging
        monkeypatch.setenv("GPBO_LOG_LEVEL", "DEBUG")
        from gopro.config import get_logger
        log = get_logger("test_debug_level")
        assert log.level == logging.DEBUG


class TestExtractLengthscales:
    """Tests for _extract_lengthscales helper."""

    def test_standard_gp(self):
        """Extract lengthscales from MAP-fitted SingleTaskGP."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll
        import torch

        X = torch.rand(10, 3, dtype=torch.double)
        Y = torch.rand(10, 1, dtype=torch.double)
        model = SingleTaskGP(X, Y, input_transform=Normalize(d=3), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        ls = step04._extract_lengthscales(model, 3)
        assert ls is not None
        assert ls.shape == (3,)
        assert np.all(ls > 0)

    def test_returns_none_for_unknown_model(self):
        """Returns None for unrecognized model types."""
        ls = step04._extract_lengthscales("not_a_model", 3)
        assert ls is None


class TestSAASBO:
    """Tests for SAASBO integration in GP-BO pipeline."""

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        n, d, m = 15, 4, 3
        X = pd.DataFrame(
            np.random.rand(n, d),
            columns=["CHIR99021_uM", "SAG_uM", "BMP4_uM", "log_harvest_day"],
            index=[f"cond_{i}" for i in range(n)],
        )
        X["fidelity"] = 1.0
        Y = pd.DataFrame(
            np.random.dirichlet(np.ones(m), size=n),
            columns=["ct_A", "ct_B", "ct_C"],
            index=X.index,
        )
        return X, Y

    def test_saasbo_returns_model_list_gp(self, small_data):
        from botorch.models import ModelListGP
        X, Y = small_data
        model, _, train_Y, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        assert isinstance(model, ModelListGP)
        assert train_Y.shape[1] == 2  # ILR: 3 -> 2

    def test_saasbo_lengthscales_extractable(self, small_data):
        X, Y = small_data
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        ls = step04._extract_lengthscales(model, X.shape[1])
        assert ls is not None
        assert len(ls) == X.shape[1]

    def test_saasbo_posterior_works(self, small_data):
        import torch
        X, Y = small_data
        model, train_X, _, _ = step04.fit_gp_botorch(
            X, Y, use_ilr=True, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        with torch.no_grad():
            post = model.posterior(train_X[:2])
        assert post.mean.shape[-1] == 2

    def test_saasbo_false_uses_standard_gp(self, small_data):
        from botorch.models import SingleTaskGP
        X, Y = small_data
        model, _, _, _ = step04.fit_gp_botorch(X, Y, use_ilr=True, use_saasbo=False)
        assert isinstance(model, SingleTaskGP)

    def test_saasbo_ignored_for_multi_fidelity(self, small_data):
        from botorch.models import SingleTaskMultiFidelityGP
        X, Y = small_data
        X = X.copy()
        X.loc[X.index[:5], "fidelity"] = 0.5
        model, _, _, _ = step04.fit_gp_botorch(X, Y, use_ilr=True, use_saasbo=True)
        assert isinstance(model, SingleTaskMultiFidelityGP)

    def test_saasbo_single_output(self, small_data):
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        X, Y = small_data
        model, _, _, _ = step04.fit_gp_botorch(
            X, Y[["ct_A"]], use_ilr=False, use_saasbo=True,
            saasbo_warmup=8, saasbo_num_samples=8, saasbo_thinning=4,
        )
        assert isinstance(model, SaasFullyBayesianSingleTaskGP)
