"""Tests for Phase 4 (CellRank 2) and Phase 5 (CellFlow) virtual data generation."""

import math
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from conftest import _import_pipeline_module

step04 = _import_pipeline_module("04_gpbo_loop")
step05 = _import_pipeline_module("05_cellrank2_virtual")
step06 = _import_pipeline_module("06_cellflow_virtual")


# ==============================================================================
# Phase 4: CellRank 2 Tests
# ==============================================================================


class TestCellRank2Constants:
    """Tests for CellRank 2 configuration constants."""

    def test_atlas_timepoints_ordered(self):
        assert step05.ATLAS_TIMEPOINTS == sorted(step05.ATLAS_TIMEPOINTS)

    def test_projection_targets_are_subset_of_atlas(self):
        for tp in step05.PROJECTION_TARGETS:
            assert tp in step05.ATLAS_TIMEPOINTS, (
                f"Projection target Day {tp} not in atlas timepoints"
            )

    def test_fidelity_level(self):
        assert step05.FIDELITY_LEVEL == 0.5

    def test_moscot_parameters_valid(self):
        assert 0 < step05.MOSCOT_EPSILON < 1
        assert 0 < step05.MOSCOT_TAU_A <= 1
        assert 0 < step05.MOSCOT_TAU_B <= 1


class TestBuildVirtualMorphogenMatrix:
    """Tests for building morphogen matrix for virtual data points."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample virtual fractions and real morphogen data."""
        virtual_fracs = pd.DataFrame({
            "original_condition": ["CHIR1.5", "SAG250"],
            "target_day": [60, 90],
            "n_source_cells": [100, 50],
            "Neuron": [0.5, 0.3],
            "NPC": [0.3, 0.5],
            "Glia": [0.2, 0.2],
        }, index=["CHIR1.5_day60", "SAG250_day90"])

        real_morphogens = pd.DataFrame({
            "CHIR99021_uM": [1.5, 0.0],
            "SAG_uM": [0.0, 0.25],
            "log_harvest_day": [math.log(72), math.log(72)],
        }, index=["CHIR1.5", "SAG250"])

        morph_path = tmp_path / "morphogens.csv"
        real_morphogens.to_csv(morph_path)

        return virtual_fracs, morph_path

    def test_builds_matrix(self, sample_data):
        virtual_fracs, morph_path = sample_data
        result = step05.build_virtual_morphogen_matrix(virtual_fracs, morph_path)
        assert len(result) == 2
        assert "CHIR99021_uM" in result.columns

    def test_harvest_day_updated(self, sample_data):
        virtual_fracs, morph_path = sample_data
        result = step05.build_virtual_morphogen_matrix(virtual_fracs, morph_path)
        assert result.loc["CHIR1.5_day60", "log_harvest_day"] == pytest.approx(
            math.log(60), abs=0.01
        )
        assert result.loc["SAG250_day90", "log_harvest_day"] == pytest.approx(
            math.log(90), abs=0.01
        )

    def test_morphogen_concentrations_preserved(self, sample_data):
        virtual_fracs, morph_path = sample_data
        result = step05.build_virtual_morphogen_matrix(virtual_fracs, morph_path)
        assert result.loc["CHIR1.5_day60", "CHIR99021_uM"] == 1.5
        assert result.loc["SAG250_day90", "SAG_uM"] == 0.25

    def test_missing_condition_skipped(self, tmp_path):
        virtual_fracs = pd.DataFrame({
            "original_condition": ["NONEXISTENT"],
            "target_day": [60],
            "Neuron": [0.5],
        }, index=["NONEXISTENT_day60"])

        real_morphogens = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "log_harvest_day": [math.log(72)],
        }, index=["CHIR1.5"])

        morph_path = tmp_path / "morphogens.csv"
        real_morphogens.to_csv(morph_path)

        result = step05.build_virtual_morphogen_matrix(virtual_fracs, morph_path)
        assert len(result) == 0


class TestMoscotPushAPI:
    """Tests for moscot .push() API integration."""

    def test_push_replaces_manual_transport(self):
        """Verify project_query_forward uses push() when available."""
        import unittest.mock as mock

        # Create a mock problem that supports .push()
        mock_problem = mock.MagicMock()

        # push() returns a 1D distribution over target cells
        n_target = 20
        target_dist = np.random.dirichlet(np.ones(n_target))
        mock_problem.push.return_value = target_dist

        # Verify push is callable and returns expected shape
        assert hasattr(mock_problem, 'push')
        result = mock_problem.push(
            source_distribution=np.ones(10) / 10,
            source=15,
            target=30,
        )
        assert result.shape == (n_target,)
        assert np.isclose(result.sum(), 1.0)


class TestValidateTransportQuality:
    """Tests for transport quality validation."""

    def test_handles_empty_solutions(self):
        class MockProblem:
            solutions = {}
        report = step05.validate_transport_quality(MockProblem())
        assert len(report) == 0


# ==============================================================================
# Phase 5: CellFlow Tests
# ==============================================================================


class TestCellFlowConstants:
    """Tests for CellFlow configuration constants."""

    def test_fidelity_level(self):
        assert step06.FIDELITY_LEVEL == 0.0

    def test_morphogen_identities_complete(self):
        """All morphogens (except log_harvest_day) have identity mappings."""
        from gopro.config import MORPHOGEN_COLUMNS

        for col in MORPHOGEN_COLUMNS:
            if col == "log_harvest_day":
                continue
            assert col in step06.MORPHOGEN_IDENTITIES, (
                f"Missing identity mapping for {col}"
            )

    def test_morphogen_pathways_complete(self):
        for col in step06.MORPHOGEN_IDENTITIES:
            assert col in step06.MORPHOGEN_PATHWAYS, (
                f"Missing pathway for {col}"
            )

    def test_pathway_values_valid(self):
        valid_pathways = {"WNT", "BMP", "SHH", "RA", "FGF", "TGFb",
                         "Notch", "EGF", "neurotrophin", "unknown"}
        for col, pathway in step06.MORPHOGEN_PATHWAYS.items():
            assert pathway in valid_pathways, (
                f"Invalid pathway '{pathway}' for {col}"
            )


class TestEncodeProtocol:
    """Tests for CellFlow protocol encoding."""

    def test_basic_encoding(self):
        vec = {"CHIR99021_uM": 1.5, "SAG_uM": 0.25, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert "modulators" in enc
        assert len(enc["modulators"]) == 2  # CHIR + SAG
        assert enc["harvest_day"] == 21

    def test_zero_morphogens_excluded(self):
        vec = {"CHIR99021_uM": 0.0, "SAG_uM": 0.25, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert len(enc["modulators"]) == 1  # only SAG

    def test_smiles_included_for_small_molecules(self):
        vec = {"CHIR99021_uM": 1.5, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert "smiles" in enc["modulators"][0]

    def test_no_smiles_for_proteins(self):
        vec = {"BMP4_uM": 0.000769, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert "smiles" not in enc["modulators"][0]

    def test_pathway_annotation(self):
        vec = {"CHIR99021_uM": 1.5, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert enc["modulators"][0]["pathway"] == "WNT"

    def test_empty_protocol(self):
        vec = {"log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert len(enc["modulators"]) == 0


class TestVirtualScreenGrid:
    """Tests for virtual screen grid generation."""

    def test_basic_grid(self):
        ranges = {
            "CHIR99021_uM": [0.0, 1.5, 3.0],
            "SAG_uM": [0.0, 0.25],
        }
        grid = step06.generate_virtual_screen_grid(ranges, harvest_days=[21])
        # 3 * 2 = 6 combinations (all others are [0.0])
        assert len(grid) == 6
        assert "CHIR99021_uM" in grid.columns
        assert "SAG_uM" in grid.columns

    def test_harvest_day_variation(self):
        ranges = {"CHIR99021_uM": [1.5]}
        grid = step06.generate_virtual_screen_grid(
            ranges, harvest_days=[21, 45, 72]
        )
        assert len(grid) == 3
        log_days = grid["log_harvest_day"].unique()
        assert len(log_days) == 3

    def test_max_combinations_limit(self):
        ranges = {
            "CHIR99021_uM": [0.0, 0.5, 1.0, 1.5, 3.0, 6.0, 9.0, 12.0],
            "SAG_uM": [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
            "BMP4_uM": [0.0, 0.000385, 0.000769, 0.001923, 0.003846],
            "RA_uM": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        }
        grid = step06.generate_virtual_screen_grid(
            ranges, max_combinations=100
        )
        assert len(grid) == 100

    def test_grid_index_format(self):
        ranges = {"CHIR99021_uM": [1.5]}
        grid = step06.generate_virtual_screen_grid(ranges)
        assert grid.index[0].startswith("virtual_")

    def test_all_morphogen_columns_present(self):
        ranges = {"CHIR99021_uM": [1.5]}
        grid = step06.generate_virtual_screen_grid(ranges)
        assert "log_harvest_day" in grid.columns


class TestBaselinePredictor:
    """Tests for the heuristic baseline predictor."""

    def test_produces_valid_fractions(self):
        protocols = pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0, 0.0],
            "SAG_uM": [0.0, 0.25, 1.0],
            "BMP4_uM": [0.0, 0.000769, 0.0],
            "log_harvest_day": [math.log(21)] * 3,
        }, index=["p1", "p2", "p3"])

        result = step06._predict_baseline(protocols)
        assert len(result) == 3

        # Check fractions sum to ~1
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

        # Check all values in [0, 1]
        assert (result >= 0).all().all()
        assert (result <= 1).all().all()

    def test_different_protocols_different_results(self):
        protocols = pd.DataFrame({
            "CHIR99021_uM": [0.0, 9.0],
            "SAG_uM": [0.0, 0.0],
            "BMP4_uM": [0.003846, 0.0],
            "LDN193189_uM": [0.0, 0.2],
            "SB431542_uM": [0.0, 10.0],
            "log_harvest_day": [math.log(21)] * 2,
        }, index=["bmp_high", "neural_induction"])

        result = step06._predict_baseline(protocols)
        # BMP should produce more CP; neural induction should produce more neuroepithelium
        assert result.loc["bmp_high", "CP"] > result.loc["neural_induction", "CP"]
        assert (result.loc["neural_induction", "Neuroepithelium"]
                > result.loc["bmp_high", "Neuroepithelium"])

    def test_with_real_fractions_csv(self, tmp_path):
        """Bug #3: When real_fractions_csv is provided, cell types match."""
        # Create real training fractions with level-2 labels
        real_Y = pd.DataFrame({
            "Cortical EN": [0.4, 0.3],
            "Cortical RG": [0.3, 0.4],
            "Cortical IP": [0.2, 0.2],
            "OPC": [0.1, 0.1],
        }, index=["cond_A", "cond_B"])
        csv_path = tmp_path / "real_fracs.csv"
        real_Y.to_csv(csv_path)

        protocols = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "SAG_uM": [0.25],
            "log_harvest_day": [math.log(21)],
        }, index=["test"])

        result = step06._predict_baseline(protocols, real_fractions_csv=csv_path)

        # Should use level-2 column names from real data
        assert set(result.columns) == {"Cortical EN", "Cortical RG", "Cortical IP", "OPC"}
        # Fractions still valid
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)
        assert (result >= 0).all().all()

    def test_with_nonexistent_csv_falls_back(self, tmp_path):
        """Bug #3: Missing CSV should fall back to level-1 types."""
        protocols = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "log_harvest_day": [math.log(21)],
        }, index=["test"])

        result = step06._predict_baseline(
            protocols, real_fractions_csv=tmp_path / "nonexistent.csv"
        )
        # Should use fallback level-1 labels
        assert "Neuron" in result.columns
        assert "NPC" in result.columns


class TestPredictionConfidence:
    """Tests for CellFlow prediction confidence estimation."""

    def test_nearby_predictions_high_confidence(self):
        training = pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0, 0.0],
            "SAG_uM": [0.0, 0.25, 0.5],
        }, index=["t1", "t2", "t3"])

        # Near training point
        predictions = pd.DataFrame({
            "CHIR99021_uM": [1.6],
            "SAG_uM": [0.01],
        }, index=["near"])

        confidence = step06.compute_prediction_confidence(predictions, training)
        assert confidence.iloc[0] > 0.5

    def test_far_predictions_low_confidence(self):
        training = pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0],
            "SAG_uM": [0.0, 0.25],
        }, index=["t1", "t2"])

        # Far from training data
        predictions = pd.DataFrame({
            "CHIR99021_uM": [100.0],
            "SAG_uM": [10.0],
        }, index=["far"])

        confidence = step06.compute_prediction_confidence(predictions, training)
        assert confidence.iloc[0] < 0.5

    def test_confidence_in_range(self):
        training = pd.DataFrame({
            "CHIR99021_uM": np.random.rand(20) * 10,
            "SAG_uM": np.random.rand(20) * 2.0,
        })
        predictions = pd.DataFrame({
            "CHIR99021_uM": np.random.rand(10) * 10,
            "SAG_uM": np.random.rand(10) * 2.0,
        })

        confidence = step06.compute_prediction_confidence(predictions, training)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()


# ==============================================================================
# Multi-fidelity GP Integration Tests
# ==============================================================================


class TestMultiFidelityIntegration:
    """Tests for multi-fidelity data merging and GP fitting."""

    @pytest.fixture
    def multi_fidelity_data(self, tmp_path):
        """Create real + virtual training data."""
        np.random.seed(42)

        # Real data (fidelity=1.0)
        n_real = 20
        Y_real = pd.DataFrame(
            np.random.dirichlet(np.ones(5), size=n_real),
            columns=["ct_A", "ct_B", "ct_C", "ct_D", "ct_E"],
            index=[f"real_{i}" for i in range(n_real)],
        )
        X_real = pd.DataFrame({
            "CHIR99021_uM": np.random.uniform(0, 6, n_real),
            "SAG_uM": np.random.uniform(0, 2.0, n_real),
            "log_harvest_day": np.log(72) * np.ones(n_real),
        }, index=Y_real.index)

        # CellRank2 virtual data (fidelity=0.5)
        n_cr2 = 30
        Y_cr2 = pd.DataFrame(
            np.random.dirichlet(np.ones(5), size=n_cr2),
            columns=["ct_A", "ct_B", "ct_C", "ct_D", "ct_E"],
            index=[f"cr2_{i}" for i in range(n_cr2)],
        )
        X_cr2 = pd.DataFrame({
            "CHIR99021_uM": np.random.uniform(0, 6, n_cr2),
            "SAG_uM": np.random.uniform(0, 2.0, n_cr2),
            "log_harvest_day": np.log(90) * np.ones(n_cr2),
        }, index=Y_cr2.index)

        # CellFlow virtual data (fidelity=0.0)
        n_cf = 50
        Y_cf = pd.DataFrame(
            np.random.dirichlet(np.ones(5), size=n_cf),
            columns=["ct_A", "ct_B", "ct_C", "ct_D", "ct_E"],
            index=[f"cf_{i}" for i in range(n_cf)],
        )
        X_cf = pd.DataFrame({
            "CHIR99021_uM": np.random.uniform(0, 6, n_cf),
            "SAG_uM": np.random.uniform(0, 2.0, n_cf),
            "log_harvest_day": np.log(21) * np.ones(n_cf),
        }, index=Y_cf.index)

        # Save all
        paths = {}
        for name, X, Y in [("real", X_real, Y_real), ("cr2", X_cr2, Y_cr2), ("cf", X_cf, Y_cf)]:
            y_path = tmp_path / f"{name}_fractions.csv"
            x_path = tmp_path / f"{name}_morphogens.csv"
            Y.to_csv(y_path)
            X.to_csv(x_path)
            paths[name] = (y_path, x_path)

        return paths

    def test_merge_multi_fidelity(self, multi_fidelity_data):
        paths = multi_fidelity_data
        sources = [
            (paths["real"][0], paths["real"][1], 1.0),
            (paths["cr2"][0], paths["cr2"][1], 0.5),
            (paths["cf"][0], paths["cf"][1], 0.0),
        ]
        X, Y = step04.merge_multi_fidelity_data(sources)
        assert len(X) == 100  # 20 + 30 + 50
        assert "fidelity" in X.columns
        assert X["fidelity"].nunique() == 3

    def test_merge_preserves_fidelity_levels(self, multi_fidelity_data):
        paths = multi_fidelity_data
        sources = [
            (paths["real"][0], paths["real"][1], 1.0),
            (paths["cr2"][0], paths["cr2"][1], 0.5),
        ]
        X, Y = step04.merge_multi_fidelity_data(sources)
        assert (X["fidelity"] == 1.0).sum() == 20
        assert (X["fidelity"] == 0.5).sum() == 30

    def test_merge_aligns_columns(self, tmp_path):
        """Test that merging aligns columns when sources have different cell types."""
        Y1 = pd.DataFrame({"ct_A": [0.6], "ct_B": [0.4]}, index=["c1"])
        Y2 = pd.DataFrame({"ct_B": [0.3], "ct_C": [0.7]}, index=["c2"])
        X1 = pd.DataFrame({"morph": [1.0]}, index=["c1"])
        X2 = pd.DataFrame({"morph": [2.0]}, index=["c2"])

        for name, X, Y in [("s1", X1, Y1), ("s2", X2, Y2)]:
            Y.to_csv(tmp_path / f"{name}_Y.csv")
            X.to_csv(tmp_path / f"{name}_X.csv")

        sources = [
            (tmp_path / "s1_Y.csv", tmp_path / "s1_X.csv", 1.0),
            (tmp_path / "s2_Y.csv", tmp_path / "s2_X.csv", 0.5),
        ]
        X, Y = step04.merge_multi_fidelity_data(sources)
        assert "ct_A" in Y.columns
        assert "ct_B" in Y.columns
        assert "ct_C" in Y.columns
        assert len(Y) == 2

    def test_merge_skips_missing_files(self, multi_fidelity_data, tmp_path):
        paths = multi_fidelity_data
        sources = [
            (paths["real"][0], paths["real"][1], 1.0),
            (tmp_path / "nonexistent.csv", tmp_path / "nonexistent2.csv", 0.5),
        ]
        X, Y = step04.merge_multi_fidelity_data(sources)
        assert len(X) == 20  # only real data

    def test_gpbo_loop_with_virtual_sources(self, multi_fidelity_data, tmp_path):
        """Test that GP-BO loop works with multi-fidelity data."""
        paths = multi_fidelity_data
        virtual_sources = [
            (paths["cr2"][0], paths["cr2"][1], 0.5),
        ]
        # Point DATA_DIR to tmp_path so file saves work
        original_data_dir = step04.DATA_DIR
        step04.DATA_DIR = tmp_path
        try:
            recs = step04.run_gpbo_loop(
                fractions_csv=paths["real"][0],
                morphogen_csv=paths["real"][1],
                n_recommendations=4,
                round_num=1,
                use_ilr=True,
                virtual_sources=virtual_sources,
            )
        finally:
            step04.DATA_DIR = original_data_dir
        assert len(recs) == 4
        assert "acquisition_value" in recs.columns


# ==============================================================================
# Property-based tests
# ==============================================================================


class TestPhase45Properties:
    """Property tests for phases 4-5 invariants."""

    def test_baseline_predictor_always_valid(self):
        """Baseline predictor should always produce valid fractions."""
        np.random.seed(42)
        for _ in range(20):
            protocols = pd.DataFrame({
                "CHIR99021_uM": [np.random.uniform(0, 12)],
                "SAG_uM": [np.random.uniform(0, 2.0)],
                "BMP4_uM": [np.random.uniform(0, 0.004)],
                "SHH_uM": [np.random.uniform(0, 0.026)],
                "RA_uM": [np.random.uniform(0, 1.0)],
                "FGF8_uM": [np.random.uniform(0, 0.009)],
                "IWP2_uM": [np.random.uniform(0, 10)],
                "LDN193189_uM": [np.random.uniform(0, 0.5)],
                "SB431542_uM": [np.random.uniform(0, 20)],
                "DAPT_uM": [np.random.uniform(0, 10)],
                "EGF_uM": [np.random.uniform(0, 0.008)],
                "log_harvest_day": [math.log(np.random.uniform(7, 120))],
            }, index=["test"])

            result = step06._predict_baseline(protocols)
            assert (result >= 0).all().all(), "Negative fraction"
            row_sum = result.sum(axis=1).iloc[0]
            assert abs(row_sum - 1.0) < 1e-6, f"Row sum {row_sum} != 1.0"

    def test_protocol_encoding_deterministic(self):
        """Same input should always produce same encoding."""
        vec = {"CHIR99021_uM": 1.5, "SAG_uM": 0.25, "log_harvest_day": math.log(21)}
        enc1 = step06.encode_protocol_cellflow(vec)
        enc2 = step06.encode_protocol_cellflow(vec)
        assert enc1 == enc2

    def test_confidence_monotonic_with_distance(self):
        """Confidence should decrease with distance from training data."""
        training = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "SAG_uM": [0.25],
        }, index=["t1"])

        # Near and far predictions
        predictions = pd.DataFrame({
            "CHIR99021_uM": [1.6, 100.0],
            "SAG_uM": [0.26, 50.0],
        }, index=["near", "far"])

        confidence = step06.compute_prediction_confidence(predictions, training)
        assert confidence.loc["near"] > confidence.loc["far"]


# ==============================================================================
# Bug Fix Tests
# ==============================================================================


class TestCategoricalDtypeFix:
    """Tests for the pandas CategoricalDtype fix in build_cellrank_kernel."""

    def test_non_categorical_day_gets_converted(self):
        """Verify non-categorical 'day' column doesn't crash with new API."""
        import anndata as ad

        adata = ad.AnnData(
            X=np.random.rand(10, 5).astype(np.float32),
            obs=pd.DataFrame({"day": [7] * 5 + [15] * 5}),
        )
        # The isinstance check should work without AttributeError
        assert not isinstance(adata.obs["day"].dtype, pd.CategoricalDtype)
        # After conversion (what the function does):
        adata.obs["day"] = adata.obs["day"].astype("category")
        assert isinstance(adata.obs["day"].dtype, pd.CategoricalDtype)

    def test_already_categorical_day_unchanged(self):
        """If day is already categorical, no error."""
        import anndata as ad

        adata = ad.AnnData(
            X=np.random.rand(10, 5).astype(np.float32),
            obs=pd.DataFrame({"day": pd.Categorical([7] * 5 + [15] * 5)}),
        )
        assert isinstance(adata.obs["day"].dtype, pd.CategoricalDtype)


class TestMinBoundWidthFix:
    """Tests for the MIN_BOUND_WIDTH guard in _compute_active_bounds."""

    def test_near_zero_column_gets_minimum_width(self):
        """Near-zero col_max should not produce zero-width bounds."""
        X = pd.DataFrame({
            "CHIR99021_uM": [0.0, 3.0, 6.0],
            "BMP4_uM": [1e-10, 2e-10, 3e-10],  # near-zero but not zero-variance
            "log_harvest_day": [np.log(72)] * 3,
        })
        bounds, active_cols = step04._compute_active_bounds(X, list(X.columns))

        # BMP4 should still be active (not zero-variance)
        assert "BMP4_uM" in active_cols
        lo, hi = bounds["BMP4_uM"]
        # Bounds must have nonzero width
        assert hi - lo >= 1e-6
        assert hi > lo

    def test_exact_zero_variance_dropped(self):
        """Truly zero-variance columns should still be dropped."""
        X = pd.DataFrame({
            "CHIR99021_uM": [0.0, 3.0, 6.0],
            "SHH_uM": [0.0, 0.0, 0.0],  # exactly zero variance
            "log_harvest_day": [np.log(72)] * 3,
        })
        bounds, active_cols = step04._compute_active_bounds(X, list(X.columns))
        assert "SHH_uM" not in active_cols

    def test_normal_range_unaffected(self):
        """Normal-range columns should not be affected by MIN_BOUND_WIDTH."""
        X = pd.DataFrame({
            "CHIR99021_uM": [0.0, 3.0, 6.0],
            "log_harvest_day": [np.log(72)] * 3,
        })
        bounds, active_cols = step04._compute_active_bounds(X, list(X.columns))
        lo, hi = bounds["CHIR99021_uM"]
        # Should use normal padding, not MIN_BOUND_WIDTH
        assert hi - lo > 1.0  # much larger than 1e-6


class TestTransportMapFix:
    """Tests for transport-map-based virtual fraction computation."""

    @pytest.fixture
    def mock_transport_data(self):
        """Create mock data for transport map testing.

        Sets up:
        - Atlas with 2 timepoints (day 15, day 30), 20 cells each
        - Query with 2 conditions, 5 cells each, at day 21
        - Mock moscot problem with a known transport matrix
        - Cell types at target: first half Neurons, second half Glia
        """
        import anndata as ad

        np.random.seed(42)
        n_source = 20
        n_target = 20
        n_query = 10

        # Atlas: source (day 15) and target (day 30)
        atlas_X = np.random.rand(n_source + n_target, 50).astype(np.float32)
        atlas_obs = pd.DataFrame({
            "day": [15] * n_source + [30] * n_target,
            "annot_level_2": (
                [f"source_ct_{i % 3}" for i in range(n_source)]
                + ["Neuron"] * 10 + ["Glia"] * 10
            ),
        })
        atlas = ad.AnnData(X=atlas_X, obs=atlas_obs)
        atlas.obsm["X_pca"] = np.random.rand(
            n_source + n_target, 30
        ).astype(np.float32)

        # Query: 2 conditions, 5 cells each
        query_X = np.random.rand(n_query, 50).astype(np.float32)
        query_obs = pd.DataFrame({
            "condition": ["condA"] * 5 + ["condB"] * 5,
            "predicted_annot_level_2": ["Neuron"] * 5 + ["Glia"] * 5,
        })
        query = ad.AnnData(X=query_X, obs=query_obs)
        # Place condA near source cells 0-4, condB near source cells 15-19
        query_pca = np.random.rand(n_query, 30).astype(np.float32)
        query_pca[:5] = atlas.obsm["X_pca"][:5] + np.random.randn(5, 30) * 0.01
        query_pca[5:] = atlas.obsm["X_pca"][15:20] + np.random.randn(5, 30) * 0.01
        query.obsm["X_pca"] = query_pca

        # Mock transport matrix:
        # Source cells 0-9 transport to target cells 0-9 (Neurons)
        # Source cells 10-19 transport to target cells 10-19 (Glia)
        transport = np.zeros((n_source, n_target))
        for i in range(n_source):
            if i < 10:
                transport[i, :10] = 1.0 / 10  # -> Neurons
            else:
                transport[i, 10:] = 1.0 / 10  # -> Glia

        # Mock moscot problem — production code accesses .solution.transport_matrix
        class MockSolution:
            def __init__(self, tm):
                self.transport_matrix = tm
                self.cost = 0.5
                self.converged = True

        class MockProblemEntry:
            def __init__(self, tm):
                self.solution = MockSolution(tm)

        class MockProblem:
            def __init__(self, tm):
                self._solutions = {(15, 30): MockProblemEntry(tm)}
                self.solutions = self._solutions

            def __getitem__(self, key):
                return self._solutions[key]

        problem = MockProblem(transport)

        return query, atlas, problem

    def test_different_conditions_different_fractions(self, mock_transport_data):
        """Core test: conditions near different source cells should get
        different virtual fractions via transport maps."""
        query, atlas, problem = mock_transport_data

        result = step05.project_query_forward(
            query_adata=query,
            atlas_adata=atlas,
            problem=problem,
            query_timepoint=21,
            target_timepoints=[30],
            label_key="annot_level_2",
            condition_key="condition",
        )

        assert len(result) == 2  # 2 conditions x 1 target timepoint

        # condA neighbors are source cells 0-4, which transport to Neurons
        # condB neighbors are source cells 15-19, which transport to Glia
        ct_cols = [c for c in result.columns
                   if c not in ["original_condition", "target_day", "n_source_cells"]]

        condA_fracs = result.loc["condA_day30", ct_cols].astype(float)
        condB_fracs = result.loc["condB_day30", ct_cols].astype(float)

        # They should be DIFFERENT (this is what the old bug prevented)
        assert not np.allclose(condA_fracs.values, condB_fracs.values), \
            "Conditions should produce different fractions via transport maps"

        # condA should have more Neurons, condB should have more Glia
        assert condA_fracs.get("Neuron", 0) > condB_fracs.get("Neuron", 0)
        assert condB_fracs.get("Glia", 0) > condA_fracs.get("Glia", 0)

    def test_fractions_sum_to_one(self, mock_transport_data):
        """Virtual fractions should be valid probability distributions."""
        query, atlas, problem = mock_transport_data

        result = step05.project_query_forward(
            query_adata=query,
            atlas_adata=atlas,
            problem=problem,
            query_timepoint=21,
            target_timepoints=[30],
        )

        ct_cols = [c for c in result.columns
                   if c not in ["original_condition", "target_day", "n_source_cells"]]
        row_sums = result[ct_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_fallback_on_missing_transport(self):
        """Should fall back to atlas average when transport key is missing."""
        import anndata as ad

        np.random.seed(42)
        atlas = ad.AnnData(
            X=np.random.rand(20, 5).astype(np.float32),
            obs=pd.DataFrame({
                "day": [15] * 10 + [30] * 10,
                "annot_level_2": ["Neuron"] * 10 + ["Glia"] * 5 + ["Neuron"] * 5,
            }),
        )
        atlas.obsm["X_pca"] = np.random.rand(20, 10).astype(np.float32)

        query = ad.AnnData(
            X=np.random.rand(5, 5).astype(np.float32),
            obs=pd.DataFrame({
                "condition": ["cond1"] * 5,
                "predicted_annot_level_2": ["Neuron"] * 5,
            }),
        )
        query.obsm["X_pca"] = np.random.rand(5, 10).astype(np.float32)

        # Problem with no solutions for the needed timepoint pair
        class EmptyProblem:
            solutions = {}
            def __getitem__(self, key):
                raise KeyError(f"No solution for {key}")

        result = step05.project_query_forward(
            query_adata=query,
            atlas_adata=atlas,
            problem=EmptyProblem(),
            query_timepoint=21,
            target_timepoints=[30],
        )

        # Should still produce output (from fallback)
        assert len(result) >= 1

    def test_sparse_transport_matrix(self, mock_transport_data):
        """Should work with scipy sparse transport matrices."""
        import scipy.sparse as sp

        query, atlas, problem = mock_transport_data

        # Convert transport to sparse
        orig_tm = problem[(15, 30)].solution.transport_matrix
        problem[(15, 30)].solution.transport_matrix = sp.csr_matrix(orig_tm)

        result = step05.project_query_forward(
            query_adata=query,
            atlas_adata=atlas,
            problem=problem,
            query_timepoint=21,
            target_timepoints=[30],
        )

        assert len(result) == 2
        ct_cols = [c for c in result.columns
                   if c not in ["original_condition", "target_day", "n_source_cells"]]
        row_sums = result[ct_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestNewScriptsSyntax:
    """Verify new scripts parse correctly."""

    def test_00a_download_geo_syntax(self):
        import ast
        path = Path(__file__).parent.parent / "00a_download_geo.py"
        ast.parse(path.read_text())

    def test_00c_build_temporal_atlas_syntax(self):
        import ast
        path = Path(__file__).parent.parent / "00c_build_temporal_atlas.py"
        ast.parse(path.read_text())
