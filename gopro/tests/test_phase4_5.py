"""Tests for Phase 4 (CellRank 2) and Phase 5 (CellFlow) virtual data generation."""

import math
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

GOPRO_DIR = Path(__file__).parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(GOPRO_DIR / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


step04 = _load("04_gpbo_loop")
step05 = _load("05_cellrank2_virtual")
step06 = _load("06_cellflow_virtual")


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
            "SAG_nM": [0.0, 250.0],
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
        assert result.loc["SAG250_day90", "SAG_nM"] == 250.0

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
        from importlib import util
        step04_spec = util.spec_from_file_location(
            "step04", str(GOPRO_DIR / "04_gpbo_loop.py")
        )
        step04_mod = util.module_from_spec(step04_spec)
        step04_spec.loader.exec_module(step04_mod)

        for col in step04_mod.MORPHOGEN_COLUMNS:
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
                         "Notch", "EGF", "unknown"}
        for col, pathway in step06.MORPHOGEN_PATHWAYS.items():
            assert pathway in valid_pathways, (
                f"Invalid pathway '{pathway}' for {col}"
            )


class TestEncodeProtocol:
    """Tests for CellFlow protocol encoding."""

    def test_basic_encoding(self):
        vec = {"CHIR99021_uM": 1.5, "SAG_nM": 250.0, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert "modulators" in enc
        assert len(enc["modulators"]) == 2  # CHIR + SAG
        assert enc["harvest_day"] == 21

    def test_zero_morphogens_excluded(self):
        vec = {"CHIR99021_uM": 0.0, "SAG_nM": 250.0, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert len(enc["modulators"]) == 1  # only SAG

    def test_smiles_included_for_small_molecules(self):
        vec = {"CHIR99021_uM": 1.5, "log_harvest_day": math.log(21)}
        enc = step06.encode_protocol_cellflow(vec)
        assert "smiles" in enc["modulators"][0]

    def test_no_smiles_for_proteins(self):
        vec = {"BMP4_ng_mL": 10.0, "log_harvest_day": math.log(21)}
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
            "SAG_nM": [0.0, 250.0],
        }
        grid = step06.generate_virtual_screen_grid(ranges, harvest_days=[21])
        # 3 * 2 = 6 combinations (all others are [0.0])
        assert len(grid) == 6
        assert "CHIR99021_uM" in grid.columns
        assert "SAG_nM" in grid.columns

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
            "SAG_nM": [0.0, 50, 100, 250, 500, 1000, 1500, 2000],
            "BMP4_ng_mL": [0.0, 5, 10, 25, 50],
            "RA_nM": [0.0, 10, 50, 100, 500, 1000],
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
            "SAG_nM": [0.0, 250.0, 1000.0],
            "BMP4_ng_mL": [0.0, 10.0, 0.0],
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
            "SAG_nM": [0.0, 0.0],
            "BMP4_ng_mL": [50.0, 0.0],
            "LDN193189_nM": [0.0, 200.0],
            "SB431542_uM": [0.0, 10.0],
            "log_harvest_day": [math.log(21)] * 2,
        }, index=["bmp_high", "neural_induction"])

        result = step06._predict_baseline(protocols)
        # BMP should produce more CP; neural induction should produce more neuroepithelium
        assert result.loc["bmp_high", "CP"] > result.loc["neural_induction", "CP"]
        assert (result.loc["neural_induction", "Neuroepithelium"]
                > result.loc["bmp_high", "Neuroepithelium"])


class TestPredictionConfidence:
    """Tests for CellFlow prediction confidence estimation."""

    def test_nearby_predictions_high_confidence(self):
        training = pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0, 0.0],
            "SAG_nM": [0.0, 250.0, 500.0],
        }, index=["t1", "t2", "t3"])

        # Near training point
        predictions = pd.DataFrame({
            "CHIR99021_uM": [1.6],
            "SAG_nM": [10.0],
        }, index=["near"])

        confidence = step06.compute_prediction_confidence(predictions, training)
        assert confidence.iloc[0] > 0.5

    def test_far_predictions_low_confidence(self):
        training = pd.DataFrame({
            "CHIR99021_uM": [1.5, 3.0],
            "SAG_nM": [0.0, 250.0],
        }, index=["t1", "t2"])

        # Far from training data
        predictions = pd.DataFrame({
            "CHIR99021_uM": [100.0],
            "SAG_nM": [10000.0],
        }, index=["far"])

        confidence = step06.compute_prediction_confidence(predictions, training)
        assert confidence.iloc[0] < 0.5

    def test_confidence_in_range(self):
        training = pd.DataFrame({
            "CHIR99021_uM": np.random.rand(20) * 10,
            "SAG_nM": np.random.rand(20) * 1000,
        })
        predictions = pd.DataFrame({
            "CHIR99021_uM": np.random.rand(10) * 10,
            "SAG_nM": np.random.rand(10) * 1000,
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
            "SAG_nM": np.random.uniform(0, 1000, n_real),
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
            "SAG_nM": np.random.uniform(0, 1000, n_cr2),
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
            "SAG_nM": np.random.uniform(0, 1000, n_cf),
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
                "SAG_nM": [np.random.uniform(0, 2000)],
                "BMP4_ng_mL": [np.random.uniform(0, 50)],
                "SHH_ng_mL": [np.random.uniform(0, 500)],
                "RA_nM": [np.random.uniform(0, 1000)],
                "FGF8_ng_mL": [np.random.uniform(0, 200)],
                "IWP2_uM": [np.random.uniform(0, 10)],
                "LDN193189_nM": [np.random.uniform(0, 500)],
                "SB431542_uM": [np.random.uniform(0, 20)],
                "DAPT_uM": [np.random.uniform(0, 10)],
                "EGF_ng_mL": [np.random.uniform(0, 50)],
                "log_harvest_day": [math.log(np.random.uniform(7, 120))],
            }, index=["test"])

            result = step06._predict_baseline(protocols)
            assert (result >= 0).all().all(), "Negative fraction"
            row_sum = result.sum(axis=1).iloc[0]
            assert abs(row_sum - 1.0) < 1e-6, f"Row sum {row_sum} != 1.0"

    def test_protocol_encoding_deterministic(self):
        """Same input should always produce same encoding."""
        vec = {"CHIR99021_uM": 1.5, "SAG_nM": 250.0, "log_harvest_day": math.log(21)}
        enc1 = step06.encode_protocol_cellflow(vec)
        enc2 = step06.encode_protocol_cellflow(vec)
        assert enc1 == enc2

    def test_confidence_monotonic_with_distance(self):
        """Confidence should decrease with distance from training data."""
        training = pd.DataFrame({
            "CHIR99021_uM": [1.5],
            "SAG_nM": [250.0],
        }, index=["t1"])

        # Near and far predictions
        predictions = pd.DataFrame({
            "CHIR99021_uM": [1.6, 100.0],
            "SAG_nM": [260.0, 50000.0],
        }, index=["near", "far"])

        confidence = step06.compute_prediction_confidence(predictions, training)
        assert confidence.loc["near"] > confidence.loc["far"]
