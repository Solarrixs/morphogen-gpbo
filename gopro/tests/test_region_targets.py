"""Tests for region targeting system (gopro/region_targets.py)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import anndata as ad
import scipy.sparse as sp

from gopro.region_targets import (
    NAMED_REGION_PROFILES,
    OFF_TARGET_LEVEL1,
    HNOCA_TO_BRAUN_REGION,
    _REQUIRED_PROFILE_KEYS,
    _STATIC_HNOCA_TO_BRAUN,
    _build_dynamic_label_map,
    build_hnoca_to_braun_label_map,
    list_named_profiles,
    build_custom_target,
    discover_available_regions,
    load_region_profile,
    load_target_profile_csv,
)


class TestNamedProfiles:
    """Tests for named region profile metadata."""

    def test_named_profiles_all_valid(self):
        """All named profiles must have all required keys."""
        assert len(NAMED_REGION_PROFILES) == 9, "Expected 9 HNOCA regions"
        for name, info in NAMED_REGION_PROFILES.items():
            missing = _REQUIRED_PROFILE_KEYS - set(info.keys())
            assert not missing, f"Profile '{name}' missing keys: {missing}"

    def test_named_profiles_hnoca_regions_match(self):
        """hnoca_region values should match HNOCA_TO_BRAUN_REGION keys."""
        for name, info in NAMED_REGION_PROFILES.items():
            assert info["hnoca_region"] in HNOCA_TO_BRAUN_REGION, (
                f"Profile '{name}' has hnoca_region='{info['hnoca_region']}' "
                f"which is not in HNOCA_TO_BRAUN_REGION"
            )

    def test_named_profiles_all_have_descriptions(self):
        """All profiles should have non-empty descriptions."""
        for name, info in NAMED_REGION_PROFILES.items():
            assert len(info["description"]) > 10, (
                f"Profile '{name}' has short/empty description"
            )

    def test_list_named_profiles_returns_df(self):
        """list_named_profiles returns a DataFrame with expected columns."""
        df = list_named_profiles()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 9
        assert "name" in df.columns
        assert "display_name" in df.columns
        assert "description" in df.columns
        assert "source" in df.columns

    def test_list_named_profiles_names_match(self):
        """All profile keys should appear in the DataFrame."""
        df = list_named_profiles()
        assert set(df["name"]) == set(NAMED_REGION_PROFILES.keys())


class TestBuildCustomTarget:
    """Tests for build_custom_target function."""

    def test_build_custom_target_sums_to_one(self):
        """Custom target output should sum to exactly 1.0."""
        target = build_custom_target({"Neuron": 0.5, "NPC": 0.3, "Glia": 0.2})
        assert abs(target.sum() - 1.0) < 1e-10

    def test_build_custom_target_preserves_ratios(self):
        """Relative proportions should be preserved."""
        target = build_custom_target({"A": 0.6, "B": 0.3, "C": 0.1})
        assert target["A"] > target["B"] > target["C"]

    def test_build_custom_target_normalizes_near_one(self):
        """Fractions summing to ~1.0 (within tolerance) should be accepted."""
        target = build_custom_target({"A": 0.51, "B": 0.51})
        # 0.51+0.51=1.02, within 0.05 tolerance
        assert abs(target.sum() - 1.0) < 1e-10

    def test_build_custom_target_rejects_bad_sum(self):
        """Fractions summing too far from 1.0 should be rejected."""
        with pytest.raises(ValueError, match="sum to"):
            build_custom_target({"A": 0.2, "B": 0.1})

    def test_build_custom_target_rejects_negative(self):
        """Negative fractions should be rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            build_custom_target({"A": 0.8, "B": -0.2})

    def test_build_custom_target_rejects_empty(self):
        """Empty dict should be rejected."""
        with pytest.raises(ValueError, match="empty"):
            build_custom_target({})

    def test_build_custom_target_exact_one(self):
        """Fractions summing to exactly 1.0 should work."""
        target = build_custom_target({"X": 1.0})
        assert target["X"] == pytest.approx(1.0)


class TestDiscoverRegions:
    """Tests for discover_available_regions using mock h5ad."""

    @pytest.fixture
    def mock_h5ad(self, tmp_path):
        """Create a minimal mock h5ad with region and cell type columns."""
        n_cells = 100
        obs = pd.DataFrame({
            "annot_region_rev2": (
                ["Dorsal telencephalon"] * 40
                + ["Ventral telencephalon"] * 30
                + ["Cerebellum"] * 20
                + ["Unspecific"] * 10
            ),
            "annot_level_3_rev2": (
                ["Cortical Neuron"] * 20
                + ["Radial Glia"] * 20
                + ["GABAergic IN"] * 30
                + ["Cerebellar NPC"] * 15
                + ["Purkinje"] * 5
                + ["Unknown"] * 10
            ),
        })
        adata = ad.AnnData(
            X=sp.csr_matrix((n_cells, 10)),
            obs=obs,
        )
        path = tmp_path / "test_atlas.h5ad"
        adata.write(str(path))
        return path

    def test_discover_regions_returns_dict(self, mock_h5ad):
        """discover_available_regions should return a dict of region metadata."""
        regions = discover_available_regions(mock_h5ad)
        assert isinstance(regions, dict)
        assert len(regions) == 3  # Unspecific should be excluded
        assert "Dorsal telencephalon" in regions
        assert "Ventral telencephalon" in regions
        assert "Cerebellum" in regions

    def test_discover_regions_excludes_unspecific(self, mock_h5ad):
        """Unspecific region should be excluded."""
        regions = discover_available_regions(mock_h5ad)
        assert "Unspecific" not in regions

    def test_discover_regions_has_metadata(self, mock_h5ad):
        """Each region should have n_cells, top_cell_types, annotation_level."""
        regions = discover_available_regions(mock_h5ad)
        for name, meta in regions.items():
            assert "n_cells" in meta
            assert "top_cell_types" in meta
            assert "annotation_level" in meta
            assert meta["n_cells"] > 0
            assert isinstance(meta["top_cell_types"], list)

    def test_discover_regions_cell_counts(self, mock_h5ad):
        """Cell counts should match the mock data."""
        regions = discover_available_regions(mock_h5ad)
        assert regions["Dorsal telencephalon"]["n_cells"] == 40
        assert regions["Ventral telencephalon"]["n_cells"] == 30
        assert regions["Cerebellum"]["n_cells"] == 20

    def test_discover_regions_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            discover_available_regions(tmp_path / "nonexistent.h5ad")

    def test_discover_regions_no_region_column(self, tmp_path):
        """Should raise ValueError if no region column exists."""
        obs = pd.DataFrame({"cell_type": ["Neuron"] * 10})
        adata = ad.AnnData(X=sp.csr_matrix((10, 5)), obs=obs)
        path = tmp_path / "no_region.h5ad"
        adata.write(str(path))
        with pytest.raises(ValueError, match="No region column"):
            discover_available_regions(path)


class TestLoadRegionProfile:
    """Tests for load_region_profile using mock h5ad."""

    @pytest.fixture
    def mock_h5ad(self, tmp_path):
        """Create a minimal mock h5ad with region and cell type columns."""
        n_cells = 100
        obs = pd.DataFrame({
            "annot_region_rev2": (
                ["Dorsal telencephalon"] * 60
                + ["Cerebellum"] * 40
            ),
            "annot_level_3_rev2": (
                ["Cortical Neuron"] * 30
                + ["Radial Glia"] * 30
                + ["Cerebellar NPC"] * 25
                + ["Purkinje"] * 15
            ),
        })
        adata = ad.AnnData(
            X=sp.csr_matrix((n_cells, 10)),
            obs=obs,
        )
        path = tmp_path / "test_atlas.h5ad"
        adata.write(str(path))
        return path

    def test_load_region_profile_returns_series(self, mock_h5ad):
        """Should return a pandas Series."""
        profile = load_region_profile("Dorsal telencephalon", mock_h5ad)
        assert isinstance(profile, pd.Series)

    def test_load_region_profile_sums_to_one(self, mock_h5ad):
        """Profile fractions should sum to 1.0."""
        profile = load_region_profile("Dorsal telencephalon", mock_h5ad)
        assert abs(profile.sum() - 1.0) < 1e-10

    def test_load_region_profile_correct_types(self, mock_h5ad):
        """Profile should contain the expected cell types for Dorsal telencephalon."""
        profile = load_region_profile("Dorsal telencephalon", mock_h5ad)
        assert "Cortical Neuron" in profile.index
        assert "Radial Glia" in profile.index
        # Each has 30 cells out of 60 = 0.5
        assert profile["Cortical Neuron"] == pytest.approx(0.5)
        assert profile["Radial Glia"] == pytest.approx(0.5)

    def test_load_region_profile_snake_case_key(self, mock_h5ad):
        """Should resolve snake_case profile keys to HNOCA region names."""
        profile = load_region_profile("dorsal_telencephalon", mock_h5ad)
        assert isinstance(profile, pd.Series)
        assert abs(profile.sum() - 1.0) < 1e-10

    def test_load_region_profile_unknown_raises(self, mock_h5ad):
        """Unknown region should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            load_region_profile("Nonexistent_Region", mock_h5ad)

    def test_load_region_profile_file_not_found(self, tmp_path):
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_region_profile("Dorsal telencephalon", tmp_path / "missing.h5ad")


class TestLoadTargetProfileCSV:
    """Tests for load_target_profile_csv function."""

    def test_two_column_format(self, tmp_path):
        """Should load a two-column CSV (cell_type, fraction)."""
        path = tmp_path / "target.csv"
        df = pd.DataFrame({"cell_type": ["A", "B"], "fraction": [0.6, 0.4]})
        df.to_csv(str(path), index=False)
        profile = load_target_profile_csv(path)
        assert abs(profile.sum() - 1.0) < 1e-10
        assert profile["A"] == pytest.approx(0.6)

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_target_profile_csv(tmp_path / "missing.csv")


class TestScoreWithTargetProfile:
    """Tests for score_all_conditions with target_profile parameter."""

    def test_score_with_target_profile(self):
        """When target_profile is provided, RSS should score against it."""
        import scanpy as sc

        # Create mock AnnData
        n_cells = 60
        obs = pd.DataFrame({
            "condition": ["cond_A"] * 30 + ["cond_B"] * 30,
            "predicted_annot_level_1": ["Neuron"] * 40 + ["NPC"] * 20,
            "predicted_annot_level_2": (
                ["ExN"] * 20 + ["InN"] * 10 + ["NPC"] * 15 + ["IP"] * 15
            ),
            "predicted_annot_level_3_rev2": (
                ["Cortical ExN"] * 20 + ["GABAergic InN"] * 10
                + ["Cerebellar NPC"] * 15 + ["IP early"] * 15
            ),
            "predicted_annot_region_rev2": (
                ["Dorsal telencephalon"] * 25 + ["Ventral telencephalon"] * 5
                + ["Cerebellum"] * 20 + ["Pons"] * 10
            ),
        })
        adata = sc.AnnData(
            X=sp.csr_matrix((n_cells, 5)),
            obs=obs,
        )

        # Create target profile matching cond_A's composition
        target = pd.Series({"ExN": 0.67, "InN": 0.33})

        # Fake braun profiles (used as fallback)
        braun = pd.DataFrame(
            {"Neuron": [0.5], "Radial glia": [0.5]},
            index=["Dorsal telencephalon"],
        )

        from conftest import _import_pipeline_module
        step03 = _import_pipeline_module("03_fidelity_scoring")

        report = step03.score_all_conditions(
            query_adata=adata,
            braun_profiles=braun,
            condition_key="condition",
            target_profile=target,
        )

        assert isinstance(report, pd.DataFrame)
        assert "composite_fidelity" in report.columns
        assert "rss_score" in report.columns
        # cond_A has ExN and InN which match target; cond_B has NPC and IP which don't
        assert report.loc["cond_A", "rss_score"] > report.loc["cond_B", "rss_score"]


class TestRefactoredImports:
    """Test that refactored imports from region_targets still work in step 03."""

    def test_off_target_level1_accessible(self):
        """OFF_TARGET_LEVEL1 should be importable from both modules."""
        from conftest import _import_pipeline_module
        step03 = _import_pipeline_module("03_fidelity_scoring")
        from gopro.region_targets import OFF_TARGET_LEVEL1 as rt_off_target

        assert step03.OFF_TARGET_LEVEL1 is rt_off_target

    def test_hnoca_to_braun_region_accessible(self):
        """HNOCA_TO_BRAUN_REGION should be importable from both modules."""
        from conftest import _import_pipeline_module
        step03 = _import_pipeline_module("03_fidelity_scoring")
        from gopro.region_targets import HNOCA_TO_BRAUN_REGION as rt_mapping

        assert step03.HNOCA_TO_BRAUN_REGION is rt_mapping

    def test_build_hnoca_to_braun_label_map_accessible(self):
        """build_hnoca_to_braun_label_map should be importable from both modules."""
        from conftest import _import_pipeline_module
        step03 = _import_pipeline_module("03_fidelity_scoring")
        from gopro.region_targets import build_hnoca_to_braun_label_map as rt_func

        assert step03.build_hnoca_to_braun_label_map is rt_func
        # Verify it still returns the correct mapping
        label_map = step03.build_hnoca_to_braun_label_map()
        assert isinstance(label_map, dict)
        assert "Neuron" in label_map
        assert label_map["NPC"] == "Radial glia"


class TestDynamicLabelMap:
    """Tests for dynamic HNOCA→Braun label map discovery."""

    @pytest.fixture
    def mock_hnoca_obs(self):
        """HNOCA obs with region and level-1 labels."""
        return pd.DataFrame({
            "annot_level_1": (
                ["Neuron"] * 40
                + ["NPC"] * 30
                + ["Glioblast"] * 20
                + ["OPC"] * 10
            ),
            "annot_region_rev2": (
                # Neurons cluster in Dorsal telencephalon
                ["Dorsal telencephalon"] * 35 + ["Ventral telencephalon"] * 5
                # NPCs spread across regions
                + ["Dorsal telencephalon"] * 15 + ["Cerebellum"] * 15
                # Glioblasts in Ventral telencephalon
                + ["Ventral telencephalon"] * 18 + ["Dorsal telencephalon"] * 2
                # OPCs in Cerebellum
                + ["Cerebellum"] * 10
            ),
        })

    @pytest.fixture
    def mock_braun_obs(self):
        """Braun obs with region and CellClass labels."""
        return pd.DataFrame({
            "CellClass": (
                ["Neuron"] * 40
                + ["Radial glia"] * 30
                + ["Glioblast"] * 20
                + ["Oligo"] * 10
            ),
            "SummarizedRegion": (
                # Neurons cluster in Dorsal telencephalon (same as HNOCA)
                ["Dorsal telencephalon"] * 35 + ["Ventral telencephalon"] * 5
                # Radial glia spread like NPC
                + ["Dorsal telencephalon"] * 15 + ["Cerebellum"] * 15
                # Glioblasts in Ventral telencephalon (same as HNOCA)
                + ["Ventral telencephalon"] * 18 + ["Dorsal telencephalon"] * 2
                # Oligo in Cerebellum (same as OPC)
                + ["Cerebellum"] * 10
            ),
        })

    def test_dynamic_map_matches_expected(self, mock_hnoca_obs, mock_braun_obs):
        """Dynamic map should find correct mappings from region co-occurrence."""
        dynamic = _build_dynamic_label_map(mock_hnoca_obs, mock_braun_obs)
        assert dynamic["Neuron"] == "Neuron"
        assert dynamic["NPC"] == "Radial glia"
        assert dynamic["Glioblast"] == "Glioblast"
        assert dynamic["OPC"] == "Oligo"

    def test_dynamic_map_returns_dict(self, mock_hnoca_obs, mock_braun_obs):
        """Should return a dict."""
        dynamic = _build_dynamic_label_map(mock_hnoca_obs, mock_braun_obs)
        assert isinstance(dynamic, dict)
        assert len(dynamic) > 0

    def test_dynamic_map_missing_region_col(self, mock_braun_obs):
        """Should raise ValueError when HNOCA obs lacks region column."""
        bad_hnoca = pd.DataFrame({"annot_level_1": ["Neuron"] * 10})
        with pytest.raises(ValueError, match="No region column"):
            _build_dynamic_label_map(bad_hnoca, mock_braun_obs)

    def test_dynamic_map_missing_braun_region(self, mock_hnoca_obs):
        """Should raise ValueError when Braun obs lacks SummarizedRegion."""
        bad_braun = pd.DataFrame({"CellClass": ["Neuron"] * 10})
        with pytest.raises(ValueError, match="SummarizedRegion"):
            _build_dynamic_label_map(mock_hnoca_obs, bad_braun)

    def test_build_label_map_no_args_returns_static(self):
        """Calling with no args should return the static curated map."""
        label_map = build_hnoca_to_braun_label_map()
        assert label_map == _STATIC_HNOCA_TO_BRAUN

    def test_build_label_map_with_obs_uses_dynamic(
        self, mock_hnoca_obs, mock_braun_obs,
    ):
        """When obs data is provided, dynamic mappings should be applied."""
        label_map = build_hnoca_to_braun_label_map(
            hnoca_obs=mock_hnoca_obs, braun_obs=mock_braun_obs,
        )
        # Dynamic should have discovered Neuron→Neuron, NPC→Radial glia, etc.
        assert label_map["Neuron"] == "Neuron"
        assert label_map["NPC"] == "Radial glia"
        # Static fallback labels not in mock data should still be present
        assert "PSC" in label_map
        assert "MC" in label_map

    def test_build_label_map_overrides_win(
        self, mock_hnoca_obs, mock_braun_obs,
    ):
        """User overrides should take highest precedence."""
        label_map = build_hnoca_to_braun_label_map(
            hnoca_obs=mock_hnoca_obs,
            braun_obs=mock_braun_obs,
            overrides={"Neuron": "CustomClass", "NewLabel": "CustomTarget"},
        )
        assert label_map["Neuron"] == "CustomClass"
        assert label_map["NewLabel"] == "CustomTarget"

    def test_build_label_map_dynamic_failure_falls_back(self):
        """If dynamic discovery fails, static map should be returned."""
        # Pass obs that will trigger ValueError (missing region col)
        bad_hnoca = pd.DataFrame({"annot_level_1": ["Neuron"] * 10})
        bad_braun = pd.DataFrame({"CellClass": ["Neuron"] * 10})
        label_map = build_hnoca_to_braun_label_map(
            hnoca_obs=bad_hnoca, braun_obs=bad_braun,
        )
        # Should fall back to static map
        assert label_map == _STATIC_HNOCA_TO_BRAUN

    def test_build_label_map_only_hnoca_obs_uses_static(self):
        """If only one obs is provided, should use static map."""
        hnoca = pd.DataFrame({
            "annot_level_1": ["Neuron"] * 10,
            "annot_region_rev2": ["Dorsal telencephalon"] * 10,
        })
        label_map = build_hnoca_to_braun_label_map(hnoca_obs=hnoca)
        assert label_map == _STATIC_HNOCA_TO_BRAUN


class TestFBaxisRank:
    """Tests for FBaxis_rank continuous A-P regionalization."""

    def test_ap_positions_cover_all_named_regions(self):
        """All 9 HNOCA regions should have A-P positions."""
        from gopro.region_targets import BRAIN_REGION_AP_POSITIONS, HNOCA_TO_BRAUN_REGION
        for region in HNOCA_TO_BRAUN_REGION:
            assert region in BRAIN_REGION_AP_POSITIONS, f"Missing A-P position for {region}"

    def test_ap_positions_monotonic_ordering(self):
        """Forebrain < midbrain < hindbrain on the A-P axis."""
        from gopro.region_targets import BRAIN_REGION_AP_POSITIONS
        assert BRAIN_REGION_AP_POSITIONS["Dorsal telencephalon"] < BRAIN_REGION_AP_POSITIONS["Dorsal midbrain"]
        assert BRAIN_REGION_AP_POSITIONS["Dorsal midbrain"] < BRAIN_REGION_AP_POSITIONS["Cerebellum"]
        assert BRAIN_REGION_AP_POSITIONS["Cerebellum"] < BRAIN_REGION_AP_POSITIONS["Medulla"]

    def test_compute_fbaxis_rank_from_region_fractions(self):
        """Weighted A-P score from region fraction vectors."""
        from gopro.region_targets import compute_fbaxis_rank, BRAIN_REGION_AP_POSITIONS
        # Condition purely in Dorsal telencephalon → score = 0.0
        region_fracs = pd.DataFrame({
            "Dorsal telencephalon": [1.0, 0.0],
            "Medulla": [0.0, 1.0],
        }, index=["forebrain_cond", "hindbrain_cond"])
        scores = compute_fbaxis_rank(pd.DataFrame(), region_fractions=region_fracs)
        assert scores["forebrain_cond"] == pytest.approx(0.0)
        assert scores["hindbrain_cond"] == pytest.approx(1.0)

    def test_compute_fbaxis_rank_mixed_regions(self):
        """Mixed region fractions give intermediate A-P score."""
        from gopro.region_targets import compute_fbaxis_rank
        region_fracs = pd.DataFrame({
            "Dorsal telencephalon": [0.5],
            "Medulla": [0.5],
        }, index=["mixed"])
        scores = compute_fbaxis_rank(pd.DataFrame(), region_fractions=region_fracs)
        assert 0.0 < scores["mixed"] < 1.0
        assert scores["mixed"] == pytest.approx(0.5)  # (0.0 + 1.0) / 2

    def test_compute_fbaxis_rank_dominant_region_fallback(self):
        """Falls back to dominant_region column when region_fractions not given."""
        from gopro.region_targets import compute_fbaxis_rank, BRAIN_REGION_AP_POSITIONS
        df = pd.DataFrame({
            "dominant_region": ["Dorsal telencephalon", "Medulla", "Cerebellum"],
        }, index=["c1", "c2", "c3"])
        scores = compute_fbaxis_rank(df)
        assert scores["c1"] == pytest.approx(BRAIN_REGION_AP_POSITIONS["Dorsal telencephalon"])
        assert scores["c2"] == pytest.approx(BRAIN_REGION_AP_POSITIONS["Medulla"])
        assert scores["c3"] == pytest.approx(BRAIN_REGION_AP_POSITIONS["Cerebellum"])

    def test_compute_fbaxis_rank_unmapped_region(self):
        """Unmapped regions get 0.5 (midpoint) with a warning."""
        from gopro.region_targets import compute_fbaxis_rank
        df = pd.DataFrame({
            "dominant_region": ["UnknownRegion"],
        }, index=["c1"])
        scores = compute_fbaxis_rank(df)
        assert scores["c1"] == pytest.approx(0.5)

    def test_compute_fbaxis_rank_no_region_col_raises(self):
        """Should raise ValueError when no region info is available."""
        from gopro.region_targets import compute_fbaxis_rank
        df = pd.DataFrame({"cell_type_A": [0.5]}, index=["c1"])
        with pytest.raises(ValueError, match="dominant_region"):
            compute_fbaxis_rank(df)

    def test_build_ap_target_profile_sums_to_one(self):
        """Target profile from A-P position should sum to 1."""
        from gopro.region_targets import build_ap_target_profile
        for fbaxis in [0.0, 0.3, 0.5, 0.7, 1.0]:
            profile = build_ap_target_profile(fbaxis)
            assert profile.sum() == pytest.approx(1.0)

    def test_build_ap_target_profile_peaks_at_target(self):
        """Profile should peak at the region closest to target_fbaxis."""
        from gopro.region_targets import build_ap_target_profile, BRAIN_REGION_AP_POSITIONS
        # Target 0.0 → should peak at Dorsal telencephalon
        profile = build_ap_target_profile(0.0)
        assert profile.idxmax() == "Dorsal telencephalon"
        # Target 1.0 → should peak at Medulla
        profile = build_ap_target_profile(1.0)
        assert profile.idxmax() == "Medulla"

    def test_build_ap_target_profile_out_of_range_raises(self):
        """Target outside [0, 1] should raise ValueError."""
        from gopro.region_targets import build_ap_target_profile
        with pytest.raises(ValueError, match="must be in"):
            build_ap_target_profile(-0.1)
        with pytest.raises(ValueError, match="must be in"):
            build_ap_target_profile(1.5)

    def test_build_ap_target_profile_invalid_width_raises(self):
        """Zero or negative width should raise ValueError."""
        from gopro.region_targets import build_ap_target_profile
        with pytest.raises(ValueError, match="width must be positive"):
            build_ap_target_profile(0.5, width=0)
        with pytest.raises(ValueError, match="width must be positive"):
            build_ap_target_profile(0.5, width=-0.1)
