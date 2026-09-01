"""Smoke tests for ngb_refactored_horizon_specific.py.

These tests build small synthetic shapefiles that mimic the expected
workspace layout (one thermal + one reservoir shapefile per horizon) and run
the full horizon-specific workflow end-to-end, verifying that:

- Every configured horizon produces a valid (0-1) favorability GeoTIFF.
- The aggregated best-horizon, weighted-mean, PRIMARY, and SECONDARY maps
  are produced with the expected shape/CRS.
- The horizon id mapping CSV and summary CSV are written and consistent.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import Point

import ngb_refactored_horizon_specific as workflow


N_POINTS = 60
BOUNDS = (3500000.0, 5800000.0, 3510000.0, 5810000.0)  # EPSG:31467-like extent


def _make_random_gdf(rng, data_col, crs):
    xs = rng.uniform(BOUNDS[0], BOUNDS[2], N_POINTS)
    ys = rng.uniform(BOUNDS[1], BOUNDS[3], N_POINTS)
    values = rng.uniform(0.0, 1.0, N_POINTS)
    return gpd.GeoDataFrame(
        {data_col: values},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )


@pytest.fixture()
def synthetic_workspace(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)

    thermal_dir = tmp_path / "workspace" / "geothermal" / "thermal_component"
    reservoir_dir = tmp_path / "workspace" / "geothermal" / "geologic_component"
    thermal_dir.mkdir(parents=True)
    reservoir_dir.mkdir(parents=True)

    for horizon_name, cfg in workflow.HORIZONS.items():
        thermal_gdf = _make_random_gdf(rng, cfg["thermal_data_col"], workflow.TARGET_CRS)
        reservoir_gdf = _make_random_gdf(rng, cfg["reservoir_data_col"], workflow.TARGET_CRS)

        thermal_gdf.to_file(thermal_dir / f"{horizon_name}_thermal.shp")
        reservoir_gdf.to_file(reservoir_dir / f"{horizon_name}_reservoir.shp")

    output_dir = tmp_path / "output"
    horizon_output_dir = output_dir / "favourability" / "geothermal" / "horizon_specific"

    monkeypatch.setattr(workflow, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(workflow, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(workflow, "THERMAL_DIR", str(thermal_dir))
    monkeypatch.setattr(workflow, "RESERVOIR_DIR", str(reservoir_dir))
    monkeypatch.setattr(workflow, "HORIZON_OUTPUT_DIR", str(horizon_output_dir))
    # Keep the grid small for a fast smoke test.
    monkeypatch.setattr(workflow, "NX", 24)
    monkeypatch.setattr(workflow, "NY", 20)

    return horizon_output_dir


def test_run_horizon_specific_workflow_end_to_end(synthetic_workspace):
    result = workflow.run_horizon_specific_workflow()

    horizon_output_dir = str(synthetic_workspace)

    # One GeoTIFF per configured horizon, each valued in [0, 1].
    assert set(result["horizon_scores"].keys()) == set(workflow.HORIZONS.keys())
    for horizon_name, grid in result["horizon_scores"].items():
        assert grid.shape == (workflow.NY, workflow.NX)
        assert np.nanmin(grid) >= 0.0
        assert np.nanmax(grid) <= 1.0

        tif_path = os.path.join(
            horizon_output_dir, f"{horizon_name}_geothermal_favourability.tif"
        )
        assert os.path.exists(tif_path)
        with rasterio.open(tif_path) as src:
            assert src.width == workflow.NX
            assert src.height == workflow.NY
            assert src.crs.to_string().upper().endswith("31467")

    # Aggregated outputs.
    for fname in (
        "best_horizon_geothermal_favourability.tif",
        "weighted_mean_geothermal_favourability.tif",
        "PRIMARY_geothermal_favourability.tif",
        "SECONDARY_geothermal_favourability.tif",
        "best_horizon_id.tif",
    ):
        assert os.path.exists(os.path.join(horizon_output_dir, fname))

    # SECONDARY should include at least as many favorable cells as PRIMARY.
    primary_valid = np.isfinite(result["primary_map"]).sum()
    secondary_valid = np.isfinite(result["secondary_map"]).sum()
    assert secondary_valid >= primary_valid

    # Horizon id mapping matches the number of configured horizons.
    mapping_path = os.path.join(horizon_output_dir, "best_horizon_id_mapping.csv")
    mapping_df = pd.read_csv(mapping_path)
    assert len(mapping_df) == len(workflow.HORIZONS)
    assert set(mapping_df["horizon_name"]) == set(workflow.HORIZONS.keys())

    # best_horizon_id values must be valid indices into the mapping.
    assert result["best_horizon_idx"].min() >= 0
    assert result["best_horizon_idx"].max() < len(workflow.HORIZONS)

    # Summary CSV has one row per horizon plus the four aggregate rows.
    summary_path = os.path.join(horizon_output_dir, "summary.csv")
    summary_df = pd.read_csv(summary_path)
    expected_rows = len(workflow.HORIZONS) + 4  # best_horizon, weighted_mean, PRIMARY, SECONDARY
    assert len(summary_df) == expected_rows


def test_missing_layers_are_skipped_not_fatal(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)

    thermal_dir = tmp_path / "workspace" / "geothermal" / "thermal_component"
    reservoir_dir = tmp_path / "workspace" / "geothermal" / "geologic_component"
    thermal_dir.mkdir(parents=True)
    reservoir_dir.mkdir(parents=True)

    # Only provide layers for a single horizon.
    horizon_name = "detfurth"
    cfg = workflow.HORIZONS[horizon_name]
    thermal_gdf = _make_random_gdf(rng, cfg["thermal_data_col"], workflow.TARGET_CRS)
    reservoir_gdf = _make_random_gdf(rng, cfg["reservoir_data_col"], workflow.TARGET_CRS)
    thermal_gdf.to_file(thermal_dir / f"{horizon_name}_thermal.shp")
    reservoir_gdf.to_file(reservoir_dir / f"{horizon_name}_reservoir.shp")

    output_dir = tmp_path / "output"
    horizon_output_dir = output_dir / "favourability" / "geothermal" / "horizon_specific"

    monkeypatch.setattr(workflow, "WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(workflow, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(workflow, "THERMAL_DIR", str(thermal_dir))
    monkeypatch.setattr(workflow, "RESERVOIR_DIR", str(reservoir_dir))
    monkeypatch.setattr(workflow, "HORIZON_OUTPUT_DIR", str(horizon_output_dir))
    monkeypatch.setattr(workflow, "NX", 16)
    monkeypatch.setattr(workflow, "NY", 12)

    result = workflow.run_horizon_specific_workflow()

    assert list(result["horizon_scores"].keys()) == [horizon_name]


def test_find_layer_file_missing_directory_returns_none(tmp_path):
    assert workflow.find_layer_file(str(tmp_path / "does_not_exist"), "detfurth") is None


def test_normalize_and_clean_handles_nan_and_out_of_range_values():
    grid = np.array([[0.2, np.nan], [1.5, -3.0]])
    cleaned = workflow.normalize_and_clean(grid, assume_0_1=False)
    assert np.all(np.isfinite(cleaned))
    assert cleaned.min() >= 0.0
    assert cleaned.max() <= 1.0
