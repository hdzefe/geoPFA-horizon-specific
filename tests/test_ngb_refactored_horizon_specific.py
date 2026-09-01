"""Tests for ngb_refactored_horizon_specific.py using synthetic shapefiles."""

import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ngb_refactored_horizon_specific as ngb

CRS = "EPSG:31467"


def _random_points_gdf(n, xmin, ymin, xmax, ymax, col, seed):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(xmin, xmax, n)
    ys = rng.uniform(ymin, ymax, n)
    values = rng.uniform(0, 1, n)
    gdf = gpd.GeoDataFrame(
        {col: values}, geometry=[Point(x, y) for x, y in zip(xs, ys)], crs=CRS
    )
    return gdf


def _make_workspace(tmp_path, xmin=0, ymin=0, xmax=1000, ymax=1000):
    workspace_dir = tmp_path / "workspace"
    thermal_dir = workspace_dir / "geothermal" / "thermal_component"
    geologic_dir = workspace_dir / "geothermal" / "geologic_component"
    thermal_dir.mkdir(parents=True)
    geologic_dir.mkdir(parents=True)

    # Basin outline (polygon)
    basin_poly = Polygon(
        [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    )
    basin_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[basin_poly], crs=CRS)
    basin_gdf.to_file(geologic_dir / "north_german_basin_.shp")

    # Salt penalty layer (already 0-1 scaled distance penalty)
    salt_gdf = _random_points_gdf(20, xmin, ymin, xmax, ymax, "value", seed=1)
    salt_gdf.to_file(geologic_dir / "salt_sweetspot_0p5_2km.shp")

    # Horizon-specific layers for "detfurth" (uses reservoir override)
    detfurth_thermal = _random_points_gdf(
        20, xmin, ymin, xmax, ymax, "value", seed=2
    )
    detfurth_thermal.to_file(thermal_dir / "detfurth_thermal_use.shp")

    detfurth_reservoir = _random_points_gdf(
        20, xmin, ymin, xmax, ymax, "value", seed=3
    )
    detfurth_reservoir.to_file(geologic_dir / "sand_zones_detfurth_.shp")

    # Horizon-specific layers for "k42" (normal naming)
    k42_thermal = _random_points_gdf(20, xmin, ymin, xmax, ymax, "value", seed=4)
    k42_thermal.to_file(thermal_dir / "k42_thermal_use.shp")

    k42_reservoir = _random_points_gdf(20, xmin, ymin, xmax, ymax, "value", seed=5)
    k42_reservoir.to_file(geologic_dir / "k4-2-Potential.shp")

    # "k43" has reservoir layer but NO thermal layer -> should fall back to
    # regional heat_flow_basin layer.
    k43_reservoir = _random_points_gdf(20, xmin, ymin, xmax, ymax, "value", seed=6)
    k43_reservoir.to_file(geologic_dir / "k4-3-Potential.shp")

    heat_flow_basin = _random_points_gdf(
        20, xmin, ymin, xmax, ymax, "value", seed=7
    )
    heat_flow_basin.to_file(thermal_dir / "heat_flow_basin.shp")

    # "k44" has neither thermal nor reservoir -> fully skipped.

    return str(workspace_dir)


def test_run_horizon_workflow_end_to_end(tmp_path):
    workspace_dir = _make_workspace(tmp_path)
    output_dir = str(tmp_path / "output")

    result = ngb.run_horizon_workflow(
        workspace_dir=workspace_dir,
        output_dir=output_dir,
        nx=10,
        ny=10,
    )

    out_dir = result["output_dir"]
    assert out_dir.endswith(
        os.path.join("favourability", "geothermal", "horizon_specific")
    )

    # Horizon with full data should produce a GeoTIFF.
    detfurth_path = os.path.join(out_dir, "detfurth_geothermal_favourability.tif")
    assert os.path.isfile(detfurth_path)
    with rasterio.open(detfurth_path) as ds:
        assert ds.crs.to_string() == CRS
        assert ds.nodata == -9999.0
        assert ds.dtypes[0] == "float32"
        data = ds.read(1)
        assert data.shape == (10, 10)
        assert np.all((data[data != -9999.0] >= 0.0) & (data[data != -9999.0] <= 1.0))

    # Horizon relying on thermal fallback should still succeed.
    k43_path = os.path.join(out_dir, "k43_geothermal_favourability.tif")
    assert os.path.isfile(k43_path)

    # k42 has both layers directly.
    k42_path = os.path.join(out_dir, "k42_geothermal_favourability.tif")
    assert os.path.isfile(k42_path)

    # Fully-missing horizon (k44) should not crash the workflow; its
    # diagnostics should be marked as skipped.
    diag_by_name = {d["horizon"]: d for d in result["horizon_diagnostics"]}
    assert diag_by_name["k44"]["status"] == "skipped"
    assert diag_by_name["detfurth"]["status"] == "ok"
    assert diag_by_name["k42"]["status"] == "ok"
    assert diag_by_name["k43"]["status"] == "ok"
    assert diag_by_name["k43"]["thermal_source"] == "heat_flow_basin"

    # Aggregated outputs.
    for fname in [
        "best_horizon_geothermal_favourability.tif",
        "PRIMARY_geothermal_favourability.tif",
        "SECONDARY_geothermal_favourability.tif",
        "best_horizon_id.tif",
        "best_horizon_id_mapping.csv",
        "summary.csv",
    ]:
        assert os.path.isfile(os.path.join(out_dir, fname)), f"missing {fname}"

    with rasterio.open(os.path.join(out_dir, "best_horizon_id.tif")) as ds:
        assert ds.dtypes[0] == "int16"
        assert ds.nodata == 0
        ids = ds.read(1)
        # Only horizons 1..17 or 0 (none) should appear.
        assert set(np.unique(ids)).issubset(set(range(0, len(ngb.HORIZON_NAMES) + 1)))

    mapping_df = pd.read_csv(os.path.join(out_dir, "best_horizon_id_mapping.csv"))
    assert len(mapping_df) == len(ngb.HORIZON_NAMES) + 1
    assert mapping_df.iloc[0]["horizon_name"] == "none"

    summary_df = pd.read_csv(os.path.join(out_dir, "summary.csv"))
    assert "detfurth" in summary_df["horizon"].values


def test_missing_basin_uses_fallback_extent(tmp_path):
    workspace_dir = tmp_path / "empty_workspace"
    (workspace_dir / "geothermal" / "thermal_component").mkdir(parents=True)
    (workspace_dir / "geothermal" / "geologic_component").mkdir(parents=True)
    output_dir = str(tmp_path / "output2")

    result = ngb.run_horizon_workflow(
        workspace_dir=str(workspace_dir),
        output_dir=output_dir,
        nx=5,
        ny=5,
        fallback_extent=[0, 0, 100, 100],
    )
    assert result["extent"] == [0, 0, 100, 100]
    # All horizons should be skipped gracefully (no crash).
    for diag in result["horizon_diagnostics"]:
        assert diag["status"] == "skipped"


def test_normalize_grid_handles_constant_and_nan():
    grid = np.array([[1.0, 1.0], [np.nan, 1.0]])
    normed = ngb.normalize_grid(grid)
    assert np.isnan(normed[1, 0])
    assert normed[0, 0] == 0.0


def test_safe_stack_argmax_all_nan_cell():
    stack = np.full((3, 2, 2), np.nan)
    stack[0, 0, 0] = 0.5
    best_value, best_index = ngb.safe_stack_argmax(stack)
    assert best_index[0, 0] == 1
    assert best_index[1, 1] == 0
    assert np.isnan(best_value[1, 1])
