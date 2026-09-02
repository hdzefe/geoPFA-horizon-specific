"""Tests for ngb_refactored_horizon_specific_tif_input.py using synthetic
GeoTIFF thermal layers and synthetic shapefiles."""

import csv
import os

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")

from shapely.geometry import Polygon, box

import ngb_refactored_horizon_specific_tif_input as ngb


EPSG = 25832
EXTENT = (0.0, 1000.0, 0.0, 1000.0)  # xmin, xmax, ymin, ymax


def _write_synthetic_tif(path, ny, nx, value_fn):
    """Write a synthetic single-band GeoTIFF covering EXTENT."""
    xmin, xmax, ymin, ymax = EXTENT
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    grid = np.fromfunction(value_fn, (ny, nx)).astype("float32")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", crs=f"EPSG:{EPSG}", transform=transform,
    ) as dst:
        dst.write(grid, 1)
    return grid


def _write_synthetic_reservoir_shp(path, geom):
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=f"EPSG:{EPSG}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gdf.to_file(path)


def _write_synthetic_basin_shp(path):
    xmin, xmax, ymin, ymax = EXTENT
    _write_synthetic_reservoir_shp(path, box(xmin, ymin, xmax, ymax))


def test_load_thermal_tif_resamples_to_requested_grid(tmp_path):
    tif_path = str(tmp_path / "aal1_thermal_use.tif")
    _write_synthetic_tif(tif_path, 20, 20, lambda y, x: (y + x) / 40.0)

    grid, extent, crs = ngb.load_thermal_tif(tif_path, nx=10, ny=10)

    assert grid.shape == (10, 10)
    assert extent == EXTENT
    assert crs is not None


def test_load_thermal_tif_missing_file_returns_none(tmp_path):
    grid, extent, crs = ngb.load_thermal_tif(str(tmp_path / "missing.tif"), 10, 10)
    assert grid is None
    assert extent is None
    assert crs is None


def test_normalize_grid_handles_constant_and_nan():
    constant = np.full((5, 5), 3.0)
    assert np.all(ngb.normalize_grid(constant) == 1.0)

    with_nan = np.array([[0.0, 10.0], [np.nan, 5.0]])
    normed = ngb.normalize_grid(with_nan)
    assert normed[0, 0] == 0.0
    assert normed[0, 1] == 1.0
    assert normed[1, 0] == 0.0  # NaN -> 0


def test_load_and_interpolate_reservoir_rasterizes_polygon(tmp_path):
    shp_path = str(tmp_path / "aal1-Potential.shp")
    # polygon covering the left half of the extent
    _write_synthetic_reservoir_shp(shp_path, box(0, 0, 500, 1000))

    grid = ngb.load_and_interpolate_reservoir(shp_path, EXTENT, nx=10, ny=10, target_epsg=EPSG)

    assert grid.shape == (10, 10)
    assert grid.max() == 1.0
    # left half should have coverage, right half should not
    assert grid[:, 0].sum() > 0
    assert grid[:, -1].sum() == 0


def test_load_and_interpolate_reservoir_missing_file_returns_zeros(tmp_path):
    grid = ngb.load_and_interpolate_reservoir(
        str(tmp_path / "missing.shp"), EXTENT, nx=10, ny=10, target_epsg=EPSG
    )
    assert grid.shape == (10, 10)
    assert np.all(grid == 0)


def test_load_and_interpolate_reservoir_none_path_returns_zeros():
    grid = ngb.load_and_interpolate_reservoir(None, EXTENT, nx=10, ny=10, target_epsg=EPSG)
    assert np.all(grid == 0)


def test_safe_stack_argmax_all_nan_cell():
    stack = np.array(
        [
            [[1.0, np.nan], [np.nan, np.nan]],
            [[0.5, np.nan], [np.nan, np.nan]],
        ]
    )
    best_val, best_idx = ngb.safe_stack_argmax(stack)
    assert best_val[0, 0] == 1.0
    assert best_idx[0, 0] == 0
    assert np.isnan(best_val[1, 1])
    assert best_idx[1, 1] == -1


def test_run_horizon_workflow_end_to_end(tmp_path):
    thermal_dir = tmp_path / "thermal"
    reservoir_dir = tmp_path / "reservoirs"
    output_dir = tmp_path / "output"

    horizons = {
        "aal1": {
            "label": "Aalenian 1",
            "thermal_tif": "aal1_thermal_use.tif",
            "reservoir_shp": "aal1-Potential.shp",
            "base_weight": 0.85,
            "evidence_weight": 0.15,
            "confidence_weight": 1.0,
        },
        "bj1": {
            "label": "Bajocian 1",
            "thermal_tif": "bj1_thermal_use.tif",
            "reservoir_shp": "bj1-Potential.shp",
            "base_weight": 0.85,
            "evidence_weight": 0.15,
            "confidence_weight": 1.0,
        },
        "valanginian": {
            "label": "Valanginian",
            "thermal_tif": "valanginian_thermal_use.tif",
            "reservoir_shp": None,
            "base_weight": 0.85,
            "evidence_weight": 0.15,
            "confidence_weight": 1.0,
        },
        "missing_thermal": {
            "label": "Missing Thermal Horizon",
            "thermal_tif": "does_not_exist_thermal_use.tif",
            "reservoir_shp": None,
            "base_weight": 0.85,
            "evidence_weight": 0.15,
            "confidence_weight": 1.0,
        },
    }

    _write_synthetic_tif(
        str(thermal_dir / "aal1_thermal_use.tif"), 25, 25, lambda y, x: (y + x) / 50.0
    )
    _write_synthetic_tif(
        str(thermal_dir / "bj1_thermal_use.tif"), 25, 25, lambda y, x: (25 - y) / 25.0
    )
    _write_synthetic_tif(
        str(thermal_dir / "valanginian_thermal_use.tif"), 25, 25, lambda y, x: 0.0 * y + 0.5
    )

    _write_synthetic_reservoir_shp(
        str(reservoir_dir / "aal1-Potential.shp"), box(0, 0, 1000, 1000)
    )
    _write_synthetic_reservoir_shp(
        str(reservoir_dir / "bj1-Potential.shp"), box(0, 0, 500, 500)
    )

    basin_shp = tmp_path / "basin" / "north_german_basin_.shp"
    _write_synthetic_basin_shp(str(basin_shp))

    salt_shp = tmp_path / "salt" / "salt.shp"
    _write_synthetic_reservoir_shp(str(salt_shp), Polygon([(900, 900), (1000, 900), (1000, 1000), (900, 1000)]))

    result = ngb.run_horizon_workflow(
        thermal_tif_dir=str(thermal_dir),
        reservoir_shp_dir=str(reservoir_dir),
        basin_shp_path=str(basin_shp),
        salt_shp_path=str(salt_shp),
        output_dir=str(output_dir),
        nx=15,
        ny=15,
        target_epsg=EPSG,
        salt_buffer_m=0.0,
        horizons=horizons,
    )

    out_dir = result["output_dir"]
    assert os.path.isdir(out_dir)

    # 3 valid horizons produced output; missing_thermal was skipped.
    assert set(result["horizon_grids"].keys()) == {"aal1", "bj1", "valanginian"}
    for name in ["aal1", "bj1", "valanginian"]:
        assert os.path.isfile(os.path.join(out_dir, f"{name}_geothermal_favourability.tif"))
    assert not os.path.isfile(
        os.path.join(out_dir, "missing_thermal_geothermal_favourability.tif")
    )

    for fname in [
        "best_horizon_geothermal_favourability.tif",
        "PRIMARY_geothermal_favourability.tif",
        "SECONDARY_geothermal_favourability.tif",
        "best_horizon_id.tif",
        "best_horizon_id_mapping.csv",
        "summary.csv",
    ]:
        assert os.path.isfile(os.path.join(out_dir, fname)), fname

    # valanginian has no reservoir shapefile -> base_score should be zero
    # everywhere for it, so its favorability grid should be all zero.
    assert np.allclose(result["horizon_grids"]["valanginian"], 0.0)

    # aal1 covers the full extent with reservoir potential, so its
    # favorability should be > 0 somewhere.
    assert result["horizon_grids"]["aal1"].max() > 0.0

    with open(os.path.join(out_dir, "best_horizon_id_mapping.csv")) as f:
        rows = list(csv.DictReader(f))
    assert {r["horizon"] for r in rows} == {"aal1", "bj1", "valanginian"}

    with open(os.path.join(out_dir, "summary.csv")) as f:
        summary_rows = list(csv.DictReader(f))
    statuses = {r["horizon"]: r["status"] for r in summary_rows}
    assert statuses["missing_thermal"] == "skipped_no_thermal"
    assert statuses["aal1"] == "ok"


def test_run_horizon_workflow_missing_basin_falls_back_to_tif_extent(tmp_path):
    thermal_dir = tmp_path / "thermal"
    output_dir = tmp_path / "output"

    horizons = {
        "aal1": {
            "label": "Aalenian 1",
            "thermal_tif": "aal1_thermal_use.tif",
            "reservoir_shp": None,
            "base_weight": 0.85,
            "evidence_weight": 0.15,
            "confidence_weight": 1.0,
        }
    }
    _write_synthetic_tif(
        str(thermal_dir / "aal1_thermal_use.tif"), 10, 10, lambda y, x: (y + x) / 20.0
    )

    result = ngb.run_horizon_workflow(
        thermal_tif_dir=str(thermal_dir),
        reservoir_shp_dir=str(tmp_path / "no_reservoirs"),
        basin_shp_path=str(tmp_path / "no_basin.shp"),
        salt_shp_path=str(tmp_path / "no_salt.shp"),
        output_dir=str(output_dir),
        nx=10,
        ny=10,
        target_epsg=EPSG,
        horizons=horizons,
    )

    assert result["extent"] is None
    assert "aal1" in result["horizon_grids"]
