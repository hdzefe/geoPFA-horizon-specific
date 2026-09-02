"""Tests for the horizon-specific VoterVeto workflow."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

import ngb_horizon_specific_voterveto as ngb

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, Polygon  # noqa: E402

TEST_EXTENT = (3_400_000.0, 5_800_000.0, 3_410_000.0, 5_810_000.0)


@pytest.fixture()
def grid():
    return ngb.build_grid(TEST_EXTENT, nx=20, ny=20)


def write_thermal_tif(path, grid, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.ny,
        width=grid.nx,
        count=1,
        dtype="float32",
        crs=rasterio.crs.CRS.from_epsg(ngb.TARGET_EPSG),
        transform=grid.transform,
    ) as dst:
        dst.write(np.asarray(values, dtype="float32"), 1)
    return path


def box(xmin, ymin, xmax, ymax):
    return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_all_18_horizons_are_configured():
    assert len(ngb.HORIZONS) == 18
    expected = {
        "detfurth", "tsf1", "tsf2", "k42", "k43", "k44", "het1", "het2",
        "sin1", "sin2", "pli1", "pli2", "toa1", "toa2", "aal1", "bj1",
        "valanginian", "bueckeberg",
    }
    assert set(ngb.HORIZONS) == expected


def test_horizon_config_is_complete():
    for name, config in ngb.HORIZONS.items():
        for key in (
            "label",
            "thermal_layer",
            "reservoir_layer",
            "reservoir_source_path",
            "reservoir_value_col",
            "evidence_layers",
            "base_weight",
            "evidence_weight",
            "confidence_weight",
        ):
            assert key in config, f"{name} misses '{key}'"
        assert config["thermal_layer"].endswith("_thermal_use")
        assert config["evidence_layers"], f"{name} has no evidence layers"
        assert config["reservoir_source_path"].endswith(".shp")
        # Either a categorical mapping or a continuous min/max range.
        assert (
            "reservoir_value_mapping" in config
            or "reservoir_value_min" in config
        )


def test_horizon_weights_match_specification():
    assert ngb.HORIZONS["detfurth"]["base_weight"] == pytest.approx(0.85)
    assert ngb.HORIZONS["detfurth"]["evidence_weight"] == pytest.approx(0.15)
    assert ngb.HORIZONS["valanginian"]["confidence_weight"] == pytest.approx(0.7)
    assert ngb.HORIZONS["bueckeberg"]["confidence_weight"] == pytest.approx(0.7)
    for name, config in ngb.HORIZONS.items():
        total = config["base_weight"] + config["evidence_weight"]
        assert total == pytest.approx(1.0), name


def test_horizon_ids_are_unique_and_one_based():
    ids = list(ngb.HORIZON_IDS.values())
    assert ids == list(range(1, 19))
    assert len(set(ids)) == len(ids)


def test_target_epsg_is_31467():
    assert ngb.TARGET_EPSG == 31467


# ---------------------------------------------------------------------------
# Value mappings
# ---------------------------------------------------------------------------


def test_map_reservoir_values_categorical_german():
    gdf = gpd.GeoDataFrame(
        {"Potential": ["hoch", "Mittel", "niedrig", "unbekannt"]},
        geometry=[Point(0, 0)] * 4,
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    )
    scores = ngb.map_reservoir_values(
        gdf, "Potential", ngb.RESERVOIR_POTENTIAL_MAPPING_DE
    )
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.6)
    assert scores[2] == pytest.approx(0.2)
    assert np.isnan(scores[3])


def test_map_reservoir_values_handles_umlaut_variants():
    gdf = gpd.GeoDataFrame(
        {"Potential": ["eingeschränkt", "eingeschraenkt"]},
        geometry=[Point(0, 0)] * 2,
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    )
    scores = ngb.map_reservoir_values(
        gdf, "Potential", ngb.RESERVOIR_POTENTIAL_MAPPING_DE_RESTRICTED
    )
    assert np.allclose(scores, 0.2)


def test_map_reservoir_values_thickness_classes():
    gdf = gpd.GeoDataFrame(
        {"quality": ["high (> 20 m)", "moderate (10-20 m)", "low (< 10 m)"]},
        geometry=[Point(0, 0)] * 3,
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    )
    scores = ngb.map_reservoir_values(
        gdf, "quality", ngb.RESERVOIR_THICKNESS_MAPPING_EN
    )
    assert np.allclose(scores, [1.0, 0.6, 0.2])


def test_map_reservoir_values_binary_codes():
    gdf = gpd.GeoDataFrame(
        {"LEGNR": [2, 3, 7, 11, 99]},
        geometry=[Point(0, 0)] * 5,
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    )
    scores = ngb.map_reservoir_values(
        gdf, "LEGNR", ngb.RESERVOIR_BINARY_MAPPING_BUECKEBERG
    )
    assert np.allclose(scores[:4], 1.0)
    assert np.isnan(scores[4])


def test_map_reservoir_values_continuous_isolines():
    gdf = gpd.GeoDataFrame(
        {"sand_share": [30.0, 60.0, 90.0, 120.0]},
        geometry=[Point(0, 0)] * 4,
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    )
    scores = ngb.map_reservoir_values(
        gdf, "sand_share", None, value_min=30.0, value_max=90.0
    )
    assert np.allclose(scores, [0.0, 0.5, 1.0, 1.0])


def test_map_reservoir_values_missing_column():
    gdf = gpd.GeoDataFrame(
        {"other": [1]}, geometry=[Point(0, 0)], crs=f"EPSG:{ngb.TARGET_EPSG}"
    )
    with pytest.raises(KeyError):
        ngb.map_reservoir_values(
            gdf, "Potential", ngb.RESERVOIR_POTENTIAL_MAPPING_DE
        )


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_grid_geometry(grid):
    assert grid.shape == (20, 20)
    xs, ys = grid.cell_centers()
    assert ys[0] > ys[-1]  # row 0 is the northern-most row
    assert xs[0] > grid.xmin


def test_build_grid_rejects_invalid_extent():
    with pytest.raises(ValueError):
        ngb.build_grid((10.0, 10.0, 0.0, 0.0))


def test_grid_points_gdf_roundtrip(grid):
    values = np.arange(grid.ny * grid.nx, dtype="float64").reshape(grid.shape)
    gdf = ngb.grid_points_gdf(grid, values)
    assert len(gdf) == grid.nx * grid.ny
    assert gdf.crs.to_epsg() == ngb.TARGET_EPSG

    from geopfa import transformation

    raster = transformation.rasterize_model_2d(gdf, "value")
    assert np.allclose(raster, values)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_horizons_composite_and_ids():
    a = np.array([[0.1, 0.9], [np.nan, 0.4]])
    b = np.array([[0.5, 0.2], [np.nan, 0.8]])
    result = ngb.aggregate_horizons(
        {"detfurth": a, "tsf1": b}, primary_quantile=0.5, secondary_quantile=0.25
    )

    assert np.allclose(result.composite[0], [0.5, 0.9])
    assert np.isnan(result.composite[1, 0])
    assert result.composite[1, 1] == pytest.approx(0.8)

    assert result.best_horizon_id[0, 0] == ngb.HORIZON_IDS["tsf1"]
    assert result.best_horizon_id[0, 1] == ngb.HORIZON_IDS["detfurth"]
    assert result.best_horizon_id[1, 0] == 0
    assert result.best_horizon_id.dtype == np.int16


def test_aggregate_horizons_quantile_masks():
    composite_source = {
        "detfurth": np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        )
    }
    result = ngb.aggregate_horizons(
        composite_source, primary_quantile=0.90, secondary_quantile=0.75
    )
    primary_cells = np.count_nonzero(np.isfinite(result.primary))
    secondary_cells = np.count_nonzero(np.isfinite(result.secondary))
    assert primary_cells < secondary_cells
    assert result.primary_threshold >= result.secondary_threshold
    # Every primary cell is also a secondary cell.
    assert np.all(np.isfinite(result.secondary)[np.isfinite(result.primary)])


def test_aggregate_horizons_requires_input():
    with pytest.raises(ValueError):
        ngb.aggregate_horizons({})


def test_mask_by_threshold_with_nan_threshold():
    composite = np.array([[0.5, 0.9]])
    masked = ngb.mask_by_threshold(composite, float("nan"))
    assert np.all(np.isnan(masked))


def test_apply_masks():
    array = np.ones((2, 2))
    basin = np.array([[True, True], [False, True]])
    salt = np.array([[False, True], [False, False]])
    masked = ngb.apply_masks(array, basin, salt)
    assert masked[0, 0] == 1.0
    assert np.isnan(masked[0, 1])  # salt veto
    assert np.isnan(masked[1, 0])  # outside basin
    assert masked[1, 1] == 1.0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_geotiff_uses_project_crs(tmp_path, grid):
    array = np.full(grid.shape, 0.5)
    array[0, 0] = np.nan
    path = ngb.export_geotiff(array, str(tmp_path / "out.tif"), grid)

    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 31467
        assert src.width == grid.nx and src.height == grid.ny
        data = src.read(1, masked=True)
        assert data.mask[0, 0]
        assert data[1, 1] == pytest.approx(0.5)


def test_export_geotiff_int16(tmp_path, grid):
    array = np.zeros(grid.shape, dtype="int16")
    array[0, 0] = 7
    path = ngb.export_geotiff(
        array, str(tmp_path / "id.tif"), grid, dtype="int16"
    )
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "int16"
        assert src.read(1)[0, 0] == 7


def test_export_horizon_id_mapping(tmp_path):
    path = ngb.export_horizon_id_mapping(
        str(tmp_path / "mapping.csv"), ["detfurth", "tsf1"]
    )
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["horizon_name"] == "nodata"
    assert rows[1]["horizon_name"] == "detfurth"
    assert rows[1]["horizon_id"] == str(ngb.HORIZON_IDS["detfurth"])
    assert rows[2]["label"] == ngb.HORIZONS["tsf1"]["label"]


def test_export_summary_statistics(tmp_path):
    results = {"detfurth": np.array([[0.2, 0.8], [np.nan, 0.5]])}
    aggregation = ngb.aggregate_horizons(results)
    path = ngb.export_summary_statistics(
        str(tmp_path / "summary.csv"), results, aggregation
    )
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["horizon"] for row in rows]
    assert names == ["detfurth", "COMPOSITE", "PRIMARY", "SECONDARY"]
    assert rows[0]["valid_cells"] == "3"
    assert float(rows[0]["max"]) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def test_load_raster_to_grid_resamples(tmp_path, grid):
    coarse = ngb.build_grid(TEST_EXTENT, nx=5, ny=5)
    values = np.arange(25, dtype="float32").reshape(5, 5)
    path = write_thermal_tif(str(tmp_path / "thermal.tif"), coarse, values)

    loaded = ngb.load_raster_to_grid(path, grid)
    assert loaded.shape == grid.shape
    assert np.nanmax(loaded) <= 24.0
    assert np.nanmin(loaded) >= 0.0


def test_point_evidence_grid_decays_with_distance(grid, monkeypatch):
    monkeypatch.setattr(ngb, "EVIDENCE_SEARCH_RADIUS_M", 5000.0)
    center = Point(
        (TEST_EXTENT[0] + TEST_EXTENT[2]) / 2,
        (TEST_EXTENT[1] + TEST_EXTENT[3]) / 2,
    )
    gdf = gpd.GeoDataFrame(geometry=[center], crs=f"EPSG:{ngb.TARGET_EPSG}")
    evidence = ngb.point_evidence_grid(gdf, grid)
    assert evidence.shape == grid.shape
    assert evidence.max() == pytest.approx(1.0)
    assert evidence[0, 0] < evidence[grid.ny // 2, grid.nx // 2]


def test_certainty_factor_variants():
    factors = ngb.certainty_factor(["sicher", "unsicher", 80, 0.5, None])
    assert factors[0] == pytest.approx(1.0)
    assert factors[1] == pytest.approx(0.6)
    assert factors[2] == pytest.approx(0.8)
    assert factors[3] == pytest.approx(0.5)
    assert factors[4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_project(tmp_path, monkeypatch, grid):
    """Create a miniature project with three real horizons."""
    thermal_dir = tmp_path / "thermal"
    reservoir_dir = tmp_path / "reservoirs"
    evidence_dir = tmp_path / "evidence"
    for directory in (thermal_dir, reservoir_dir, evidence_dir):
        directory.mkdir()

    xs, ys = grid.cell_centers()
    xx, _ = np.meshgrid(xs, ys)
    base = (xx - grid.xmin) / (grid.xmax - grid.xmin)

    horizons = ["tsf1", "het1", "bueckeberg"]
    for offset, name in enumerate(horizons):
        config = ngb.HORIZONS[name]
        write_thermal_tif(
            str(thermal_dir / f"{config['thermal_layer']}.tif"),
            grid,
            (base + 0.1 * offset).astype("float32"),
        )

    mid_x = (grid.xmin + grid.xmax) / 2

    gpd.GeoDataFrame(
        {"Potential": ["hoch", "mittel"]},
        geometry=[
            box(grid.xmin, grid.ymin, mid_x, grid.ymax),
            box(mid_x, grid.ymin, grid.xmax, grid.ymax),
        ],
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    ).to_file(reservoir_dir / "tsf1-Potential.shp")

    gpd.GeoDataFrame(
        {"quality": ["high (> 20 m)"]},
        geometry=[box(grid.xmin, grid.ymin, grid.xmax, grid.ymax)],
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    ).to_file(reservoir_dir / "Het1_Potential.shp")

    gpd.GeoDataFrame(
        {"LEGNR": [2, 99]},
        geometry=[
            box(grid.xmin, grid.ymin, mid_x, grid.ymax),
            box(mid_x, grid.ymin, grid.xmax, grid.ymax),
        ],
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    ).to_file(reservoir_dir / "bueckeberg_group_sandstone_lbeg.shp")

    gpd.GeoDataFrame(
        geometry=[Point(mid_x, (grid.ymin + grid.ymax) / 2)],
        crs=f"EPSG:{ngb.TARGET_EPSG}",
    ).to_file(evidence_dir / "tsf1_borehole_poroperm_evidence.shp")

    monkeypatch.setattr(ngb, "THERMAL_TIF_DIR", str(thermal_dir))
    monkeypatch.setattr(ngb, "RESERVOIR_SHP_DIR", str(reservoir_dir))
    monkeypatch.setattr(ngb, "RAW_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(ngb, "EVIDENCE_DIR", str(evidence_dir))
    monkeypatch.setattr(ngb, "BASIN_SHP_PATH", "")
    monkeypatch.setattr(ngb, "SALT_SHP_PATH", "")
    return {"root": tmp_path, "horizons": horizons}


def test_run_pipeline_writes_expected_products(synthetic_project, tmp_path, grid):
    output_dir = tmp_path / "output"
    result = ngb.run_pipeline(
        horizons=synthetic_project["horizons"],
        output_dir=str(output_dir),
        grid=grid,
    )

    target = os.path.join(str(output_dir), ngb.HORIZON_OUTPUT_SUBDIR)
    assert result["output_dir"] == target
    assert not result["skipped"]

    for name in synthetic_project["horizons"]:
        path = os.path.join(target, f"{name}_geothermal_favourability.tif")
        assert os.path.isfile(path)
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 31467
            assert src.shape == grid.shape

    for filename in (
        "best_horizon_geothermal_favourability.tif",
        "PRIMARY_geothermal_favourability.tif",
        "SECONDARY_geothermal_favourability.tif",
        "best_horizon_id.tif",
        "best_horizon_id_mapping.csv",
        "summary.csv",
    ):
        assert os.path.isfile(os.path.join(target, filename)), filename

    aggregation = result["aggregation"]
    stack = np.stack(
        [result["horizon_results"][n] for n in aggregation.horizon_names]
    )
    with np.errstate(invalid="ignore"):
        expected = np.nanmax(stack, axis=0)
    finite = np.isfinite(expected)
    assert np.allclose(aggregation.composite[finite], expected[finite])

    with rasterio.open(os.path.join(target, "best_horizon_id.tif")) as src:
        ids = src.read(1)
    assert set(np.unique(ids)).issubset(
        {0} | {ngb.HORIZON_IDS[n] for n in synthetic_project["horizons"]}
    )


def test_run_pipeline_applies_reservoir_veto(synthetic_project, tmp_path, grid):
    result = ngb.run_pipeline(
        horizons=["bueckeberg"], output_dir=str(tmp_path / "out"), grid=grid
    )
    favorability = result["horizon_results"]["bueckeberg"]
    # LEGNR 99 is not part of the mapping -> eastern half must be NaN.
    assert np.all(np.isnan(favorability[:, grid.nx // 2 + 1 :]))
    assert np.any(np.isfinite(favorability[:, : grid.nx // 2]))


def test_run_pipeline_confidence_weight_scales_favorability(
    synthetic_project, tmp_path, grid
):
    result = ngb.run_pipeline(
        horizons=["bueckeberg"], output_dir=str(tmp_path / "out2"), grid=grid
    )
    favorability = result["horizon_results"]["bueckeberg"]
    assert np.nanmax(favorability) <= (
        ngb.HORIZONS["bueckeberg"]["confidence_weight"] + 1e-9
    )


def test_run_pipeline_skips_horizons_without_thermal(
    synthetic_project, tmp_path, grid
):
    result = ngb.run_pipeline(
        horizons=["tsf1", "detfurth"],
        output_dir=str(tmp_path / "out3"),
        grid=grid,
    )
    assert result["skipped"] == ["detfurth"]
    assert set(result["horizon_results"]) == {"tsf1"}


def test_run_pipeline_rejects_unknown_horizon(synthetic_project, tmp_path, grid):
    with pytest.raises(KeyError):
        ngb.run_pipeline(
            horizons=["not_a_horizon"], output_dir=str(tmp_path), grid=grid
        )


def test_run_pipeline_raises_without_any_input(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(ngb, "THERMAL_TIF_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(ngb, "BASIN_SHP_PATH", "")
    monkeypatch.setattr(ngb, "SALT_SHP_PATH", "")
    with pytest.raises(RuntimeError):
        ngb.run_pipeline(horizons=["tsf1"], output_dir=str(tmp_path), grid=grid)


def test_cli_list_horizons(capsys):
    assert ngb.main(["--list-horizons"]) == 0
    out = capsys.readouterr().out
    assert "detfurth" in out and "bueckeberg" in out
    assert len(out.strip().splitlines()) == 18
