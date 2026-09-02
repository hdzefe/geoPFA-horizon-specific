"""Tests for the horizon-specific geothermal favourability workflow.

The tests build a small synthetic dataset (basin outline, salt structures,
thermal GeoTIFFs, reservoir shapefiles and evidence layers) in a temporary
directory, then exercise the full workflow for all 18 configured horizons plus
a number of edge cases (missing thermal raster, missing reservoir shapefile,
unknown attribute values, degenerate data).
"""

from __future__ import annotations

import copy
import csv
import os
import sys

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, box  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ngb_refactored_horizon_specific_tif_input as ngb  # noqa: E402


EPSG = 31467
XMIN, YMIN, XMAX, YMAX = 3_400_000.0, 5_800_000.0, 3_500_000.0, 5_900_000.0


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _write_shapefile(path, geometries, attributes=None, epsg=EPSG):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = dict(attributes or {})
    data["geometry"] = geometries
    gdf = gpd.GeoDataFrame(data, crs=f"EPSG:{epsg}")
    gdf.to_file(path)
    return path


def _write_tif(path, array, epsg=EPSG, bounds=(XMIN, YMIN, XMAX, YMAX),
               nodata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.asarray(array, dtype="float32")
    transform = rasterio.transform.from_bounds(*bounds, arr.shape[1], arr.shape[0])
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": None if epsg is None else f"EPSG:{epsg}",
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def _reservoir_values_for(config):
    """Pick attribute values that the horizon configuration can map."""
    mapping = config.get("reservoir_value_mapping")
    if mapping is not None:
        keys = list(mapping.keys())
        return [keys[-1], keys[0]]
    lo = config.get("reservoir_value_min", 0.0)
    hi = config.get("reservoir_value_max", 1.0)
    return [hi, lo]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Create a complete synthetic dataset and the matching horizon config."""
    root = tmp_path_factory.mktemp("ngb_data")
    thermal_dir = root / "thermal"
    evidence_dir = root / "evidence"
    reservoir_dir = root / "reservoirs"
    out_dir = root / "output"

    basin = _write_shapefile(
        str(root / "basin" / "north_german_basin_.shp"),
        [box(XMIN, YMIN, XMAX, YMAX)],
        {"name": ["basin"]},
    )

    salt = _write_shapefile(
        str(root / "salt" / "Salzstrukturen_Inspee__v1_poly.shp"),
        [box(XMIN + 5_000, YMIN + 5_000, XMIN + 20_000, YMIN + 20_000)],
        {"name": ["salt"]},
    )

    gradient = np.tile(np.linspace(20.0, 140.0, 40), (40, 1))

    horizons = copy.deepcopy(ngb.HORIZONS)
    for name, config in horizons.items():
        _write_tif(str(thermal_dir / f"{config['thermal_layer']}.tif"), gradient)

        shp_path = reservoir_dir / f"{name}_reservoir.shp"
        values = _reservoir_values_for(config)
        geometries = [
            box(XMIN, YMIN + 50_000, XMAX, YMAX),   # northern half
            box(XMIN, YMIN, XMAX, YMIN + 50_000),   # southern half
        ]
        attributes = {config["reservoir_value_col"]: values}
        if config.get("reservoir_certainty_col"):
            attributes[config["reservoir_certainty_col"]] = [100.0, 50.0]
        _write_shapefile(str(shp_path), geometries, attributes)
        config["reservoir_source_path"] = str(shp_path)

        first_evidence = (config.get("evidence_layers") or [None])[0]
        if first_evidence:
            _write_shapefile(
                str(evidence_dir / f"{first_evidence}.shp"),
                [Point(XMIN + 60_000, YMIN + 60_000)],
                {"id": [1]},
            )

    return {
        "root": root,
        "basin": basin,
        "salt": salt,
        "thermal_dir": str(thermal_dir),
        "evidence_dir": str(evidence_dir),
        "reservoir_dir": str(reservoir_dir),
        "output_dir": str(out_dir),
        "horizons": horizons,
    }


@pytest.fixture(scope="module")
def workflow_config(dataset):
    return ngb.WorkflowConfig(
        nx=48,
        ny=48,
        target_epsg=EPSG,
        thermal_dir=dataset["thermal_dir"],
        evidence_dir=dataset["evidence_dir"],
        basin_shp_path=dataset["basin"],
        salt_shp_path=dataset["salt"],
        output_dir=dataset["output_dir"],
        horizons=dataset["horizons"],
    )


@pytest.fixture(scope="module")
def workflow_result(workflow_config):
    ngb.setup_logging(0)
    return ngb.run_workflow(workflow_config)


@pytest.fixture()
def grid():
    return ngb.Grid(extent=(XMIN, YMIN, XMAX, YMAX), nx=32, ny=32, epsg=EPSG)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


EXPECTED_HORIZONS = [
    "detfurth", "tsf1", "tsf2", "k42", "k43", "k44", "het1", "het2",
    "sin1", "sin2", "pli1", "pli2", "toa1", "toa2", "aal1", "bj1",
    "valanginian", "bueckeberg",
]


def test_all_eighteen_horizons_are_configured():
    assert list(ngb.HORIZONS) == EXPECTED_HORIZONS
    assert len(ngb.HORIZONS) == 18


@pytest.mark.parametrize("name", EXPECTED_HORIZONS)
def test_horizon_config_is_complete(name):
    config = ngb.HORIZONS[name]
    assert config["thermal_layer"] == f"{name}_thermal_use"
    assert config["reservoir_layer"]
    assert config["reservoir_source_path"].endswith(".shp")
    assert config["reservoir_value_col"]
    assert config["evidence_layers"]
    assert 0.0 < config["base_weight"] <= 1.0
    assert 0.0 <= config["evidence_weight"] <= 1.0
    assert 0.0 < config["confidence_weight"] <= 1.0
    assert config["base_weight"] + config["evidence_weight"] == pytest.approx(1.0)
    has_mapping = "reservoir_value_mapping" in config
    has_range = "reservoir_value_min" in config and "reservoir_value_max" in config
    assert has_mapping or has_range


def test_target_epsg_is_31467():
    assert ngb.TARGET_EPSG == 31467
    assert ngb.Grid(extent=(0, 0, 10, 10), nx=2, ny=2).crs == "EPSG:31467"


def test_specific_column_names_preserved():
    assert ngb.HORIZONS["detfurth"]["reservoir_value_col"] == "sand_share"
    assert ngb.HORIZONS["detfurth"]["reservoir_certainty_col"] == "certainity"
    assert ngb.HORIZONS["detfurth"]["reservoir_value_min"] == 30.0
    assert ngb.HORIZONS["detfurth"]["reservoir_value_max"] == 90.0
    assert ngb.HORIZONS["het1"]["reservoir_value_col"] == "quality"
    assert ngb.HORIZONS["sin1"]["reservoir_value_col"] == "reservoirq"
    assert ngb.HORIZONS["pli2"]["reservoir_value_col"] == "reservoir"
    assert ngb.HORIZONS["valanginian"]["reservoir_value_col"] == "OBJECTID"
    assert ngb.HORIZONS["bueckeberg"]["reservoir_value_col"] == "LEGNR"
    assert ngb.HORIZONS["valanginian"]["confidence_weight"] == 0.7
    assert ngb.HORIZONS["bueckeberg"]["confidence_weight"] == 0.7
    assert ngb.HORIZONS["detfurth"]["base_weight"] == 0.85
    assert ngb.HORIZONS["detfurth"]["evidence_weight"] == 0.15


def test_k4_horizons_use_restricted_mapping():
    for name in ("k42", "k43"):
        mapping = ngb.HORIZONS[name]["reservoir_value_mapping"]
        assert mapping is ngb.RESERVOIR_POTENTIAL_MAPPING_DE_RESTRICTED
        assert mapping["eingeschränkt"] == 0.2
        assert mapping["eingeschraenkt"] == 0.2
    assert (
        ngb.HORIZONS["k44"]["reservoir_value_mapping"]
        is ngb.RESERVOIR_POTENTIAL_MAPPING_DE
    )


# ---------------------------------------------------------------------------
# Value mapping helpers
# ---------------------------------------------------------------------------


def test_map_reservoir_values_german_classes():
    values = ngb.map_reservoir_values(
        ["hoch", "Mittel", " niedrig ", "unbekannt", None],
        value_mapping=ngb.RESERVOIR_POTENTIAL_MAPPING_DE,
    )
    assert list(values) == [1.0, 0.6, 0.2, 0.0, 0.0]


def test_map_reservoir_values_english_thickness_classes():
    values = ngb.map_reservoir_values(
        ["low (< 10 m)", "moderate (10-20 m)", "HIGH (> 20 M)"],
        value_mapping=ngb.RESERVOIR_THICKNESS_MAPPING_EN,
    )
    assert list(values) == [0.2, 0.6, 1.0]


def test_map_reservoir_values_binary_codes():
    values = ngb.map_reservoir_values(
        [2, 3.0, 7, 11, 99],
        value_mapping=ngb.RESERVOIR_BINARY_MAPPING_BUECKEBERG,
    )
    assert list(values) == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_map_reservoir_values_continuous_isolines():
    values = ngb.map_reservoir_values(
        [30.0, 60.0, 90.0, 120.0], value_min=30.0, value_max=90.0
    )
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1.0)
    assert values[3] == pytest.approx(1.0)  # clipped


def test_normalise_array_handles_constant_and_nan():
    assert np.allclose(ngb.normalise_array(np.full((2, 2), 5.0)), 0.0)
    out = ngb.normalise_array(np.array([np.nan, 0.0, 10.0]))
    assert np.isnan(out[0]) and out[1] == 0.0 and out[2] == 1.0
    assert np.all(np.isnan(ngb.normalise_array(np.full(3, np.nan))))


def test_resolve_column_is_case_and_truncation_tolerant():
    assert ngb.resolve_column(["Potential"], "Potential") == "Potential"
    assert ngb.resolve_column(["POTENTIAL"], "Potential") == "POTENTIAL"
    assert ngb.resolve_column(["reservoirq"], "reservoirqu") == "reservoirq"
    assert ngb.resolve_column(["other"], "Potential") is None


def test_normalise_certainty_numeric_and_textual():
    assert np.allclose(ngb.normalise_certainty([100.0, 50.0, 0.0]), [1.0, 0.5, 0.0])
    assert np.allclose(
        ngb.normalise_certainty(["gesichert", "vermutet", "???"]), [1.0, 0.4, 1.0]
    )


def test_safe_quantile_without_positive_values():
    assert ngb.safe_quantile(np.zeros((4, 4)), 0.9) is None
    assert ngb.safe_quantile(np.array([1.0, 2.0, 3.0]), 0.5) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_grid_geometry(grid):
    assert grid.shape == (32, 32)
    assert grid.cell_size == (pytest.approx(3125.0), pytest.approx(3125.0))
    xs, ys = grid.cell_centres()
    assert xs.shape == grid.shape
    assert ys[0, 0] > ys[-1, 0]  # north-up


def test_grid_rejects_invalid_input():
    with pytest.raises(ValueError):
        ngb.Grid(extent=(10, 10, 0, 0), nx=4, ny=4)
    with pytest.raises(ValueError):
        ngb.Grid(extent=(0, 0, 10, 10), nx=0, ny=4)


# ---------------------------------------------------------------------------
# Thermal GeoTIFF loading
# ---------------------------------------------------------------------------


def test_load_thermal_tif_resamples_to_grid(tmp_path, grid):
    path = _write_tif(
        str(tmp_path / "t.tif"), np.tile(np.linspace(0.0, 100.0, 10), (10, 1))
    )
    thermal = ngb.load_thermal_tif(path, grid)
    assert thermal.shape == grid.shape
    assert np.nanmin(thermal) >= 0.0 and np.nanmax(thermal) <= 1.0
    row = thermal[grid.ny // 2]
    assert row[-1] > row[0]


def test_load_thermal_tif_missing_file_returns_none(tmp_path, grid):
    assert ngb.load_thermal_tif(str(tmp_path / "nope.tif"), grid) is None
    assert ngb.load_thermal_tif("", grid) is None


def test_load_thermal_tif_honours_nodata(tmp_path, grid):
    arr = np.full((10, 10), 50.0)
    arr[:5, :] = -9999.0
    path = _write_tif(str(tmp_path / "nd.tif"), arr, nodata=-9999.0)
    thermal = ngb.load_thermal_tif(path, grid)
    assert np.isnan(thermal).any()
    assert np.isfinite(thermal[-1, :]).any()


def test_load_thermal_tif_all_nodata_returns_none(tmp_path, grid):
    path = _write_tif(
        str(tmp_path / "empty.tif"), np.full((10, 10), -9999.0), nodata=-9999.0
    )
    assert ngb.load_thermal_tif(path, grid) is None


def test_load_thermal_tif_reprojects_other_crs(tmp_path, grid):
    path = _write_tif(
        str(tmp_path / "utm.tif"),
        np.full((10, 10), 80.0),
        epsg=25832,
        bounds=(500_000, 5_800_000, 600_000, 5_900_000),
    )
    thermal = ngb.load_thermal_tif(path, grid)
    assert thermal is None or thermal.shape == grid.shape


def test_thermal_tif_path_for_uses_configured_layer_name():
    path = ngb.thermal_tif_path_for("aal1", ngb.HORIZONS["aal1"], "/tmp/thermal")
    assert path.endswith(os.path.join("thermal", "aal1_thermal_use.tif"))


# ---------------------------------------------------------------------------
# Reservoir rasterisation
# ---------------------------------------------------------------------------


def test_load_and_rasterize_reservoir_categorical(tmp_path, grid):
    path = _write_shapefile(
        str(tmp_path / "res.shp"),
        [box(XMIN, YMIN, XMAX, (YMIN + YMAX) / 2),
         box(XMIN, (YMIN + YMAX) / 2, XMAX, YMAX)],
        {"Potential": ["hoch", "niedrig"]},
    )
    raster = ngb.load_and_rasterize_reservoir(
        path, grid, value_col="Potential",
        value_mapping=ngb.RESERVOIR_POTENTIAL_MAPPING_DE,
    )
    assert raster.shape == grid.shape
    assert raster[0, 0] == pytest.approx(0.2)   # north = niedrig
    assert raster[-1, 0] == pytest.approx(1.0)  # south = hoch


def test_load_and_rasterize_reservoir_applies_certainty(tmp_path, grid):
    path = _write_shapefile(
        str(tmp_path / "detfurth.shp"),
        [box(XMIN, YMIN, XMAX, YMAX)],
        {"sand_share": [90.0], "certainity": [50.0]},
    )
    raster = ngb.load_and_rasterize_reservoir(
        path, grid, value_col="sand_share",
        value_min=30.0, value_max=90.0, certainty_col="certainity",
    )
    assert raster[0, 0] == pytest.approx(0.5)


def test_load_and_rasterize_reservoir_missing_column_falls_back(tmp_path, grid):
    path = _write_shapefile(
        str(tmp_path / "nocol.shp"), [box(XMIN, YMIN, XMAX, YMAX)], {"foo": [1]}
    )
    raster = ngb.load_and_rasterize_reservoir(
        path, grid, value_col="Potential",
        value_mapping=ngb.RESERVOIR_POTENTIAL_MAPPING_DE,
    )
    assert np.allclose(raster, 1.0)


def test_load_and_rasterize_reservoir_missing_file_returns_none(tmp_path, grid):
    assert ngb.load_and_rasterize_reservoir(
        str(tmp_path / "absent.shp"), grid, value_col="Potential"
    ) is None


def test_reservoir_outside_extent_is_all_zero(tmp_path, grid):
    path = _write_shapefile(
        str(tmp_path / "far.shp"),
        [box(XMIN - 500_000, YMIN - 500_000, XMIN - 400_000, YMIN - 400_000)],
        {"Potential": ["hoch"]},
    )
    raster = ngb.load_and_rasterize_reservoir(
        path, grid, value_col="Potential",
        value_mapping=ngb.RESERVOIR_POTENTIAL_MAPPING_DE,
    )
    assert np.allclose(raster, 0.0)


# ---------------------------------------------------------------------------
# Evidence and salt
# ---------------------------------------------------------------------------


def test_evidence_point_layer_creates_decay_halo(tmp_path, grid):
    _write_shapefile(
        str(tmp_path / "aal1_borehole_poroperm_evidence.shp"),
        [Point((XMIN + XMAX) / 2, (YMIN + YMAX) / 2)],
        {"id": [1]},
    )
    layer = ngb.load_evidence_layer(
        "aal1_borehole_poroperm_evidence", grid, str(tmp_path),
        influence_radius=30_000,
    )
    assert layer.shape == grid.shape
    assert layer.max() > 0.9
    assert layer[0, 0] == pytest.approx(0.0)


def test_evidence_polygon_layer_is_burned_directly(tmp_path, grid):
    _write_shapefile(
        str(tmp_path / "poly_evidence.shp"),
        [box(XMIN, YMIN, (XMIN + XMAX) / 2, YMAX)],
        {"id": [1]},
    )
    layer = ngb.load_evidence_layer("poly_evidence", grid, str(tmp_path))
    assert layer[0, 0] == 1.0
    assert layer[0, -1] == 0.0


def test_missing_evidence_layers_give_zero_grid(tmp_path, grid):
    evidence, used = ngb.load_evidence_layers(
        ngb.HORIZONS["tsf1"], grid, str(tmp_path)
    )
    assert used == []
    assert np.allclose(evidence, 0.0)


def test_salt_penalty(tmp_path, grid):
    path = _write_shapefile(
        str(tmp_path / "salt.shp"),
        [box(XMIN, YMIN, (XMIN + XMAX) / 2, (YMIN + YMAX) / 2)],
        {"id": [1]},
    )
    penalty = ngb.load_salt_penalty(path, grid)
    assert penalty[-1, 0] == pytest.approx(1.0)
    assert penalty[0, -1] == pytest.approx(0.0)


def test_missing_salt_layer_yields_zero_penalty(tmp_path, grid):
    penalty = ngb.load_salt_penalty(str(tmp_path / "nope.shp"), grid)
    assert np.allclose(penalty, 0.0)


def test_salt_penalty_strength_is_validated(tmp_path, grid):
    with pytest.raises(ValueError):
        ngb.load_salt_penalty(str(tmp_path / "nope.shp"), grid, strength=2.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_horizon_applies_weights_and_salt(dataset, grid):
    config = dataset["horizons"]["tsf1"]
    salt = np.zeros(grid.shape)
    result = ngb.score_horizon(
        "tsf1", config, grid, salt,
        thermal_dir=dataset["thermal_dir"], evidence_dir=dataset["evidence_dir"],
    )
    assert result.scored
    assert np.nanmax(result.score) <= 1.0
    assert np.nanmin(result.score) >= 0.0

    salt_full = np.ones(grid.shape)
    vetoed = ngb.score_horizon(
        "tsf1", config, grid, salt_full,
        thermal_dir=dataset["thermal_dir"], evidence_dir=dataset["evidence_dir"],
    )
    assert np.allclose(np.nan_to_num(vetoed.score), 0.0)


def test_score_horizon_confidence_weight_scales_result(dataset, grid):
    base = copy.deepcopy(dataset["horizons"]["tsf1"])
    reduced = copy.deepcopy(base)
    reduced["confidence_weight"] = 0.5
    salt = np.zeros(grid.shape)
    kwargs = dict(
        thermal_dir=dataset["thermal_dir"], evidence_dir=dataset["evidence_dir"]
    )
    full_score = ngb.score_horizon("tsf1", base, grid, salt, **kwargs).score
    half_score = ngb.score_horizon("tsf1", reduced, grid, salt, **kwargs).score
    assert np.allclose(
        np.nan_to_num(half_score), 0.5 * np.nan_to_num(full_score), atol=1e-9
    )


def test_score_horizon_skips_missing_thermal(dataset, grid, tmp_path):
    result = ngb.score_horizon(
        "tsf1", dataset["horizons"]["tsf1"], grid, np.zeros(grid.shape),
        thermal_dir=str(tmp_path), evidence_dir=dataset["evidence_dir"],
    )
    assert result.skipped
    assert "thermal" in result.skip_reason
    assert result.score is None


def test_score_horizon_skips_missing_reservoir(dataset, grid, tmp_path):
    config = copy.deepcopy(dataset["horizons"]["tsf1"])
    config["reservoir_source_path"] = str(tmp_path / "absent.shp")
    result = ngb.score_horizon(
        "tsf1", config, grid, np.zeros(grid.shape),
        thermal_dir=dataset["thermal_dir"], evidence_dir=dataset["evidence_dir"],
    )
    assert result.skipped
    assert "reservoir" in result.skip_reason


def test_score_horizon_respects_basin_mask(dataset, grid):
    mask = np.zeros(grid.shape, dtype=bool)
    mask[: grid.ny // 2] = True
    result = ngb.score_horizon(
        "tsf1", dataset["horizons"]["tsf1"], grid, np.zeros(grid.shape),
        basin_mask=mask, thermal_dir=dataset["thermal_dir"],
        evidence_dir=dataset["evidence_dir"],
    )
    assert np.all(np.isnan(result.score[grid.ny // 2:]))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _result(name, array):
    res = ngb.HorizonResult(name=name, label=name.upper())
    res.score = np.asarray(array, dtype=float)
    res.statistics = ngb.compute_statistics(res.score)
    return res


def test_aggregate_picks_cellwise_maximum():
    grid = ngb.Grid(extent=(0, 0, 10, 10), nx=2, ny=2)
    a = _result("a", [[0.1, 0.9], [0.4, 0.4]])
    b = _result("b", [[0.5, 0.2], [0.2, np.nan]])
    agg = ngb.aggregate_horizons([a, b], grid)
    assert np.allclose(agg.best_horizon, [[0.5, 0.9], [0.4, 0.4]])
    assert agg.best_horizon_id.tolist() == [[2, 1], [1, 1]]
    assert agg.horizon_order == ["a", "b"]


def test_aggregate_marks_all_nan_cells_with_id_zero():
    grid = ngb.Grid(extent=(0, 0, 10, 10), nx=2, ny=1)
    a = _result("a", [[np.nan, 0.5]])
    agg = ngb.aggregate_horizons([a], grid)
    assert agg.best_horizon_id.tolist() == [[0, 1]]
    assert np.isnan(agg.best_horizon[0, 0])


def test_aggregate_sweetspot_thresholds():
    grid = ngb.Grid(extent=(0, 0, 10, 10), nx=10, ny=1)
    values = np.linspace(0.1, 1.0, 10).reshape(1, 10)
    agg = ngb.aggregate_horizons([_result("a", values)], grid)
    assert agg.primary_threshold > agg.secondary_threshold
    assert agg.primary_mask.sum() <= agg.secondary_mask.sum()
    assert agg.primary_mask.sum() >= 1


def test_aggregate_without_positive_values_gives_empty_masks():
    grid = ngb.Grid(extent=(0, 0, 10, 10), nx=2, ny=2)
    agg = ngb.aggregate_horizons([_result("a", np.zeros((2, 2)))], grid)
    assert agg.primary_threshold is None
    assert agg.primary_mask.sum() == 0
    assert agg.secondary_mask.sum() == 0


def test_aggregate_requires_at_least_one_scored_horizon():
    grid = ngb.Grid(extent=(0, 0, 10, 10), nx=2, ny=2)
    skipped = ngb.HorizonResult(name="x", label="X", skipped=True)
    with pytest.raises(ValueError):
        ngb.aggregate_horizons([skipped], grid)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_geotiff_writes_target_crs(tmp_path, grid):
    data = np.full(grid.shape, 0.5)
    data[0, 0] = np.nan
    path = ngb.export_geotiff(data, str(tmp_path / "out.tif"), grid)
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 31467
        assert src.width == grid.nx and src.height == grid.ny
        assert src.nodata == ngb.NODATA_VALUE
        band = src.read(1)
    assert band[0, 0] == pytest.approx(ngb.NODATA_VALUE)
    assert band[1, 1] == pytest.approx(0.5)


def test_export_geotiff_rejects_shape_mismatch(tmp_path, grid):
    with pytest.raises(ValueError):
        ngb.export_geotiff(np.zeros((2, 2)), str(tmp_path / "bad.tif"), grid)


# ---------------------------------------------------------------------------
# End-to-end workflow
# ---------------------------------------------------------------------------


def test_workflow_scores_all_eighteen_horizons(workflow_result):
    results = workflow_result["results"]
    assert len(results) == 18
    assert all(res.scored for res in results), [
        (r.name, r.skip_reason) for r in results if not r.scored
    ]


def test_workflow_writes_expected_outputs(workflow_result, workflow_config):
    out_dir = workflow_config.horizon_output_dir
    for name in EXPECTED_HORIZONS:
        assert os.path.exists(
            os.path.join(out_dir, f"{name}_geothermal_favourability.tif")
        )
    for filename in (
        "best_horizon_geothermal_favourability.tif",
        "PRIMARY_geothermal_favourability.tif",
        "SECONDARY_geothermal_favourability.tif",
        "best_horizon_id.tif",
        "best_horizon_id_mapping.csv",
        "summary.csv",
    ):
        assert os.path.exists(os.path.join(out_dir, filename)), filename


def test_workflow_outputs_use_epsg_31467(workflow_result, workflow_config):
    path = os.path.join(
        workflow_config.horizon_output_dir, "detfurth_geothermal_favourability.tif"
    )
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 31467
        assert (src.width, src.height) == (workflow_config.nx, workflow_config.ny)


def test_workflow_best_horizon_is_the_maximum(workflow_result):
    agg = workflow_result["aggregation"]
    stack = np.stack(
        [r.score for r in workflow_result["results"] if r.scored], axis=0
    )
    expected = np.nanmax(stack, axis=0)
    valid = np.isfinite(expected)
    assert np.allclose(agg.best_horizon[valid], expected[valid])
    assert set(np.unique(agg.best_horizon_id)) <= set(range(0, 19))


def test_workflow_sweetspots_are_nested(workflow_result):
    agg = workflow_result["aggregation"]
    assert np.all(agg.secondary_mask[agg.primary_mask > 0] > 0)
    assert agg.primary_mask.sum() <= agg.secondary_mask.sum()


def test_workflow_salt_area_is_suppressed(workflow_result, workflow_config):
    grid = workflow_result["grid"]
    salt = ngb.load_salt_penalty(workflow_config.salt_shp_path, grid)
    best = workflow_result["aggregation"].best_horizon
    assert np.allclose(np.nan_to_num(best[salt > 0]), 0.0)


def test_summary_csv_contents(workflow_result, workflow_config):
    path = os.path.join(workflow_config.horizon_output_dir, "summary.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["horizon"] for row in rows]
    assert names[:18] == EXPECTED_HORIZONS
    assert names[-1] == "best_horizon"
    assert all(row["status"] == "scored" for row in rows[:18])


def test_best_horizon_id_mapping_csv(workflow_result, workflow_config):
    path = os.path.join(
        workflow_config.horizon_output_dir, "best_horizon_id_mapping.csv"
    )
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["horizon"] for row in rows] == EXPECTED_HORIZONS
    assert [int(row["horizon_id"]) for row in rows] == list(range(1, 19))
    assert sum(int(row["cells_best"]) for row in rows) > 0


def test_workflow_skips_horizon_without_thermal_raster(dataset, tmp_path):
    horizons = copy.deepcopy(dataset["horizons"])
    broken = copy.deepcopy(horizons["bj1"])
    broken["thermal_layer"] = "does_not_exist_thermal_use"
    subset = {"tsf1": horizons["tsf1"], "bj1": broken}
    config = ngb.WorkflowConfig(
        nx=24, ny=24, target_epsg=EPSG,
        thermal_dir=dataset["thermal_dir"],
        evidence_dir=dataset["evidence_dir"],
        basin_shp_path=dataset["basin"],
        salt_shp_path=dataset["salt"],
        output_dir=str(tmp_path / "out"),
        horizons=subset,
    )
    result = ngb.run_workflow(config)
    statuses = {res.name: res.scored for res in result["results"]}
    assert statuses == {"tsf1": True, "bj1": False}
    assert result["aggregation"].horizon_order == ["tsf1"]
    assert not os.path.exists(
        os.path.join(config.horizon_output_dir,
                     "bj1_geothermal_favourability.tif")
    )


def test_workflow_errors_when_no_horizon_can_be_scored(dataset, tmp_path):
    config = ngb.WorkflowConfig(
        nx=16, ny=16, target_epsg=EPSG,
        thermal_dir=str(tmp_path / "empty_thermal"),
        evidence_dir=dataset["evidence_dir"],
        basin_shp_path=dataset["basin"],
        salt_shp_path=dataset["salt"],
        output_dir=str(tmp_path / "out2"),
        horizons={"tsf1": dataset["horizons"]["tsf1"]},
    )
    result = ngb.run_workflow(config)
    assert result["aggregation"] is None
    assert os.path.exists(os.path.join(config.horizon_output_dir, "summary.csv"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_defaults():
    args = ngb.build_arg_parser().parse_args([])
    assert args.nx == ngb.NX and args.ny == ngb.NY
    assert args.epsg == 31467


def test_cli_horizon_subset():
    args = ngb.build_arg_parser().parse_args(["--horizons", "tsf1", "bj1"])
    cfg = ngb.config_from_args(args)
    assert list(cfg.horizons) == ["tsf1", "bj1"]


def test_cli_rejects_unknown_horizon():
    args = ngb.build_arg_parser().parse_args(["--horizons", "nonsense"])
    with pytest.raises(SystemExit):
        ngb.config_from_args(args)


def test_main_reports_error_for_missing_basin(tmp_path):
    code = ngb.main(
        [
            "--basin", str(tmp_path / "missing.shp"),
            "--output-dir", str(tmp_path / "out"),
            "--quiet",
        ]
    )
    assert code == 1
