# geoPFA-horizon-specific

Modular horizon-specific geothermal favourability analysis using GeoPFA.

`ngb_horizon_specific_voterveto.py` builds the geothermal favourability of the
North German Basin **per horizon** and only afterwards aggregates the horizon
results into composite products. All data and products use **EPSG:31467**
(Gauss-Krüger zone 3).

## Workflow

1. **Per horizon** (all 18 horizons):
   * load `{thermal_layer}.tif` from `THERMAL_TIF_DIR` (rasterio, resampled to
     the model grid),
   * load and rasterize the horizon reservoir shapefile using the horizon's
     attribute column and value mapping,
   * load the horizon evidence layers (rasters, polygons or points; point
     layers become a distance-decay evidence grid),
   * combine everything with `geopfa.layer_combination.VoterVeto` using the
     horizon `base_weight` / `evidence_weight`, then scale by
     `confidence_weight`,
   * export `{horizon}_geothermal_favourability.tif`.
2. **Aggregate** the horizon grids:
   * `best_horizon_geothermal_favourability.tif` — composite (max across horizons),
   * `PRIMARY_geothermal_favourability.tif` — top 10 % of the composite,
   * `SECONDARY_geothermal_favourability.tif` — top 25 % of the composite,
   * `best_horizon_id.tif` — 1-based id of the best horizon per cell (0 = nodata),
   * `best_horizon_id_mapping.csv` and `summary.csv`.

All products are written to
`OUTPUT_DIR/favourability/geothermal/horizon_specific/`.

## Horizons

| id | horizon | reservoir attribute | value mapping | base / evidence / confidence |
| -- | ------- | ------------------- | ------------- | ---------------------------- |
| 1 | detfurth | `sand_share` (+ `certainity`) | linear 30–90 % | 0.85 / 0.15 / 1.0 |
| 2 | tsf1 | `Potential` | niedrig/mittel/hoch | 0.90 / 0.10 / 1.0 |
| 3 | tsf2 | `Potential` | niedrig/mittel/hoch | 0.90 / 0.10 / 1.0 |
| 4 | k42 | `Potential` | + eingeschränkt | 0.90 / 0.10 / 1.0 |
| 5 | k43 | `Potential` | + eingeschränkt | 0.90 / 0.10 / 1.0 |
| 6 | k44 | `Potential` | niedrig/mittel/hoch | 0.90 / 0.10 / 1.0 |
| 7 | het1 | `quality` | thickness classes (EN) | 0.90 / 0.10 / 1.0 |
| 8 | het2 | `quality` | thickness classes (EN) | 0.90 / 0.10 / 1.0 |
| 9 | sin1 | `reservoirq` | thickness classes (EN) | 0.90 / 0.10 / 1.0 |
| 10 | sin2 | `reservoirq` | thickness classes (EN) | 0.90 / 0.10 / 1.0 |
| 11 | pli1 | `reservoirq` | thickness classes (EN) | 0.90 / 0.10 / 1.0 |
| 12 | pli2 | `reservoir` | thickness classes (EN) | 0.90 / 0.10 / 1.0 |
| 13 | toa1 | `Potential` | niedrig/mittel/hoch | 0.90 / 0.10 / 1.0 |
| 14 | toa2 | `Potential` | niedrig/mittel/hoch | 0.90 / 0.10 / 1.0 |
| 15 | aal1 | `Potential` | niedrig/mittel/hoch | 0.90 / 0.10 / 1.0 |
| 16 | bj1 | `Potential` | niedrig/mittel/hoch | 0.90 / 0.10 / 1.0 |
| 17 | valanginian | `OBJECTID` | sandstone codes 1/2/3 | 0.90 / 0.10 / 0.7 |
| 18 | bueckeberg | `LEGNR` | sandstone codes 2/3/7/11 | 0.90 / 0.10 / 0.7 |

`python ngb_horizon_specific_voterveto.py --list-horizons` prints the full
configuration.

## Configuration

Every input/output location is a module constant that can be overridden with an
environment variable:

| variable | meaning |
| -------- | ------- |
| `NGB_THERMAL_TIF_DIR` | directory with the `*_thermal_use.tif` rasters |
| `NGB_RAW_DATA_DIR` | root of the raw vector data |
| `NGB_RESERVOIR_SHP_DIR` | reservoir shapefiles |
| `NGB_EVIDENCE_DIR` | evidence layers |
| `NGB_BASIN_SHP_PATH` | basin outline (extent + valid mask) |
| `NGB_SALT_SHP_PATH` | salt structures (vetoed) |
| `NGB_OUTPUT_DIR` | root output directory |
| `NGB_GRID_NX`, `NGB_GRID_NY` | model grid size |
| `NGB_TARGET_EPSG` | project CRS (default `31467`) |
| `NGB_PRIMARY_QUANTILE`, `NGB_SECONDARY_QUANTILE` | sweetspot quantiles |
| `NGB_EVIDENCE_SEARCH_RADIUS_M` | radius for point-evidence decay |

## Usage

```bash
pip install -r requirements.txt

# all 18 horizons
python ngb_horizon_specific_voterveto.py

# a subset, custom grid and output location
python ngb_horizon_specific_voterveto.py --horizons detfurth tsf1 --nx 400 --ny 400 \
    --output-dir D:\geoPFA_Projects\north_german_basin\data\output
```

Horizons whose thermal raster is missing are skipped with a warning and listed
in the run summary.

## Tests

```bash
pip install -r requirements.txt pytest
pytest
```
