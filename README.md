# geoPFA-horizon-specific

Modular horizon-specific geothermal favourability analysis for the North German
Basin.

`ngb_refactored_horizon_specific_tif_input.py` is a refactor of the original
GeoPFA production script. All geological logic is preserved; only two things
changed:

1. **Thermal source** — the thermal criterion is read from the pre-computed
   `{horizon}_thermal_use.tif` GeoTIFFs with `rasterio` instead of being
   interpolated from GeotiS point data.
2. **Aggregation** — every horizon is scored independently and the results are
   aggregated afterwards (best horizon, best-horizon-ID grid, PRIMARY/SECONDARY
   sweet spots) instead of a global `VoterVeto` combination.

All rasters are written in **EPSG:31467** (DHDN / Gauss-Krüger zone 3).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Development resolution (default `NX, NY = 320, 320`):

```bash
python ngb_refactored_horizon_specific_tif_input.py
```

Publication resolution:

```bash
python ngb_refactored_horizon_specific_tif_input.py --nx 512 --ny 512
```

Process a subset of horizons and write a log file:

```bash
python ngb_refactored_horizon_specific_tif_input.py --horizons detfurth tsf1 --log-file run.log
```

## Configuration

Paths are module constants that can be overridden with environment variables or
CLI switches, so the same script runs on the production machine and in CI.

| Environment variable | CLI switch | Default |
| --- | --- | --- |
| `NGB_PROJECT_ROOT` | – | `D:\geoPFA_Projects\north_german_basin` |
| `NGB_THERMAL_TIF_DIR` | `--thermal-dir` | `<root>\data\output\favourability\geothermal\horizon_specific` |
| `NGB_RAW_DATA_DIR` | – | `<root>\data\raw` |
| `NGB_EVIDENCE_SHP_DIR` | `--evidence-dir` | `<raw>\evidence` |
| `NGB_BASIN_SHP_PATH` | `--basin` | `<raw>\north_german_basin_extention\north_german_basin_.shp` |
| `NGB_SALT_SHP_PATH` | `--salt` | `<raw>\salt_structures\Salzstrukturen_Inspee__v1_poly.shp` |
| `NGB_OUTPUT_DIR` | `--output-dir` | `<root>\data\output` |
| `NGB_TARGET_EPSG` | `--epsg` | `31467` |
| `NGB_NX` / `NGB_NY` | `--nx` / `--ny` | `320` |
| `NGB_SALT_PENALTY` | – | `1.0` |
| `NGB_EVIDENCE_RADIUS` | – | `15000` (m) |

## Horizons

All 18 horizons are configured with their exact reservoir shapefile, attribute
column, value mapping and weights:

`detfurth`, `tsf1`, `tsf2`, `k42`, `k43`, `k44`, `het1`, `het2`, `sin1`,
`sin2`, `pli1`, `pli2`, `toa1`, `toa2`, `aal1`, `bj1`, `valanginian`,
`bueckeberg`.

Reservoir attributes are translated with the original mappings:

* `Potential` → `niedrig` / `mittel` / `hoch` (plus `eingeschränkt` for
  K4-2 and K4-3)
* `quality` / `reservoirq` / `reservoir` → `low (< 10 m)` /
  `moderate (10-20 m)` / `high (> 20 m)`
* `sand_share` (Detfurth isolines) → linear 30 %–90 %, multiplied by the
  `certainity` column
* `OBJECTID` (Valanginian) and `LEGNR` (Bückeberg) → sandstone class codes

Scoring per horizon:

```
base_score     = thermal * reservoir
horizon_score  = base_weight * base_score + evidence_weight * evidence
horizon_score *= confidence_weight
horizon_score *= (1 - salt_penalty)
```

A horizon whose thermal GeoTIFF or reservoir shapefile is missing is skipped
with a warning (and recorded in `summary.csv`) instead of aborting the run.

## Output

Written to `<OUTPUT_DIR>/favourability/geothermal/horizon_specific`:

```
detfurth_geothermal_favourability.tif
tsf1_geothermal_favourability.tif
...                                       (one per scored horizon)
best_horizon_geothermal_favourability.tif
PRIMARY_geothermal_favourability.tif      (top 10 %)
SECONDARY_geothermal_favourability.tif    (top 25 %)
best_horizon_id.tif
best_horizon_id_mapping.csv
summary.csv
```

## Tests

```bash
python -m pytest tests -q
```

The suite builds a synthetic dataset (basin, salt, thermal GeoTIFFs, reservoir
shapefiles, evidence points) and covers all 18 horizons, the value mappings,
missing data handling, aggregation, sweet-spot masks, exports and the CLI.
