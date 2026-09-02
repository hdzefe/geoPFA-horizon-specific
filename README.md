# geoPFA-horizon-specific
Modular horizon-specific geothermal favorability analysis using GeoPFA

## `ngb_refactored_horizon_specific_tif_input.py`

This script builds horizon-specific geothermal **favorability** maps by
combining **pre-computed thermal usefulness GeoTIFF rasters** (e.g. generated
from GeotiS temperature points + TUNB base surfaces + thickness correction)
with **reservoir potential shapefiles**, a basin outline shapefile, and a
salt-structure penalty shapefile.

Unlike `ngb_refactored_horizon_specific.py`, this variant does **not**
interpolate thermal point data on the fly — it assumes each horizon's
thermal layer already exists as a GeoTIFF (`{horizon}_thermal_use.tif`) and
loads it directly with `rasterio`, resampling to the configured grid
resolution only if needed.

### Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `ngb_refactored_horizon_specific_tif_input.py` and change **one
   line** to control the output resolution:
   ```python
   NX, NY = 320, 320  # Development: ~2-3 min
   # NX, NY = 512, 512  # Publication: ~15-20 min
   # NX, NY = 800, 800  # Ultra-high: ~30-40 min
   ```
3. Point the script at your data. All input/output locations can be
   overridden with environment variables so you never need to edit the
   defaults hard-coded in the script:

   | Environment variable | Purpose |
   |---|---|
   | `NGB_THERMAL_TIF_DIR` | Directory containing `{horizon}_thermal_use.tif` files |
   | `NGB_RESERVOIR_SHP_DIR` | Directory containing reservoir potential shapefiles |
   | `NGB_BASIN_SHP_PATH` | Path to the basin outline shapefile |
   | `NGB_SALT_SHP_PATH` | Path to the salt structures shapefile |
   | `NGB_OUTPUT_DIR` | Root output directory |
   | `NGB_TARGET_EPSG` | EPSG code used for shapefiles without a CRS (default `25832`) |
   | `NGB_SALT_BUFFER_M` | Buffer distance (m) around salt structures used for the penalty zone (default `2000`) |

4. Run it:
   ```bash
   python ngb_refactored_horizon_specific_tif_input.py
   ```

### Scoring logic (per horizon)

```
thermal_grid   = load {horizon}_thermal_use.tif, resampled to (NY, NX)
reservoir_grid = rasterize {horizon}-Potential.shp to (NY, NX), normalized [0, 1]
base_score     = thermal_grid * reservoir_grid
evidence_grid  = 0  (placeholder)
horizon_score  = confidence_weight * (0.85 * base_score + 0.15 * evidence_grid)
salt_penalty   = rasterized, buffered salt structures, [0, 1]
horizon_score_final = clip(horizon_score * (1 - salt_penalty), 0, 1)
```

### Outputs

Written under `OUTPUT_DIR/favourability/geothermal/horizon_specific/`:

- `{horizon}_geothermal_favourability.tif` — one per horizon
- `best_horizon_geothermal_favourability.tif` — elementwise max across horizons
- `PRIMARY_geothermal_favourability.tif` — top 10% of the best-horizon grid
- `SECONDARY_geothermal_favourability.tif` — top 25% of the best-horizon grid
- `best_horizon_id.tif` — index of the winning horizon per cell
- `best_horizon_id_mapping.csv` — index → horizon name/label lookup
- `summary.csv` — per-horizon mean/max/coverage stats and processing status

### Error handling

The workflow never crashes on missing input data:

- Missing thermal TIF → the horizon is skipped (logged as a warning, recorded
  as `skipped_no_thermal` in `summary.csv`).
- Missing reservoir shapefile → an all-zero reservoir grid is used (logged as
  a warning). Horizons `valanginian` and `bueckeberg` have no reservoir
  shapefile by design.
- Missing salt shapefile → no salt penalty is applied (logged as a warning).

### Tests

Run the test suite (uses synthetic in-memory GeoTIFF/shapefile fixtures, no
real project data required):

```bash
pytest tests/test_ngb_refactored_horizon_specific_tif_input.py -v
```
