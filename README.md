# geoPFA-horizon-specific
Modular horizon-specific geothermal favorability analysis using GeoPFA

## Overview

`ngb_refactored_horizon_specific.py` performs a horizon-specific geothermal
favorability analysis for the North German Basin (TUNB model), scoring each
of 17 target horizons independently (thermal × reservoir co-occurrence,
weighted by base/evidence/confidence weights, and salt-structure penalty),
then aggregates the results into a "best horizon" map plus PRIMARY/SECONDARY
sweetspot masks.

## Installation

```bash
pip install -r requirements.txt
```

`geopfa` requires the native GDAL library (`osgeo` Python bindings), which
is not pulled in automatically by `pip install geopfa`. On Debian/Ubuntu:

```bash
sudo apt-get install -y gdal-bin libgdal-dev
pip install "gdal==$(gdal-config --version)"
```

## Quick Start

1. Verify your workspace has thermal + reservoir layers in:
   - `WORKSPACE_DIR/geothermal/thermal_component/*.shp`
   - `WORKSPACE_DIR/geothermal/geologic_component/*.shp`

2. For **development** (quick testing):
   ```python
   NX, NY = 320, 320  # ~2-3 min runtime
   ```

3. For **publication** (high quality):
   ```python
   NX, NY = 512, 512  # ~15-20 min runtime
   ```

4. Set `WORKSPACE_DIR`/`OUTPUT_DIR` (either edit the constants at the top of
   the script, or set the `NGB_WORKSPACE_DIR`/`NGB_OUTPUT_DIR` environment
   variables) and run:
   ```bash
   python ngb_refactored_horizon_specific.py
   ```

5. Check results in:
   `OUTPUT_DIR/favourability/geothermal/horizon_specific/`

### Changing resolution

Edit the single `NX, NY` line near the top of the script:

```python
NX, NY = 320, 320  # development
NX, NY = 512, 512  # publication
NX, NY = 800, 800  # ultra-high
```

No other code changes are needed — all grid calculations auto-scale.

## Thermal Layer Handling

For each horizon the script first looks for a horizon-specific thermal
layer (`{horizon_name}_thermal_use.shp`). If it is not found, it prints a
warning and falls back to the regional `heat_flow_basin.shp` /
`heat_flow_germany.shp` layers. If no thermal data is available at all, the
horizon is skipped (marked in `summary.csv`) without stopping the rest of
the workflow.

## Salt Structures

The pre-processed `salt_sweetspot_0p5_2km.shp` distance-penalty layer
(0–1 scale, 0–2000 m linear ramp) is interpolated once and applied uniformly
to all horizons via `horizon_score × (1 - salt_penalty)`.

> Salt structures are penalized via a 2000 m linear distance decay function,
> uniformly applied across all horizons following standard PFA methodology
> (Pauling et al., 2023).

## Output Maps

- 17 individual horizon favorability maps (GeoTIFF)
- `best_horizon_geothermal_favourability.tif` — max favorability across horizons
- `PRIMARY_geothermal_favourability.tif` — top 10% sweetspots
- `SECONDARY_geothermal_favourability.tif` — top 25% backup zones
- `best_horizon_id.tif` — which horizon is best at each cell (int16)
- `best_horizon_id_mapping.csv` — horizon ID → name/label mapping
- `summary.csv` — per-horizon and aggregate statistics

All GeoTIFFs use CRS `EPSG:31467`, NoData `-9999.0`, LZW compression, and
`float32` data (except `best_horizon_id.tif`, which is `int16` with NoData
`0`). Scores range from 0.0 (unfavorable) to 1.0 (highly favorable).

## Testing

```bash
python -m pytest tests/ -v
```

The tests build a small synthetic workspace (basin outline, salt layer, and
a subset of horizon thermal/reservoir layers) and run the full workflow at
low resolution, validating output structure, graceful handling of missing
layers, and the aggregation/export logic.
