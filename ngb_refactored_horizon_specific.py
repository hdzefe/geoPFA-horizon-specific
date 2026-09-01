"""
HORIZON-SPECIFIC GEOTHERMAL FAVORABILITY ANALYSIS
For: North German Basin (TUNB Model)
Target: Scientific publication

QUICK START:
────────────
1. Verify workspace has thermal + reservoir layers in:
   - WORKSPACE_DIR/geothermal/thermal_component/*.shp
   - WORKSPACE_DIR/geothermal/geologic_component/*.shp

2. For DEVELOPMENT (quick testing):
   NX, NY = 320, 320  # ~2-3 min runtime

3. For PUBLICATION (high quality):
   NX, NY = 512, 512  # ~15-20 min runtime

4. Run:
   python ngb_refactored_horizon_specific.py

5. Check results:
   OUTPUT_DIR/favourability/geothermal/horizon_specific/

CHANGING RESOLUTION:
────────────────────
Edit the line with NX, NY:
  NX, NY = 320, 320  (development)
  NX, NY = 512, 512  (publication)
  NX, NY = 800, 800  (ultra-high)

No other code changes needed!

OUTPUT MAPS:
────────────
- 17 individual horizon favorability maps (GeoTIFF)
- best_horizon_geothermal_favourability.tif (max across horizons)
- PRIMARY_geothermal_favourability.tif (top 10% sweetspots)
- SECONDARY_geothermal_favourability.tif (top 25% backup)
- best_horizon_id.tif (which horizon is best at each cell)
- best_horizon_id_mapping.csv
- summary.csv (overall statistics)

All maps have:
- CRS: EPSG:31467
- Scale: 0.0 (unfavorable) to 1.0 (highly favorable)
- NoData: -9999
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds

from geopfa.processing import Processing

# ============================================================================
# WORKSPACE / OUTPUT CONFIGURATION
# ============================================================================
# Root directory containing the "geothermal" workspace with
# thermal_component/ and geologic_component/ sub-directories.
WORKSPACE_DIR = os.environ.get("NGB_WORKSPACE_DIR", "./workspace")

# Root directory where results are written. The script creates:
#   OUTPUT_DIR/favourability/geothermal/horizon_specific/
OUTPUT_DIR = os.environ.get("NGB_OUTPUT_DIR", "./output")

THERMAL_COMPONENT_DIR = os.path.join(WORKSPACE_DIR, "geothermal", "thermal_component")
GEOLOGIC_COMPONENT_DIR = os.path.join(WORKSPACE_DIR, "geothermal", "geologic_component")

BASIN_OUTLINE_LAYER = "north_german_basin_"
SALT_LAYER = "salt_sweetspot_0p5_2km"

# Fallback thermal layers (regional), tried in this order if a horizon-
# specific thermal layer is not found.
FALLBACK_THERMAL_LAYERS = ["heat_flow_basin", "heat_flow_germany"]

# ============================================================================
# RESOLUTION CONFIGURATION (Easy to Change)
# ============================================================================
# Development: 320x320 (quick testing, ~2-3 min)
# Publication: 512x512 (high quality, ~15-20 min)
# Ultra-high: 800x800 (maximum detail, ~30-40 min)

NX, NY = 320, 320  # <- Change this ONE line for different resolution

# No other code modifications needed - all calculations auto-scale!

# ============================================================================
# CRS / EXPORT CONFIGURATION
# ============================================================================
OUTPUT_CRS = "EPSG:31467"
NODATA_VALUE = -9999.0

# ============================================================================
# HORIZON CONFIGURATION
# ============================================================================
HORIZONS = {
    # PRODUCING HYDROTHERMAL SYSTEMS
    "detfurth": {
        "label": "Detfurth Sandstone (Middle Bunter)",
        "thermal_layer": "detfurth_thermal_use",
        "reservoir_layer": "sand_zones_detfurth_",  # NOTE: used instead of detfurth_sand_quality
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "k44": {
        "label": "K4-4 Contorta Sandstone (Rhaetian - Exter Fm)",
        "thermal_layer": "k44_thermal_use",
        "reservoir_layer": "k4-4-Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "k43": {
        "label": "K4-3 Postera Sandstone (Rhaetian - Exter Fm)",
        "thermal_layer": "k43_thermal_use",
        "reservoir_layer": "k4-3-Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "k42": {
        "label": "K4-2 Postera Sandstone (Rhaetian - Exter Fm)",
        "thermal_layer": "k42_thermal_use",
        "reservoir_layer": "k4-2-Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "aal1": {
        "label": "AAL1 Aalenian",
        "thermal_layer": "aal1_thermal_use",
        "reservoir_layer": "aal1-Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    # BALNEOLOGY (SPA/WELLNESS) SYSTEMS
    "het1": {
        "label": "HET1 Hettangian",
        "thermal_layer": "het1_thermal_use",
        "reservoir_layer": "Het1_Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "het2": {
        "label": "HET2 Hettangian",
        "thermal_layer": "het2_thermal_use",
        "reservoir_layer": "Het2_Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "sin1": {
        "label": "SIN1 Sinemurian",
        "thermal_layer": "sin1_thermal_use",
        "reservoir_layer": "Sin1_Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "sin2": {
        "label": "SIN2 Sinemurian",
        "thermal_layer": "sin2_thermal_use",
        "reservoir_layer": "Sin2_Potential",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    # POTENTIAL SYSTEMS
    "tsf2": {
        "label": "TSF2 Upper Schilfsandstein",
        "thermal_layer": "tsf2_thermal_use",
        "reservoir_layer": "Tsf2-Potential",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "tsf1": {
        "label": "TSF1 Lower Schilfsandstein",
        "thermal_layer": "tsf1_thermal_use",
        "reservoir_layer": "tsf1-Potential",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "pli1": {
        "label": "PLI1 Pliensbachian",
        "thermal_layer": "pli1_thermal_use",
        "reservoir_layer": "Pli1_Potential",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.8,
    },
    "pli2": {
        "label": "PLI2 Pliensbachian",
        "thermal_layer": "pli2_thermal_use",
        "reservoir_layer": "Pli2_Potential",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.8,
    },
    "toa1": {
        "label": "TOA1 Toarcian",
        "thermal_layer": "toa1_thermal_use",
        "reservoir_layer": "toa1-Potential",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.8,
    },
    "toa2": {
        "label": "TOA2 Toarcian",
        "thermal_layer": "toa2_thermal_use",
        "reservoir_layer": "toa2-Potential",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.8,
    },
    "bj1": {
        "label": "BJ1 Bajocian",
        "thermal_layer": "bj1_thermal_use",
        "reservoir_layer": "bj1-Potential",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.6,
    },
    "valanginian": {
        "label": "Valanginian (Lower Cretaceous)",
        "thermal_layer": "valanginian_thermal_use",
        "reservoir_layer": "valanginian_sandstone",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.5,
    },
    "bueckeberg": {
        "label": "Bückeberg Group (Lower Cretaceous)",
        "thermal_layer": "bueckeberg_thermal_use",
        "reservoir_layer": "bueckeberg_group_sandstone_lbeg",
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.5,
    },
}

HORIZON_NAMES = list(HORIZONS.keys())


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def find_layer_path(component_dir, layer_name):
    """Locate a shapefile for `layer_name` inside `component_dir`.

    Tries an exact match first (`{layer_name}.shp`), then falls back to a
    case-insensitive search. Returns None if nothing is found.
    """
    exact_path = os.path.join(component_dir, f"{layer_name}.shp")
    if os.path.isfile(exact_path):
        return exact_path

    if not os.path.isdir(component_dir):
        return None

    target = f"{layer_name}.shp".lower()
    for fname in os.listdir(component_dir):
        if fname.lower() == target:
            return os.path.join(component_dir, fname)
    return None


def load_layer_gdf(component_dir, layer_name):
    """Load a shapefile layer as a GeoDataFrame, or return None if missing."""
    path = find_layer_path(component_dir, layer_name)
    if path is None:
        return None
    try:
        return gpd.read_file(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ⚠ Failed to read '{layer_name}' from {path}: {exc}")
        return None


def pick_data_col(gdf, preferred_cols=("value", "score", "tuse", "reservoir_score")):
    """Pick the numeric column to interpolate from a GeoDataFrame.

    Prefers well-known column names, otherwise falls back to the first
    numeric, non-geometry column.
    """
    for col in preferred_cols:
        if col in gdf.columns:
            return col
    for col in gdf.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_numeric_dtype(gdf[col]):
            return col
    return None


def normalize_grid(grid):
    """Min-max normalize a 2D array to [0, 1], ignoring NaNs."""
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    gmin, gmax = np.nanmin(finite), np.nanmax(finite)
    if gmax - gmin <= 0:
        return np.where(np.isfinite(grid), 0.0, np.nan)
    return (grid - gmin) / (gmax - gmin)


def interpolate_layer_to_grid(gdf, data_col, nx, ny, extent, interp_method="linear"):
    """Interpolate a point/polygon layer to an (ny, nx) grid using geoPFA.

    Builds a minimal `pfa` dict as expected by
    `geopfa.processing.Processing.interpolate_points`, runs the
    interpolation, and reshapes the resulting values back into a 2D grid.
    """
    pfa = {
        "criteria": {
            "geothermal": {
                "components": {
                    "layer_component": {
                        "layers": {
                            "layer": {
                                "data": gdf,
                                "data_col": data_col,
                                "units": "score_0_1",
                            }
                        }
                    }
                }
            }
        }
    }
    pfa = Processing.interpolate_points(
        pfa=pfa,
        criteria="geothermal",
        component="layer_component",
        layer="layer",
        interp_method=interp_method,
        nx=nx,
        ny=ny,
        extent=extent,
    )
    model_gdf = pfa["criteria"]["geothermal"]["components"]["layer_component"][
        "layers"
    ]["layer"]["model"]
    model_col = pfa["criteria"]["geothermal"]["components"]["layer_component"][
        "layers"
    ]["layer"]["model_data_col"]
    grid = model_gdf[model_col].to_numpy().reshape(ny, nx)
    return grid


def get_basin_extent(basin_gdf, fallback_extent=None):
    """Return [xmin, ymin, xmax, ymax] from a basin outline GeoDataFrame."""
    if basin_gdf is not None and len(basin_gdf) > 0:
        bounds = basin_gdf.total_bounds
        return [float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])]
    if fallback_extent is not None:
        print("  ⚠ Basin outline not found - using fallback extent")
        return list(fallback_extent)
    raise RuntimeError(
        "Basin outline layer not found and no fallback extent provided. "
        "Cannot determine grid extent."
    )


def load_thermal_grid(horizon_name, horizon_cfg, nx, ny, extent):
    """Flexible thermal layer loading with graceful fallback.

    Order of attempts:
      1. Horizon-specific layer `{horizon_name}_thermal_use.shp`
      2. Regional fallback layers (`heat_flow_basin`, `heat_flow_germany`)
      3. None (horizon will be skipped downstream)
    """
    thermal_layer_name = horizon_cfg["thermal_layer"]
    gdf = load_layer_gdf(THERMAL_COMPONENT_DIR, thermal_layer_name)
    if gdf is not None:
        data_col = pick_data_col(gdf)
        if data_col is not None:
            grid = interpolate_layer_to_grid(gdf, data_col, nx, ny, extent)
            print(f"  ✓ {horizon_name} thermal loaded ({thermal_layer_name})")
            return normalize_grid(grid), thermal_layer_name

    print(f"  ⚠ {horizon_name} thermal NOT FOUND - using fallback")
    for fallback_name in FALLBACK_THERMAL_LAYERS:
        gdf = load_layer_gdf(THERMAL_COMPONENT_DIR, fallback_name)
        if gdf is None:
            continue
        data_col = pick_data_col(gdf)
        if data_col is None:
            continue
        grid = interpolate_layer_to_grid(gdf, data_col, nx, ny, extent)
        print(f"    → using regional fallback '{fallback_name}'")
        return normalize_grid(grid), fallback_name

    print(f"    → no thermal data available for {horizon_name}; skipping horizon")
    return None, None


def compute_salt_penalty(nx, ny, extent):
    """Load and interpolate the salt sweetspot penalty layer to the grid.

    Returns a 0-1 scaled penalty grid (0 = no penalty, 1 = full veto).
    Returns an all-zero grid (no penalty applied) if the layer is missing.
    """
    gdf = load_layer_gdf(GEOLOGIC_COMPONENT_DIR, SALT_LAYER)
    if gdf is None:
        print(f"  ⚠ Salt layer '{SALT_LAYER}' NOT FOUND - no salt penalty applied")
        return np.zeros((ny, nx), dtype=float)

    data_col = pick_data_col(gdf)
    if data_col is None:
        print(f"  ⚠ Salt layer '{SALT_LAYER}' has no usable numeric column - no salt penalty applied")
        return np.zeros((ny, nx), dtype=float)

    grid = interpolate_layer_to_grid(gdf, data_col, nx, ny, extent)
    grid = normalize_grid(grid)
    grid = np.where(np.isfinite(grid), grid, 0.0)
    return np.clip(grid, 0.0, 1.0)


def score_horizon(thermal_grid, reservoir_grid, base_weight, evidence_weight, confidence_weight, salt_penalty):
    """Compute the final horizon favorability grid.

    Follows the exact formula:
      base_score = thermal * reservoir
      evidence_grid = 0 (placeholder for future evidence layers)
      horizon_score = base_weight * base_score + evidence_weight * evidence_grid
      horizon_score = confidence_weight * horizon_score
      horizon_score_final = horizon_score * (1 - salt_penalty)
      horizon_score_final = clip(horizon_score_final, 0, 1)
    """
    base_score = thermal_grid * reservoir_grid
    evidence_grid = np.zeros_like(base_score)
    horizon_score = base_weight * base_score + evidence_weight * evidence_grid
    horizon_score = confidence_weight * horizon_score
    horizon_score_final = horizon_score * (1.0 - salt_penalty)
    return np.clip(horizon_score_final, 0.0, 1.0)


def export_geotiff(array, path, extent, crs=OUTPUT_CRS, nodata=NODATA_VALUE):
    """Export a 2D array to a GeoTIFF with the required publication settings."""
    ny, nx = array.shape
    transform = from_bounds(extent[0], extent[1], extent[2], extent[3], nx, ny)
    out_array = np.where(np.isfinite(array), array, nodata).astype("float32")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(out_array, 1)


def export_int_geotiff(array, path, extent, crs=OUTPUT_CRS, nodata=0):
    """Export a 2D int16 array (e.g., horizon ID grid) to a GeoTIFF."""
    ny, nx = array.shape
    transform = from_bounds(extent[0], extent[1], extent[2], extent[3], nx, ny)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="int16",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(array.astype("int16"), 1)


def safe_stack_argmax(horizon_stack):
    """Compute per-cell best horizon value & 1-indexed horizon id, NaN-safe.

    Returns (best_value_grid, best_index_grid) where best_index_grid is 0
    where no horizon has a valid (finite, > 0) score.
    """
    valid_mask = np.isfinite(horizon_stack) & (horizon_stack > 0)
    any_valid = valid_mask.any(axis=0)

    filled = np.where(valid_mask, horizon_stack, -np.inf)
    best_index = np.argmax(filled, axis=0) + 1
    best_index = np.where(any_valid, best_index, 0).astype("int16")

    best_value = np.where(any_valid, np.max(filled, axis=0), np.nan)

    return best_value, best_index


# ============================================================================
# MAIN WORKFLOW
# ============================================================================
def run_horizon_workflow(
    workspace_dir=None,
    output_dir=None,
    nx=None,
    ny=None,
    fallback_extent=None,
):
    """Run the full horizon-specific geothermal favorability workflow.

    Parameters
    ----------
    workspace_dir : str, optional
        Overrides module-level WORKSPACE_DIR.
    output_dir : str, optional
        Overrides module-level OUTPUT_DIR.
    nx, ny : int, optional
        Overrides module-level NX, NY grid resolution.
    fallback_extent : list, optional
        [xmin, ymin, xmax, ymax] used if the basin outline layer is missing.

    Returns
    -------
    dict
        Summary of results including output directory and per-horizon stats.
    """
    global THERMAL_COMPONENT_DIR, GEOLOGIC_COMPONENT_DIR

    workspace_dir = workspace_dir or WORKSPACE_DIR
    output_dir = output_dir or OUTPUT_DIR
    nx = nx or NX
    ny = ny or NY

    THERMAL_COMPONENT_DIR = os.path.join(workspace_dir, "geothermal", "thermal_component")
    GEOLOGIC_COMPONENT_DIR = os.path.join(workspace_dir, "geothermal", "geologic_component")

    horizon_out_dir = os.path.join(
        output_dir, "favourability", "geothermal", "horizon_specific"
    )
    os.makedirs(horizon_out_dir, exist_ok=True)

    start_time = time.time()

    print("=" * 60)
    print("[STEP 1] Determining grid extent from basin outline...")
    basin_gdf = load_layer_gdf(GEOLOGIC_COMPONENT_DIR, BASIN_OUTLINE_LAYER)
    extent = get_basin_extent(basin_gdf, fallback_extent=fallback_extent)
    print(f"  Extent: {extent}")
    print(f"  Grid resolution: {nx} x {ny}")

    print("\n[STEP 2] Loading and interpolating salt structure penalty...")
    salt_penalty = compute_salt_penalty(nx, ny, extent)
    print(
        f"  Salt penalty range: [{np.nanmin(salt_penalty):.3f}, "
        f"{np.nanmax(salt_penalty):.3f}]"
    )

    print("\n[STEP 3] Scoring Horizons...")
    print("-" * 45)
    horizon_results = {}
    horizon_diagnostics = []

    for horizon_name, horizon_cfg in HORIZONS.items():
        h_start = time.time()
        thermal_grid, thermal_source = load_thermal_grid(
            horizon_name, horizon_cfg, nx, ny, extent
        )

        reservoir_gdf = load_layer_gdf(
            GEOLOGIC_COMPONENT_DIR, horizon_cfg["reservoir_layer"]
        )
        reservoir_grid = None
        if reservoir_gdf is not None:
            data_col = pick_data_col(reservoir_gdf)
            if data_col is not None:
                reservoir_grid = normalize_grid(
                    interpolate_layer_to_grid(reservoir_gdf, data_col, nx, ny, extent)
                )
        if reservoir_grid is None:
            print(
                f"  ⚠ {horizon_name}: reservoir layer "
                f"'{horizon_cfg['reservoir_layer']}' NOT FOUND - skipping horizon"
            )

        if thermal_grid is None or reservoir_grid is None:
            horizon_results[horizon_name] = np.full((ny, nx), np.nan)
            horizon_diagnostics.append(
                {
                    "horizon": horizon_name,
                    "status": "skipped",
                    "thermal_source": thermal_source,
                    "min": np.nan,
                    "max": np.nan,
                    "nonzero": 0,
                    "elapsed_s": round(time.time() - h_start, 2),
                }
            )
            continue

        h_score = score_horizon(
            thermal_grid,
            reservoir_grid,
            horizon_cfg["base_weight"],
            horizon_cfg["evidence_weight"],
            horizon_cfg["confidence_weight"],
            salt_penalty,
        )
        horizon_results[horizon_name] = h_score

        nonzero = int(np.nansum(h_score > 0))
        print(
            f"  ✓ {horizon_name}: min={np.nanmin(h_score):.3f}, "
            f"max={np.nanmax(h_score):.3f}, nonzero={nonzero}/{h_score.size}"
        )
        horizon_diagnostics.append(
            {
                "horizon": horizon_name,
                "status": "ok",
                "thermal_source": thermal_source,
                "min": float(np.nanmin(h_score)),
                "max": float(np.nanmax(h_score)),
                "nonzero": nonzero,
                "elapsed_s": round(time.time() - h_start, 2),
            }
        )

        out_path = os.path.join(
            horizon_out_dir, f"{horizon_name}_geothermal_favourability.tif"
        )
        export_geotiff(h_score, out_path, extent)

    print("\n[STEP 4] Aggregating horizon results...")
    print("-" * 45)
    horizon_stack = np.stack(
        [horizon_results[h] for h in HORIZON_NAMES], axis=0
    )  # Shape: (17, ny, nx)

    best_horizon_geothermal, best_horizon_index = safe_stack_argmax(horizon_stack)

    finite_scores = best_horizon_geothermal[
        np.isfinite(best_horizon_geothermal) & (best_horizon_geothermal > 0)
    ]
    valid_cells = int(np.isfinite(best_horizon_geothermal).sum())
    total_cells = best_horizon_geothermal.size
    print("Aggregation Results:")
    print(
        f"  min={np.nanmin(best_horizon_geothermal) if finite_scores.size else float('nan'):.3f}, "
        f"median={np.nanmedian(best_horizon_geothermal) if finite_scores.size else float('nan'):.3f}, "
        f"max={np.nanmax(best_horizon_geothermal) if finite_scores.size else float('nan'):.3f}"
    )
    print(
        f"  cells with valid horizon: {valid_cells}/{total_cells} "
        f"({100.0 * valid_cells / total_cells:.0f}%)"
    )

    if finite_scores.size > 0:
        primary_threshold = float(np.quantile(finite_scores, 0.90))
        primary = np.where(
            best_horizon_geothermal >= primary_threshold, best_horizon_geothermal, np.nan
        )
        secondary_threshold = float(np.quantile(finite_scores, 0.75))
        secondary = np.where(
            best_horizon_geothermal >= secondary_threshold, best_horizon_geothermal, np.nan
        )
    else:
        primary_threshold = float("nan")
        secondary_threshold = float("nan")
        primary = np.full_like(best_horizon_geothermal, np.nan)
        secondary = np.full_like(best_horizon_geothermal, np.nan)

    print("\nPRIMARY (top 10%):")
    print(f"  threshold={primary_threshold:.3f}")
    print(f"  cells: {int(np.isfinite(primary).sum())}")
    print("\nSECONDARY (top 25%):")
    print(f"  threshold={secondary_threshold:.3f}")
    print(f"  cells: {int(np.isfinite(secondary).sum())}")

    print("\n[STEP 5] Exporting aggregated outputs...")
    export_geotiff(
        best_horizon_geothermal,
        os.path.join(horizon_out_dir, "best_horizon_geothermal_favourability.tif"),
        extent,
    )
    export_geotiff(
        primary, os.path.join(horizon_out_dir, "PRIMARY_geothermal_favourability.tif"), extent
    )
    export_geotiff(
        secondary,
        os.path.join(horizon_out_dir, "SECONDARY_geothermal_favourability.tif"),
        extent,
    )
    export_int_geotiff(
        best_horizon_index, os.path.join(horizon_out_dir, "best_horizon_id.tif"), extent
    )

    mapping_df = pd.DataFrame(
        {
            "horizon_id": [0] + [i + 1 for i in range(len(HORIZON_NAMES))],
            "horizon_name": ["none"] + HORIZON_NAMES,
            "label": ["No valid horizon"]
            + [HORIZONS[h]["label"] for h in HORIZON_NAMES],
        }
    )
    mapping_df.to_csv(
        os.path.join(horizon_out_dir, "best_horizon_id_mapping.csv"), index=False
    )

    summary_df = pd.DataFrame(horizon_diagnostics)
    summary_df.loc["aggregate"] = {
        "horizon": "best_horizon",
        "status": "ok" if finite_scores.size else "no_valid_data",
        "thermal_source": None,
        "min": float(np.nanmin(best_horizon_geothermal)) if finite_scores.size else np.nan,
        "max": float(np.nanmax(best_horizon_geothermal)) if finite_scores.size else np.nan,
        "nonzero": int(valid_cells),
        "elapsed_s": round(time.time() - start_time, 2),
    }
    summary_df.to_csv(os.path.join(horizon_out_dir, "summary.csv"), index=False)

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] Total runtime: {total_elapsed:.1f}s")
    print(f"[DONE] Outputs written to: {horizon_out_dir}")

    return {
        "output_dir": horizon_out_dir,
        "extent": extent,
        "horizon_diagnostics": horizon_diagnostics,
        "primary_threshold": primary_threshold,
        "secondary_threshold": secondary_threshold,
        "elapsed_s": total_elapsed,
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_horizon_workflow()
