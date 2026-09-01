"""
ngb_refactored_horizon_specific.py
===================================

Horizon-specific geothermal favorability analysis for the North German Basin
(NGB) geoPFA workflow.

This script replaces the previous monolithic workflow (``ngb_1.txt`` /
``ngb_2.txt``), which combined all 17+ reservoir horizons globally using a
single ``VoterVeto.do_voter_veto()`` aggregation step. That approach made it
impossible to see individual horizon favorability, hid backup reservoir
options, and was limited to a coarse 160x160 grid.

Instead, this workflow:

1. Loads the raw thermal and reservoir shapefiles for each horizon from the
   workspace directory.
2. Interpolates each layer onto a shared model grid using
   ``geopfa.processing.Processing.interpolate_points()`` (2D only).
3. Scores each of the (up to) 17 reservoir horizons *independently*:
   ``horizon_score = base_weight * (thermal * reservoir) + evidence_weight * evidence``
   ``horizon_score_final = confidence_weight * horizon_score``
   (the global ``VoterVeto.do_voter_veto()`` step is intentionally skipped).
4. Exports one GeoTIFF favorability map per horizon.
5. Aggregates across horizons (best-horizon max + optional weighted mean) to
   produce a composite "best available reservoir" map, plus PRIMARY
   (top 10%) and SECONDARY (top 25%) sweet-spot / backup maps.
6. Exports summary statistics and a horizon-id -> horizon-name mapping.

Usage
-----
1. Confirm all thermal and reservoir shapefiles exist under:
     ``WORKSPACE_DIR/geothermal/thermal_component/*.shp``
     ``WORKSPACE_DIR/geothermal/geologic_component/*.shp``
2. Verify the horizon names in ``HORIZONS`` below match your shapefile
   naming convention (each shapefile name must contain the horizon's
   ``thermal_key`` / ``reservoir_key`` substring).
3. Adjust ``NX`` / ``NY`` for the desired output resolution (e.g. 320, 400,
   512). Higher resolution increases interpolation time (roughly 2-5x for
   512x512 vs. the legacy 160x160 grid).
4. Run: ``python ngb_refactored_horizon_specific.py``
5. Inspect ``OUTPUT_DIR/favourability/geothermal/horizon_specific`` for the
   resulting GeoTIFFs and CSV summaries.
"""

from __future__ import annotations

import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

from geopfa.processing import Processing

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Root directory containing the raw geoPFA workspace (thermal + geologic
# shapefiles). Update this to point at your workspace.
WORKSPACE_DIR = "./workspace"

# Root directory where favorability outputs will be written.
OUTPUT_DIR = "./output"

# Model grid resolution. Increase these for higher-resolution (less
# pixelated) output maps. The legacy workflow used 160x160.
NX = 320
NY = 320

# Coordinate reference system for the North German Basin.
TARGET_CRS = "EPSG:31467"

# Optional manual override of the model grid extent as
# (x_min, y_min, x_max, y_max). If None, the extent is computed automatically
# from the union of bounds of all discovered horizon layers.
EXTENT_OVERRIDE = None

# Interpolation method passed to Processing.interpolate_points().
INTERP_METHOD = "linear"

# Quantile thresholds used to build the PRIMARY / SECONDARY sweet-spot maps.
PRIMARY_QUANTILE = 0.90
SECONDARY_QUANTILE = 0.75

THERMAL_DIR = os.path.join(WORKSPACE_DIR, "geothermal", "thermal_component")
RESERVOIR_DIR = os.path.join(WORKSPACE_DIR, "geothermal", "geologic_component")
EVIDENCE_DIR = os.path.join(
    WORKSPACE_DIR, "geothermal", "geothermal_evidence_component"
)

HORIZON_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR, "favourability", "geothermal", "horizon_specific"
)

# ---------------------------------------------------------------------------
# HORIZONS CONFIGURATION
# ---------------------------------------------------------------------------
# Each horizon entry defines:
#   label              : human readable name
#   thermal_key         : substring used to locate the thermal shapefile in
#                          THERMAL_DIR (e.g. "*<thermal_key>*.shp")
#   reservoir_key        : substring used to locate the reservoir shapefile in
#                          RESERVOIR_DIR
#   thermal_data_col     : column in the thermal shapefile holding the score
#   reservoir_data_col   : column in the reservoir shapefile holding the score
#   evidence_layers      : list of evidence shapefile key substrings
#                          (searched in EVIDENCE_DIR). Empty by default --
#                          this is a placeholder the user can populate from
#                          their evidence manifest.
#   evidence_data_col    : column in evidence shapefiles holding the score
#   base_weight          : weight applied to (thermal * reservoir)
#   evidence_weight      : weight applied to the evidence term
#   confidence_weight    : overall confidence multiplier for this horizon
#   aggregation_weight   : weight used when combining horizons via a
#                          weighted mean (higher = more proven/producing)
HORIZONS = {
    "detfurth": {
        "label": "Detfurth Sandstone",
        "thermal_key": "detfurth",
        "reservoir_key": "detfurth",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 2.0,
    },
    "tsf1": {
        "label": "Schilfsandstein Upper",
        "thermal_key": "tsf1",
        "reservoir_key": "tsf1",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.0,
    },
    "tsf2": {
        "label": "Schilfsandstein Lower",
        "thermal_key": "tsf2",
        "reservoir_key": "tsf2",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.0,
    },
    "k42": {
        "label": "Exter Formation K4-2",
        "thermal_key": "k42",
        "reservoir_key": "k42",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 2.0,
    },
    "k43": {
        "label": "Exter Formation K4-3",
        "thermal_key": "k43",
        "reservoir_key": "k43",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 2.0,
    },
    "k44": {
        "label": "Exter Formation K4-4",
        "thermal_key": "k44",
        "reservoir_key": "k44",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 2.0,
    },
    "het1": {
        "label": "Hettangian Upper",
        "thermal_key": "het1",
        "reservoir_key": "het1",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.5,
    },
    "het2": {
        "label": "Hettangian Lower",
        "thermal_key": "het2",
        "reservoir_key": "het2",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.5,
    },
    "sin1": {
        "label": "Sinemurian Upper",
        "thermal_key": "sin1",
        "reservoir_key": "sin1",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.5,
    },
    "sin2": {
        "label": "Sinemurian Lower",
        "thermal_key": "sin2",
        "reservoir_key": "sin2",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.5,
    },
    "pli1": {
        "label": "Pliensbachian Upper",
        "thermal_key": "pli1",
        "reservoir_key": "pli1",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 0.5,
    },
    "pli2": {
        "label": "Pliensbachian Lower",
        "thermal_key": "pli2",
        "reservoir_key": "pli2",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 0.5,
    },
    "toa1": {
        "label": "Toarcian Upper",
        "thermal_key": "toa1",
        "reservoir_key": "toa1",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.0,
    },
    "toa2": {
        "label": "Toarcian Lower",
        "thermal_key": "toa2",
        "reservoir_key": "toa2",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 1.0,
    },
    "aal1": {
        "label": "Aalenian",
        "thermal_key": "aal1",
        "reservoir_key": "aal1",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 2.0,
    },
    "bj1": {
        "label": "Bajocian",
        "thermal_key": "bj1",
        "reservoir_key": "bj1",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 0.5,
    },
    "valanginian": {
        "label": "Valanginian",
        "thermal_key": "valanginian",
        "reservoir_key": "valanginian",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 0.5,
    },
    "bueckeberg": {
        "label": "Bueckeberg Group",
        "thermal_key": "bueckeberg",
        "reservoir_key": "bueckeberg",
        "thermal_data_col": "tuse",
        "reservoir_data_col": "reservoir_score",
        "evidence_layers": [],
        "evidence_data_col": "score",
        "base_weight": 1.0,
        "evidence_weight": 0.0,
        "confidence_weight": 1.0,
        "aggregation_weight": 0.5,
    },
}


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def find_layer_file(directory, key):
    """Locate a shapefile within ``directory`` whose name contains ``key``.

    Parameters
    ----------
    directory : str
        Directory to search for ``*.shp`` files.
    key : str
        Case-insensitive substring expected to appear in the shapefile name.

    Returns
    -------
    str or None
        Path to the first matching shapefile, or ``None`` if the directory
        does not exist or no match is found.
    """
    if not os.path.isdir(directory):
        return None

    matches = sorted(
        f
        for f in glob.glob(os.path.join(directory, "*.shp"))
        if key.lower() in os.path.basename(f).lower()
    )
    return matches[0] if matches else None


def resolve_data_col(gdf, preferred_col):
    """Resolve which column of ``gdf`` holds the score to interpolate.

    Falls back to the first numeric, non-geometry column if
    ``preferred_col`` is not present, printing a warning in that case.
    """
    if preferred_col in gdf.columns:
        return preferred_col

    numeric_cols = [
        c
        for c in gdf.columns
        if c != "geometry" and pd.api.types.is_numeric_dtype(gdf[c])
    ]
    if numeric_cols:
        print(
            f"  [WARN] Column '{preferred_col}' not found; "
            f"using fallback numeric column '{numeric_cols[0]}'."
        )
        return numeric_cols[0]

    raise ValueError(
        f"No usable numeric data column found (looked for '{preferred_col}')."
    )


def load_layer_gdf(path, target_crs):
    """Read a shapefile into a GeoDataFrame and reproject it to ``target_crs``."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Layer '{path}' has no CRS set; cannot reproject.")
    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf


def compute_basin_extent(gdfs):
    """Compute (x_min, y_min, x_max, y_max) as the union of bounds of ``gdfs``."""
    bounds = np.array([gdf.total_bounds for gdf in gdfs])
    x_min = float(np.min(bounds[:, 0]))
    y_min = float(np.min(bounds[:, 1]))
    x_max = float(np.max(bounds[:, 2]))
    y_max = float(np.max(bounds[:, 3]))
    return (x_min, y_min, x_max, y_max)


def interpolate_layer_to_grid(gdf, data_col, nx, ny, extent, interp_method=INTERP_METHOD):
    """Interpolate a point/polygon layer onto a regular (ny, nx) grid.

    Wraps ``geopfa.processing.Processing.interpolate_points()`` (2D only) so
    that all interpolation goes through the geoPFA library rather than a
    bespoke re-implementation.
    """
    mini_pfa = {
        "criteria": {
            "_criteria": {
                "components": {
                    "_component": {
                        "layers": {
                            "_layer": {
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

    mini_pfa = Processing.interpolate_points(
        pfa=mini_pfa,
        criteria="_criteria",
        component="_component",
        layer="_layer",
        nx=nx,
        ny=ny,
        extent=list(extent),
        interp_method=interp_method,
    )

    model = mini_pfa["criteria"]["_criteria"]["components"]["_component"][
        "layers"
    ]["_layer"]["model"]
    col = mini_pfa["criteria"]["_criteria"]["components"]["_component"][
        "layers"
    ]["_layer"]["model_data_col"]

    return model[col].to_numpy().reshape(ny, nx)


def normalize_and_clean(grid, assume_0_1=True, fill_value=0.0):
    """Normalize a grid to the [0, 1] range and replace non-finite values.

    Parameters
    ----------
    grid : numpy.ndarray
        Input grid, possibly containing NaN/Inf values.
    assume_0_1 : bool
        If True and the finite values already lie within [0, 1] (with a
        small tolerance), the grid is left as-is (only NaN/Inf are cleaned).
        Otherwise the grid is min-max normalized.
    fill_value : float
        Value used to replace NaN/Inf cells.
    """
    grid = np.asarray(grid, dtype=float)
    finite_mask = np.isfinite(grid)

    if not finite_mask.any():
        return np.full_like(grid, fill_value)

    finite_vals = grid[finite_mask]
    g_min = float(np.min(finite_vals))
    g_max = float(np.max(finite_vals))

    if assume_0_1 and g_min >= -1e-6 and g_max <= 1.0 + 1e-6:
        out = grid.copy()
    elif g_max > g_min:
        out = (grid - g_min) / (g_max - g_min)
    else:
        out = np.full_like(grid, fill_value)

    out = np.where(np.isfinite(out), out, fill_value)
    return np.clip(out, 0.0, 1.0)


def load_evidence_grid(cfg, nx, ny, extent):
    """Load and average configured evidence layers onto the grid.

    Returns a zero grid (neutral placeholder, no evidence contribution) when
    no evidence layers are configured or found -- this is intentional; see
    module docstring notes on evidence layers.
    """
    evidence_layers = cfg.get("evidence_layers", [])
    if not evidence_layers:
        return np.zeros((ny, nx), dtype=float)

    grids = []
    for key in evidence_layers:
        path = find_layer_file(EVIDENCE_DIR, key)
        if path is None:
            print(f"  [WARN] Evidence layer '{key}' not found in {EVIDENCE_DIR}.")
            continue
        gdf = load_layer_gdf(path, TARGET_CRS)
        data_col = resolve_data_col(gdf, cfg.get("evidence_data_col", "score"))
        grid = interpolate_layer_to_grid(gdf, data_col, nx, ny, extent)
        grids.append(normalize_and_clean(grid))

    if not grids:
        return np.zeros((ny, nx), dtype=float)

    return np.nanmean(np.stack(grids, axis=0), axis=0)


def export_geotiff(grid, path, extent, crs=TARGET_CRS, dtype="float32"):
    """Export a (ny, nx) grid as a north-up GeoTIFF.

    The interpolation grid is built with row 0 = y_min (south) increasing
    northward, so the array is flipped vertically before writing to match
    the GeoTIFF convention of row 0 = north (y_max).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    x_min, y_min, x_max, y_max = extent
    ny, nx = grid.shape
    transform = from_bounds(x_min, y_min, x_max, y_max, nx, ny)

    data = np.flipud(grid).astype(dtype)
    nodata = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else None

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)


def compute_horizon_stats(name, grid, extra=None):
    """Compute summary statistics for a single favorability grid."""
    finite_vals = grid[np.isfinite(grid)]
    row = {"horizon": name, "n_valid_cells": int(finite_vals.size), "n_total_cells": int(grid.size)}

    if finite_vals.size == 0:
        row.update({"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan, "pct_area_gt_0_5": np.nan})
    else:
        row.update(
            {
                "min": float(np.min(finite_vals)),
                "max": float(np.max(finite_vals)),
                "mean": float(np.mean(finite_vals)),
                "median": float(np.median(finite_vals)),
                "std": float(np.std(finite_vals)),
                "pct_area_gt_0_5": float(100.0 * np.mean(finite_vals > 0.5)),
            }
        )

    if extra:
        row.update(extra)

    return row


# ---------------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------------


def run_horizon_specific_workflow():
    print(f"[CONFIG] Grid resolution: {NX} x {NY}")
    print(f"[CONFIG] Target CRS: {TARGET_CRS}")
    print(f"[CONFIG] Output directory: {HORIZON_OUTPUT_DIR}")
    print(f"[CONFIG] {len(HORIZONS)} horizons configured: {', '.join(HORIZONS)}")

    os.makedirs(HORIZON_OUTPUT_DIR, exist_ok=True)

    # --- Discover + load raw thermal / reservoir layers for each horizon ---
    horizon_inputs = {}
    extent_gdfs = []
    missing_horizons = []

    for horizon_name, cfg in HORIZONS.items():
        thermal_path = find_layer_file(THERMAL_DIR, cfg["thermal_key"])
        reservoir_path = find_layer_file(RESERVOIR_DIR, cfg["reservoir_key"])

        if thermal_path is None or reservoir_path is None:
            print(
                f"[SKIP] {horizon_name}: missing "
                f"{'thermal layer' if thermal_path is None else ''}"
                f"{' and ' if thermal_path is None and reservoir_path is None else ''}"
                f"{'reservoir layer' if reservoir_path is None else ''} "
                f"in workspace."
            )
            missing_horizons.append(horizon_name)
            continue

        thermal_gdf = load_layer_gdf(thermal_path, TARGET_CRS)
        reservoir_gdf = load_layer_gdf(reservoir_path, TARGET_CRS)

        horizon_inputs[horizon_name] = (thermal_gdf, reservoir_gdf, cfg)
        extent_gdfs.extend([thermal_gdf, reservoir_gdf])

    if not horizon_inputs:
        raise RuntimeError(
            "No horizon layers were found. Verify that WORKSPACE_DIR "
            f"('{WORKSPACE_DIR}') contains matching shapefiles under "
            f"'{THERMAL_DIR}' and '{RESERVOIR_DIR}'."
        )

    extent = EXTENT_OVERRIDE or compute_basin_extent(extent_gdfs)
    print(f"[GRID] Model extent (xmin, ymin, xmax, ymax): {extent}")

    # --- Score each horizon independently (thermal * reservoir * evidence) ---
    horizon_scores = {}
    stats_rows = []

    for horizon_name, (thermal_gdf, reservoir_gdf, cfg) in horizon_inputs.items():
        print(f"\n[HORIZON] {horizon_name} ({cfg['label']})")

        thermal_col = resolve_data_col(thermal_gdf, cfg["thermal_data_col"])
        reservoir_col = resolve_data_col(reservoir_gdf, cfg["reservoir_data_col"])

        thermal_grid = interpolate_layer_to_grid(thermal_gdf, thermal_col, NX, NY, extent)
        reservoir_grid = interpolate_layer_to_grid(reservoir_gdf, reservoir_col, NX, NY, extent)

        thermal_grid = normalize_and_clean(thermal_grid)
        reservoir_grid = normalize_and_clean(reservoir_grid)
        evidence_grid = load_evidence_grid(cfg, NX, NY, extent)

        horizon_score = cfg["base_weight"] * (thermal_grid * reservoir_grid) + (
            cfg["evidence_weight"] * evidence_grid
        )
        horizon_score_final = cfg["confidence_weight"] * horizon_score
        horizon_score_final = np.clip(
            np.where(np.isfinite(horizon_score_final), horizon_score_final, 0.0), 0.0, 1.0
        )

        horizon_scores[horizon_name] = horizon_score_final

        out_path = os.path.join(HORIZON_OUTPUT_DIR, f"{horizon_name}_geothermal_favourability.tif")
        export_geotiff(horizon_score_final, out_path, extent)
        print(f"  -> exported {out_path}")

        stats_rows.append(
            compute_horizon_stats(
                horizon_name,
                horizon_score_final,
                extra={"aggregation_weight": cfg["aggregation_weight"]},
            )
        )

    # --- Aggregate across horizons ---
    print("\n[AGGREGATION] Combining horizon-specific favorability maps...")
    horizon_names = list(horizon_scores.keys())
    stack = np.stack([horizon_scores[h] for h in horizon_names], axis=0)  # (n_horizons, ny, nx)

    best_horizon_score = np.nanmax(stack, axis=0)
    best_horizon_idx = np.nanargmax(stack, axis=0).astype("int16")

    weights = np.array([HORIZONS[h]["aggregation_weight"] for h in horizon_names], dtype=float)
    weighted_mean_score = np.tensordot(weights, stack, axes=(0, 0)) / np.sum(weights)

    finite_best = best_horizon_score[np.isfinite(best_horizon_score)]
    q_primary = float(np.quantile(finite_best, PRIMARY_QUANTILE))
    q_secondary = float(np.quantile(finite_best, SECONDARY_QUANTILE))

    primary_map = np.where(best_horizon_score >= q_primary, best_horizon_score, np.nan)
    secondary_map = np.where(best_horizon_score >= q_secondary, best_horizon_score, np.nan)

    export_geotiff(best_horizon_score, os.path.join(HORIZON_OUTPUT_DIR, "best_horizon_geothermal_favourability.tif"), extent)
    export_geotiff(weighted_mean_score, os.path.join(HORIZON_OUTPUT_DIR, "weighted_mean_geothermal_favourability.tif"), extent)
    export_geotiff(primary_map, os.path.join(HORIZON_OUTPUT_DIR, "PRIMARY_geothermal_favourability.tif"), extent)
    export_geotiff(secondary_map, os.path.join(HORIZON_OUTPUT_DIR, "SECONDARY_geothermal_favourability.tif"), extent)
    export_geotiff(best_horizon_idx, os.path.join(HORIZON_OUTPUT_DIR, "best_horizon_id.tif"), extent, dtype="int16")

    id_mapping = pd.DataFrame(
        {
            "horizon_id": range(len(horizon_names)),
            "horizon_name": horizon_names,
            "label": [HORIZONS[h]["label"] for h in horizon_names],
        }
    )
    id_mapping.to_csv(os.path.join(HORIZON_OUTPUT_DIR, "best_horizon_id_mapping.csv"), index=False)

    stats_rows.append(
        compute_horizon_stats("best_horizon", best_horizon_score, extra={"aggregation_weight": np.nan})
    )
    stats_rows.append(
        compute_horizon_stats("weighted_mean", weighted_mean_score, extra={"aggregation_weight": np.nan})
    )
    stats_rows.append(
        compute_horizon_stats("PRIMARY", primary_map, extra={"aggregation_weight": np.nan, "quantile_threshold": q_primary})
    )
    stats_rows.append(
        compute_horizon_stats("SECONDARY", secondary_map, extra={"aggregation_weight": np.nan, "quantile_threshold": q_secondary})
    )

    summary_df = pd.DataFrame(stats_rows)
    summary_df.to_csv(os.path.join(HORIZON_OUTPUT_DIR, "summary.csv"), index=False)

    if missing_horizons:
        print(
            f"\n[NOTE] {len(missing_horizons)} horizon(s) skipped due to missing "
            f"layers: {', '.join(missing_horizons)}"
        )

    print(f"\n[SUCCESS] Horizon-specific favorability workflow complete. "
          f"{len(horizon_names)}/{len(HORIZONS)} horizons processed.")
    print(f"[SUCCESS] Outputs written to: {HORIZON_OUTPUT_DIR}")

    return {
        "horizon_scores": horizon_scores,
        "best_horizon_score": best_horizon_score,
        "weighted_mean_score": weighted_mean_score,
        "primary_map": primary_map,
        "secondary_map": secondary_map,
        "best_horizon_idx": best_horizon_idx,
        "extent": extent,
        "summary": summary_df,
    }


if __name__ == "__main__":
    run_horizon_specific_workflow()
