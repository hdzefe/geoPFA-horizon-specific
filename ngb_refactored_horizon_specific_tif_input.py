"""Horizon-specific geothermal favorability analysis using pre-computed GeoTIFF
thermal layers combined with reservoir potential shapefiles.

Unlike ``ngb_refactored_horizon_specific.py`` (which interpolates thermal point
data on the fly using ``geopfa.processing.Processing.interpolate_points``),
this script assumes the horizon-specific thermal "usefulness" layers have
*already* been generated as GeoTIFF rasters (e.g. from GeotiS temperature
points + TUNB base surfaces + thickness correction). Those rasters are loaded
directly with ``rasterio`` and, if necessary, resampled to the configured
grid resolution.

Reservoir potential polygons are still supplied as shapefiles and are
rasterized onto the same grid as the thermal layer.

Usage
-----
Change the resolution by editing the single ``NX, NY`` line below, then run::

    python ngb_refactored_horizon_specific_tif_input.py

All input directories can be overridden with environment variables (see the
CONFIGURATION section) which also makes the script easy to exercise in tests
without touching the hard-coded default paths.
"""

from __future__ import annotations

import os
import csv
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "rasterio is required to run this script. Install it with "
        "'pip install -r requirements.txt'."
    ) from exc

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "geopandas is required to run this script. Install it with "
        "'pip install -r requirements.txt'."
    ) from exc

from scipy import ndimage

# ============================================================================
# RESOLUTION CONFIGURATION
# ============================================================================
# Change ONLY this line to control the resolution of every output raster.
#   Development: 320 x 320  (~2-3 min)
#   Publication: 512 x 512  (~15-20 min)
#   Ultra-high:  800 x 800  (~30-40 min)
NX, NY = 320, 320

# ============================================================================
# WORKSPACE / OUTPUT CONFIGURATION
# ============================================================================
# All paths can be overridden via environment variables, which keeps the
# hard-coded defaults (matching the user's Windows project layout) usable
# for production runs while still allowing tests/CI to point at temporary
# synthetic data.
THERMAL_TIF_DIR = os.environ.get(
    "NGB_THERMAL_TIF_DIR",
    r"D:\geoPFA_Projects\north_german_basin\data\output\favourability\geothermal\horizon_specific",
)
RESERVOIR_SHP_DIR = os.environ.get(
    "NGB_RESERVOIR_SHP_DIR",
    r"D:\geoPFA_Projects\north_german_basin\data\raw\reservoirs",
)
BASIN_SHP_PATH = os.environ.get(
    "NGB_BASIN_SHP_PATH",
    r"D:\geoPFA_Projects\north_german_basin\data\raw\north_german_basin_extention\north_german_basin_.shp",
)
SALT_SHP_PATH = os.environ.get(
    "NGB_SALT_SHP_PATH",
    r"D:\geoPFA_Projects\north_german_basin\data\raw\salt_structures\Salzstrukturen_Inspee__v1_poly.shp",
)
OUTPUT_DIR = os.environ.get(
    "NGB_OUTPUT_DIR",
    r"D:\geoPFA_Projects\north_german_basin\data\output",
)

# EPSG code used when a shapefile has no CRS defined, and as the fallback CRS
# for written rasters if no other CRS can be determined. ETRS89 / UTM zone
# 32N is the typical CRS used for North German Basin datasets.
TARGET_EPSG = int(os.environ.get("NGB_TARGET_EPSG", "25832"))

# Buffer distance (meters) used to build the salt structure penalty zone.
SALT_BUFFER_M = float(os.environ.get("NGB_SALT_BUFFER_M", "2000"))

OUTPUT_SUBDIR = os.path.join("favourability", "geothermal", "horizon_specific")

# ============================================================================
# HORIZON CONFIGURATION
# ============================================================================
# reservoir_shp is None for horizons for which no reservoir potential
# shapefile was provided by the user (valanginian, bueckeberg); the workflow
# falls back to an all-zero reservoir grid (with a warning) for those.
HORIZONS: Dict[str, Dict[str, object]] = {
    "detfurth": {
        "label": "Detfurth Sandstone",
        "thermal_tif": "detfurth_thermal_use.tif",
        "reservoir_shp": "detfurth-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "tsf1": {
        "label": "Tsf1",
        "thermal_tif": "tsf1_thermal_use.tif",
        "reservoir_shp": "tsf1-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "tsf2": {
        "label": "Tsf2",
        "thermal_tif": "tsf2_thermal_use.tif",
        "reservoir_shp": "Tsf2-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "k42": {
        "label": "K4-2",
        "thermal_tif": "k42_thermal_use.tif",
        "reservoir_shp": "k4-2-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "k43": {
        "label": "K4-3",
        "thermal_tif": "k43_thermal_use.tif",
        "reservoir_shp": "k4-3-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "k44": {
        "label": "K4-4",
        "thermal_tif": "k44_thermal_use.tif",
        "reservoir_shp": "k4-4-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "aal1": {
        "label": "Aalenian 1",
        "thermal_tif": "aal1_thermal_use.tif",
        "reservoir_shp": "aal1-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "het1": {
        "label": "Hettangian 1",
        "thermal_tif": "het1_thermal_use.tif",
        "reservoir_shp": "Het1_Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "het2": {
        "label": "Hettangian 2",
        "thermal_tif": "het2_thermal_use.tif",
        "reservoir_shp": "Het2_Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "sin1": {
        "label": "Sinemurian 1",
        "thermal_tif": "sin1_thermal_use.tif",
        "reservoir_shp": "Sin1_Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "sin2": {
        "label": "Sinemurian 2",
        "thermal_tif": "sin2_thermal_use.tif",
        "reservoir_shp": "Sin2_Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "pli1": {
        "label": "Pliensbachian 1",
        "thermal_tif": "pli1_thermal_use.tif",
        "reservoir_shp": "Pli1_Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "pli2": {
        "label": "Pliensbachian 2",
        "thermal_tif": "pli2_thermal_use.tif",
        "reservoir_shp": "Pli2_Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "toa1": {
        "label": "Toarcian 1",
        "thermal_tif": "toa1_thermal_use.tif",
        "reservoir_shp": "toa1-Potential.shp",
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "toa2": {
        "label": "Toarcian 2",
        "thermal_tif": "toa2_thermal_use.tif",
        "reservoir_shp": "toa2-Potential.shp",
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
    "bueckeberg": {
        "label": "Bueckeberg",
        "thermal_tif": "bueckeberg_thermal_use.tif",
        "reservoir_shp": None,
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
}


# ============================================================================
# HELPERS
# ============================================================================
Extent = Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    """Normalize a grid to [0, 1], safely handling constants and NaNs."""
    grid = np.asarray(grid, dtype=float)
    finite = np.isfinite(grid)
    if not finite.any():
        return np.zeros_like(grid)

    gmin = np.nanmin(grid[finite])
    gmax = np.nanmax(grid[finite])
    out = np.zeros_like(grid)
    if gmax - gmin < 1e-12:
        # Constant (non-NaN) grid: treat as fully favorable if positive, else 0.
        out[finite] = 1.0 if gmax > 0 else 0.0
        return out

    out[finite] = (grid[finite] - gmin) / (gmax - gmin)
    out[~finite] = 0.0
    return out


def load_thermal_tif(
    tif_path: str, nx: int, ny: int
) -> Tuple[Optional[np.ndarray], Optional[Extent], Optional[object]]:
    """Load a thermal GeoTIFF and resample it to (ny, nx).

    Returns ``(grid, extent, crs)`` or ``(None, None, None)`` if the file is
    missing or cannot be read.
    """
    if not tif_path or not os.path.isfile(tif_path):
        print(f"  ⚠ Thermal TIF not found: {tif_path}")
        return None, None, None

    try:
        with rasterio.open(tif_path) as src:
            thermal_grid = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                thermal_grid = np.where(thermal_grid == nodata, np.nan, thermal_grid)
            bounds = src.bounds
            crs = src.crs
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ⚠ Failed to read thermal TIF '{tif_path}': {exc}")
        return None, None, None

    if thermal_grid.shape != (ny, nx):
        zoom_y = ny / thermal_grid.shape[0]
        zoom_x = nx / thermal_grid.shape[1]
        # Resample NaNs safely by filling then re-masking.
        nan_mask = ~np.isfinite(thermal_grid)
        filled = np.where(nan_mask, 0.0, thermal_grid)
        resampled = ndimage.zoom(filled, (zoom_y, zoom_x), order=1)
        if nan_mask.any():
            mask_resampled = ndimage.zoom(
                nan_mask.astype(float), (zoom_y, zoom_x), order=1
            )
            resampled = np.where(mask_resampled > 0.5, np.nan, resampled)
        thermal_grid = resampled

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return thermal_grid, extent, crs


def rasterize_geometries(
    gdf: "gpd.GeoDataFrame", extent: Extent, nx: int, ny: int, buffer_m: float = 0.0
) -> np.ndarray:
    """Rasterize a GeoDataFrame's (optionally buffered) geometries to a
    binary [0, 1] grid of shape (ny, nx) covering ``extent``."""
    if gdf is None or len(gdf) == 0:
        return np.zeros((ny, nx))

    geoms = gdf.geometry
    if buffer_m:
        geoms = geoms.buffer(buffer_m)

    xmin, xmax, ymin, ymax = extent
    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    shapes = [(geom, 1) for geom in geoms if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((ny, nx))

    grid = rasterize(
        shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="float64",
    )
    return grid


def load_and_interpolate_reservoir(
    shp_path: Optional[str], extent: Extent, nx: int, ny: int, target_epsg: int
) -> np.ndarray:
    """Load a reservoir potential shapefile and rasterize it to the grid."""
    if not shp_path:
        return np.zeros((ny, nx))
    if not os.path.isfile(shp_path):
        print(f"  ⚠ Reservoir shapefile not found: {shp_path}")
        return np.zeros((ny, nx))

    try:
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=target_epsg)
        else:
            gdf = gdf.to_crs(epsg=target_epsg)

        if len(gdf) == 0:
            return np.zeros((ny, nx))

        grid = rasterize_geometries(gdf, extent, nx, ny)
        return normalize_grid(grid)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ⚠ Reservoir shapefile load failed: {exc}")
        return np.zeros((ny, nx))


def load_salt_penalty(
    shp_path: Optional[str], extent: Extent, nx: int, ny: int, target_epsg: int,
    buffer_m: float,
) -> np.ndarray:
    """Load salt structures and build a [0, 1] penalty grid (1 = fully
    penalized). Returns all-zeros (no penalty) if unavailable."""
    if not shp_path:
        print("  ⚠ No salt structure shapefile configured - skipping penalty")
        return np.zeros((ny, nx))
    if not os.path.isfile(shp_path):
        print(f"  ⚠ Salt structure shapefile not found: {shp_path}")
        return np.zeros((ny, nx))

    try:
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=target_epsg)
        else:
            gdf = gdf.to_crs(epsg=target_epsg)

        if len(gdf) == 0:
            return np.zeros((ny, nx))

        grid = rasterize_geometries(gdf, extent, nx, ny, buffer_m=buffer_m)
        return np.clip(grid, 0.0, 1.0)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ⚠ Salt structure shapefile load failed: {exc}")
        return np.zeros((ny, nx))


def get_basin_extent(basin_shp_path: str, target_epsg: int) -> Optional[Extent]:
    """Return (xmin, xmax, ymin, ymax) from the basin outline shapefile, or
    None if it cannot be loaded."""
    if not basin_shp_path or not os.path.isfile(basin_shp_path):
        print(f"  ⚠ Basin outline not found: {basin_shp_path}")
        return None
    try:
        gdf = gpd.read_file(basin_shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=target_epsg)
        else:
            gdf = gdf.to_crs(epsg=target_epsg)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        return (xmin, xmax, ymin, ymax)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ⚠ Failed to load basin outline: {exc}")
        return None


def write_geotiff(
    path: str, grid: np.ndarray, extent: Extent, crs: object, nodata: float = -9999.0
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ny, nx = grid.shape
    xmin, xmax, ymin, ymax = extent
    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    out_grid = np.where(np.isfinite(grid), grid, nodata).astype("float32")
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
    ) as dst:
        dst.write(out_grid, 1)


def safe_stack_argmax(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given a stack of shape (n_horizons, ny, nx), return the elementwise
    max value grid and the argmax index grid. Cells that are NaN in every
    horizon get value NaN and index -1."""
    all_nan = ~np.isfinite(stack).any(axis=0)
    filled = np.where(np.isfinite(stack), stack, -np.inf)
    best_idx = np.argmax(filled, axis=0)
    best_val = np.take_along_axis(filled, best_idx[np.newaxis, ...], axis=0)[0]
    best_val = np.where(all_nan, np.nan, best_val)
    best_idx = np.where(all_nan, -1, best_idx)
    return best_val, best_idx


# ============================================================================
# MAIN WORKFLOW
# ============================================================================
def run_horizon_workflow(
    thermal_tif_dir: str = THERMAL_TIF_DIR,
    reservoir_shp_dir: str = RESERVOIR_SHP_DIR,
    basin_shp_path: str = BASIN_SHP_PATH,
    salt_shp_path: str = SALT_SHP_PATH,
    output_dir: str = OUTPUT_DIR,
    nx: int = NX,
    ny: int = NY,
    target_epsg: int = TARGET_EPSG,
    salt_buffer_m: float = SALT_BUFFER_M,
    horizons: Dict[str, Dict[str, object]] = HORIZONS,
) -> Dict[str, object]:
    """Run the full horizon-specific favorability workflow and write outputs.

    Returns a dict summarizing what was produced (mostly useful for tests).
    """
    print("=" * 60)
    print("[STEP 1] Determining grid extent...")
    extent = get_basin_extent(basin_shp_path, target_epsg)

    out_dir = os.path.join(output_dir, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Grid resolution: {nx} x {ny}")
    print("[STEP 2] Loading salt structure penalty (applies to all horizons)...")
    salt_grid = None  # computed lazily once we know the extent

    horizon_grids: Dict[str, np.ndarray] = {}
    crs = None
    summary_rows = []

    print("[STEP 3] Processing horizons...")
    for name, cfg in horizons.items():
        print(f"  -- {name} ({cfg['label']}) --")
        thermal_path = os.path.join(thermal_tif_dir, str(cfg["thermal_tif"]))
        thermal_grid, tif_extent, tif_crs = load_thermal_tif(thermal_path, nx, ny)
        if thermal_grid is None:
            print(f"  ⚠ Skipping horizon '{name}': thermal TIF unavailable")
            summary_rows.append(
                {"horizon": name, "label": cfg["label"], "status": "skipped_no_thermal",
                 "mean": "", "max": "", "cells_gt_0p5": ""}
            )
            continue

        horizon_extent = extent if extent is not None else tif_extent
        if crs is None:
            crs = tif_crs

        reservoir_shp = cfg.get("reservoir_shp")
        reservoir_path = (
            os.path.join(reservoir_shp_dir, str(reservoir_shp)) if reservoir_shp else None
        )
        reservoir_grid = load_and_interpolate_reservoir(
            reservoir_path, horizon_extent, nx, ny, target_epsg
        )

        thermal_grid_n = normalize_grid(thermal_grid)
        base_score = thermal_grid_n * reservoir_grid

        evidence_grid = np.zeros((ny, nx))  # placeholder, per spec

        base_weight = float(cfg.get("base_weight", 0.85))
        evidence_weight = float(cfg.get("evidence_weight", 0.15))
        confidence_weight = float(cfg.get("confidence_weight", 1.0))

        horizon_score = base_weight * base_score + evidence_weight * evidence_grid
        horizon_score = confidence_weight * horizon_score

        if salt_grid is None:
            salt_grid = load_salt_penalty(
                salt_shp_path, horizon_extent, nx, ny, target_epsg, salt_buffer_m
            )

        horizon_score_final = horizon_score * (1 - salt_grid)
        horizon_score_final = np.clip(horizon_score_final, 0.0, 1.0)

        horizon_grids[name] = horizon_score_final

        out_path = os.path.join(out_dir, f"{name}_geothermal_favourability.tif")
        write_geotiff(out_path, horizon_score_final, horizon_extent, crs)

        summary_rows.append(
            {
                "horizon": name,
                "label": cfg["label"],
                "status": "ok",
                "mean": float(np.nanmean(horizon_score_final)),
                "max": float(np.nanmax(horizon_score_final)),
                "cells_gt_0p5": int(np.sum(horizon_score_final > 0.5)),
            }
        )
        print(
            f"     mean={summary_rows[-1]['mean']:.4f} "
            f"max={summary_rows[-1]['max']:.4f}"
        )

    print("[STEP 4] Aggregating horizons...")
    result: Dict[str, object] = {
        "horizon_grids": horizon_grids,
        "extent": extent,
        "output_dir": out_dir,
    }

    if not horizon_grids:
        print("  ⚠ No horizons produced valid output - skipping aggregation")
        _write_summary_csv(os.path.join(out_dir, "summary.csv"), summary_rows)
        return result

    names = list(horizon_grids.keys())
    stack = np.stack([horizon_grids[n] for n in names], axis=0)
    best_val, best_idx = safe_stack_argmax(stack)

    agg_extent = extent if extent is not None else (0.0, float(nx), 0.0, float(ny))

    best_path = os.path.join(out_dir, "best_horizon_geothermal_favourability.tif")
    write_geotiff(best_path, best_val, agg_extent, crs)

    finite_vals = best_val[np.isfinite(best_val)]
    if finite_vals.size:
        p90 = np.nanpercentile(finite_vals, 90)
        p75 = np.nanpercentile(finite_vals, 75)
    else:
        p90 = p75 = np.nan

    primary = np.where(np.isfinite(best_val) & (best_val >= p90), best_val, np.nan)
    secondary = np.where(np.isfinite(best_val) & (best_val >= p75), best_val, np.nan)

    write_geotiff(
        os.path.join(out_dir, "PRIMARY_geothermal_favourability.tif"), primary,
        agg_extent, crs,
    )
    write_geotiff(
        os.path.join(out_dir, "SECONDARY_geothermal_favourability.tif"), secondary,
        agg_extent, crs,
    )

    best_idx_path = os.path.join(out_dir, "best_horizon_id.tif")
    write_geotiff(
        best_idx_path, best_idx.astype(float), agg_extent, crs, nodata=-1.0
    )

    mapping_path = os.path.join(out_dir, "best_horizon_id_mapping.csv")
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "horizon", "label"])
        for i, n in enumerate(names):
            writer.writerow([i, n, horizons[n]["label"]])

    _write_summary_csv(os.path.join(out_dir, "summary.csv"), summary_rows)

    print("[DONE] Horizon-specific favorability workflow complete.")
    print(f"  Outputs written to: {out_dir}")

    result.update(
        {
            "names": names,
            "best_val": best_val,
            "best_idx": best_idx,
            "primary": primary,
            "secondary": secondary,
            "p90": p90,
            "p75": p75,
        }
    )
    return result


def _write_summary_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["horizon", "label", "status", "mean", "max", "cells_gt_0p5"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_horizon_workflow()
