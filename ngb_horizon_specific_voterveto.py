"""Horizon-specific geothermal favourability for the North German Basin.

Workflow
--------
The favourability model is built **per horizon** and only afterwards
aggregated into composite products:

1. For **each** of the 18 horizons:

   * load the horizon thermal-use GeoTIFF (``rasterio``),
   * load and rasterize the horizon reservoir shapefile (``geopandas``),
   * load the horizon evidence layers (borehole poro/perm, deep
     hydrothermal sites, ...),
   * combine them with :class:`geopfa.layer_combination.VoterVeto` using the
     horizon specific ``base_weight`` / ``evidence_weight`` /
     ``confidence_weight``,
   * export ``{horizon}_geothermal_favourability.tif``.

2. Aggregate the 18 horizon grids into

   * ``best_horizon_geothermal_favourability.tif`` (max across horizons),
   * ``PRIMARY_geothermal_favourability.tif`` (top 10 % of the composite),
   * ``SECONDARY_geothermal_favourability.tif`` (top 25 % of the composite),
   * ``best_horizon_id.tif`` (1-indexed id of the best horizon per cell),
   * ``best_horizon_id_mapping.csv`` and ``summary.csv``.

All products are written in **EPSG:31467** (Gauss-Krueger zone 3), which is
the project CRS of the North German Basin study.

All input/output locations are module level constants that can be overridden
through environment variables, so the script runs unchanged on the production
machine and inside tests.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("ngb_horizon_specific")

# ---------------------------------------------------------------------------
# Project constants
# ---------------------------------------------------------------------------

#: Project CRS. Gauss-Krueger zone 3 -- hard-coded for the whole workflow.
TARGET_EPSG = int(os.environ.get("NGB_TARGET_EPSG", "31467"))

#: Directory holding the ``*_thermal_use.tif`` rasters.
THERMAL_TIF_DIR = os.environ.get(
    "NGB_THERMAL_TIF_DIR",
    r"D:\geoPFA_Projects\north_german_basin\data\output\favourability\geothermal\horizon_specific",
)

#: Root directory of the raw vector data (reservoirs, evidence, ...).
RAW_DATA_DIR = os.environ.get(
    "NGB_RAW_DATA_DIR",
    r"D:\geoPFA_Projects\north_german_basin\data\raw",
)

#: Directory holding the reservoir shapefiles.
RESERVOIR_SHP_DIR = os.environ.get(
    "NGB_RESERVOIR_SHP_DIR",
    os.path.join(RAW_DATA_DIR, "reservoirs"),
)

#: Directory holding the evidence layers (rasters or vector files).
EVIDENCE_DIR = os.environ.get(
    "NGB_EVIDENCE_DIR",
    os.path.join(RAW_DATA_DIR, "evidence"),
)

#: Outline of the basin -- defines the modelling extent and the valid mask.
BASIN_SHP_PATH = os.environ.get(
    "NGB_BASIN_SHP_PATH",
    os.path.join(
        RAW_DATA_DIR, "north_german_basin_extention", "north_german_basin_.shp"
    ),
)

#: Salt structures -- vetoed (excluded) from the favourability maps.
SALT_SHP_PATH = os.environ.get(
    "NGB_SALT_SHP_PATH",
    os.path.join(
        RAW_DATA_DIR, "salt_structures", "Salzstrukturen_Inspee__v1_poly.shp"
    ),
)

#: Root output directory.
OUTPUT_DIR = os.environ.get(
    "NGB_OUTPUT_DIR",
    r"D:\geoPFA_Projects\north_german_basin\data\output",
)

#: Sub directory of ``OUTPUT_DIR`` receiving all products of this script.
HORIZON_OUTPUT_SUBDIR = os.path.join(
    "favourability", "geothermal", "horizon_specific"
)

#: Model grid size (number of columns / rows).
GRID_NX = int(os.environ.get("NGB_GRID_NX", "300"))
GRID_NY = int(os.environ.get("NGB_GRID_NY", "300"))

#: Prior favourability used to derive the voter intercept ``w0``.
PRIOR_FAVORABILITY = float(os.environ.get("NGB_PRIOR_FAVORABILITY", "0.5"))

#: Normalisation method handed to ``VoterVeto.do_voter_veto``.
NORMALIZE_METHOD = os.environ.get("NGB_NORMALIZE_METHOD", "minmax")

#: Quantiles used to derive the PRIMARY / SECONDARY sweetspot masks.
PRIMARY_QUANTILE = float(os.environ.get("NGB_PRIMARY_QUANTILE", "0.90"))
SECONDARY_QUANTILE = float(os.environ.get("NGB_SECONDARY_QUANTILE", "0.75"))

#: Search radius (metres) used to convert point evidence into a grid.
EVIDENCE_SEARCH_RADIUS_M = float(
    os.environ.get("NGB_EVIDENCE_SEARCH_RADIUS_M", "15000")
)

#: NoData value used in the exported GeoTIFFs.
FLOAT_NODATA = -9999.0
INT_NODATA = 0

# ---------------------------------------------------------------------------
# Value mappings (exact definitions of the original production script)
# ---------------------------------------------------------------------------

RESERVOIR_POTENTIAL_MAPPING_DE = {
    "niedrig": 0.2,
    "mittel": 0.6,
    "hoch": 1.0,
}

RESERVOIR_POTENTIAL_MAPPING_DE_RESTRICTED = {
    "niedrig": 0.2,
    "eingeschraenkt": 0.2,
    "eingeschränkt": 0.2,
    "mittel": 0.6,
    "hoch": 1.0,
}

RESERVOIR_THICKNESS_MAPPING_EN = {
    "low (< 10 m)": 0.2,
    "moderate (10-20 m)": 0.6,
    "high (> 20 m)": 1.0,
}

RESERVOIR_BINARY_MAPPING_VALANGINIAN = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
}

RESERVOIR_BINARY_MAPPING_BUECKEBERG = {
    2: 1.0,
    3: 1.0,
    7: 1.0,
    11: 1.0,
}

# ---------------------------------------------------------------------------
# Horizon configuration -- ALL 18 horizons
# ---------------------------------------------------------------------------

HORIZONS: Dict[str, Dict[str, Any]] = {
    "detfurth": {
        "label": "Detfurth Sandstone",
        "thermal_layer": "detfurth_thermal_use",
        "reservoir_layer": "detfurth_sand_quality",
        "reservoir_source_path": os.path.join(
            RAW_DATA_DIR, "sand_zones_midbunter", "sand_zones_detfurth_.shp"
        ),
        "reservoir_value_col": "sand_share",
        "reservoir_value_min": 30.0,
        "reservoir_value_max": 90.0,
        "reservoir_certainty_col": "certainity",
        "evidence_layers": [
            "detfurth_borehole_poroperm_evidence",
            "detfurth_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.85,
        "evidence_weight": 0.15,
        "confidence_weight": 1.0,
    },
    "tsf1": {
        "label": "Tsf1 Potential",
        "thermal_layer": "tsf1_thermal_use",
        "reservoir_layer": "tsf1-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "tsf1-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE,
        "evidence_layers": ["tsf1_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "tsf2": {
        "label": "Tsf2 Potential",
        "thermal_layer": "tsf2_thermal_use",
        "reservoir_layer": "Tsf2-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "Tsf2-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE,
        "evidence_layers": ["tsf2_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "k42": {
        "label": "K4-2 Potential",
        "thermal_layer": "k42_thermal_use",
        "reservoir_layer": "k4-2-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "k4-2-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE_RESTRICTED,
        "evidence_layers": [
            "k42_borehole_poroperm_evidence",
            "k42_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "k43": {
        "label": "K4-3 Potential",
        "thermal_layer": "k43_thermal_use",
        "reservoir_layer": "k4-3-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "k4-3-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE_RESTRICTED,
        "evidence_layers": [
            "k43_borehole_poroperm_evidence",
            "k43_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "k44": {
        "label": "K4-4 Potential",
        "thermal_layer": "k44_thermal_use",
        "reservoir_layer": "k4-4-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "k4-4-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE,
        "evidence_layers": [
            "k44_borehole_poroperm_evidence",
            "k44_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "het1": {
        "label": "Het1 Potential",
        "thermal_layer": "het1_thermal_use",
        "reservoir_layer": "Het1_Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "Het1_Potential.shp"
        ),
        "reservoir_value_col": "quality",
        "reservoir_value_mapping": RESERVOIR_THICKNESS_MAPPING_EN,
        "evidence_layers": [
            "het1_borehole_poroperm_evidence",
            "het1_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "het2": {
        "label": "Het2 Potential",
        "thermal_layer": "het2_thermal_use",
        "reservoir_layer": "Het2_Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "Het2_Potential.shp"
        ),
        "reservoir_value_col": "quality",
        "reservoir_value_mapping": RESERVOIR_THICKNESS_MAPPING_EN,
        "evidence_layers": [
            "het2_borehole_poroperm_evidence",
            "het2_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "sin1": {
        "label": "Sin1 Potential",
        "thermal_layer": "sin1_thermal_use",
        "reservoir_layer": "Sin1_Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "Sin1_Potential.shp"
        ),
        "reservoir_value_col": "reservoirq",
        "reservoir_value_mapping": RESERVOIR_THICKNESS_MAPPING_EN,
        "evidence_layers": [
            "sin1_borehole_poroperm_evidence",
            "sin1_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "sin2": {
        "label": "Sin2 Potential",
        "thermal_layer": "sin2_thermal_use",
        "reservoir_layer": "Sin2_Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "Sin2_Potential.shp"
        ),
        "reservoir_value_col": "reservoirq",
        "reservoir_value_mapping": RESERVOIR_THICKNESS_MAPPING_EN,
        "evidence_layers": [
            "sin2_borehole_poroperm_evidence",
            "sin2_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "pli1": {
        "label": "Pli1 Potential",
        "thermal_layer": "pli1_thermal_use",
        "reservoir_layer": "Pli1_Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "Pli1_Potential.shp"
        ),
        "reservoir_value_col": "reservoirq",
        "reservoir_value_mapping": RESERVOIR_THICKNESS_MAPPING_EN,
        "evidence_layers": ["pli1_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "pli2": {
        "label": "Pli2 Potential",
        "thermal_layer": "pli2_thermal_use",
        "reservoir_layer": "Pli2_Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "Pli2_Potential.shp"
        ),
        "reservoir_value_col": "reservoir",
        "reservoir_value_mapping": RESERVOIR_THICKNESS_MAPPING_EN,
        "evidence_layers": ["pli2_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "toa1": {
        "label": "Toa1 Potential",
        "thermal_layer": "toa1_thermal_use",
        "reservoir_layer": "toa1-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "toa1-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE,
        "evidence_layers": ["toa1_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "toa2": {
        "label": "Toa2 Potential",
        "thermal_layer": "toa2_thermal_use",
        "reservoir_layer": "toa2-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "toa2-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE,
        "evidence_layers": ["toa2_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "aal1": {
        "label": "Aal1 Potential",
        "thermal_layer": "aal1_thermal_use",
        "reservoir_layer": "aal1-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "aal1-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE,
        "evidence_layers": [
            "aal1_borehole_poroperm_evidence",
            "aal1_deep_hydrothermal_sites_evidence",
        ],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "bj1": {
        "label": "Bj1 Potential",
        "thermal_layer": "bj1_thermal_use",
        "reservoir_layer": "bj1-Potential",
        "reservoir_source_path": os.path.join(
            RESERVOIR_SHP_DIR, "bj1-Potential.shp"
        ),
        "reservoir_value_col": "Potential",
        "reservoir_value_mapping": RESERVOIR_POTENTIAL_MAPPING_DE,
        "evidence_layers": ["bj1_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 1.0,
    },
    "valanginian": {
        "label": "Valanginian Sandstone",
        "thermal_layer": "valanginian_thermal_use",
        "reservoir_layer": "valanginian_sandstone",
        "reservoir_source_path": os.path.join(
            RAW_DATA_DIR, "lower_cretaceous_lbeg", "valanginian_sandstone.shp"
        ),
        "reservoir_value_col": "OBJECTID",
        "reservoir_value_mapping": RESERVOIR_BINARY_MAPPING_VALANGINIAN,
        "evidence_layers": ["valanginian_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.7,
    },
    "bueckeberg": {
        "label": "Bueckeberg Group Sandstone",
        "thermal_layer": "bueckeberg_thermal_use",
        "reservoir_layer": "bueckeberg_group_sandstone_lbeg",
        "reservoir_source_path": os.path.join(
            RAW_DATA_DIR,
            "lower_cretaceous_lbeg",
            "bueckeberg_group_sandstone_lbeg.shp",
        ),
        "reservoir_value_col": "LEGNR",
        "reservoir_value_mapping": RESERVOIR_BINARY_MAPPING_BUECKEBERG,
        "evidence_layers": ["bueckeberg_borehole_poroperm_evidence"],
        "base_weight": 0.90,
        "evidence_weight": 0.10,
        "confidence_weight": 0.7,
    },
}

#: Stable 1-based ids used for ``best_horizon_id.tif``.
HORIZON_IDS: Dict[str, int] = {
    name: index for index, name in enumerate(HORIZONS, start=1)
}

#: Default extent (EPSG:31467) used when no basin outline is available.
DEFAULT_EXTENT = (3_300_000.0, 5_780_000.0, 3_700_000.0, 6_060_000.0)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """Regular model grid (north-up, row 0 is the northern-most row)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    nx: int
    ny: int

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self) -> float:
        return (self.ymax - self.ymin) / self.ny

    @property
    def transform(self):
        from rasterio.transform import from_bounds

        return from_bounds(
            self.xmin, self.ymin, self.xmax, self.ymax, self.nx, self.ny
        )

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ``(x, y)`` cell centre coordinate vectors.

        ``x`` increases west to east, ``y`` decreases north to south so that
        ``y[0]`` matches raster row 0.
        """
        xs = self.xmin + (np.arange(self.nx) + 0.5) * self.dx
        ys = self.ymax - (np.arange(self.ny) + 0.5) * self.dy
        return xs, ys

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = self.cell_centers()
        return np.meshgrid(xs, ys)


def build_grid(
    extent: Sequence[float], nx: int = GRID_NX, ny: int = GRID_NY
) -> Grid:
    """Create a :class:`Grid` from ``(xmin, ymin, xmax, ymax)``."""
    xmin, ymin, xmax, ymax = (float(v) for v in extent)
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"Invalid extent: {extent!r}")
    if nx < 1 or ny < 1:
        raise ValueError(f"Invalid grid size: nx={nx}, ny={ny}")
    return Grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, nx=nx, ny=ny)


def grid_points_gdf(grid: Grid, values: np.ndarray, col: str = "value"):
    """Return the grid as a point ``GeoDataFrame`` understood by geopfa.

    ``geopfa`` rasterizes point GeoDataFrames internally, therefore every
    layer handed to :class:`~geopfa.layer_combination.VoterVeto` has to be a
    complete, regular point grid.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    if values.shape != grid.shape:
        raise ValueError(
            f"Values shape {values.shape} does not match grid shape {grid.shape}."
        )

    xx, yy = grid.meshgrid()
    geoms = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
    gdf = gpd.GeoDataFrame(
        {col: np.asarray(values, dtype=float).ravel()},
        geometry=geoms,
        crs=f"EPSG:{TARGET_EPSG}",
    )
    return gdf


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _iter_candidate_dirs() -> Iterable[str]:
    for directory in (RESERVOIR_SHP_DIR, RAW_DATA_DIR, EVIDENCE_DIR):
        if directory and os.path.isdir(directory):
            yield directory


def resolve_vector_path(configured_path: str) -> Optional[str]:
    """Resolve a configured vector path to an existing file.

    The horizon configuration carries the production paths.  When the script
    runs somewhere else (test machine, Linux server) the file is looked up by
    name below :data:`RESERVOIR_SHP_DIR` / :data:`RAW_DATA_DIR`.
    """
    if not configured_path:
        return None

    if os.path.isfile(configured_path):
        return configured_path

    name = os.path.basename(configured_path.replace("\\", "/"))
    for directory in _iter_candidate_dirs():
        direct = os.path.join(directory, name)
        if os.path.isfile(direct):
            return direct
        for found in Path(directory).rglob(name):
            if found.is_file():
                return str(found)
    return None


def thermal_tif_path(horizon_config: Mapping[str, Any]) -> Optional[str]:
    """Return the thermal GeoTIFF path of a horizon, if it exists."""
    layer = horizon_config["thermal_layer"]
    candidates = [f"{layer}.tif", f"{layer}.tiff"]
    for candidate in candidates:
        path = os.path.join(THERMAL_TIF_DIR, candidate)
        if os.path.isfile(path):
            return path
    return None


def evidence_path(layer_name: str) -> Optional[str]:
    """Locate an evidence layer (raster or vector) by layer name."""
    suffixes = (".tif", ".tiff", ".shp", ".gpkg", ".geojson", ".csv")
    for suffix in suffixes:
        path = os.path.join(EVIDENCE_DIR, f"{layer_name}{suffix}")
        if os.path.isfile(path):
            return path
    if os.path.isdir(EVIDENCE_DIR):
        for suffix in suffixes:
            for found in Path(EVIDENCE_DIR).rglob(f"{layer_name}{suffix}"):
                if found.is_file():
                    return str(found)
    return None


# ---------------------------------------------------------------------------
# Raster / vector loading
# ---------------------------------------------------------------------------


def load_raster_to_grid(path: str, grid: Grid) -> np.ndarray:
    """Read ``path`` and reproject/resample it onto ``grid``.

    Returns a ``float64`` array of shape ``grid.shape`` with ``np.nan`` for
    nodata cells.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject

    destination = np.full(grid.shape, np.nan, dtype="float64")
    dst_crs = CRS.from_epsg(TARGET_EPSG)

    with rasterio.open(path) as src:
        source = src.read(1, masked=True).astype("float64").filled(np.nan)
        src_crs = src.crs if src.crs is not None else dst_crs
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def read_vector(path: str):
    """Read a vector file and reproject it to :data:`TARGET_EPSG`."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        LOGGER.warning(
            "%s has no CRS; assuming EPSG:%s.", path, TARGET_EPSG
        )
        gdf = gdf.set_crs(f"EPSG:{TARGET_EPSG}")
    elif gdf.crs.to_epsg() != TARGET_EPSG:
        gdf = gdf.to_crs(f"EPSG:{TARGET_EPSG}")
    return gdf


def rasterize_gdf(gdf, grid: Grid, value_col: str) -> np.ndarray:
    """Rasterize polygon/line geometries of ``gdf`` using ``value_col``."""
    from rasterio.features import rasterize as rio_rasterize

    shapes = [
        (geom, float(value))
        for geom, value in zip(gdf.geometry, gdf[value_col])
        if geom is not None and not geom.is_empty and np.isfinite(value)
    ]
    if not shapes:
        return np.full(grid.shape, np.nan, dtype="float64")

    array = rio_rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=np.nan,
        dtype="float64",
        all_touched=False,
    )
    return array


def normalize_mapping_key(value: Any) -> Any:
    """Normalise a raw attribute value for mapping lookups."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return value


def map_reservoir_values(
    gdf,
    value_col: str,
    value_mapping: Optional[Mapping[Any, float]] = None,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> np.ndarray:
    """Convert reservoir attributes to favourability scores in ``[0, 1]``.

    Two modes are supported, matching the original production script:

    * categorical attributes (``Potential``, ``quality``, ``reservoirq``,
      ``OBJECTID``, ``LEGNR``) via ``value_mapping``;
    * continuous attributes (Detfurth ``sand_share`` isolines) linearly
      rescaled between ``value_min`` and ``value_max``.
    """
    if value_col not in gdf.columns:
        raise KeyError(
            f"Column '{value_col}' not found; available columns: "
            f"{[c for c in gdf.columns if c != 'geometry']}"
        )

    raw = gdf[value_col]

    if value_mapping is not None:
        lookup = {normalize_mapping_key(k): float(v) for k, v in value_mapping.items()}
        scores = np.array(
            [lookup.get(normalize_mapping_key(v), np.nan) for v in raw],
            dtype="float64",
        )
        return scores

    values = np.asarray(raw, dtype="float64")
    if value_min is None:
        value_min = float(np.nanmin(values))
    if value_max is None:
        value_max = float(np.nanmax(values))
    if math.isclose(value_max, value_min):
        return np.where(np.isfinite(values), 1.0, np.nan)

    scores = (values - value_min) / (value_max - value_min)
    return np.clip(scores, 0.0, 1.0)


def load_and_rasterize_reservoir(
    horizon_config: Mapping[str, Any], grid: Grid
) -> Optional[np.ndarray]:
    """Load the horizon reservoir shapefile and rasterize it onto ``grid``."""
    configured = horizon_config.get("reservoir_source_path")
    path = resolve_vector_path(configured) if configured else None
    if path is None:
        LOGGER.warning(
            "Reservoir source not found for layer '%s' (%s).",
            horizon_config.get("reservoir_layer"),
            configured,
        )
        return None

    gdf = read_vector(path)
    if len(gdf) == 0:
        LOGGER.warning("Reservoir file %s is empty.", path)
        return None

    scores = map_reservoir_values(
        gdf,
        horizon_config["reservoir_value_col"],
        horizon_config.get("reservoir_value_mapping"),
        horizon_config.get("reservoir_value_min"),
        horizon_config.get("reservoir_value_max"),
    )

    certainty_col = horizon_config.get("reservoir_certainty_col")
    if certainty_col and certainty_col in gdf.columns:
        scores = scores * certainty_factor(gdf[certainty_col])

    gdf = gdf.assign(_reservoir_score=scores)
    gdf = gdf[np.isfinite(gdf["_reservoir_score"])]
    if len(gdf) == 0:
        LOGGER.warning(
            "No mappable reservoir values in %s (column '%s').",
            path,
            horizon_config["reservoir_value_col"],
        )
        return None

    return rasterize_gdf(gdf, grid, "_reservoir_score")


#: Down-weighting applied to reservoir polygons flagged as uncertain.
CERTAINTY_FACTORS = {
    "sicher": 1.0,
    "certain": 1.0,
    "wahrscheinlich": 0.8,
    "probable": 0.8,
    "vermutet": 0.6,
    "unsicher": 0.6,
    "uncertain": 0.6,
}


def certainty_factor(series) -> np.ndarray:
    """Translate a certainty column into a multiplicative factor."""
    factors = []
    for value in series:
        key = normalize_mapping_key(value)
        if isinstance(key, str):
            factors.append(CERTAINTY_FACTORS.get(key, 1.0))
        elif isinstance(key, (int, float)) and np.isfinite(key):
            # Numeric certainty is interpreted as a percentage or a fraction.
            factors.append(float(key) / 100.0 if key > 1 else float(key))
        else:
            factors.append(1.0)
    return np.clip(np.asarray(factors, dtype="float64"), 0.0, 1.0)


def load_evidence_layer(layer_name: str, grid: Grid) -> Optional[np.ndarray]:
    """Load a single evidence layer onto ``grid``.

    Rasters are resampled, polygons rasterized and point layers converted
    into a distance-decay evidence grid within
    :data:`EVIDENCE_SEARCH_RADIUS_M`.
    """
    path = evidence_path(layer_name)
    if path is None:
        LOGGER.info("Evidence layer '%s' not found - skipped.", layer_name)
        return None

    suffix = os.path.splitext(path)[1].lower()
    if suffix in {".tif", ".tiff"}:
        return load_raster_to_grid(path, grid)

    gdf = read_vector(path)
    if len(gdf) == 0:
        LOGGER.warning("Evidence layer '%s' is empty.", layer_name)
        return None

    geom_types = set(gdf.geom_type.dropna().unique())
    if geom_types & {"Point", "MultiPoint"}:
        weights = None
        for candidate in ("weight", "score", "evidence", "value"):
            if candidate in gdf.columns:
                weights = np.asarray(gdf[candidate], dtype="float64")
                break
        return point_evidence_grid(gdf, grid, weights)

    value_col = None
    for candidate in ("weight", "score", "evidence", "value"):
        if candidate in gdf.columns:
            value_col = candidate
            break
    if value_col is None:
        gdf = gdf.assign(_evidence=1.0)
        value_col = "_evidence"
    return rasterize_gdf(gdf, grid, value_col)


def point_evidence_grid(
    gdf, grid: Grid, weights: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convert point evidence into a linear distance-decay grid."""
    xs = np.asarray(
        [geom.x for geom in gdf.geometry if geom is not None], dtype="float64"
    )
    ys = np.asarray(
        [geom.y for geom in gdf.geometry if geom is not None], dtype="float64"
    )
    if xs.size == 0:
        return np.zeros(grid.shape, dtype="float64")

    if weights is None or weights.size != xs.size:
        weights = np.ones_like(xs)
    weights = np.nan_to_num(weights, nan=1.0)

    gx, gy = grid.meshgrid()
    evidence = np.zeros(grid.shape, dtype="float64")
    radius = max(EVIDENCE_SEARCH_RADIUS_M, 1e-6)

    for x, y, weight in zip(xs, ys, weights):
        distance = np.hypot(gx - x, gy - y)
        contribution = weight * np.clip(1.0 - distance / radius, 0.0, None)
        evidence = np.maximum(evidence, contribution)

    max_value = float(np.nanmax(evidence)) if evidence.size else 0.0
    if max_value > 0:
        evidence = evidence / max_value
    return evidence


def load_evidence_layers(
    layer_names: Sequence[str], grid: Grid
) -> Dict[str, np.ndarray]:
    """Load all evidence layers of a horizon, skipping missing ones."""
    evidence: Dict[str, np.ndarray] = {}
    for name in layer_names:
        array = load_evidence_layer(name, grid)
        if array is not None:
            evidence[name] = array
    return evidence


def load_mask(path: str, grid: Grid) -> Optional[np.ndarray]:
    """Rasterize a polygon file into a boolean mask (``True`` inside)."""
    resolved = resolve_vector_path(path) if path else None
    if resolved is None:
        LOGGER.info("Mask source %s not available.", path)
        return None
    gdf = read_vector(resolved)
    if len(gdf) == 0:
        return None
    gdf = gdf.assign(_mask=1.0)
    array = rasterize_gdf(gdf, grid, "_mask")
    return np.isfinite(array) & (array > 0)


def determine_extent() -> Tuple[float, float, float, float]:
    """Determine the modelling extent.

    Priority: basin outline > first available thermal GeoTIFF >
    :data:`DEFAULT_EXTENT`.
    """
    basin_path = resolve_vector_path(BASIN_SHP_PATH)
    if basin_path is not None:
        gdf = read_vector(basin_path)
        if len(gdf) > 0:
            xmin, ymin, xmax, ymax = (float(v) for v in gdf.total_bounds)
            return (xmin, ymin, xmax, ymax)

    import rasterio
    from rasterio.warp import transform_bounds

    for config in HORIZONS.values():
        path = thermal_tif_path(config)
        if path is None:
            continue
        with rasterio.open(path) as src:
            bounds = src.bounds
            if src.crs is not None and src.crs.to_epsg() != TARGET_EPSG:
                bounds = transform_bounds(
                    src.crs, f"EPSG:{TARGET_EPSG}", *bounds
                )
            return tuple(float(v) for v in bounds)  # type: ignore[return-value]

    LOGGER.warning("No basin outline or thermal raster found; using default extent.")
    return DEFAULT_EXTENT


# ---------------------------------------------------------------------------
# PFA construction and VoterVeto
# ---------------------------------------------------------------------------


def build_pfa(
    grid: Grid,
    thermal_grid: np.ndarray,
    reservoir_grid: Optional[np.ndarray],
    evidence_grids: Mapping[str, np.ndarray],
    base_weight: float,
    evidence_weight: float,
    prior_favorability: float = PRIOR_FAVORABILITY,
) -> Dict[str, Any]:
    """Build the geopfa PFA configuration of a single horizon.

    The ``base`` component holds the thermal and reservoir layers and carries
    ``base_weight``; the ``evidence`` component holds the horizon evidence
    layers and carries ``evidence_weight``.
    """
    base_layers: Dict[str, Any] = {
        "thermal_use": {
            "model": grid_points_gdf(grid, thermal_grid),
            "model_data_col": "value",
            "transformation_method": "none",
            "weight": 1.0,
        }
    }
    if reservoir_grid is not None:
        base_layers["reservoir_quality"] = {
            "model": grid_points_gdf(grid, reservoir_grid),
            "model_data_col": "value",
            "transformation_method": "none",
            "weight": 1.0,
        }

    components: Dict[str, Any] = {
        "base": {
            "weight": float(base_weight),
            "pr0": float(prior_favorability),
            "layers": base_layers,
        }
    }

    if evidence_grids:
        components["evidence"] = {
            "weight": float(evidence_weight),
            "pr0": float(prior_favorability),
            "layers": {
                name: {
                    "model": grid_points_gdf(grid, array),
                    "model_data_col": "value",
                    "transformation_method": "none",
                    "weight": 1.0,
                }
                for name, array in evidence_grids.items()
            },
        }

    return {
        "criteria": {
            "geothermal": {
                "weight": 1.0,
                "components": components,
            }
        }
    }


def run_voter_veto(pfa: Mapping[str, Any], grid: Grid) -> np.ndarray:
    """Run ``VoterVeto.do_voter_veto`` and return the favourability grid."""
    from geopfa import transformation
    from geopfa.layer_combination import VoterVeto

    result = VoterVeto.do_voter_veto(
        dict(pfa),
        normalize_method=NORMALIZE_METHOD,
        component_veto=False,
        criteria_veto=True,
        normalize=True,
        norm_to=1,
    )
    favorability = transformation.rasterize_model_2d(
        result["pr_norm"], "favorability"
    ).astype("float64")

    if favorability.shape != grid.shape:
        raise ValueError(
            f"VoterVeto returned shape {favorability.shape}, expected {grid.shape}."
        )
    return favorability


def process_horizon(
    horizon_name: str,
    horizon_config: Mapping[str, Any],
    grid: Grid,
    basin_mask: Optional[np.ndarray] = None,
    salt_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Run the full VoterVeto workflow for a single horizon.

    Returns the horizon favourability grid, or ``None`` if the horizon cannot
    be modelled (missing thermal raster).
    """
    LOGGER.info("Processing horizon '%s' (%s)", horizon_name, horizon_config["label"])

    thermal_path = thermal_tif_path(horizon_config)
    if thermal_path is None:
        LOGGER.warning(
            "Thermal raster '%s.tif' not found in %s - horizon '%s' skipped.",
            horizon_config["thermal_layer"],
            THERMAL_TIF_DIR,
            horizon_name,
        )
        return None

    thermal_grid = load_raster_to_grid(thermal_path, grid)
    if not np.any(np.isfinite(thermal_grid)):
        LOGGER.warning(
            "Thermal raster for '%s' contains no valid data - skipped.",
            horizon_name,
        )
        return None

    reservoir_grid = load_and_rasterize_reservoir(horizon_config, grid)
    evidence_grids = load_evidence_layers(
        horizon_config.get("evidence_layers", []), grid
    )

    pfa = build_pfa(
        grid,
        thermal_grid,
        reservoir_grid,
        evidence_grids,
        base_weight=float(horizon_config["base_weight"]),
        evidence_weight=float(horizon_config["evidence_weight"]),
    )

    favorability = run_voter_veto(pfa, grid)

    # Reservoir veto: no reservoir -> no horizon-specific resource.
    if reservoir_grid is not None:
        favorability = np.where(
            np.isfinite(reservoir_grid) & (reservoir_grid > 0),
            favorability,
            np.nan,
        )

    # Confidence weight scales the horizon favourability.
    favorability = favorability * float(horizon_config.get("confidence_weight", 1.0))

    favorability = apply_masks(favorability, basin_mask, salt_mask)
    return favorability


def apply_masks(
    array: np.ndarray,
    basin_mask: Optional[np.ndarray],
    salt_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Restrict to the basin outline and veto salt structures."""
    result = np.asarray(array, dtype="float64").copy()
    if basin_mask is not None:
        result = np.where(basin_mask, result, np.nan)
    if salt_mask is not None:
        result = np.where(salt_mask, np.nan, result)
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggregationResult:
    """Composite products derived from the individual horizon grids."""

    horizon_names: List[str]
    composite: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    best_horizon_id: np.ndarray
    primary_threshold: float
    secondary_threshold: float


def aggregate_horizons(
    horizon_results: Mapping[str, np.ndarray],
    primary_quantile: float = PRIMARY_QUANTILE,
    secondary_quantile: float = SECONDARY_QUANTILE,
) -> AggregationResult:
    """Aggregate horizon favourability grids into the composite products."""
    if not horizon_results:
        raise ValueError("No horizon results to aggregate.")

    names = [name for name in HORIZONS if name in horizon_results]
    names += [name for name in horizon_results if name not in HORIZONS]

    stack = np.stack([np.asarray(horizon_results[n], dtype="float64") for n in names])
    all_nan = np.all(np.isnan(stack), axis=0)

    # ``-inf`` fill keeps NaN cells out of the max/argmax without emitting
    # "All-NaN slice" warnings; fully empty cells are masked afterwards.
    filled = np.where(np.isnan(stack), -np.inf, stack)
    composite = np.where(all_nan, np.nan, np.max(filled, axis=0))
    best_index = np.argmax(filled, axis=0)
    # Map the stack index onto the stable, 1-based horizon id.
    id_lookup = np.array(
        [HORIZON_IDS.get(name, index + 1) for index, name in enumerate(names)],
        dtype="int16",
    )
    best_horizon_id = np.where(all_nan, INT_NODATA, id_lookup[best_index]).astype(
        "int16"
    )

    valid = composite[np.isfinite(composite) & (composite > 0)]
    if valid.size:
        primary_threshold = float(np.quantile(valid, primary_quantile))
        secondary_threshold = float(np.quantile(valid, secondary_quantile))
    else:
        primary_threshold = float("nan")
        secondary_threshold = float("nan")

    primary = mask_by_threshold(composite, primary_threshold)
    secondary = mask_by_threshold(composite, secondary_threshold)

    return AggregationResult(
        horizon_names=names,
        composite=composite,
        primary=primary,
        secondary=secondary,
        best_horizon_id=best_horizon_id,
        primary_threshold=primary_threshold,
        secondary_threshold=secondary_threshold,
    )


def mask_by_threshold(composite: np.ndarray, threshold: float) -> np.ndarray:
    """Keep composite values ``>= threshold``, everything else becomes NaN."""
    if not np.isfinite(threshold):
        return np.full_like(composite, np.nan)
    with np.errstate(invalid="ignore"):
        return np.where(composite >= threshold, composite, np.nan)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_geotiff(
    array: np.ndarray, path: str, grid: Grid, dtype: str = "float32"
) -> str:
    """Write ``array`` as a single band GeoTIFF in EPSG:31467."""
    import rasterio
    from rasterio.crs import CRS

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    data = np.asarray(array)
    if np.issubdtype(np.dtype(dtype), np.integer):
        nodata: float = INT_NODATA
        data = np.nan_to_num(data, nan=INT_NODATA).astype(dtype)
    else:
        nodata = FLOAT_NODATA
        data = np.where(np.isfinite(data), data, FLOAT_NODATA).astype(dtype)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.ny,
        width=grid.nx,
        count=1,
        dtype=dtype,
        crs=CRS.from_epsg(TARGET_EPSG),
        transform=grid.transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)
    return path


def export_horizon_id_mapping(path: str, horizon_names: Sequence[str]) -> str:
    """Write the ``best_horizon_id`` legend as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["horizon_id", "horizon_name", "label"])
        writer.writerow([INT_NODATA, "nodata", "No favourable horizon"])
        for name in horizon_names:
            writer.writerow(
                [
                    HORIZON_IDS.get(name, ""),
                    name,
                    HORIZONS.get(name, {}).get("label", name),
                ]
            )
    return path


def summarize_grid(array: np.ndarray) -> Dict[str, float]:
    """Descriptive statistics of a favourability grid."""
    values = np.asarray(array, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "valid_cells": 0,
            "min": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
        }
    return {
        "valid_cells": int(finite.size),
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def export_summary_statistics(
    path: str,
    horizon_results: Mapping[str, np.ndarray],
    aggregation: Optional[AggregationResult] = None,
) -> str:
    """Write per-horizon and composite statistics as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "horizon",
        "horizon_id",
        "label",
        "base_weight",
        "evidence_weight",
        "confidence_weight",
        "valid_cells",
        "min",
        "mean",
        "median",
        "max",
        "p90",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, array in horizon_results.items():
            config = HORIZONS.get(name, {})
            row = {
                "horizon": name,
                "horizon_id": HORIZON_IDS.get(name, ""),
                "label": config.get("label", name),
                "base_weight": config.get("base_weight", ""),
                "evidence_weight": config.get("evidence_weight", ""),
                "confidence_weight": config.get("confidence_weight", ""),
            }
            row.update(summarize_grid(array))
            writer.writerow(row)

        if aggregation is not None:
            for label, array in (
                ("COMPOSITE", aggregation.composite),
                ("PRIMARY", aggregation.primary),
                ("SECONDARY", aggregation.secondary),
            ):
                row = {
                    "horizon": label,
                    "horizon_id": "",
                    "label": label.title(),
                    "base_weight": "",
                    "evidence_weight": "",
                    "confidence_weight": "",
                }
                row.update(summarize_grid(array))
                writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def horizon_output_dir(output_dir: str = OUTPUT_DIR) -> str:
    return os.path.join(output_dir, HORIZON_OUTPUT_SUBDIR)


def run_pipeline(
    horizons: Optional[Sequence[str]] = None,
    output_dir: str = OUTPUT_DIR,
    grid: Optional[Grid] = None,
) -> Dict[str, Any]:
    """Run the complete horizon-specific workflow.

    Returns a dictionary with the grid, the per-horizon favourability arrays,
    the aggregation result and the list of written files.
    """
    selected = list(horizons) if horizons else list(HORIZONS)
    unknown = [name for name in selected if name not in HORIZONS]
    if unknown:
        raise KeyError(f"Unknown horizon(s): {unknown}")

    if grid is None:
        grid = build_grid(determine_extent(), GRID_NX, GRID_NY)
    LOGGER.info(
        "Model grid: %s x %s cells, extent %s, EPSG:%s",
        grid.nx,
        grid.ny,
        grid.extent,
        TARGET_EPSG,
    )

    basin_mask = load_mask(BASIN_SHP_PATH, grid)
    salt_mask = load_mask(SALT_SHP_PATH, grid)

    target_dir = horizon_output_dir(output_dir)
    os.makedirs(target_dir, exist_ok=True)

    horizon_results: Dict[str, np.ndarray] = {}
    skipped: List[str] = []
    written: List[str] = []

    for name in selected:
        favorability = process_horizon(
            name, HORIZONS[name], grid, basin_mask, salt_mask
        )
        if favorability is None:
            skipped.append(name)
            continue
        horizon_results[name] = favorability
        written.append(
            export_geotiff(
                favorability,
                os.path.join(target_dir, f"{name}_geothermal_favourability.tif"),
                grid,
            )
        )

    if not horizon_results:
        raise RuntimeError(
            "No horizon could be processed. Check THERMAL_TIF_DIR "
            f"({THERMAL_TIF_DIR}) and the reservoir/evidence data locations."
        )

    aggregation = aggregate_horizons(horizon_results)

    written.append(
        export_geotiff(
            aggregation.composite,
            os.path.join(
                target_dir, "best_horizon_geothermal_favourability.tif"
            ),
            grid,
        )
    )
    written.append(
        export_geotiff(
            aggregation.primary,
            os.path.join(target_dir, "PRIMARY_geothermal_favourability.tif"),
            grid,
        )
    )
    written.append(
        export_geotiff(
            aggregation.secondary,
            os.path.join(target_dir, "SECONDARY_geothermal_favourability.tif"),
            grid,
        )
    )
    written.append(
        export_geotiff(
            aggregation.best_horizon_id,
            os.path.join(target_dir, "best_horizon_id.tif"),
            grid,
            dtype="int16",
        )
    )
    written.append(
        export_horizon_id_mapping(
            os.path.join(target_dir, "best_horizon_id_mapping.csv"),
            aggregation.horizon_names,
        )
    )
    written.append(
        export_summary_statistics(
            os.path.join(target_dir, "summary.csv"),
            horizon_results,
            aggregation,
        )
    )

    if skipped:
        LOGGER.warning("Skipped horizons (missing input data): %s", ", ".join(skipped))

    return {
        "grid": grid,
        "horizon_results": horizon_results,
        "aggregation": aggregation,
        "skipped": skipped,
        "output_dir": target_dir,
        "written": written,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Horizon-specific geothermal favourability (VoterVeto per horizon, "
            "then aggregation) for the North German Basin, EPSG:31467."
        )
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=None,
        help="Subset of horizons to process (default: all 18).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Root output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--nx", type=int, default=GRID_NX, help="Grid columns (default: %(default)s)."
    )
    parser.add_argument(
        "--ny", type=int, default=GRID_NY, help="Grid rows (default: %(default)s)."
    )
    parser.add_argument(
        "--list-horizons",
        action="store_true",
        help="Print the horizon configuration and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.list_horizons:
        for name, config in HORIZONS.items():
            print(
                f"{HORIZON_IDS[name]:2d} {name:<12} {config['label']:<30} "
                f"thermal={config['thermal_layer']}.tif "
                f"reservoir={config['reservoir_layer']} "
                f"base={config['base_weight']} "
                f"evidence={config['evidence_weight']} "
                f"confidence={config['confidence_weight']}"
            )
        return 0

    grid = build_grid(determine_extent(), args.nx, args.ny)
    result = run_pipeline(
        horizons=args.horizons, output_dir=args.output_dir, grid=grid
    )
    LOGGER.info(
        "Wrote %s files to %s", len(result["written"]), result["output_dir"]
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
