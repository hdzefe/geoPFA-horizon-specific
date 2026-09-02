#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Horizon-specific geothermal favourability mapping for the North German Basin.

This script is a *refactor* of the original monolithic GeoPFA workflow.  The
geological logic (horizon configuration, reservoir column names, value
mappings, evidence layers, weights and salt penalty) is preserved exactly as it
was in the production script.  Only two things changed:

1. **Thermal source** -- the thermal criterion is no longer interpolated from
   GeotiS point measurements.  Instead the pre-computed, depth-corrected
   ``{horizon}_thermal_use.tif`` GeoTIFF rasters are read with ``rasterio`` and
   resampled onto the analysis grid.
2. **Aggregation** -- the global ``VoterVeto`` layer combination has been
   replaced with an independent, per-horizon scoring pass followed by an
   explicit aggregation step (best horizon, best-horizon-ID grid, PRIMARY and
   SECONDARY sweet-spot masks).

Everything else -- normalisation, clipping, evidence weighting, confidence
weighting and the salt-structure penalty -- behaves exactly as before.

Coordinate reference system
---------------------------
All rasters are written in **EPSG:31467** (DHDN / Gauss-Krueger zone 3), which
is the project CRS for the North German Basin.  Input vector and raster layers
are re-projected to this CRS on the fly when they carry a different CRS.

Usage
-----
Run with the built-in defaults (development resolution)::

    python ngb_refactored_horizon_specific_tif_input.py

Publication resolution::

    python ngb_refactored_horizon_specific_tif_input.py --nx 512 --ny 512

Every path can be overridden with an environment variable (see the
``Configuration`` section below) or with a command line switch, so the script
runs unchanged on the production Windows machine and in CI.

Outputs (written to ``<OUTPUT_DIR>/favourability/geothermal/horizon_specific``)::

    detfurth_geothermal_favourability.tif
    tsf1_geothermal_favourability.tif
    ...
    best_horizon_geothermal_favourability.tif
    PRIMARY_geothermal_favourability.tif
    SECONDARY_geothermal_favourability.tif
    best_horizon_id.tif
    best_horizon_id_mapping.csv
    summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("ngb.horizon_specific")

# ---------------------------------------------------------------------------
# Optional third-party imports.
#
# ``rasterio``/``geopandas`` are hard requirements for a real run, but importing
# them lazily keeps the module importable (and unit-testable in parts) on
# machines where the geospatial stack is not installed.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.warp import reproject

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover - exercised only without rasterio
    rasterio = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]
    transform_from_bounds = None  # type: ignore[assignment]
    reproject = None  # type: ignore[assignment]
    HAS_RASTERIO = False

try:  # pragma: no cover - trivial import guard
    import geopandas as gpd
    from rasterio import features as rasterio_features

    HAS_GEOPANDAS = True
except ImportError:  # pragma: no cover - exercised only without geopandas
    gpd = None  # type: ignore[assignment]
    rasterio_features = None  # type: ignore[assignment]
    HAS_GEOPANDAS = False


# ===========================================================================
# Configuration
# ===========================================================================

#: Target CRS for **all** outputs.  DHDN / Gauss-Krueger zone 3.
TARGET_EPSG: int = int(os.environ.get("NGB_TARGET_EPSG", "31467"))

#: Analysis grid size.  320x320 for development, 512x512 for publication.
NX: int = int(os.environ.get("NGB_NX", "320"))
NY: int = int(os.environ.get("NGB_NY", "320"))

#: Project root on the production machine.
PROJECT_ROOT: str = os.environ.get(
    "NGB_PROJECT_ROOT", r"D:\geoPFA_Projects\north_german_basin"
)

#: Directory holding the pre-computed ``{horizon}_thermal_use.tif`` rasters.
THERMAL_TIF_DIR: str = os.environ.get(
    "NGB_THERMAL_TIF_DIR",
    os.path.join(
        PROJECT_ROOT,
        "data",
        "output",
        "favourability",
        "geothermal",
        "horizon_specific",
    ),
)

#: Root directory of the raw data.  Used to relocate the per-horizon reservoir
#: shapefile paths when the project is not mounted at ``PROJECT_ROOT``.
RAW_DATA_DIR: str = os.environ.get(
    "NGB_RAW_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "raw")
)

#: Directory with the evidence layers (boreholes, deep hydrothermal sites).
EVIDENCE_SHP_DIR: str = os.environ.get(
    "NGB_EVIDENCE_SHP_DIR", os.path.join(RAW_DATA_DIR, "evidence")
)

#: Basin outline defining the analysis extent and the study-area mask.
BASIN_SHP_PATH: str = os.environ.get(
    "NGB_BASIN_SHP_PATH",
    os.path.join(
        RAW_DATA_DIR, "north_german_basin_extention", "north_german_basin_.shp"
    ),
)

#: Salt structures used for the penalty term.
SALT_SHP_PATH: str = os.environ.get(
    "NGB_SALT_SHP_PATH",
    os.path.join(RAW_DATA_DIR, "salt_structures", "Salzstrukturen_Inspee__v1_poly.shp"),
)

#: Output root.  The horizon-specific sub-directory is created below it.
OUTPUT_DIR: str = os.environ.get(
    "NGB_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "data", "output")
)

#: Relative sub-directory for the results of this workflow.
OUTPUT_SUBDIR: str = os.path.join(
    "favourability", "geothermal", "horizon_specific"
)

#: Suffix of the pre-computed thermal rasters.
THERMAL_TIF_SUFFIX: str = "_thermal_use.tif"

#: Nodata value used for the exported float rasters.
NODATA_VALUE: float = -9999.0

#: Strength of the salt-structure penalty (1.0 == full veto inside salt).
SALT_PENALTY_STRENGTH: float = float(os.environ.get("NGB_SALT_PENALTY", "1.0"))

#: Search radius (in CRS units, i.e. metres) around an evidence point within
#: which the evidence support decays linearly from 1.0 to 0.0.
EVIDENCE_INFLUENCE_RADIUS: float = float(
    os.environ.get("NGB_EVIDENCE_RADIUS", "15000")
)

#: Quantiles defining the PRIMARY / SECONDARY sweet-spot masks.
PRIMARY_QUANTILE: float = 0.90
SECONDARY_QUANTILE: float = 0.75


# ---------------------------------------------------------------------------
# Value mappings (exact from the original script)
# ---------------------------------------------------------------------------

RESERVOIR_POTENTIAL_MAPPING_DE: Dict[Any, float] = {
    "niedrig": 0.2,
    "mittel": 0.6,
    "hoch": 1.0,
}

RESERVOIR_POTENTIAL_MAPPING_DE_RESTRICTED: Dict[Any, float] = {
    "niedrig": 0.2,
    "eingeschraenkt": 0.2,
    "eingeschränkt": 0.2,
    "mittel": 0.6,
    "hoch": 1.0,
}

RESERVOIR_THICKNESS_MAPPING_EN: Dict[Any, float] = {
    "low (< 10 m)": 0.2,
    "moderate (10-20 m)": 0.6,
    "high (> 20 m)": 1.0,
}

RESERVOIR_BINARY_MAPPING_VALANGINIAN: Dict[Any, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
}

RESERVOIR_BINARY_MAPPING_BUECKEBERG: Dict[Any, float] = {
    2: 1.0,
    3: 1.0,
    7: 1.0,
    11: 1.0,
}


# ---------------------------------------------------------------------------
# Horizon configuration -- ALL 18 horizons, exactly as in the original script
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
            RAW_DATA_DIR, "reservoirs", "tsf1-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "Tsf2-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "k4-2-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "k4-3-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "k4-4-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "Het1_Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "Het2_Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "Sin1_Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "Sin2_Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "Pli1_Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "Pli2_Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "toa1-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "toa2-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "aal1-Potential.shp"
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
            RAW_DATA_DIR, "reservoirs", "bj1-Potential.shp"
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
        "base_weight": 0.9,
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


# ===========================================================================
# Small helpers
# ===========================================================================


class MissingDependencyError(RuntimeError):
    """Raised when a required geospatial dependency is not installed."""


def require_rasterio() -> None:
    """Raise a helpful error when ``rasterio`` is unavailable."""
    if not HAS_RASTERIO:
        raise MissingDependencyError(
            "rasterio is required for raster I/O. Install it with "
            "'pip install rasterio'."
        )


def require_geopandas() -> None:
    """Raise a helpful error when ``geopandas`` is unavailable."""
    if not HAS_GEOPANDAS:
        raise MissingDependencyError(
            "geopandas is required for vector I/O. Install it with "
            "'pip install geopandas'."
        )


def setup_logging(verbosity: int = 1, log_file: Optional[str] = None) -> None:
    """Configure the module logger.

    Parameters
    ----------
    verbosity:
        ``0`` -> WARNING, ``1`` -> INFO, ``>=2`` -> DEBUG.
    log_file:
        Optional path of an additional log file.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    LOGGER.setLevel(level)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)

    LOGGER.propagate = False


def normalise_key(value: Any) -> Any:
    """Normalise a lookup key so mappings tolerate real-world attribute noise.

    Strings are lower-cased and stripped; numeric values that represent whole
    numbers are converted to ``int`` so that ``2.0`` matches a mapping key of
    ``2``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
        return value
    if isinstance(value, np.generic):
        return normalise_key(value.item())
    return value


def build_normalised_mapping(mapping: Mapping[Any, float]) -> Dict[Any, float]:
    """Return a copy of ``mapping`` whose keys are normalised."""
    return {normalise_key(key): float(val) for key, val in mapping.items()}


def normalise_array(
    data: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Linearly rescale ``data`` to ``[0, 1]`` and clip.

    ``NaN`` values are preserved.  A degenerate range (``vmax <= vmin``) yields
    zeros, which mirrors the behaviour of the original script.
    """
    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, np.nan, dtype=float)

    lo = float(np.nanmin(arr[finite])) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr[finite])) if vmax is None else float(vmax)

    out = np.full(arr.shape, np.nan, dtype=float)
    if hi <= lo:
        out[finite] = 0.0
        return out

    out[finite] = (arr[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def safe_quantile(data: np.ndarray, q: float) -> Optional[float]:
    """Quantile of the strictly positive, finite values of ``data``.

    Returns ``None`` when there is no valid data, so callers can degrade
    gracefully instead of raising.
    """
    arr = np.asarray(data, dtype=float)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return None
    return float(np.quantile(valid, q))


# ===========================================================================
# Grid definition
# ===========================================================================


@dataclass(frozen=True)
class Grid:
    """Regular analysis grid in the target CRS.

    Attributes
    ----------
    extent:
        ``(xmin, ymin, xmax, ymax)`` in target-CRS units.
    nx, ny:
        Number of columns / rows.
    epsg:
        EPSG code of the grid CRS (always :data:`TARGET_EPSG` in production).
    """

    extent: Tuple[float, float, float, float]
    nx: int
    ny: int
    epsg: int = TARGET_EPSG

    def __post_init__(self) -> None:
        xmin, ymin, xmax, ymax = self.extent
        if not (xmax > xmin and ymax > ymin):
            raise ValueError(f"Invalid extent: {self.extent!r}")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"Invalid grid size: {self.nx}x{self.ny}")

    @property
    def shape(self) -> Tuple[int, int]:
        """Array shape ``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def transform(self):  # pragma: no cover - thin rasterio wrapper
        """Affine transform of the grid (north-up)."""
        require_rasterio()
        xmin, ymin, xmax, ymax = self.extent
        return transform_from_bounds(xmin, ymin, xmax, ymax, self.nx, self.ny)

    @property
    def crs(self) -> str:
        """CRS string, e.g. ``"EPSG:31467"``."""
        return f"EPSG:{self.epsg}"

    @property
    def cell_size(self) -> Tuple[float, float]:
        """Cell size ``(dx, dy)`` in CRS units."""
        xmin, ymin, xmax, ymax = self.extent
        return ((xmax - xmin) / self.nx, (ymax - ymin) / self.ny)

    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the ``(x, y)`` coordinates of all cell centres."""
        xmin, ymin, xmax, ymax = self.extent
        dx, dy = self.cell_size
        xs = xmin + dx * (np.arange(self.nx) + 0.5)
        ys = ymax - dy * (np.arange(self.ny) + 0.5)
        return np.meshgrid(xs, ys)

    def empty(self, fill: float = np.nan) -> np.ndarray:
        """Return an array matching the grid, filled with ``fill``."""
        return np.full(self.shape, fill, dtype=float)


def expand_extent(
    extent: Tuple[float, float, float, float], fraction: float = 0.0
) -> Tuple[float, float, float, float]:
    """Grow ``extent`` by ``fraction`` of its width/height on every side."""
    xmin, ymin, xmax, ymax = extent
    dx = (xmax - xmin) * fraction
    dy = (ymax - ymin) * fraction
    return (xmin - dx, ymin - dy, xmax + dx, ymax + dy)


# ===========================================================================
# Vector helpers
# ===========================================================================


def read_vector(path: str, target_epsg: int = TARGET_EPSG):
    """Read a vector file and re-project it to ``target_epsg``.

    A layer without CRS information is *assumed* to already be in the target
    CRS -- this matches the behaviour of the original script and is logged.
    """
    require_geopandas()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vector file not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        LOGGER.warning(
            "Layer '%s' has no CRS; assuming EPSG:%s", path, target_epsg
        )
        gdf = gdf.set_crs(epsg=target_epsg)
    elif gdf.crs.to_epsg() != target_epsg:
        LOGGER.debug(
            "Re-projecting '%s' from %s to EPSG:%s", path, gdf.crs, target_epsg
        )
        gdf = gdf.to_crs(epsg=target_epsg)
    return gdf


def load_basin_grid(
    basin_shp_path: str,
    nx: int,
    ny: int,
    target_epsg: int = TARGET_EPSG,
    margin: float = 0.0,
) -> Tuple[Grid, np.ndarray]:
    """Build the analysis grid and the basin mask from the basin outline.

    Returns
    -------
    (grid, mask)
        ``mask`` is a boolean array (``True`` inside the basin).
    """
    gdf = read_vector(basin_shp_path, target_epsg)
    if gdf.empty:
        raise ValueError(f"Basin outline '{basin_shp_path}' contains no features")

    bounds = tuple(float(v) for v in gdf.total_bounds)  # (xmin, ymin, xmax, ymax)
    extent = expand_extent(bounds, margin)  # type: ignore[arg-type]
    grid = Grid(extent=extent, nx=nx, ny=ny, epsg=target_epsg)

    mask = rasterize_geometries(gdf.geometry, grid, fill=0.0, default_value=1.0) > 0.5
    LOGGER.info(
        "Analysis grid: %dx%d cells, extent=%s, cell size=%.1f x %.1f m",
        grid.nx,
        grid.ny,
        tuple(round(v, 1) for v in grid.extent),
        grid.cell_size[0],
        grid.cell_size[1],
    )
    LOGGER.info(
        "Basin mask covers %d of %d cells (%.1f %%)",
        int(mask.sum()),
        mask.size,
        100.0 * mask.sum() / mask.size,
    )
    return grid, mask


def rasterize_geometries(
    geometries: Iterable,
    grid: Grid,
    values: Optional[Sequence[float]] = None,
    fill: float = np.nan,
    default_value: float = 1.0,
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterise geometries onto ``grid``.

    When ``values`` is given it must have the same length as ``geometries`` and
    supplies the burn value for each feature; otherwise ``default_value`` is
    burned everywhere.
    """
    require_geopandas()
    require_rasterio()

    geoms = [geom for geom in geometries if geom is not None and not geom.is_empty]
    if values is None:
        shapes = [(geom, float(default_value)) for geom in geoms]
    else:
        pairs = [
            (geom, float(val))
            for geom, val in zip(geometries, values)
            if geom is not None and not geom.is_empty and val is not None
            and np.isfinite(val)
        ]
        shapes = pairs

    if not shapes:
        return grid.empty(fill)

    return rasterio_features.rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=fill,
        all_touched=all_touched,
        dtype="float64",
    )


def resolve_column(columns: Sequence[str], wanted: str) -> Optional[str]:
    """Resolve a column name case-insensitively.

    Real shapefiles frequently differ in case (``POTENTIAL`` vs ``Potential``)
    or are truncated by the DBF 10-character limit, so an exact match is tried
    first, then a case-insensitive match, then a truncated prefix match.
    """
    if wanted in columns:
        return wanted
    lowered = {str(col).lower(): col for col in columns}
    if wanted.lower() in lowered:
        return lowered[wanted.lower()]
    truncated = wanted.lower()[:10]
    if truncated in lowered:
        return lowered[truncated]
    for low, original in lowered.items():
        if low.startswith(truncated) or truncated.startswith(low):
            return original
    return None


# ===========================================================================
# Thermal loading (NEW: pre-computed GeoTIFF instead of interpolation)
# ===========================================================================


def load_thermal_tif(
    thermal_tif_path: str,
    grid: Grid,
    normalise: bool = True,
) -> Optional[np.ndarray]:
    """Load a pre-computed ``{horizon}_thermal_use.tif`` onto the grid.

    The raster is re-projected/resampled (bilinear) to the analysis grid when
    its CRS, extent or shape differ.  Nodata cells become ``NaN``.

    Returns ``None`` when the file does not exist, so the caller can skip the
    horizon with a warning instead of aborting the whole run.
    """
    if not thermal_tif_path or not os.path.exists(thermal_tif_path):
        LOGGER.warning("Thermal raster not found: %s", thermal_tif_path)
        return None

    require_rasterio()

    with rasterio.open(thermal_tif_path) as src:
        data = src.read(1).astype(float)
        src_nodata = src.nodata
        src_crs = src.crs
        src_transform = src.transform

        if src_nodata is not None:
            data = np.where(np.isclose(data, float(src_nodata)), np.nan, data)

        if src_crs is None:
            LOGGER.warning(
                "Thermal raster '%s' has no CRS; assuming %s",
                thermal_tif_path,
                grid.crs,
            )
            src_crs = rasterio.crs.CRS.from_epsg(grid.epsg)

        destination = np.full(grid.shape, np.nan, dtype=float)
        reproject(
            source=data,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    if not np.isfinite(destination).any():
        LOGGER.warning(
            "Thermal raster '%s' has no valid data on the analysis grid",
            thermal_tif_path,
        )
        return None

    LOGGER.debug(
        "Thermal raster '%s' loaded: min=%.3f max=%.3f",
        os.path.basename(thermal_tif_path),
        float(np.nanmin(destination)),
        float(np.nanmax(destination)),
    )

    if normalise:
        destination = normalise_array(destination)
    return destination


def thermal_tif_path_for(horizon_name: str, config: Mapping[str, Any],
                         thermal_dir: str) -> str:
    """Return the expected thermal GeoTIFF path for a horizon."""
    layer = str(config.get("thermal_layer") or f"{horizon_name}_thermal_use")
    filename = layer if layer.lower().endswith(".tif") else f"{layer}.tif"
    return os.path.join(thermal_dir, filename)


# ===========================================================================
# Reservoir loading
# ===========================================================================


def map_reservoir_values(
    series,
    value_mapping: Optional[Mapping[Any, float]] = None,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
) -> np.ndarray:
    """Convert a reservoir attribute column into favourability values in [0, 1].

    Two modes, exactly as in the original script:

    * **categorical** -- ``value_mapping`` translates class labels
      (``niedrig``/``mittel``/``hoch``, ``low (< 10 m)``/... or the LBEG
      sandstone codes) into scores.  Unmapped classes score ``0``.
    * **continuous** -- ``value_min``/``value_max`` linearly rescale numeric
      values (e.g. the Detfurth ``sand_share`` isolines 30 % ... 90 %).
    """
    raw = list(series)

    if value_mapping is not None:
        lookup = build_normalised_mapping(value_mapping)
        out = np.zeros(len(raw), dtype=float)
        unmapped: Dict[Any, int] = {}
        for i, value in enumerate(raw):
            key = normalise_key(value)
            if key in lookup:
                out[i] = lookup[key]
            elif key is not None:
                unmapped[key] = unmapped.get(key, 0) + 1
        if unmapped:
            LOGGER.warning(
                "Unmapped reservoir classes (scored 0.0): %s",
                ", ".join(f"{k!r} x{v}" for k, v in sorted(
                    unmapped.items(), key=lambda item: str(item[0]))),
            )
        return np.clip(out, 0.0, 1.0)

    numeric = np.array(
        [np.nan if value is None else _to_float(value) for value in raw],
        dtype=float,
    )
    if value_min is None or value_max is None:
        return normalise_array(numeric)
    return normalise_array(numeric, vmin=value_min, vmax=value_max)


def _to_float(value: Any) -> float:
    """Best-effort float conversion; non-numeric values become ``NaN``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_and_rasterize_reservoir(
    shp_path: str,
    grid: Grid,
    value_col: Optional[str] = None,
    value_mapping: Optional[Mapping[Any, float]] = None,
    value_min: Optional[float] = None,
    value_max: Optional[float] = None,
    certainty_col: Optional[str] = None,
    target_epsg: int = TARGET_EPSG,
) -> Optional[np.ndarray]:
    """Load a reservoir shapefile and rasterise it into a ``[0, 1]`` grid.

    Returns ``None`` when the shapefile is missing so the horizon can be
    skipped with a warning.
    """
    if not shp_path or not os.path.exists(shp_path):
        LOGGER.warning("Reservoir shapefile not found: %s", shp_path)
        return None

    gdf = read_vector(shp_path, target_epsg)
    if gdf.empty:
        LOGGER.warning("Reservoir shapefile '%s' contains no features", shp_path)
        return None

    columns = list(gdf.columns)
    if value_col is None:
        LOGGER.info(
            "No reservoir value column configured for '%s'; using presence/absence",
            os.path.basename(shp_path),
        )
        values = np.ones(len(gdf), dtype=float)
    else:
        resolved = resolve_column(columns, value_col)
        if resolved is None:
            LOGGER.warning(
                "Reservoir column '%s' not found in '%s' (available: %s); "
                "falling back to presence/absence",
                value_col,
                os.path.basename(shp_path),
                ", ".join(str(c) for c in columns if c != "geometry"),
            )
            values = np.ones(len(gdf), dtype=float)
        else:
            if resolved != value_col:
                LOGGER.info(
                    "Reservoir column '%s' resolved to '%s' in '%s'",
                    value_col,
                    resolved,
                    os.path.basename(shp_path),
                )
            values = map_reservoir_values(
                gdf[resolved],
                value_mapping=value_mapping,
                value_min=value_min,
                value_max=value_max,
            )

    if certainty_col:
        resolved_cert = resolve_column(columns, certainty_col)
        if resolved_cert is None:
            LOGGER.warning(
                "Certainty column '%s' not found in '%s'; ignoring",
                certainty_col,
                os.path.basename(shp_path),
            )
        else:
            certainty = normalise_certainty(gdf[resolved_cert])
            values = values * certainty

    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    raster = rasterize_geometries(gdf.geometry, grid, values=values, fill=0.0)
    raster = np.clip(np.nan_to_num(raster, nan=0.0), 0.0, 1.0)

    LOGGER.debug(
        "Reservoir '%s': %d features, mean score on grid=%.4f",
        os.path.basename(shp_path),
        len(gdf),
        float(np.nanmean(raster)),
    )
    return raster


#: Textual certainty classes encountered in the Detfurth sand-zone layer.
CERTAINTY_MAPPING: Dict[Any, float] = {
    "gesichert": 1.0,
    "sicher": 1.0,
    "certain": 1.0,
    "high": 1.0,
    "hoch": 1.0,
    "wahrscheinlich": 0.7,
    "moderate": 0.7,
    "mittel": 0.7,
    "vermutet": 0.4,
    "unsicher": 0.4,
    "uncertain": 0.4,
    "low": 0.4,
    "niedrig": 0.4,
}


def normalise_certainty(series) -> np.ndarray:
    """Convert a certainty column into a multiplicative factor in ``[0, 1]``.

    Numeric columns are rescaled (values >1 are treated as percentages),
    textual columns use :data:`CERTAINTY_MAPPING`; unknown entries default to
    ``1.0`` so that a poorly populated certainty column never zeroes a horizon.
    """
    raw = list(series)
    numeric = np.array([_to_float(value) for value in raw], dtype=float)

    if np.isfinite(numeric).any() and np.isfinite(numeric).sum() >= len(raw) / 2:
        finite_max = float(np.nanmax(numeric[np.isfinite(numeric)]))
        if finite_max > 1.0:
            numeric = numeric / (100.0 if finite_max <= 100.0 else finite_max)
        return np.clip(np.nan_to_num(numeric, nan=1.0), 0.0, 1.0)

    out = np.ones(len(raw), dtype=float)
    for i, value in enumerate(raw):
        key = normalise_key(value)
        if key in CERTAINTY_MAPPING:
            out[i] = CERTAINTY_MAPPING[key]
    return out


# ===========================================================================
# Evidence layers (logic preserved from the original script)
# ===========================================================================


def evidence_path_candidates(layer_name: str, evidence_dir: str) -> List[str]:
    """Return plausible file locations for an evidence layer."""
    candidates = []
    for ext in (".shp", ".gpkg", ".geojson"):
        candidates.append(os.path.join(evidence_dir, f"{layer_name}{ext}"))
    return candidates


def load_evidence_layer(
    layer_name: str,
    grid: Grid,
    evidence_dir: str,
    influence_radius: float = EVIDENCE_INFLUENCE_RADIUS,
    target_epsg: int = TARGET_EPSG,
) -> Optional[np.ndarray]:
    """Load a single evidence layer and convert it to a ``[0, 1]`` support grid.

    Polygon evidence is burned directly; point/line evidence (boreholes with
    poro-perm data, deep hydrothermal sites) receives a linearly decaying
    influence halo of ``influence_radius`` metres, exactly as in the original
    workflow.

    Returns ``None`` when no source file exists for the layer.
    """
    path = next(
        (p for p in evidence_path_candidates(layer_name, evidence_dir)
         if os.path.exists(p)),
        None,
    )
    if path is None:
        LOGGER.debug("Evidence layer '%s' not available; skipped", layer_name)
        return None

    gdf = read_vector(path, target_epsg)
    if gdf.empty:
        LOGGER.warning("Evidence layer '%s' contains no features", layer_name)
        return None

    geom_types = set(gdf.geom_type.unique())
    if geom_types <= {"Polygon", "MultiPolygon"}:
        raster = rasterize_geometries(gdf.geometry, grid, fill=0.0, default_value=1.0)
        return np.clip(np.nan_to_num(raster, nan=0.0), 0.0, 1.0)

    return evidence_distance_support(gdf, grid, influence_radius)


def evidence_distance_support(gdf, grid: Grid, influence_radius: float) -> np.ndarray:
    """Linear distance-decay support around evidence features."""
    if influence_radius <= 0:
        raise ValueError("influence_radius must be positive")

    xs, ys = grid.cell_centres()
    support = np.zeros(grid.shape, dtype=float)

    coords: List[Tuple[float, float]] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        representative = geom.representative_point()
        coords.append((float(representative.x), float(representative.y)))

    if not coords:
        return support

    for x0, y0 in coords:
        distance = np.hypot(xs - x0, ys - y0)
        contribution = np.clip(1.0 - distance / influence_radius, 0.0, 1.0)
        support = np.maximum(support, contribution)

    return support


def load_evidence_layers(
    horizon_config: Mapping[str, Any],
    grid: Grid,
    evidence_dir: str,
    influence_radius: float = EVIDENCE_INFLUENCE_RADIUS,
    target_epsg: int = TARGET_EPSG,
) -> Tuple[np.ndarray, List[str]]:
    """Combine all evidence layers configured for a horizon.

    Multiple evidence layers are averaged (equal weight), matching the original
    script.  When no evidence layer is available the function returns a zero
    grid, i.e. the horizon score reduces to ``base_weight * base_score``.

    Returns
    -------
    (evidence_grid, used_layer_names)
    """
    layers: List[np.ndarray] = []
    used: List[str] = []

    for layer_name in horizon_config.get("evidence_layers", []) or []:
        layer = load_evidence_layer(
            layer_name,
            grid,
            evidence_dir,
            influence_radius=influence_radius,
            target_epsg=target_epsg,
        )
        if layer is None:
            continue
        layers.append(layer)
        used.append(layer_name)

    if not layers:
        return grid.empty(0.0), used

    stacked = np.stack(layers, axis=0)
    combined = np.nanmean(stacked, axis=0)
    return np.clip(np.nan_to_num(combined, nan=0.0), 0.0, 1.0), used


# ===========================================================================
# Salt penalty
# ===========================================================================


def load_salt_penalty(
    salt_shp_path: str,
    grid: Grid,
    strength: float = SALT_PENALTY_STRENGTH,
    target_epsg: int = TARGET_EPSG,
) -> np.ndarray:
    """Rasterise the salt structures into a penalty grid in ``[0, 1]``.

    A cell inside a salt structure receives a penalty of ``strength``; the
    favourability is later multiplied by ``(1 - penalty)``.
    """
    if not (0.0 <= strength <= 1.0):
        raise ValueError("Salt penalty strength must be within [0, 1]")

    if not salt_shp_path or not os.path.exists(salt_shp_path):
        LOGGER.warning(
            "Salt structure layer not found (%s); no salt penalty applied",
            salt_shp_path,
        )
        return grid.empty(0.0)

    gdf = read_vector(salt_shp_path, target_epsg)
    if gdf.empty:
        LOGGER.warning("Salt structure layer is empty; no salt penalty applied")
        return grid.empty(0.0)

    raster = rasterize_geometries(gdf.geometry, grid, fill=0.0, default_value=1.0)
    penalty = np.clip(np.nan_to_num(raster, nan=0.0), 0.0, 1.0) * strength
    LOGGER.info(
        "Salt penalty applied to %d cells (strength %.2f)",
        int((penalty > 0).sum()),
        strength,
    )
    return penalty


# ===========================================================================
# Raster export
# ===========================================================================


def export_geotiff(
    data: np.ndarray,
    path: str,
    grid: Grid,
    nodata: float = NODATA_VALUE,
    dtype: str = "float32",
) -> str:
    """Write ``data`` as a single-band GeoTIFF in the grid CRS (EPSG:31467)."""
    require_rasterio()

    arr = np.asarray(data, dtype=float)
    if arr.shape != grid.shape:
        raise ValueError(
            f"Array shape {arr.shape} does not match grid shape {grid.shape}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    out = np.where(np.isfinite(arr), arr, nodata)
    profile = {
        "driver": "GTiff",
        "height": grid.ny,
        "width": grid.nx,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out.astype(dtype), 1)

    LOGGER.info("Wrote %s", path)
    return path


def write_csv(path: str, fieldnames: Sequence[str],
              rows: Iterable[Mapping[str, Any]]) -> str:
    """Write a CSV file with a stable column order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    LOGGER.info("Wrote %s", path)
    return path


# ===========================================================================
# Per-horizon scoring
# ===========================================================================


@dataclass
class HorizonResult:
    """Result of scoring a single horizon."""

    name: str
    label: str
    score: Optional[np.ndarray] = None
    skipped: bool = False
    skip_reason: str = ""
    evidence_layers_used: List[str] = field(default_factory=list)
    output_path: str = ""
    statistics: Dict[str, float] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        """``True`` when the horizon produced a favourability grid."""
        return self.score is not None and not self.skipped


def compute_statistics(score: np.ndarray) -> Dict[str, float]:
    """Descriptive statistics of a favourability grid."""
    arr = np.asarray(score, dtype=float)
    valid = arr[np.isfinite(arr)]
    positive = valid[valid > 0]
    if valid.size == 0:
        return {
            "valid_cells": 0,
            "positive_cells": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
        }
    return {
        "valid_cells": int(valid.size),
        "positive_cells": int(positive.size),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p90": float(np.quantile(positive, 0.90)) if positive.size else 0.0,
    }


def score_horizon(
    horizon_name: str,
    horizon_config: Mapping[str, Any],
    grid: Grid,
    salt_penalty: np.ndarray,
    basin_mask: Optional[np.ndarray] = None,
    thermal_dir: str = THERMAL_TIF_DIR,
    evidence_dir: str = EVIDENCE_SHP_DIR,
    target_epsg: int = TARGET_EPSG,
    influence_radius: float = EVIDENCE_INFLUENCE_RADIUS,
) -> HorizonResult:
    """Score a single horizon.

    The scoring follows the original formula::

        base_score     = thermal * reservoir
        horizon_score  = base_weight * base_score + evidence_weight * evidence
        horizon_score *= confidence_weight
        horizon_score *= (1 - salt_penalty)

    A missing thermal raster or a missing reservoir shapefile causes the
    horizon to be skipped (with a warning) rather than aborting the run.
    """
    label = str(horizon_config.get("label", horizon_name))
    result = HorizonResult(name=horizon_name, label=label)
    LOGGER.info("--- Scoring horizon '%s' (%s) ---", horizon_name, label)

    thermal_path = thermal_tif_path_for(horizon_name, horizon_config, thermal_dir)
    thermal = load_thermal_tif(thermal_path, grid)
    if thermal is None:
        result.skipped = True
        result.skip_reason = f"missing or empty thermal raster: {thermal_path}"
        LOGGER.warning("Skipping horizon '%s': %s", horizon_name, result.skip_reason)
        return result

    reservoir_path = horizon_config.get("reservoir_source_path")
    reservoir = load_and_rasterize_reservoir(
        reservoir_path,
        grid,
        value_col=horizon_config.get("reservoir_value_col"),
        value_mapping=horizon_config.get("reservoir_value_mapping"),
        value_min=horizon_config.get("reservoir_value_min"),
        value_max=horizon_config.get("reservoir_value_max"),
        certainty_col=horizon_config.get("reservoir_certainty_col"),
        target_epsg=target_epsg,
    )
    if reservoir is None:
        result.skipped = True
        result.skip_reason = f"missing reservoir layer: {reservoir_path}"
        LOGGER.warning("Skipping horizon '%s': %s", horizon_name, result.skip_reason)
        return result

    evidence, used_layers = load_evidence_layers(
        horizon_config,
        grid,
        evidence_dir,
        influence_radius=influence_radius,
        target_epsg=target_epsg,
    )
    result.evidence_layers_used = used_layers
    if used_layers:
        LOGGER.info(
            "Horizon '%s': evidence layers used: %s",
            horizon_name,
            ", ".join(used_layers),
        )
    else:
        LOGGER.info("Horizon '%s': no evidence layers available", horizon_name)

    base_weight = float(horizon_config.get("base_weight", 1.0))
    evidence_weight = float(horizon_config.get("evidence_weight", 0.0))
    confidence_weight = float(horizon_config.get("confidence_weight", 1.0))

    thermal_filled = np.nan_to_num(thermal, nan=0.0)
    base_score = thermal_filled * reservoir
    score = base_weight * base_score + evidence_weight * evidence
    score = confidence_weight * score
    score = score * (1.0 - np.clip(salt_penalty, 0.0, 1.0))
    score = np.clip(score, 0.0, 1.0)

    # Evidence must never create favourability where the thermal criterion has
    # no data at all -- the original workflow masked those cells out.
    score = np.where(np.isfinite(thermal), score, np.nan)

    if basin_mask is not None:
        score = np.where(basin_mask, score, np.nan)

    result.score = score
    result.statistics = compute_statistics(score)
    LOGGER.info(
        "Horizon '%s': mean=%.4f max=%.4f (%d positive cells)",
        horizon_name,
        result.statistics["mean"],
        result.statistics["max"],
        result.statistics["positive_cells"],
    )
    return result


# ===========================================================================
# Aggregation (NEW: replaces the global VoterVeto)
# ===========================================================================


@dataclass
class AggregationResult:
    """Aggregated results across all successfully scored horizons."""

    best_horizon: np.ndarray
    best_horizon_id: np.ndarray
    primary_mask: np.ndarray
    secondary_mask: np.ndarray
    primary_threshold: Optional[float]
    secondary_threshold: Optional[float]
    horizon_order: List[str]


def aggregate_horizons(
    results: Sequence[HorizonResult],
    grid: Grid,
    primary_quantile: float = PRIMARY_QUANTILE,
    secondary_quantile: float = SECONDARY_QUANTILE,
) -> AggregationResult:
    """Aggregate the per-horizon favourability grids.

    * ``best_horizon`` -- cell-wise maximum across all scored horizons.
    * ``best_horizon_id`` -- 1-based index of the horizon achieving that
      maximum (``0`` where no horizon has data).
    * ``primary_mask`` / ``secondary_mask`` -- sweet-spot masks defined by the
      given quantiles of the strictly positive best-horizon values.
    """
    scored = [res for res in results if res.scored]
    if not scored:
        raise ValueError("No horizon could be scored; nothing to aggregate")

    stack = np.stack([np.asarray(res.score, dtype=float) for res in scored], axis=0)
    order = [res.name for res in scored]

    all_nan = np.all(~np.isfinite(stack), axis=0)
    filled = np.where(np.isfinite(stack), stack, -np.inf)

    best = np.max(filled, axis=0)
    best_idx = np.argmax(filled, axis=0)

    best_horizon = np.where(all_nan, np.nan, best)
    best_horizon_id = np.where(all_nan, 0, best_idx + 1).astype(np.int32)

    primary_threshold = safe_quantile(best_horizon, primary_quantile)
    secondary_threshold = safe_quantile(best_horizon, secondary_quantile)

    if primary_threshold is None or secondary_threshold is None:
        LOGGER.warning(
            "Best-horizon grid has no positive values; sweet-spot masks are empty"
        )
        primary_mask = np.zeros(grid.shape, dtype=float)
        secondary_mask = np.zeros(grid.shape, dtype=float)
    else:
        valid = np.isfinite(best_horizon)
        primary_mask = np.where(
            valid & (best_horizon >= primary_threshold), 1.0, 0.0
        )
        secondary_mask = np.where(
            valid & (best_horizon >= secondary_threshold), 1.0, 0.0
        )
        LOGGER.info(
            "PRIMARY threshold (q=%.2f): %.4f -> %d cells",
            primary_quantile,
            primary_threshold,
            int(primary_mask.sum()),
        )
        LOGGER.info(
            "SECONDARY threshold (q=%.2f): %.4f -> %d cells",
            secondary_quantile,
            secondary_threshold,
            int(secondary_mask.sum()),
        )

    return AggregationResult(
        best_horizon=best_horizon,
        best_horizon_id=best_horizon_id,
        primary_mask=primary_mask,
        secondary_mask=secondary_mask,
        primary_threshold=primary_threshold,
        secondary_threshold=secondary_threshold,
        horizon_order=order,
    )


# ===========================================================================
# Workflow
# ===========================================================================


@dataclass
class WorkflowConfig:
    """Runtime configuration of the workflow."""

    nx: int = NX
    ny: int = NY
    target_epsg: int = TARGET_EPSG
    thermal_dir: str = THERMAL_TIF_DIR
    evidence_dir: str = EVIDENCE_SHP_DIR
    basin_shp_path: str = BASIN_SHP_PATH
    salt_shp_path: str = SALT_SHP_PATH
    output_dir: str = OUTPUT_DIR
    horizons: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: dict(HORIZONS)
    )
    salt_penalty_strength: float = SALT_PENALTY_STRENGTH
    evidence_influence_radius: float = EVIDENCE_INFLUENCE_RADIUS
    primary_quantile: float = PRIMARY_QUANTILE
    secondary_quantile: float = SECONDARY_QUANTILE

    @property
    def horizon_output_dir(self) -> str:
        """Directory receiving all rasters and tables of this workflow."""
        return os.path.join(self.output_dir, OUTPUT_SUBDIR)


def run_workflow(config: Optional[WorkflowConfig] = None) -> Dict[str, Any]:
    """Execute the complete horizon-specific favourability workflow.

    Returns a dictionary with the per-horizon results, the aggregation result
    and the list of files written.
    """
    cfg = config or WorkflowConfig()
    require_rasterio()
    require_geopandas()

    LOGGER.info("=" * 78)
    LOGGER.info("North German Basin -- horizon-specific geothermal favourability")
    LOGGER.info("Grid: %d x %d | CRS: EPSG:%s", cfg.nx, cfg.ny, cfg.target_epsg)
    LOGGER.info("Thermal rasters : %s", cfg.thermal_dir)
    LOGGER.info("Evidence layers : %s", cfg.evidence_dir)
    LOGGER.info("Output          : %s", cfg.horizon_output_dir)
    LOGGER.info("=" * 78)

    grid, basin_mask = load_basin_grid(
        cfg.basin_shp_path, cfg.nx, cfg.ny, target_epsg=cfg.target_epsg
    )
    salt_penalty = load_salt_penalty(
        cfg.salt_shp_path,
        grid,
        strength=cfg.salt_penalty_strength,
        target_epsg=cfg.target_epsg,
    )

    out_dir = cfg.horizon_output_dir
    os.makedirs(out_dir, exist_ok=True)

    results: List[HorizonResult] = []
    written: List[str] = []

    for horizon_name, horizon_config in cfg.horizons.items():
        result = score_horizon(
            horizon_name,
            horizon_config,
            grid,
            salt_penalty,
            basin_mask=basin_mask,
            thermal_dir=cfg.thermal_dir,
            evidence_dir=cfg.evidence_dir,
            target_epsg=cfg.target_epsg,
            influence_radius=cfg.evidence_influence_radius,
        )
        if result.scored:
            path = os.path.join(
                out_dir, f"{horizon_name}_geothermal_favourability.tif"
            )
            export_geotiff(result.score, path, grid)
            result.output_path = path
            written.append(path)
        results.append(result)

    scored = [res for res in results if res.scored]
    LOGGER.info(
        "%d of %d horizons scored successfully", len(scored), len(results)
    )

    aggregation: Optional[AggregationResult] = None
    if scored:
        aggregation = aggregate_horizons(
            results,
            grid,
            primary_quantile=cfg.primary_quantile,
            secondary_quantile=cfg.secondary_quantile,
        )

        written.append(
            export_geotiff(
                aggregation.best_horizon,
                os.path.join(
                    out_dir, "best_horizon_geothermal_favourability.tif"
                ),
                grid,
            )
        )
        written.append(
            export_geotiff(
                aggregation.primary_mask,
                os.path.join(out_dir, "PRIMARY_geothermal_favourability.tif"),
                grid,
            )
        )
        written.append(
            export_geotiff(
                aggregation.secondary_mask,
                os.path.join(out_dir, "SECONDARY_geothermal_favourability.tif"),
                grid,
            )
        )
        written.append(
            export_geotiff(
                aggregation.best_horizon_id.astype(float),
                os.path.join(out_dir, "best_horizon_id.tif"),
                grid,
                nodata=0.0,
                dtype="int32",
            )
        )
        written.append(
            write_best_horizon_mapping(aggregation, results, out_dir)
        )
    else:
        LOGGER.error(
            "No horizon could be scored -- check the thermal rasters in %s",
            cfg.thermal_dir,
        )

    written.append(write_summary(results, aggregation, out_dir))

    return {
        "grid": grid,
        "results": results,
        "aggregation": aggregation,
        "written": written,
        "output_dir": out_dir,
    }


def write_best_horizon_mapping(
    aggregation: AggregationResult,
    results: Sequence[HorizonResult],
    out_dir: str,
) -> str:
    """Write the ID -> horizon lookup table for ``best_horizon_id.tif``."""
    labels = {res.name: res.label for res in results}
    counts = {
        int(value): int(count)
        for value, count in zip(*np.unique(aggregation.best_horizon_id,
                                           return_counts=True))
    }
    rows = [
        {
            "horizon_id": idx,
            "horizon": name,
            "label": labels.get(name, name),
            "cells_best": counts.get(idx, 0),
        }
        for idx, name in enumerate(aggregation.horizon_order, start=1)
    ]
    return write_csv(
        os.path.join(out_dir, "best_horizon_id_mapping.csv"),
        ["horizon_id", "horizon", "label", "cells_best"],
        rows,
    )


def write_summary(
    results: Sequence[HorizonResult],
    aggregation: Optional[AggregationResult],
    out_dir: str,
) -> str:
    """Write the per-horizon summary statistics table."""
    fieldnames = [
        "horizon",
        "label",
        "status",
        "skip_reason",
        "evidence_layers_used",
        "valid_cells",
        "positive_cells",
        "min",
        "max",
        "mean",
        "median",
        "p90",
        "output",
    ]
    rows = []
    for res in results:
        stats = res.statistics
        rows.append(
            {
                "horizon": res.name,
                "label": res.label,
                "status": "scored" if res.scored else "skipped",
                "skip_reason": res.skip_reason,
                "evidence_layers_used": ";".join(res.evidence_layers_used),
                "valid_cells": stats.get("valid_cells", 0),
                "positive_cells": stats.get("positive_cells", 0),
                "min": stats.get("min", ""),
                "max": stats.get("max", ""),
                "mean": stats.get("mean", ""),
                "median": stats.get("median", ""),
                "p90": stats.get("p90", ""),
                "output": os.path.basename(res.output_path) if res.output_path else "",
            }
        )

    if aggregation is not None:
        stats = compute_statistics(aggregation.best_horizon)
        rows.append(
            {
                "horizon": "best_horizon",
                "label": "Best horizon (max across horizons)",
                "status": "aggregated",
                "skip_reason": "",
                "evidence_layers_used": "",
                "valid_cells": stats["valid_cells"],
                "positive_cells": stats["positive_cells"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "median": stats["median"],
                "p90": stats["p90"],
                "output": "best_horizon_geothermal_favourability.tif",
            }
        )

    return write_csv(os.path.join(out_dir, "summary.csv"), fieldnames, rows)


# ===========================================================================
# Command line interface
# ===========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Horizon-specific geothermal favourability mapping for the North "
            "German Basin (thermal input from pre-computed GeoTIFFs)."
        )
    )
    parser.add_argument("--nx", type=int, default=NX, help="grid columns")
    parser.add_argument("--ny", type=int, default=NY, help="grid rows")
    parser.add_argument(
        "--epsg", type=int, default=TARGET_EPSG, help="target EPSG code"
    )
    parser.add_argument("--thermal-dir", default=THERMAL_TIF_DIR)
    parser.add_argument("--evidence-dir", default=EVIDENCE_SHP_DIR)
    parser.add_argument("--basin", default=BASIN_SHP_PATH)
    parser.add_argument("--salt", default=SALT_SHP_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--horizons",
        nargs="*",
        default=None,
        help="subset of horizon keys to process (default: all 18)",
    )
    parser.add_argument("--log-file", default=None)
    parser.add_argument(
        "-v", "--verbose", action="count", default=1, help="increase verbosity"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only log warnings and errors"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    """Translate parsed CLI arguments into a :class:`WorkflowConfig`."""
    horizons = dict(HORIZONS)
    if args.horizons:
        unknown = [name for name in args.horizons if name not in HORIZONS]
        if unknown:
            raise SystemExit(
                f"Unknown horizon(s): {', '.join(unknown)}. "
                f"Available: {', '.join(HORIZONS)}"
            )
        horizons = {name: HORIZONS[name] for name in args.horizons}

    return WorkflowConfig(
        nx=args.nx,
        ny=args.ny,
        target_epsg=args.epsg,
        thermal_dir=args.thermal_dir,
        evidence_dir=args.evidence_dir,
        basin_shp_path=args.basin,
        salt_shp_path=args.salt,
        output_dir=args.output_dir,
        horizons=horizons,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(0 if args.quiet else args.verbose, log_file=args.log_file)

    try:
        run_workflow(config_from_args(args))
    except (MissingDependencyError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
