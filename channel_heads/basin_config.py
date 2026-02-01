"""Basin configuration data from Goren & Shelef (2024).

This module contains configuration data for the 18 elongated mountain ranges
analyzed in the paper "Channel concavity controls planform complexity of
branching drainage networks" (Earth Surf. Dynam., 12, 1347–1369, 2024).

The data is extracted from Table A1 in Appendix A of the paper.

Reference:
    Goren, L. and Shelef, E.: Channel concavity controls planform complexity
    of branching drainage networks, Earth Surf. Dynam., 12, 1347–1369,
    https://doi.org/10.5194/esurf-12-1347-2024, 2024.

Usage:
    from channel_heads.basin_config import BASIN_CONFIG, get_basin_config

    # Get z_th for a specific basin
    config = get_basin_config("inyo")
    z_th = config["z_th"]

    # Or access the full DataFrame
    df = BASIN_CONFIG
    inyo_z_th = df.loc["inyo", "z_th"]
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path

# Basin configuration data from Table A1 of Goren & Shelef (2024)
# Columns:
#   - name: Basin name (lowercase for easy lookup)
#   - full_name: Full official name from the paper
#   - lat: Latitude of highest peak
#   - lon: Longitude of highest peak
#   - z_max: Maximum elevation (m)
#   - z_th: Elevation threshold (m) - minimum closed contour defining range bounds
#   - area_km2: Total drainage area of main basins (km²)
#   - theta: Best-suited concavity index (median, 25th, 75th percentiles)
#   - delta_L: Lengthwise asymmetry (median, 25th, 75th percentiles)
#   - delta_chi: Chi difference (median, 25th, 75th percentiles)
#   - aridity_index: Aridity index (median, 25th, 75th percentiles)
#   - num_pairs: Number of paired flow pathways analyzed

_BASIN_DATA = [
    {
        "name": "taiwan",
        "full_name": "Central Mountain Range, Taiwan",
        "lat": 23.47,
        "lon": 120.96,
        "z_max": 3917,
        "z_th": 80,
        "area_km2": 11991,
        "theta": 0.59,
        "theta_25": 0.51,
        "theta_75": 0.68,
        "delta_L": 1.16,
        "delta_L_25": 0.60,
        "delta_L_75": 1.58,
        "delta_chi": 0.58,
        "delta_chi_25": 0.25,
        "delta_chi_75": 1.10,
        "aridity_index": 2.29,
        "aridity_index_25": 1.81,
        "aridity_index_75": 3.39,
        "num_pairs": 57010,
    },
    {
        "name": "clanalpine",
        "full_name": "Clan Alpine Mountains, Nevada",
        "lat": 39.69,
        "lon": -117.87,
        "z_max": 2677,
        "z_th": 1700,
        "area_km2": 204,
        "theta": 0.34,
        "theta_25": 0.25,
        "theta_75": 0.41,
        "delta_L": 0.27,
        "delta_L_25": 0.12,
        "delta_L_75": 0.48,
        "delta_chi": 0.31,
        "delta_chi_25": 0.14,
        "delta_chi_75": 0.56,
        "aridity_index": 0.16,
        "aridity_index_25": 0.14,
        "aridity_index_75": 0.17,
        "num_pairs": 1547,
    },
    {
        "name": "daqing",
        "full_name": "Daquing Shan, China",
        "lat": 40.71,
        "lon": 109.12,
        "z_max": 2293,
        "z_th": 1200,
        "area_km2": 297,
        "theta": 0.55,
        "theta_25": 0.50,
        "theta_75": 0.64,
        "delta_L": 0.82,
        "delta_L_25": 0.40,
        "delta_L_75": 1.21,
        "delta_chi": 0.38,
        "delta_chi_25": 0.15,
        "delta_chi_75": 0.76,
        "aridity_index": 0.19,
        "aridity_index_25": 0.17,
        "aridity_index_75": 0.20,
        "num_pairs": 1961,
    },
    {
        "name": "finisterre",
        "full_name": "Finisterre Range, Papua New Guinea",
        "lat": -5.95,
        "lon": 146.38,
        "z_max": 4096,
        "z_th": 400,
        "area_km2": 8227,
        "theta": 0.64,
        "theta_25": 0.62,
        "theta_75": 0.70,
        "delta_L": 0.93,
        "delta_L_25": 0.49,
        "delta_L_75": 1.37,
        "delta_chi": 0.55,
        "delta_chi_25": 0.23,
        "delta_chi_75": 0.99,
        "aridity_index": 2.14,
        "aridity_index_25": 1.87,
        "aridity_index_75": 2.67,
        "num_pairs": 31675,
    },
    {
        "name": "humboldt",
        "full_name": "Humboldt Range, Nevada",
        "lat": 40.52,
        "lon": -118.17,
        "z_max": 2984,
        "z_th": 1450,
        "area_km2": 291,
        "theta": 0.46,
        "theta_25": 0.38,
        "theta_75": 0.60,
        "delta_L": 0.34,
        "delta_L_25": 0.13,
        "delta_L_75": 0.61,
        "delta_chi": 0.31,
        "delta_chi_25": 0.12,
        "delta_chi_75": 0.65,
        "aridity_index": 0.16,
        "aridity_index_25": 0.14,
        "aridity_index_75": 0.20,
        "num_pairs": 2313,
    },
    {
        "name": "inyo",
        "full_name": "Inyo Mountains, California",
        "lat": 36.71,
        "lon": -117.96,
        "z_max": 3363,
        "z_th": 1200,
        "area_km2": 266,
        "theta": 0.35,
        "theta_25": 0.28,
        "theta_75": 0.48,
        "delta_L": 0.17,
        "delta_L_25": 0.08,
        "delta_L_75": 0.34,
        "delta_chi": 0.38,
        "delta_chi_25": 0.15,
        "delta_chi_75": 0.71,
        "aridity_index": 0.17,
        "aridity_index_25": 0.13,
        "aridity_index_75": 0.21,
        "num_pairs": 2419,
    },
    {
        "name": "kammanassie",
        "full_name": "Kammanassie Mountains, South Africa",
        "lat": -33.62,
        "lon": 22.94,
        "z_max": 1935,
        "z_th": 630,
        "area_km2": 353,
        "theta": 0.39,
        "theta_25": 0.32,
        "theta_75": 0.48,
        "delta_L": 0.35,
        "delta_L_25": 0.17,
        "delta_L_75": 0.62,
        "delta_chi": 0.37,
        "delta_chi_25": 0.16,
        "delta_chi_75": 0.69,
        "aridity_index": 0.27,
        "aridity_index_25": 0.26,
        "aridity_index_75": 0.29,
        "num_pairs": 2858,
    },
    {
        "name": "luliang",
        "full_name": "Lüliang Mountains, China",
        "lat": 39.27,
        "lon": 112.96,
        "z_max": 2391,
        "z_th": 1100,
        "area_km2": 1036,
        "theta": 0.37,
        "theta_25": 0.36,
        "theta_75": 0.38,
        "delta_L": 0.46,
        "delta_L_25": 0.21,
        "delta_L_75": 0.77,
        "delta_chi": 0.39,
        "delta_chi_25": 0.18,
        "delta_chi_75": 0.78,
        "aridity_index": 0.37,
        "aridity_index_25": 0.34,
        "aridity_index_75": 0.42,
        "num_pairs": 5685,
    },
    {
        "name": "panamint",
        "full_name": "Panamint Range, California",
        "lat": 36.17,
        "lon": -117.09,
        "z_max": 3344,
        "z_th": 800,
        "area_km2": 713,
        "theta": 0.50,
        "theta_25": 0.42,
        "theta_75": 0.61,
        "delta_L": 0.40,
        "delta_L_25": 0.13,
        "delta_L_75": 0.80,
        "delta_chi": 0.43,
        "delta_chi_25": 0.20,
        "delta_chi_75": 0.84,
        "aridity_index": 0.10,
        "aridity_index_25": 0.07,
        "aridity_index_75": 0.14,
        "num_pairs": 6852,
    },
    {
        "name": "sakhalin",
        "full_name": "Sakhalin Mountains, Russia",
        "lat": 47.07,
        "lon": 142.88,
        "z_max": 1028,
        "z_th": 60,
        "area_km2": 556,
        "theta": 0.42,
        "theta_25": 0.37,
        "theta_75": 0.45,
        "delta_L": 0.56,
        "delta_L_25": 0.25,
        "delta_L_75": 1.02,
        "delta_chi": 0.37,
        "delta_chi_25": 0.18,
        "delta_chi_75": 0.68,
        "aridity_index": 1.32,
        "aridity_index_25": 1.26,
        "aridity_index_75": 1.39,
        "num_pairs": 5774,
    },
    {
        "name": "vallefertil",
        "full_name": "Sierra del Valle Fértil, Argentina",
        "lat": -30.44,
        "lon": -67.80,
        "z_max": 2311,
        "z_th": 1050,
        "area_km2": 696,
        "theta": 0.54,
        "theta_25": 0.47,
        "theta_75": 0.67,
        "delta_L": 0.90,
        "delta_L_25": 0.45,
        "delta_L_75": 1.36,
        "delta_chi": 0.46,
        "delta_chi_25": 0.21,
        "delta_chi_75": 0.77,
        "aridity_index": 0.14,
        "aridity_index_25": 0.14,
        "aridity_index_75": 0.15,
        "num_pairs": 6415,
    },
    {
        "name": "sierramadre",
        "full_name": "Sierra Madre del Sur, Mexico",
        "lat": 17.52,
        "lon": -100.31,
        "z_max": 3105,
        "z_th": 380,
        "area_km2": 8406,
        "theta": 0.67,
        "theta_25": 0.62,
        "theta_75": 0.70,
        "delta_L": 1.01,
        "delta_L_25": 0.52,
        "delta_L_75": 1.47,
        "delta_chi": 0.52,
        "delta_chi_25": 0.26,
        "delta_chi_75": 0.86,
        "aridity_index": 0.67,
        "aridity_index_25": 0.61,
        "aridity_index_75": 0.77,
        "num_pairs": 17271,
    },
    {
        "name": "sierranevada_spain",
        "full_name": "Sierra Nevada, Spain",
        "lat": 37.05,
        "lon": -3.31,
        "z_max": 3446,
        "z_th": 1200,
        "area_km2": 628,
        "theta": 0.24,
        "theta_25": 0.18,
        "theta_75": 0.29,
        "delta_L": 0.19,
        "delta_L_25": 0.08,
        "delta_L_75": 0.34,
        "delta_chi": 0.36,
        "delta_chi_25": 0.15,
        "delta_chi_75": 0.69,
        "aridity_index": 0.49,
        "aridity_index_25": 0.41,
        "aridity_index_75": 0.59,
        "num_pairs": 6046,
    },
    {
        "name": "piedepalo",
        "full_name": "Sierra Pie de Palo, Argentina",
        "lat": -31.32,
        "lon": -67.92,
        "z_max": 3157,
        "z_th": 650,
        "area_km2": 882,
        "theta": 0.25,
        "theta_25": 0.21,
        "theta_75": 0.28,
        "delta_L": 0.17,
        "delta_L_25": 0.07,
        "delta_L_75": 0.34,
        "delta_chi": 0.26,
        "delta_chi_25": 0.12,
        "delta_chi_75": 0.53,
        "aridity_index": 0.16,
        "aridity_index_25": 0.15,
        "aridity_index_75": 0.17,
        "num_pairs": 9731,
    },
    {
        "name": "toano",
        "full_name": "Toano Range, Nevada",
        "lat": 40.50,
        "lon": -114.30,
        "z_max": 2914,
        "z_th": 1710,
        "area_km2": 364,
        "theta": 0.22,
        "theta_25": 0.12,
        "theta_75": 0.32,
        "delta_L": 0.18,
        "delta_L_25": 0.09,
        "delta_L_75": 0.32,
        "delta_chi": 0.34,
        "delta_chi_25": 0.14,
        "delta_chi_75": 0.68,
        "aridity_index": 0.19,
        "aridity_index_25": 0.16,
        "aridity_index_75": 0.22,
        "num_pairs": 3721,
    },
    {
        "name": "troodos",
        "full_name": "Troodos Mountains, Cyprus",
        "lat": 34.94,
        "lon": 32.86,
        "z_max": 1949,
        "z_th": 200,
        "area_km2": 1899,
        "theta": 0.51,
        "theta_25": 0.48,
        "theta_75": 0.51,
        "delta_L": 0.84,
        "delta_L_25": 0.42,
        "delta_L_75": 1.26,
        "delta_chi": 0.42,
        "delta_chi_25": 0.18,
        "delta_chi_75": 0.81,
        "aridity_index": 0.35,
        "aridity_index_25": 0.30,
        "aridity_index_75": 0.41,
        "num_pairs": 18104,
    },
    {
        "name": "tsugaru",
        "full_name": "Tsugaru Peninsula, Japan",
        "lat": 40.97,
        "lon": 140.55,
        "z_max": 666,
        "z_th": 30,
        "area_km2": 263,
        "theta": 0.37,
        "theta_25": 0.28,
        "theta_75": 0.50,
        "delta_L": 0.53,
        "delta_L_25": 0.28,
        "delta_L_75": 0.82,
        "delta_chi": 0.26,
        "delta_chi_25": 0.11,
        "delta_chi_75": 0.52,
        "aridity_index": 1.45,
        "aridity_index_25": 1.42,
        "aridity_index_75": 1.49,
        "num_pairs": 2343,
    },
    {
        "name": "yoro",
        "full_name": "Yoro Mountains, Japan",
        "lat": 35.28,
        "lon": 136.51,
        "z_max": 885,
        "z_th": 130,
        "area_km2": 84,
        "theta": 0.82,
        "theta_25": 0.77,
        "theta_75": 0.89,
        "delta_L": 1.38,
        "delta_L_25": 0.96,
        "delta_L_75": 1.65,
        "delta_chi": 0.35,
        "delta_chi_25": 0.17,
        "delta_chi_75": 0.79,
        "aridity_index": 1.76,
        "aridity_index_25": 1.67,
        "aridity_index_75": 1.82,
        "num_pairs": 726,
    },
]


def _create_basin_dataframe() -> pd.DataFrame:
    """Create the basin configuration DataFrame from raw data."""
    df = pd.DataFrame(_BASIN_DATA)
    df = df.set_index("name")
    return df


# Main basin configuration DataFrame
BASIN_CONFIG: pd.DataFrame = _create_basin_dataframe()
"""DataFrame containing basin configuration data from Goren & Shelef (2024).

Index:
    name (str): Lowercase basin identifier

Columns:
    full_name (str): Full official name from the paper
    lat (float): Latitude of highest peak
    lon (float): Longitude of highest peak
    z_max (int): Maximum elevation (m)
    z_th (int): Elevation threshold (m) for masking
    area_km2 (int): Total drainage area (km²)
    theta (float): Best-suited concavity index (median)
    theta_25 (float): 25th percentile of theta
    theta_75 (float): 75th percentile of theta
    delta_L (float): Lengthwise asymmetry (median)
    delta_L_25 (float): 25th percentile of delta_L
    delta_L_75 (float): 75th percentile of delta_L
    delta_chi (float): Chi difference (median)
    delta_chi_25 (float): 25th percentile of delta_chi
    delta_chi_75 (float): 75th percentile of delta_chi
    aridity_index (float): Aridity index (median)
    aridity_index_25 (float): 25th percentile of aridity index
    aridity_index_75 (float): 75th percentile of aridity index
    num_pairs (int): Number of paired flow pathways analyzed
"""


# Mapping from local DEM names to paper basin names (for convenience)
LOCAL_TO_PAPER_BASIN: Dict[str, str] = {
    # Original 7 basins
    "inyo": "inyo",
    "humboldt": "humboldt",
    "calnalpine": "clanalpine",  # Note: different spelling
    "daqing": "daqing",
    "luliang": "luliang",
    "kammanasie": "kammanassie",  # Note: different spelling
    "finisterre": "finisterre",
    # Additional basins from Goren & Shelef (2024)
    "taiwan": "taiwan",
    "panamint": "panamint",
    "sakhalin": "sakhalin",
    "vallefertil": "vallefertil",
    "sierramadre": "sierramadre",
    "sierranevadaspain": "sierranevada_spain",
    "piedepalo": "piedepalo",
    "toano": "toano",
    "troodos": "troodos",
    "tsugaru": "tsugaru",
    "yoro": "yoro",
}
"""Mapping from local DEM file names to paper basin names."""


def get_basin_config(basin_name: str) -> Dict[str, Any]:
    """Get configuration for a specific basin.

    Parameters
    ----------
    basin_name : str
        Basin name (case-insensitive). Can be either the paper name
        or a local DEM name.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all configuration parameters for the basin.

    Raises
    ------
    KeyError
        If basin_name is not found.

    Example
    -------
    >>> config = get_basin_config("inyo")
    >>> print(f"z_th for Inyo: {config['z_th']} m")
    z_th for Inyo: 1200 m
    """
    name = basin_name.lower()

    # Check if it's a local name that needs mapping
    if name in LOCAL_TO_PAPER_BASIN:
        name = LOCAL_TO_PAPER_BASIN[name]

    if name not in BASIN_CONFIG.index:
        available = ", ".join(sorted(BASIN_CONFIG.index.tolist()))
        raise KeyError(f"Basin '{basin_name}' not found. Available basins: {available}")

    return BASIN_CONFIG.loc[name].to_dict()


def get_z_th(basin_name: str) -> int:
    """Get the elevation threshold (z_th) for a basin.

    This is a convenience function for the most commonly used parameter.

    Parameters
    ----------
    basin_name : str
        Basin name (case-insensitive).

    Returns
    -------
    int
        Elevation threshold in meters.

    Example
    -------
    >>> z_th = get_z_th("inyo")
    >>> print(f"Mask elevations below {z_th} m")
    Mask elevations below 1200 m
    """
    return int(get_basin_config(basin_name)["z_th"])


def list_basins() -> pd.DataFrame:
    """List all available basins with key parameters.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with basin names and key parameters.

    Example
    -------
    >>> basins = list_basins()
    >>> print(basins[["full_name", "z_th", "theta"]])
    """
    return BASIN_CONFIG[["full_name", "z_th", "z_max", "theta", "delta_L", "aridity_index"]].copy()


def get_reference_delta_L(basin_name: str) -> Dict[str, float]:
    """Get reference lengthwise asymmetry values from the paper.

    Parameters
    ----------
    basin_name : str
        Basin name (case-insensitive).

    Returns
    -------
    Dict[str, float]
        Dictionary with 'median', 'p25', and 'p75' keys.

    Example
    -------
    >>> ref = get_reference_delta_L("inyo")
    >>> print(f"Reference ΔL: {ref['median']:.2f} ({ref['p25']:.2f} - {ref['p75']:.2f})")
    Reference ΔL: 0.17 (0.08 - 0.34)
    """
    config = get_basin_config(basin_name)
    return {
        "median": config["delta_L"],
        "p25": config["delta_L_25"],
        "p75": config["delta_L_75"],
    }
