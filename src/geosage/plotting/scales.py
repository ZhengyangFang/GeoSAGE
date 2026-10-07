"""Data-derived display policies shared by static plots and the 3D viewer."""

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex


def field_scale(values, *, centered=False):
    """Full finite range; never silently clip outliers or change model values."""
    values = np.asarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Display values must be nonempty and finite.")
    low, high = float(values.min()), float(values.max())
    diverging = centered or low < 0 < high
    if diverging:
        bound = max(abs(low), abs(high)) or 1.0
        limits = [-bound, bound]
    elif low == high:
        pad = abs(low) * 0.05 or 1.0
        limits = [0.0, 1.0] if low == 0 else [low - pad, high + pad]
    else:
        limits = [low, high]
    return {
        "limits": limits,
        "cmap": "RdBu_r" if diverging else "viridis",
        "data_range": [low, high],
        "range_policy": "full range",
    }


def symmetric_percentile_scale(values, *, lower=2.0, upper=98.0, padding=0.01, mask=None):
    """Return robust, zero-centred limits from finite percentile endpoints.

    Values outside the limits remain in the plot and saturate at the end
    colours, so isolated extremes do not wash out the main model structure.
    """
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("Percentile limits must satisfy 0 <= lower < upper <= 100.")
    if not np.isfinite(padding) or padding < 0.0:
        raise ValueError("Colour-scale padding must be a finite non-negative value.")
    values = np.asarray(values)
    valid = np.isfinite(values)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("Display mask must match the values shape.")
        valid &= mask
    finite = values[valid]
    if not finite.size:
        raise ValueError("Display values must contain at least one finite value.")
    low, high = (float(v) for v in np.percentile(finite, [lower, upper]))
    bound = max(abs(low), abs(high)) + float(padding)
    if not np.isfinite(bound) or bound == 0.0:
        bound = 1.0
    return {
        "limits": [-bound, bound],
        "cmap": "RdBu_r",
        "data_range": [float(finite.min()), float(finite.max())],
        "percentile_range": [low, high],
        "percentiles": [float(lower), float(upper)],
        "padding": float(padding),
        "range_policy": (
            f"symmetric {lower:g}–{upper:g} percentile range plus {padding:g} padding"
        ),
    }


def category_colors(labels):
    """One colour per actual ID; sparse IDs are never treated as continuous."""
    ids = sorted(int(v) for v in np.unique(labels) if v != 0)
    palette = colormaps["tab10" if len(ids) <= 10 else "tab20"]
    colors = {
        str(uid): to_hex(palette(i))
        if len(ids) <= 20
        else to_hex(colormaps["hsv"]((i * 0.61803398875) % 1))
        for i, uid in enumerate(ids)
    }
    if np.any(np.asarray(labels) == 0):
        colors["0"] = "#e5e7eb"
    return colors


def coordinate_scale(*edges):
    """Metres for local surveys, kilometres for regional models."""
    return (1000.0, "km") if max(float(np.ptp(a)) for a in edges) >= 2000 else (1.0, "m")


def middle_index(edges):
    """Choose the cell nearest the physical midpoint, including uneven meshes."""
    edges = np.asarray(edges)
    centers = (edges[:-1] + edges[1:]) * 0.5
    return int(np.argmin(abs(centers - (edges[0] + edges[-1]) * 0.5)))
