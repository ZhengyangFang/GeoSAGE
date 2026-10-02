from __future__ import annotations

from pathlib import Path

import numpy as np


def _parse_mesh_axis_line(line: str, n: int) -> np.ndarray:
    vals = np.array([float(v) for v in line.strip().split()], dtype=float)
    if vals.size != n:
        raise ValueError(f"Expected {n} values, got {vals.size}: {line[:80]}...")
    return vals


def read_ubc_tensor_mesh(mesh_file: str | Path):
    """Read a simple UBC tensor mesh and return shape, edges, and centers.

    UBC 3D meshes store the z origin at the top. The returned z edges are
    reversed to ascending bottom-to-top order so that k=0 aligns with the
    deepest layer in saved core-model arrays used by the notebooks.
    """
    mesh_file = Path(mesh_file)
    lines = [ln for ln in mesh_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 5:
        raise ValueError(f"Invalid mesh file: {mesh_file}")

    nx, ny, nz = [int(v) for v in lines[0].split()]
    x0, y0, z0 = [float(v) for v in lines[1].split()]
    hx = _parse_mesh_axis_line(lines[2], nx)
    hy = _parse_mesh_axis_line(lines[3], ny)
    hz = _parse_mesh_axis_line(lines[4], nz)

    x_edges = x0 + np.r_[0.0, np.cumsum(hx)]
    y_edges = y0 + np.r_[0.0, np.cumsum(hy)]
    z_edges_top_to_bottom = z0 - np.r_[0.0, np.cumsum(hz)]
    z_edges = z_edges_top_to_bottom[::-1]

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return (nx, ny, nz), x_edges, y_edges, z_edges, x_centers, y_centers, z_centers
