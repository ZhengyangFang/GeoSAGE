"""
Gravity + Magnetics joint inversion workflow,
refactored into a callable function for later agent use.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import discretize

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from pyproj import CRS, Transformer
from discretize.utils import active_from_xyz
from simpeg.potential_fields import gravity, magnetics
from simpeg import (
    maps,
    data,
    inverse_problem,
    data_misfit,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)

# ------------------------------------------------------------------
# Default configuration constants.
# ------------------------------------------------------------------
DEFAULT_SELECT_REGION = [510000, 535000, 4290000, 4320000]  # min_e, max_e, min_n, max_n
DEFAULT_TARGET_GRV_DATA = "ISO"   # Options: CBA, FAA, ISO, Gzz
DEFAULT_GRAVITY_COMPONENT = "gz"  # Options: gx, gy, gz, gxx, gxy, gxz, gyy, gyz, gzz

# Gravity component units mapping
GRAVITY_UNITS = {
    "gx": "mGal",
    "gy": "mGal",
    "gz": "mGal",
    "gxx": "E",
    "gxy": "E",
    "gxz": "E",
    "gyy": "E",
    "gyz": "E",
    "gzz": "E",
}

DEFAULT_STD_GRV = 0.25
DEFAULT_STD_MAG = 10.0
DEFAULT_STD_GRV_RELATIVE = False
DEFAULT_STD_MAG_RELATIVE = False
DEFAULT_FLIGHT_HEIGHT_FT = 1000.0

DEFAULT_INCLINATION = 62.0
DEFAULT_DECLINATION = 15.0
DEFAULT_FIELD_STRENGTH = 50686.0  # nT

DEFAULT_REG_COEFFICIENT = (1.0, 1.0)   # (gravity, magnetic)
DEFAULT_REG_GRV_NORM = (1.0, 2.0, 2.0, 2.0)
DEFAULT_REG_MAG_NORM = (1.0, 2.0, 2.0, 2.0)

DEFAULT_GRV_ALPHA = (20.0, 1.0, 1.0, 1.0)    # (alpha_s, alpha_x, alpha_y, alpha_z)
DEFAULT_MAG_ALPHA = (10.0, 1.0, 1.0, 1.0)

DEFAULT_CGLAMBDA = 1e3
DEFAULT_BETA0_RATIO = 10.0
DEFAULT_BETA_COOLING = 0.8

DEFAULT_GRV_BOUNDS = (-10.0, 10.0)
DEFAULT_MAG_BOUNDS = (-10.0, 10.0)

DEFAULT_MAX_GNCG = 50
DEFAULT_MAX_LS = 10
DEFAULT_MAX_CG = 1000
DEFAULT_TOL_CG = 1e-2
DEFAULT_TOL_X = 1e-2

DEFAULT_MAX_IRLS_ITER = 100
DEFAULT_IRLS_START = 5e4
DEFAULT_IRLS_MINDELTA = 1e-2
DEFAULT_IRLS_BETA_TOL = 1e-2


def run_joint_inversion(
    project_name: str = "Hannah",
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    select_region: list[float] | tuple[float, float, float, float] | None = None,
    target_grv_data: str = DEFAULT_TARGET_GRV_DATA,
    gravity_component: str = DEFAULT_GRAVITY_COMPONENT,
    std_grv: float = DEFAULT_STD_GRV,
    std_mag: float = DEFAULT_STD_MAG,
    std_grv_relative: bool = False,
    std_mag_relative: bool = False,
    flight_height_ft: float = DEFAULT_FLIGHT_HEIGHT_FT,
    inclination: float = DEFAULT_INCLINATION,
    declination: float = DEFAULT_DECLINATION,
    field_strength: float = DEFAULT_FIELD_STRENGTH,
    grv_alpha: tuple[float, float, float, float] = DEFAULT_GRV_ALPHA,
    mag_alpha: tuple[float, float, float, float] = DEFAULT_MAG_ALPHA,
    reg_grv_norm: tuple[float, float, float, float] = DEFAULT_REG_GRV_NORM,
    reg_mag_norm: tuple[float, float, float, float] = DEFAULT_REG_MAG_NORM,
    reg_coefficient: tuple[float, float] = DEFAULT_REG_COEFFICIENT,
    cross_gradient_lambda: float = DEFAULT_CGLAMBDA,
    beta0_ratio: float = DEFAULT_BETA0_RATIO,
    beta_cooling: float = DEFAULT_BETA_COOLING,
    grv_bounds: tuple[float, float] = DEFAULT_GRV_BOUNDS,
    mag_bounds: tuple[float, float] = DEFAULT_MAG_BOUNDS,
    maxGNCG: int = DEFAULT_MAX_GNCG,
    maxLS: int = DEFAULT_MAX_LS,
    maxCG: int = DEFAULT_MAX_CG,
    tolCG: float = DEFAULT_TOL_CG,
    tolX: float = DEFAULT_TOL_X,
    maxIRLSiter: int = DEFAULT_MAX_IRLS_ITER,
    IRLSstart: float = DEFAULT_IRLS_START,
    IRLS_mindelta: float = DEFAULT_IRLS_MINDELTA,
    IRLSbeta_tol: float = DEFAULT_IRLS_BETA_TOL,
    make_plots: bool = True,
) -> dict:
    """
    Run the full gravity + magnetic joint inversion workflow.

    Parameters
    ----------
    project_name : str
        
    input_dir : Path or str, optional
        
    output_dir : Path or str, optional
        
    select_region : [min_e, max_e, min_n, max_n], optional
    gravity_component : str, optional
        Gravity component to invert. Options: gx, gy, gz, gxx, gxy, gxz, gyy, gyz, gzz.
        Default is "gz" (vertical gravity acceleration).
    std_grv : float, optional
        Standard deviation for gravity data (absolute or relative).
    std_mag : float, optional
        Standard deviation for magnetic data (absolute or relative).
    std_grv_relative : bool, optional
        If True, std_grv is treated as a relative fraction of observed data.
        E.g., 0.05 means 5% relative uncertainty. Default is False (absolute).
    std_mag_relative : bool, optional
        If True, std_mag is treated as a relative fraction of observed data.
        E.g., 0.10 means 10% relative uncertainty. Default is False (absolute).


    Returns
    -------
    result : dict
        {
          "mesh": mesh,
          "mesh_core": mesh_core,
          "ind_active": ind_active (bool array),
          "dens_core_3d": dens_core_3d,
          "susc_core_3d": susc_core_3d,
          "recovered_model": recovered_model (2*nC),
          "paths": { ... ... },
          "runtime_hours": float,
        }
    """

    tic_total = time.time()
    gravity_component = (gravity_component or "").lower()

    # Resolve input/output paths.
    input_dir = Path(input_dir) if input_dir is not None else Path(project_name)
    output_root = Path(output_dir) if output_dir is not None else Path(f"{project_name}_Inversion")

    # Input files.
    input_grv = input_dir / f"{project_name}_gravity_data.csv"
    input_mag = input_dir / f"{project_name}_magnetic_data.csv"
    input_topo = input_dir / f"{project_name}_topo.tif"
    input_msh = input_dir / f"{project_name}_mesh.msh"
    input_msh_core = input_dir / f"{project_name}_mesh_core.msh"

    # Output directories and file paths.
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[run_joint_inversion] OUTPUT_DIR = {output_root.resolve()}")

    out_obd_dir = output_root / "observed_data"
    out_obd_dir.mkdir(parents=True, exist_ok=True)
    obs_grv_UBC_path = out_obd_dir / "gravity.obs"
    obs_mag_UBC_path = out_obd_dir / "magnetics.obs"

    iter_model_dir = output_root / "iteration_model"
    iter_model_dir.mkdir(parents=True, exist_ok=True)

    out_msh_dir = output_root / "mesh"
    out_msh_dir.mkdir(parents=True, exist_ok=True)
    msh_UBC_path = out_msh_dir / "mesh.msh"
    msh_core_UBC_path = out_msh_dir / "mesh_core.msh"

    out_topo_dir = output_root / "topo"
    out_topo_dir.mkdir(parents=True, exist_ok=True)
    topo_UBC_path = out_topo_dir / "topography.xyz"

    out_res_dir = output_root / "inversion_result"
    out_res_dir.mkdir(parents=True, exist_ok=True)
    res_image_dir = out_res_dir / "image"
    res_image_dir.mkdir(parents=True, exist_ok=True)

    recovered_model_path = out_res_dir / "inversion_result_dens_susc.npy"
    dens_act_path = out_res_dir / "inversion_result_density_active.npy"
    susc_act_path = out_res_dir / "inversion_result_susceptibility_active.npy"

    pred_grv_path = out_res_dir / "dpred_gravity.npy"
    pred_mag_path = out_res_dir / "dpred_magnetics.npy"

    dens_UBC_path = out_res_dir / "joint_density_full_UBC.txt"
    susc_UBC_path = out_res_dir / "joint_susceptibility_full_UBC.txt"
    dens_core_path = out_res_dir / "joint_density_core.npy"
    susc_core_path = out_res_dir / "joint_susceptibility_core.npy"
    dens_core_UBC_path = out_res_dir / "joint_density_core_UBC.txt"
    susc_core_UBC_path = out_res_dir / "joint_susceptibility_core_UBC.txt"

    # Parse runtime parameters.
    if select_region is None:
        select_region = list(DEFAULT_SELECT_REGION)
    min_e, max_e, min_n, max_n = select_region
    flight_h = flight_height_ft / 3.2808399

    grv_alpha_s, grv_alpha_x, grv_alpha_y, grv_alpha_z = grv_alpha
    mag_alpha_s, mag_alpha_x, mag_alpha_y, mag_alpha_z = mag_alpha
    reg_grv_norm = list(reg_grv_norm)
    reg_mag_norm = list(reg_mag_norm)
    grv_lb, grv_ub = grv_bounds
    mag_lb, mag_ub = mag_bounds

    # 1) Load gravity data.
    print(20*"=","loading gravity data",20*"=")
    df_grv = pd.read_csv(input_grv)
    transformer = None

    if "Easting" in df_grv.columns and "Northing" in df_grv.columns:
        easting_grv_raw = df_grv["Easting"].values
        northing_grv_raw = df_grv["Northing"].values
        
        if "Longitude" in df_grv.columns and "Latitude" in df_grv.columns:
            lon_grv_raw = df_grv["Longitude"].values
            lat_grv_raw = df_grv["Latitude"].values
            zone = int(np.floor((lon_grv_raw.mean() + 180) / 6) + 1)
            hemisphere = "north" if lat_grv_raw.mean() >= 0 else "south"
            print(f"UTM Zone: {zone}{' Northern' if hemisphere=='north' else ' Southern'}")
            crs_utm = CRS.from_epsg(32600 + zone) if hemisphere == "north" else CRS.from_epsg(32700 + zone)
            transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    else:
        lon_grv_raw = df_grv['Longitude'].values
        lat_grv_raw = df_grv['Latitude'].values
        zone = int(np.floor((lon_grv_raw.mean() + 180) / 6) + 1)
        hemisphere = 'north' if lat_grv_raw.mean() >= 0 else 'south'
        print(f"UTM Zone: {zone}{' Northern' if hemisphere=='north' else ' Southern'}")

        crs_utm = CRS.from_epsg(32600 + zone) if hemisphere == 'north' else CRS.from_epsg(32700 + zone)
        transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
        easting_grv_raw, northing_grv_raw = transformer.transform(lon_grv_raw, lat_grv_raw)
        df_grv['Easting'] = easting_grv_raw
        df_grv['Northing'] = northing_grv_raw

    height_grv_raw = df_grv["Height"].values

    # Resolve gravity column robustly (case-insensitive + common aliases).
    cols_lower = {str(c).strip().lower(): c for c in df_grv.columns}

    def _first_available_col(candidates: list[str]) -> str | None:
        for cand in candidates:
            key = str(cand).strip().lower()
            if not key:
                continue
            if key in cols_lower:
                return cols_lower[key]
        return None

    target_col_req = str(target_grv_data).strip()
    target_col_req_l = target_col_req.lower()
    comp_l = str(gravity_component).strip().lower()

    # Prefer user-requested column, then graceful fallbacks.
    grv_candidates = [target_col_req]
    if comp_l in {"gxx", "gxy", "gxz", "gyy", "gyz", "gzz"} or target_col_req_l in {"gxx", "gxy", "gxz", "gyy", "gyz", "gzz"}:
        grv_candidates += ["gzz", "gravity_gradient_zz", "gravity_gzz", "iso", "cba", "faa", "sba", "og"]
    else:
        grv_candidates += ["iso", "cba", "faa", "sba", "og", "gz", "gzz"]

    grv_col = _first_available_col(grv_candidates)
    if grv_col is None:
        raise KeyError(
            f"Gravity column '{target_grv_data}' not found. "
            f"Tried aliases: {grv_candidates}. Available columns: {list(df_grv.columns)}"
        )
    if grv_col != target_col_req:
        print(f"[WARN] Gravity column '{target_col_req}' not found; using '{grv_col}'.")

    # Keep downstream labels/outputs consistent with the actually used column.
    target_grv_data = str(grv_col)
    obs_grv_raw = df_grv[target_grv_data].values

    # Auto-fix obvious unit/system mismatch between selected column and gravity component.
    grad_components = {"gxx", "gxy", "gxz", "gyy", "gyz", "gzz"}
    accel_components = {"gx", "gy", "gz"}
    anomaly_like_cols = {"og", "faa", "sba", "cba", "iso", "gz", "gravity"}
    selected_l = target_grv_data.strip().lower()

    if comp_l in grad_components and selected_l in anomaly_like_cols:
        print(
            f"[WARN] gravity_component='{gravity_component}' is a gradient, but column '{target_grv_data}' "
            "looks like anomaly/acceleration data. Auto-switching gravity_component to 'gz'."
        )
        gravity_component = "gz"
        comp_l = "gz"
    elif comp_l in accel_components and selected_l in grad_components:
        print(
            f"[WARN] gravity_component='{gravity_component}' is acceleration, but column '{target_grv_data}' "
            f"looks like gradient data. Auto-switching gravity_component to '{selected_l}'."
        )
        gravity_component = selected_l
        comp_l = selected_l

    mask_grv = (
        (easting_grv_raw > min_e) & (easting_grv_raw < max_e) &
        (northing_grv_raw > min_n) & (northing_grv_raw < max_n)
    )
    easting_grv = easting_grv_raw[mask_grv]
    northing_grv = northing_grv_raw[mask_grv]
    height_grv = height_grv_raw[mask_grv]
    obs_grv = obs_grv_raw[mask_grv]

    if easting_grv.size == 0:
        valid_xy = np.isfinite(easting_grv_raw) & np.isfinite(northing_grv_raw)
        if not np.any(valid_xy):
            raise ValueError("Gravity data has no valid Easting/Northing values.")
        auto_min_e = float(np.nanmin(easting_grv_raw[valid_xy]))
        auto_max_e = float(np.nanmax(easting_grv_raw[valid_xy]))
        auto_min_n = float(np.nanmin(northing_grv_raw[valid_xy]))
        auto_max_n = float(np.nanmax(northing_grv_raw[valid_xy]))
        print(
            "[WARN] No gravity points found in select_region "
            f"[{min_e}, {max_e}, {min_n}, {max_n}]. "
            "Auto-switching to gravity data extent: "
            f"[{auto_min_e:.2f}, {auto_max_e:.2f}, {auto_min_n:.2f}, {auto_max_n:.2f}]"
        )
        min_e, max_e, min_n, max_n = auto_min_e, auto_max_e, auto_min_n, auto_max_n
        select_region = [min_e, max_e, min_n, max_n]
        mask_grv = (
            (easting_grv_raw > min_e) & (easting_grv_raw < max_e) &
            (northing_grv_raw > min_n) & (northing_grv_raw < max_n)
        )
        easting_grv = easting_grv_raw[mask_grv]
        northing_grv = northing_grv_raw[mask_grv]
        height_grv = height_grv_raw[mask_grv]
        obs_grv = obs_grv_raw[mask_grv]
        if easting_grv.size == 0:
            raise ValueError(
                "Gravity data selection is empty even after auto-adjusting region. "
                f"Adjusted region: [{min_e}, {max_e}, {min_n}, {max_n}]"
            )

    data_grv_ori = np.column_stack([easting_grv, northing_grv, height_grv, obs_grv])
    # SimPEG expects downward-positive for acceleration components; gradient components keep their sign.
    _is_gradient = gravity_component.lower() in {"gxx", "gxy", "gxz", "gyy", "gyz", "gzz"}
    obs_grv_simpeg = obs_grv if _is_gradient else -obs_grv
    data_grv = np.column_stack([easting_grv, northing_grv, height_grv, obs_grv_simpeg])

    # 2) Load or generate inversion mesh.
    print(20*"=","loading inversion mesh",20*"=")
    if input_msh.exists():
        print(f"Found existing mesh file, reading: {input_msh}")
        mesh = discretize.TensorMesh._readUBC_3DMesh(input_msh)
    else:
        print("No existing mesh file found, generating new mesh...")

        xy = np.c_[data_grv[:, 0], data_grv[:, 1]]
        tree = cKDTree(xy)
        d, _ = tree.query(xy, k=2)
        nearest_dist = np.nanmedian(d[:, 1])
        cell_size_xy = nearest_dist
        print(f"Estimated horizontal cell size: {cell_size_xy:.2f} m")

        pad_cells_xy = 5
        expansion_factor_xy = 2
        core_length_e = max_e - min_e
        core_length_n = max_n - min_n
        n_core_e = int(np.ceil(core_length_e / cell_size_xy))
        n_core_n = int(np.ceil(core_length_n / cell_size_xy))

        hx_core = np.ones(n_core_e) * cell_size_xy
        hy_core = np.ones(n_core_n) * cell_size_xy

        pad_factors = expansion_factor_xy ** np.arange(0, pad_cells_xy + 1)
        hx_pad = cell_size_xy * pad_factors
        hy_pad = cell_size_xy * pad_factors

        hx = np.r_[hx_pad[::-1], hx_core, hx_pad]
        hy = np.r_[hy_pad[::-1], hy_core, hy_pad]

        z_variation = 200.0
        z_top = data_grv[:, 2].max() + z_variation
        cell_size_z = cell_size_xy / 8
        core_height = 2 * z_variation
        core_cells_z = int(np.ceil(core_height / cell_size_z))
        hz_core = np.ones(core_cells_z) * cell_size_z

        pad_cells_z = 30
        expansion_factor_z = 1.12
        hz_pad = cell_size_z * (expansion_factor_z ** np.arange(1, pad_cells_z + 1))
        hz = np.r_[hz_core, hz_pad]
        z0 = z_top - hz.sum()

        mesh = discretize.TensorMesh(
            [hx, hy, hz],
            x0=(min_e - hx_pad.sum(), min_n - hy_pad.sum(), z0),
        )

    mesh.write_UBC(msh_UBC_path)
    print("Wrote mesh:", msh_UBC_path)

    x_nodes = mesh.nodes_x
    y_nodes = mesh.nodes_y
    z_nodes = mesh.nodes_z

    xmin_mesh, xmax_mesh = x_nodes[0], x_nodes[-1]
    ymin_mesh, ymax_mesh = y_nodes[0], y_nodes[-1]
    zmin_mesh, zmax_mesh = z_nodes[0], z_nodes[-1]

    print("mesh range:")
    print(f"  Easting (X):  {xmin_mesh:.2f} ~ {xmax_mesh:.2f}")
    print(f"  Northing (Y): {ymin_mesh:.2f} ~ {ymax_mesh:.2f}")
    print(f"  Elevation(Z): {zmin_mesh:.2f} ~ {zmax_mesh:.2f}")
    print(f"  Cell: {mesh.shape_cells}")

    hx = mesh.h[0]
    hy = mesh.h[1]
    x_mask = np.isclose(hx, hx.min(), rtol=1e-3, atol=1e-3 * hx.min())
    y_mask = np.isclose(hy, hy.min(), rtol=1e-3, atol=1e-3 * hy.min())
    idx = np.where(x_mask)[0]
    idy = np.where(y_mask)[0]
    if (idx.size == 0) or (idy.size == 0):
        raise ValueError("No cells near min spacing found; check mesh.h.")

    x_breaks = np.where(np.diff(idx) > 1)[0]
    y_breaks = np.where(np.diff(idy) > 1)[0]
    x_segments = np.split(idx, x_breaks + 1)
    y_segments = np.split(idy, y_breaks + 1)
    x_core_seg = max(x_segments, key=len)
    y_core_seg = max(y_segments, key=len)

    ix_start, ix_end = x_core_seg[0] + 1, x_core_seg[-1]
    iy_start, iy_end = y_core_seg[0] + 1, y_core_seg[-1]
    print(f"index of core region in x direction:", ix_start, "->", ix_end)
    print(f"index of core region in y direction:", iy_start, "->", iy_end)

    hx_core = hx[ix_start:ix_end]
    hy_core = hy[iy_start:iy_end]
    hz_core = mesh.h[2].copy()

    x0_core = x_nodes[ix_start]
    y0_core = y_nodes[iy_start]
    z0_core = mesh.x0[2]

    if input_msh_core.exists():
        print(f"Found existing core mesh file, reading: {input_msh_core}")
        mesh_core = discretize.TensorMesh._readUBC_3DMesh(input_msh_core)
    else:
        print("No existing core mesh file found, generating new core mesh...")
        mesh_core = discretize.TensorMesh(
            [hx_core, hy_core, hz_core],
            x0=(x0_core, y0_core, z0_core),
        )

    cell_size_xy = float(np.min(mesh_core.h[0]))
    print("core mesh range:")
    print(f"  Easting (X):  {mesh_core.nodes_x[0]:.2f} ~ {mesh_core.nodes_x[-1]:.2f}")
    print(f"  Northing(Y):  {mesh_core.nodes_y[0]:.2f} ~ {mesh_core.nodes_y[-1]:.2f}")
    print(f"  Elevation(Z): {mesh_core.nodes_z[0]:.2f} ~ {mesh_core.nodes_z[-1]:.2f}")
    print(f"  Cell shape: {mesh_core.shape_cells}")

    mesh_core.write_UBC(msh_core_UBC_path)
    print("Wrote core mesh:", msh_core_UBC_path)

    # 3) Load topography and project to UTM.
    print(20*"=","loading local topography",20*"=")
    with rasterio.open(input_topo) as src:
        transform = src.transform
        topo_raw = src.read(1)
        width, height = src.width, src.height
        print("GeoTIFF CRS:", src.crs)

        lon_topo_min, lat_topo_max = transform * (0, 0)
        lon_topo_max, lat_topo_min = transform * (width, height)
        print("GeoTIFF range:")
        print(f"  X: {lon_topo_min:.4f} ~ {lon_topo_max:.4f}")
        print(f"  Y: {lat_topo_min:.4f} ~ {lat_topo_max:.4f}")

        transform_full = transform
        topo_raw = src.read(1)

    nrows_topo_raw, ncols_topo_raw = topo_raw.shape
    rows, cols = np.meshgrid(
        np.arange(nrows_topo_raw),
        np.arange(ncols_topo_raw),
        indexing="ij",
    )
    xs, ys = rasterio.transform.xy(transform_full, rows, cols)
    lon_topo_raw = np.array(xs)
    lat_topo_raw = np.array(ys)

    print(f"elevation range: {np.nanmin(topo_raw):.2f} ~ {np.nanmax(topo_raw):.2f}")

    if make_plots:
        plt.figure(figsize=(8, 6))
        plt.imshow(
            topo_raw,
            cmap="terrain",
            extent=[lon_topo_raw.min(), lon_topo_raw.max(),
                    lat_topo_raw.min(), lat_topo_raw.max()],
            origin="upper",
        )
        plt.title("Elevation (GeoTIFF)")
        plt.xlabel("GeoTIFF X")
        plt.ylabel("GeoTIFF Y")
        plt.colorbar(label="Elevation (m)")
        plt.tight_layout()
        plt.close()

    lon_topo_flat = lon_topo_raw.ravel()
    lat_topo_flat = lat_topo_raw.ravel()
    z_topo_flat = topo_raw.ravel()

    if 'transformer' not in locals() or transformer is None:
        raise ValueError("UTM transformer is undefined; please provide Longitude/Latitude in gravity data (or specify a projection) so topography can be projected.")

    easting_topo_flat, northing_topo_flat = transformer.transform(
        lon_topo_flat, lat_topo_flat
    )

    mask_mesh = (
        (easting_topo_flat >= xmin_mesh) & (easting_topo_flat <= xmax_mesh) &
        (northing_topo_flat >= ymin_mesh) & (northing_topo_flat <= ymax_mesh)
    )
    easting_topo_mesh = easting_topo_flat[mask_mesh]
    northing_topo_mesh = northing_topo_flat[mask_mesh]
    height_topo_mesh = z_topo_flat[mask_mesh]

    valid = np.isfinite(height_topo_mesh)
    easting_topo_mesh = easting_topo_mesh[valid]
    northing_topo_mesh = northing_topo_mesh[valid]
    height_topo_mesh = height_topo_mesh[valid]

    buffer = 100.0
    mask_topo_core = (
        (easting_topo_mesh >= min_e - buffer) & (easting_topo_mesh <= max_e + buffer) &
        (northing_topo_mesh >= min_n - buffer) & (northing_topo_mesh <= max_n + buffer)
    )
    easting_topo_core = easting_topo_mesh[mask_topo_core]
    northing_topo_core = northing_topo_mesh[mask_topo_core]
    height_topo_core = height_topo_mesh[mask_topo_core]

    if easting_topo_core.size == 0:
        print(
            "[WARN] No topo points found in core region "
            f"[{min_e}, {max_e}, {min_n}, {max_n}]. Trying fallback bounds."
        )

        fallback_bounds: list[tuple[float, float, float, float]] = []
        if easting_grv.size > 0:
            fallback_bounds.append(
                (
                    float(np.nanmin(easting_grv)),
                    float(np.nanmax(easting_grv)),
                    float(np.nanmin(northing_grv)),
                    float(np.nanmax(northing_grv)),
                )
            )
        fallback_bounds.append(
            (
                float(np.nanmin(easting_grv_raw)),
                float(np.nanmax(easting_grv_raw)),
                float(np.nanmin(northing_grv_raw)),
                float(np.nanmax(northing_grv_raw)),
            )
        )
        fallback_bounds.append(
            (
                float(mesh_core.nodes_x[0]),
                float(mesh_core.nodes_x[-1]),
                float(mesh_core.nodes_y[0]),
                float(mesh_core.nodes_y[-1]),
            )
        )

        for i_fb, (fb_min_e, fb_max_e, fb_min_n, fb_max_n) in enumerate(fallback_bounds, start=1):
            mask_fb = (
                (easting_topo_mesh >= fb_min_e - buffer) & (easting_topo_mesh <= fb_max_e + buffer) &
                (northing_topo_mesh >= fb_min_n - buffer) & (northing_topo_mesh <= fb_max_n + buffer)
            )
            if np.any(mask_fb):
                easting_topo_core = easting_topo_mesh[mask_fb]
                northing_topo_core = northing_topo_mesh[mask_fb]
                height_topo_core = height_topo_mesh[mask_fb]
                min_e, max_e, min_n, max_n = fb_min_e, fb_max_e, fb_min_n, fb_max_n
                select_region = [min_e, max_e, min_n, max_n]
                print(
                    f"[WARN] Topography fallback #{i_fb} succeeded. "
                    "Using bounds: "
                    f"[{min_e:.2f}, {max_e:.2f}, {min_n:.2f}, {max_n:.2f}]"
                )
                break

        if easting_topo_core.size == 0:
            raise ValueError(
                "No topo points found in core region after fallback. "
                f"Topography in mesh range X:[{np.nanmin(easting_topo_mesh):.2f}, {np.nanmax(easting_topo_mesh):.2f}], "
                f"Y:[{np.nanmin(northing_topo_mesh):.2f}, {np.nanmax(northing_topo_mesh):.2f}]. "
                "Check select_region and CRS."
            )

    print("core topo points:", easting_topo_core.size)
    print(f"X range: {easting_topo_core.min():.2f} ~ {easting_topo_core.max():.2f}")
    print(f"Y range: {northing_topo_core.min():.2f} ~ {northing_topo_core.max():.2f}")
    print(f"Z range: {height_topo_core.min():.2f} ~ {height_topo_core.max():.2f}")

    topo_core = np.column_stack([easting_topo_core, northing_topo_core, height_topo_core])

    if make_plots:
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            easting_topo_core,
            northing_topo_core,
            c=height_topo_core,
            s=1,
            cmap="terrain",
        )
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.title("Elevation (core region)")
        plt.colorbar(sc, label="Elevation (m)")
        plt.xlim([min_e, max_e])
        plt.ylim([min_n, max_n])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.savefig(out_topo_dir / "topo_core.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4) Load magnetic data.
    print(20*"=","loading magnetic data",20*"=")
    df_mag = pd.read_csv(input_mag)

    # Handle coordinate conversion with improved flexibility
    if {"Easting", "Northing"} <= set(df_mag.columns):
        easting_mag_raw = df_mag["Easting"].values
        northing_mag_raw = df_mag["Northing"].values
        # If Longitude/Latitude are also provided, use them to infer UTM zone
        # and create transformer for later topography/magnetic reuse
        if {"Longitude", "Latitude"} <= set(df_mag.columns):
            lon_mag_raw = df_mag["Longitude"].values
            lat_mag_raw = df_mag["Latitude"].values
            zone = int(np.floor((lon_mag_raw.mean() + 180) / 6) + 1)
            hemisphere = "north" if lat_mag_raw.mean() >= 0 else "south"
            print(f"UTM Zone: {zone}{' Northern' if hemisphere=='north' else ' Southern'}")
            crs_utm = CRS.from_epsg(32600 + zone) if hemisphere == "north" else CRS.from_epsg(32700 + zone)
            transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    elif {"Longitude", "Latitude"} <= set(df_mag.columns):
        lon_mag_raw = df_mag["Longitude"].values
        lat_mag_raw = df_mag["Latitude"].values
        zone = int(np.floor((lon_mag_raw.mean() + 180) / 6) + 1)
        hemisphere = "north" if lat_mag_raw.mean() >= 0 else "south"
        print(f"UTM Zone: {zone}{' Northern' if hemisphere=='north' else ' Southern'}")

        crs_utm = CRS.from_epsg(32600 + zone) if hemisphere == "north" else CRS.from_epsg(32700 + zone)
        transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
        easting_mag_raw, northing_mag_raw = transformer.transform(lon_mag_raw, lat_mag_raw)
        df_mag["Easting"] = easting_mag_raw
        df_mag["Northing"] = northing_mag_raw
    else:
        raise ValueError("Magnetic data needs Easting/Northing or Longitude/Latitude columns.")

    # Extract TFMA (total field magnetic anomaly)
    tfma_mag_raw = df_mag["TFMA"].to_numpy()

    mask_mag = (
        (easting_mag_raw > min_e) & (easting_mag_raw < max_e) &
        (northing_mag_raw > min_n) & (northing_mag_raw < max_n)
    )
    easting_mag = easting_mag_raw[mask_mag]
    northing_mag = northing_mag_raw[mask_mag]
    tfma_mag = tfma_mag_raw[mask_mag]

    if easting_mag.size == 0:
        raise ValueError("No magnetic points found in select_region.")

    data_mag_ori = np.zeros((easting_mag.shape[0], 4), dtype=float)
    data_mag_ori[:, 0] = easting_mag
    data_mag_ori[:, 1] = northing_mag
    data_mag_ori[:, 3] = tfma_mag

    if "Height" in df_mag.columns:
        height_mag_raw   = np.array(df_mag["Height"].values)
        data_mag_ori[:, 2] = height_mag_raw[mask_mag]
    else:
    # Interpolate receiver elevation from topography + flight height.
    # topo_core columns: [Easting, Northing, Elevation].
    # flight_h is platform height above ground (meters).
        interp_rec = griddata(
            topo_core[:, 0:2],  # topo XY
            topo_core[:, 2],    # topo elevation
            data_mag_ori[:, 0:2],  # receiver XY
            method="linear"
        )
        # Receiver elevation = ground elevation + flight height
        data_mag_ori[:, 2] = interp_rec + flight_h

    if np.any(np.isnan(data_mag_ori[:, 2])):
        print("NaNs exist in interpolated receiver elevation; using 1200 m fallback.")
        nan_ind = np.argwhere(np.isnan(data_mag_ori[:, 2]))
        data_mag_ori[nan_ind, 2] = 1200.0

    data_mag = data_mag_ori.copy()

    # 5) Build surveys and data objects.
    print(20*"=","preparing inversion",20*"=")
    print(f"Gravity component: {gravity_component}")
    receiver_grv = gravity.receivers.Point(data_grv[:, 0:3], components=gravity_component)
    source_field_grv = gravity.sources.SourceField(receiver_list=[receiver_grv])
    survey_grv = gravity.survey.Survey(source_field_grv)

    receiver_mag = magnetics.receivers.Point(data_mag[:, 0:3], components="tmi")
    source_field_mag = magnetics.sources.UniformBackgroundField(
        receiver_list=[receiver_mag],
        amplitude=field_strength,
        inclination=inclination,
        declination=declination,
    )
    survey_mag = magnetics.survey.Survey(source_field_mag)

    # Compute uncertainties (absolute or relative)
    if std_grv_relative:
        uncertainties_grv = std_grv * np.abs(data_grv[:, 3])
    else:
        uncertainties_grv = std_grv * np.ones_like(data_grv[:, 3])

    if std_mag_relative:
        uncertainties_mag = std_mag * np.abs(data_mag[:, 3])
    else:
        uncertainties_mag = std_mag * np.ones_like(data_mag[:, 3])

    data_object_grv = data.Data(survey_grv, dobs=data_grv[:, 3],
                                standard_deviation=uncertainties_grv)
    data_object_mag = data.Data(survey_mag, dobs=data_mag[:, 3],
                                standard_deviation=uncertainties_mag)

    # Get gravity unit for plotting
    grv_unit = GRAVITY_UNITS.get(gravity_component, "mGal")

    if make_plots:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        mm = utils.plot2Ddata(
            data_object_grv.survey.receiver_locations,
            -data_object_grv.dobs,
            level=True, dataloc=True,
            ncontour=12, shade=True,
            contourOpts={"cmap": "jet", "alpha": 0.85},
            levelOpts={"colors": "k", "linewidths": 0.6, "linestyles": "dashed"},
            ax=ax,
        )
        mappable = mm[0] if isinstance(mm, (tuple, list)) else mm
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(f"{target_grv_data} ({grv_unit})")
        ax.set_title(f"Observed Gravity {gravity_component.upper()} ({target_grv_data})")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
        plt.tight_layout()
        fig.savefig(out_obd_dir / f"gravity_obs_{target_grv_data}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        mm = utils.plot2Ddata(
            data_object_mag.survey.receiver_locations,
            data_object_mag.dobs,
            level=True, dataloc=True,
            ncontour=12, shade=True,
            contourOpts={"cmap": "jet", "alpha": 0.85},
            levelOpts={"colors": "k", "linewidths": 0.6, "linestyles": "dashed"},
            ax=ax,
        )
        mappable = mm[0] if isinstance(mm, (tuple, list)) else mm
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("TMI (nT)")
        ax.set_title("Observed Magnetic (TMI)")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
        plt.tight_layout()
        fig.savefig(out_obd_dir / "magnetic_obs.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 6) Interpolate topography on mesh and build active cells.
   
    cc = mesh.cell_centers
    xc = cc[:, 0]
    yc = cc[:, 1]
    z_grid = griddata(
        np.c_[easting_topo_mesh, northing_topo_mesh],
        height_topo_mesh,
        (xc, yc),
        method="nearest",
    )
    if np.isnan(z_grid).any():
        z_grid_nn = griddata(
            np.c_[easting_topo_mesh, northing_topo_mesh],
            height_topo_mesh,
            (xc, yc),
            method="nearest",
        )
        z_grid = np.where(np.isnan(z_grid), z_grid_nn, z_grid)
    if np.isnan(z_grid).any():
        z_grid = np.where(np.isnan(z_grid), np.nanmin(z_grid), z_grid)

    topo_mesh = np.c_[xc, yc, z_grid]

    topo_mesh_df = pd.DataFrame({"X": xc, "Y": yc, "Z": z_grid}).dropna()
    topo_mesh_df.to_csv(topo_UBC_path, sep=" ", index=False, header=False)
    print("Wrote topography on mesh (cell centers):", topo_UBC_path)

    ind_active = active_from_xyz(mesh, topo_mesh_df.to_numpy())
    print("Number of active cells:", int(ind_active.sum()))
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC)
    print("Active cells:", nC)

    wires = maps.Wires(("density", nC), ("susceptibility", nC))
    background_dens, background_susc = 1e-6, 1e-6
    starting_model = np.r_[background_dens * np.ones(nC),
                           background_susc * np.ones(nC)]

    # Snapshot key inversion parameters to JSON before running the inversion
    nx_core, ny_core, nz_core = mesh_core.shape_cells
    params_json_path = output_root / "inversion_params.json"
    params_snapshot = {
        "project_name": project_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "select_region": [float(x) for x in select_region],
        "gravity_component": gravity_component,
        "target_gravity_column": target_grv_data,
        "std_grv": float(std_grv),
        "std_mag": float(std_mag),
        "std_grv_relative": bool(std_grv_relative),
        "std_mag_relative": bool(std_mag_relative),
        "flight_height_ft": float(flight_height_ft),
        "inclination": float(inclination),
        "declination": float(declination),
        "field_strength": float(field_strength),
        "inv_bound": {
            "grv_lb": float(grv_lb),
            "grv_ub": float(grv_ub),
            "mag_lb": float(mag_lb),
            "mag_ub": float(mag_ub),
        },
        "reg_coefficient": [float(x) for x in reg_coefficient],
        "weight_grv": [float(x) for x in (grv_alpha_s, grv_alpha_x, grv_alpha_y, grv_alpha_z)],
        "weight_mag": [float(x) for x in (mag_alpha_s, mag_alpha_x, mag_alpha_y, mag_alpha_z)],
        "reg_grv_norm": [float(x) for x in reg_grv_norm],
        "reg_mag_norm": [float(x) for x in reg_mag_norm],
        "cross_gradient_lambda": float(cross_gradient_lambda),
        "beta0_ratio": float(beta0_ratio),
        "beta_cooling": float(beta_cooling),
        "irls": {
            "maxIRLSiter": int(maxIRLSiter),
            "IRLSstart": float(IRLSstart),
            "IRLS_mindelta": float(IRLS_mindelta),
            "IRLSbeta_tol": float(IRLSbeta_tol),
        },
        "optimization": {
            "maxGNCG": int(maxGNCG),
            "maxLS": int(maxLS),
            "maxCG": int(maxCG),
            "tolCG": float(tolCG),
            "tolX": float(tolX),
        },
        "mesh_core": {
            "shape": [int(nx_core), int(ny_core), int(nz_core)],
            "cell_size_xy": float(cell_size_xy),
        },
        "mesh": {"shape": [int(s) for s in mesh.shape_cells]},
        "n_active_cells": int(nC),
    }
    params_json_path.write_text(json.dumps(params_snapshot, ensure_ascii=False, indent=2))
    print(f"Parameter snapshot saved (JSON): {params_json_path}")

    # 7) Set up objective function, regularization, and inversion directives.
    simulation_grv = gravity.simulation.Simulation3DIntegral(
        survey=survey_grv,
        mesh=mesh,
        rhoMap=wires.density,
        active_cells=ind_active,
        engine="choclo",
    )
    simulation_mag = magnetics.simulation.Simulation3DIntegral(
        survey=survey_mag,
        mesh=mesh,
        model_type="scalar",
        chiMap=wires.susceptibility,
        active_cells=ind_active,
    )

    dmis_grv = data_misfit.L2DataMisfit(data=data_object_grv, simulation=simulation_grv)
    dmis_mag = data_misfit.L2DataMisfit(data=data_object_mag, simulation=simulation_mag)

    reg_grv = regularization.Sparse(
        mesh, active_cells=ind_active, mapping=wires.density,
        gradient_type="components",
    )
    reg_mag = regularization.Sparse(
        mesh, active_cells=ind_active, mapping=wires.susceptibility,
        gradient_type="components",
    )

    reg_grv.norms = reg_grv_norm
    reg_mag.norms = reg_mag_norm

    reg_grv.alpha_s, reg_grv.alpha_x, reg_grv.alpha_y, reg_grv.alpha_z = (
        grv_alpha_s, grv_alpha_x, grv_alpha_y, grv_alpha_z
    )
    reg_mag.alpha_s, reg_mag.alpha_x, reg_mag.alpha_y, reg_mag.alpha_z = (
        mag_alpha_s, mag_alpha_x, mag_alpha_y, mag_alpha_z
    )

    lamda = cross_gradient_lambda
    cross_grad = regularization.CrossGradient(mesh, wires, active_cells=ind_active)

    dmis = dmis_grv + dmis_mag
    reg = reg_coefficient[0] * reg_grv + reg_coefficient[1] * reg_mag + lamda * cross_grad

    opt = optimization.ProjectedGNCG(
        maxIter=maxGNCG,
        lower=np.concatenate([grv_lb * np.ones(nC), mag_lb * np.ones(nC)], 0),
        upper=np.concatenate([grv_ub * np.ones(nC), mag_ub * np.ones(nC)], 0),
        maxIterLS=maxLS,
        maxIterCG=maxCG,
        tolCG=tolCG,
        tolX=tolX,
    )

    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    starting_beta = directives.PairedBetaEstimate_ByEig(beta0_ratio=beta0_ratio)
    update_IRLS = directives.UpdateIRLS(
        f_min_change=IRLS_mindelta,
        max_irls_iterations=maxIRLSiter,
        misfit_tolerance=IRLSbeta_tol,
        chifact_start=IRLSstart,
        verbose=True,
    )
    beta_schedule = directives.PairedBetaSchedule(
        cooling_factor=beta_cooling,
        cooling_rate=1,
    )
    joint_inv_dir = directives.SimilarityMeasureInversionDirective()
    sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
    update_jacobi = directives.UpdatePreconditioner()

    save_output = directives.SimilarityMeasureSaveOutputEveryIteration(
        directory=iter_model_dir, name="Output"
    )
    save_model = directives.SaveModelEveryIteration(
        directory=iter_model_dir, name="InversionModel"
    )

    directives_list = [
        joint_inv_dir,
        sensitivity_weights,
        starting_beta,
        beta_schedule,
        update_IRLS,
        save_output,
        save_model,
        update_jacobi,
    ]

    inv = inversion.BaseInversion(inv_prob, directives_list)

    # Run inversion.
    print(20*"=","starting inversion",20*"=")
    tic_inv = time.time()
    recovered_model = inv.run(starting_model)
    runtime_hours = (time.time() - tic_inv) / 3600.0
    print(f"The inversion runtime is: {runtime_hours:.2f} [h]")

    # Persist inversion outputs.
    print(20*"=","saving inversion result",20*"=")
    np.save(recovered_model_path, recovered_model)
    print("Wrote inversion model:", recovered_model_path)

    m_dens = wires.density * recovered_model
    m_susc = wires.susceptibility * recovered_model
    np.save(dens_act_path, m_dens)
    np.save(susc_act_path, m_susc)
    print("Wrote active-cell models:")
    print(" ", dens_act_path)
    print(" ", susc_act_path)

    dpred_grv = simulation_grv.dpred(recovered_model)
    dpred_mag = simulation_mag.dpred(recovered_model)
    np.save(pred_grv_path, dpred_grv)
    np.save(pred_mag_path, dpred_mag)
    print("Wrote predicted data:")
    print(" ", pred_grv_path)
    print(" ", pred_mag_path)

    pad0_map = maps.InjectActiveCells(mesh, ind_active, 0.0)
    dens_full = pad0_map * m_dens
    susc_full = pad0_map * m_susc

    mesh.write_model_UBC(dens_UBC_path, dens_full)
    mesh.write_model_UBC(susc_UBC_path, susc_full)
    print("Wrote full UBC models:")
    print(" ", dens_UBC_path)
    print(" ", susc_UBC_path)

    np.savetxt(
        obs_grv_UBC_path,
        np.column_stack(
            [data_grv[:, 0], data_grv[:, 1], data_grv[:, 2],
             data_grv[:, 3], uncertainties_grv]
        ),
        fmt="%.6f",
    )
    np.savetxt(
        obs_mag_UBC_path,
        np.column_stack(
            [data_mag[:, 0], data_mag[:, 1], data_mag[:, 2],
             data_mag[:, 3], uncertainties_mag]
        ),
        fmt="%.6f",
    )
    print("Wrote observed gravity:", obs_grv_UBC_path)
    print("Wrote observed magnetics:", obs_mag_UBC_path)

    # 8) Extract and save core-region models.
    dens_full = discretize.TensorMesh.read_model_UBC(mesh, dens_UBC_path)
    susc_full = discretize.TensorMesh.read_model_UBC(mesh, susc_UBC_path)
    nx, ny, nz = mesh.shape_cells
    dens_3d = dens_full.reshape((nx, ny, nz), order="F")
    susc_3d = susc_full.reshape((nx, ny, nz), order="F")

    dens_core_3d = dens_3d[ix_start:ix_end, iy_start:iy_end, :]
    susc_core_3d = susc_3d[ix_start:ix_end, iy_start:iy_end, :]
    np.save(dens_core_path, dens_core_3d)
    np.save(susc_core_path, susc_core_3d)
    print("Wrote core-region 3D models:")
    print(" ", dens_core_path)
    print(" ", susc_core_path)

    dens_core_1d = dens_core_3d.reshape(mesh_core.nC, order="F")
    susc_core_1d = susc_core_3d.reshape(mesh_core.nC, order="F")
    mesh_core.write_model_UBC(dens_core_UBC_path, dens_core_1d)
    mesh_core.write_model_UBC(susc_core_UBC_path, susc_core_1d)
    print("Wrote core-region UBC models:")
    print(" ", dens_core_UBC_path)
    print(" ", susc_core_UBC_path)

    # 9) Generate diagnostic slice/section figures.
    def _find_k_from_depth(mesh, depth_target):
        zc = mesh.cell_centers[:, 2].reshape(mesh.shape_cells, order="F")
        z_line = zc[0, 0, :]
        k = int(np.argmin(np.abs(z_line - depth_target)))
        return k, float(z_line[k])


    inversion_result_slice_files: list[str] = []
    inversion_result_section_files: list[str] = []

    if make_plots:
        xc_core = mesh_core.cell_centers_x
        yc_core = mesh_core.cell_centers_y
        zc_core = mesh_core.cell_centers_z
        dxp = np.mean(mesh_core.h[0])
        dyp = np.mean(mesh_core.h[1])
        dzp = np.mean(mesh_core.h[2])
        x_min_plot = xc_core.min() - dxp / 2
        x_max_plot = xc_core.max() + dxp / 2
        y_min_plot = yc_core.min() - dyp / 2
        y_max_plot = yc_core.max() + dyp / 2
        z_min_plot = zc_core.min() - dzp / 2
        z_max_plot = zc_core.max() + dzp / 2

        vmin_grv, vmax_grv = -0.3, 0.3
        vmin_mag, vmax_mag = -0.05, 0.05

        nx_core, ny_core, nz_core = mesh_core.shape_cells

        # Select up to 10 representative horizontal slices.
        ind_act_3d = ind_active.reshape(mesh.shape_cells, order="F")
        ind_core_3d = ind_act_3d[ix_start:ix_end, iy_start:iy_end, :]

        first_full = next(
            (k for k in range(nz_core) if np.all(ind_core_3d[:, :, k])), None
        )
        if first_full is None:
            # If no fully active layer exists, start from the first layer with any active cell.
            active_any = np.where(ind_core_3d.any(axis=(0, 1)))[0]
            first_full = int(active_any.min()) if active_any.size else 0

        active_any = np.where(ind_core_3d.any(axis=(0, 1)))[0]
        last_active = int(active_any.max()) if active_any.size else nz_core - 1

        start_k = max(0, min(first_full, nz_core - 1))
        end_k = max(start_k, min(last_active, nz_core - 1))

        n_slices = min(10, end_k - start_k + 1)
        z_indices = np.unique(
            np.rint(np.linspace(start_k, end_k, num=n_slices)).astype(int)
        )
        for k_idx in z_indices:
            z_val = float(zc_core[k_idx])

            
            fig_combo, (axc1, axc2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
            imc1 = axc1.imshow(
                dens_core_3d[:, :, k_idx].T,
                origin="lower",
                extent=[x_min_plot, x_max_plot, y_min_plot, y_max_plot],
                cmap="seismic",
                vmin=vmin_grv,
                vmax=vmax_grv,
            )
            axc1.set_aspect("equal")
            axc1.set_title(f"Density k={k_idx}, z={z_val:.1f} m")
            axc1.set_xlabel("Easting (m)")
            axc1.set_ylabel("Northing (m)")
            fig_combo.colorbar(imc1, ax=axc1, label="Density contrast (g/cc)")

            imc2 = axc2.imshow(
                susc_core_3d[:, :, k_idx].T,
                origin="lower",
                extent=[x_min_plot, x_max_plot, y_min_plot, y_max_plot],
                cmap="seismic",
                vmin=vmin_mag,
                vmax=vmax_mag,
            )
            axc2.set_aspect("equal")
            axc2.set_title(f"Susceptibility k={k_idx}, z={z_val:.1f} m")
            axc2.set_xlabel("Easting (m)")
            axc2.set_ylabel("Northing (m)")
            fig_combo.colorbar(imc2, ax=axc2, label="Susceptibility (SI)")
            fig_combo.tight_layout()
            fname_combo = res_image_dir / f"combo_core_slice_k={k_idx:03d}_z={z_val:.0f}m.png"
            fig_combo.savefig(fname_combo, dpi=300)
            plt.close(fig_combo)
            inversion_result_slice_files.append(str(fname_combo))

        x_edges = mesh_core.nodes_x
        z_edges = mesh_core.nodes_z
        Xe, Ze = np.meshgrid(x_edges, z_edges, indexing="xy")  # (nz+1, nx+1)

        y_indices = np.unique(np.linspace(0, ny_core - 1, 10, dtype=int))
        for j_idx in y_indices:
            y_val = float(yc_core[j_idx])

            dens_slice = dens_core_3d[:, j_idx, :]   # (nx_core, nz_core)
            susc_slice = susc_core_3d[:, j_idx, :]   # (nx_core, nz_core)

            fig_combo, (axc1, axc2) = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

            imc1 = axc1.pcolormesh(
                Xe,
                Ze,
                dens_slice.T,   # (nz, nx)
                cmap="seismic",
                vmin=vmin_grv,
                vmax=vmax_grv,
                shading="auto",
            )
            axc1.set_title(f"Density section j={j_idx}, y={y_val:.1f} m")
            axc1.set_xlabel("Easting (m)")
            axc1.set_ylabel("Elevation (m)")
            fig_combo.colorbar(imc1, ax=axc1, label="Density contrast (g/cc)")

            imc2 = axc2.pcolormesh(
                Xe,
                Ze,
                susc_slice.T,
                cmap="seismic",
                vmin=vmin_mag,
                vmax=vmax_mag,
                shading="auto",
            )
            axc2.set_title(f"Susceptibility section j={j_idx}, y={y_val:.1f} m")
            axc2.set_xlabel("Easting (m)")
            axc2.set_ylabel("Elevation (m)")
            fig_combo.colorbar(imc2, ax=axc2, label="Susceptibility (SI)")

            fig_combo.tight_layout()
            fname_combo_sec = res_image_dir / f"combo_core_section_j={j_idx:03d}_y={y_val:.0f}m.png"
            fig_combo.savefig(fname_combo_sec, dpi=300)
            plt.close(fig_combo)
            inversion_result_section_files.append(str(fname_combo_sec))

    # 10) Save run metadata to HDF5.
    paras_path = output_root / "paras.h5"
    with h5py.File(paras_path, "w") as h5f:
        h5f.create_dataset("core_zone", data=np.asarray(select_region, dtype=float))
        h5f.attrs["gravity_component"] = gravity_component
        h5f.attrs["std_grv"] = float(std_grv)
        h5f.attrs["std_mag"] = float(std_mag)
        h5f.attrs["std_grv_relative"] = bool(std_grv_relative)
        h5f.attrs["std_mag_relative"] = bool(std_mag_relative)
        h5f.create_dataset(
            "inv_bound",
            data=np.array([grv_lb, mag_lb, grv_ub, mag_ub], dtype=float),
        )
        h5f.create_dataset(
            "weight_grv",
            data=np.array([grv_alpha_s, grv_alpha_x, grv_alpha_y, grv_alpha_z], dtype=float),
        )
        h5f.create_dataset(
            "weight_mag",
            data=np.array([mag_alpha_s, mag_alpha_x, mag_alpha_y, mag_alpha_z], dtype=float),
        )
        h5f.create_dataset("reg_grv_norm", data=np.asarray(reg_grv_norm, dtype=float))
        h5f.create_dataset("reg_mag_norm", data=np.asarray(reg_mag_norm, dtype=float))

        h5f.attrs["maxGNCG"] = int(maxGNCG)
        h5f.attrs["maxLS"] = int(maxLS)
        h5f.attrs["maxCG"] = int(maxCG)
        h5f.attrs["tolCG"] = float(tolCG)
        h5f.attrs["tolX"] = float(tolX)
        h5f.attrs["maxIRLSiter"] = int(maxIRLSiter)
        h5f.attrs["IRLSstart"] = float(IRLSstart)
        h5f.attrs["IRLS_mindelta"] = float(IRLS_mindelta)
        h5f.attrs["IRLSbeta_tol"] = float(IRLSbeta_tol)

        h5f.attrs["beta0_ratio"] = float(beta0_ratio)
        h5f.attrs["betacool"] = float(beta_cooling)
        h5f.attrs["CGlambda"] = float(cross_gradient_lambda)

        nx_core, ny_core, nz_core = mesh_core.shape_cells
        h5f.attrs["nx_core"] = int(nx_core)
        h5f.attrs["ny_core"] = int(ny_core)
        h5f.attrs["nz_core"] = int(nz_core)
        h5f.attrs["cell_size_xy"] = float(cell_size_xy)

    print("-" * 55)
    print(f"Inversion parameters saved in: {paras_path}")
    print("-" * 55)

    runtime_total_hours = (time.time() - tic_total) / 3600.0

    paths = {
        "output_root": str(output_root),
        "mesh_ubc": str(msh_UBC_path),
        "mesh_core_ubc": str(msh_core_UBC_path),
        "topography_xyz": str(topo_UBC_path),
        "obs_gravity_ubc": str(obs_grv_UBC_path),
        "obs_magnetics_ubc": str(obs_mag_UBC_path),
        "recovered_model_npy": str(recovered_model_path),
        "density_active_npy": str(dens_act_path),
        "susceptibility_active_npy": str(susc_act_path),
        "dpred_gravity_npy": str(pred_grv_path),
        "dpred_magnetics_npy": str(pred_mag_path),
        "density_full_ubc": str(dens_UBC_path),
        "susceptibility_full_ubc": str(susc_UBC_path),
        "density_core_npy": str(dens_core_path),
        "susceptibility_core_npy": str(susc_core_path),
        "density_core_ubc": str(dens_core_UBC_path),
        "susceptibility_core_ubc": str(susc_core_UBC_path),
        "paras_h5": str(paras_path),
        "paras_json": str(params_json_path),
        # Diagnostic figure paths.
        "inversion_result_slice_list": inversion_result_slice_files,
        "inversion_result_section_list": inversion_result_section_files,
    }

    return {
        "mesh": mesh,
        "mesh_core": mesh_core,
        "ind_active": ind_active,
        "dens_core_3d": dens_core_3d,
        "susc_core_3d": susc_core_3d,
        "recovered_model": recovered_model,
        "paths": paths,
        "runtime_hours": runtime_total_hours,
    }
