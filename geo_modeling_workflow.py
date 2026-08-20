"""
Build pseudo-geology models from inversion outputs.
This module prepares pseudo-geological labels from inversion results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista as pv
import discretize
from discretize.utils import active_from_xyz
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage


def build_geology_model(
    project_name: str = "Hannah",
    input_dir: str | Path | None = None,
    inversion_dir: str | Path | None = None,
    min_voxels: int = 10,
    fill_iterations: int = 3,
    unit_defs_csv: str | Path | None = None,
    unit_groups_csv: str | Path | None = None,
    unit_id_npy: str | Path | None = None,
    output_dir: str | Path | None = None,
    make_plots: bool = True,
) -> dict:
    """
    Build pseudo-geological unit/group models from inversion results.

    Parameters
    ----------
    project_name : str
        Project name used to resolve default file names.
    input_dir : Path or str, optional
        Directory containing ``*_unit_defs.csv`` and ``*_unit_groups.csv``.
        Defaults to ``Path(project_name)``.
    inversion_dir : Path or str, optional
        Read-only inversion output directory. Defaults to
        ``Path(f"{project_name}_Inversion")``.
    min_voxels : int
        Minimum connected-component size kept during geo-group cleanup.
    fill_iterations : int
        Maximum iterations for majority-vote zero filling.
    unit_defs_csv : Path or str, optional
        Optional path to unit definition CSV.
    unit_groups_csv : Path or str, optional
        Optional path to unit-to-group mapping CSV.
    unit_id_npy : Path or str, optional
        Optional precomputed unit-id volume. If provided, it is used directly.
    output_dir : Path or str, optional
        Interpretation output directory. New geology files are written below
        ``output_dir/geology_models``. If omitted, the historical behavior of
        writing below ``inversion_dir`` is retained.
    make_plots : bool
        Retained for workflow compatibility. Plot generation remains enabled
        for existing callers; lightweight test callers may set this to false.

    Returns
    -------
    dict
        Core models, classified labels, mapping metadata, and output paths.
    """
    input_dir = Path(input_dir) if input_dir is not None else Path(project_name)
    inversion_dir = Path(inversion_dir) if inversion_dir is not None else Path(f"{project_name}_Inversion")
    inversion_dir = inversion_dir.expanduser().resolve()
    output_dir = Path(output_dir) if output_dir is not None else inversion_dir
    output_dir = output_dir.expanduser().resolve()

    # Resolve input/output paths.
    input_unit = Path(unit_defs_csv) if unit_defs_csv is not None else input_dir / f"{project_name}_unit_defs.csv"
    input_unit_groups = Path(unit_groups_csv) if unit_groups_csv is not None else input_dir / f"{project_name}_unit_groups.csv"
    input_unit_id = Path(unit_id_npy) if unit_id_npy else None

    out_msh_dir = inversion_dir / "mesh"
    out_res_dir = inversion_dir / "inversion_result"

    msh_core_UBC_path = out_msh_dir / "mesh_core.msh"
    dens_core_path = out_res_dir / "joint_density_core.npy"
    susc_core_path = out_res_dir / "joint_susceptibility_core.npy"

    out_geo_dir = output_dir / "geology_models"
    out_geo_dir.mkdir(parents=True, exist_ok=True)

    out_geo_slices = out_geo_dir / "slices_and_sections"
    out_geo_slices.mkdir(parents=True, exist_ok=True)

    # 1) Load core mesh and inversion models.
    mesh_core = discretize.TensorMesh._readUBC_3DMesh(msh_core_UBC_path)
    print("mesh_core.shape_cells =", mesh_core.shape_cells)

    dens_core_3d = np.load(dens_core_path)
    susc_core_3d = np.load(susc_core_path)
    print("dens_core_3d.shape =", dens_core_3d.shape)
    print("susc_core_3d.shape =", susc_core_3d.shape)

    nx, ny, nz = dens_core_3d.shape
    assert (nx, ny, nz) == mesh_core.shape_cells, \
        f"Mesh cells {mesh_core.shape_cells} != model {dens_core_3d.shape}"

    nodes = mesh_core.nodes
    x_nodes = np.unique(nodes[:, 0])
    y_nodes = np.unique(nodes[:, 1])
    z_nodes = np.unique(nodes[:, 2])
    print("len(x_nodes), len(y_nodes), len(z_nodes) =",
          len(x_nodes), len(y_nodes), len(z_nodes))

    # -----------------------------------------
    # Active-mask from topography (below surface only)
    # -----------------------------------------
    topo_xyz_path = inversion_dir / "topo" / "topography.xyz"
    active_mask_3d = np.ones(dens_core_3d.shape, dtype=bool)
    if topo_xyz_path.exists():
        topo_xyz = np.loadtxt(topo_xyz_path)
        if topo_xyz.ndim == 1:
            topo_xyz = topo_xyz.reshape(1, -1)
        if topo_xyz.shape[1] >= 3:
            # active_from_xyz returns a flat mask in mesh ordering (Fortran-like)
            # so reshape with order='F' to align with core 3D arrays.
            ind_active = active_from_xyz(mesh_core, topo_xyz[:, :3])
            active_mask_3d = ind_active.reshape(dens_core_3d.shape, order="F")
        else:
            print(f"[WARN] {topo_xyz_path} has invalid shape {topo_xyz.shape}; using all cells as active.")
    else:
        print(f"[WARN] Topography file not found: {topo_xyz_path}; using all cells as active.")

    active_flat = active_mask_3d.ravel()
    n_total = int(active_flat.size)
    n_active = int(active_flat.sum())
    n_inactive = n_total - n_active
    print(f"Active cells (below topography): {n_active}/{n_total} ({100.0*n_active/n_total:.2f}%)")
    print(f"Inactive cells (air/above topography): {n_inactive}/{n_total} ({100.0*n_inactive/n_total:.2f}%)")

    # 2) Build unit_id model from CSV ranges or precomputed labels.
    df_unit = pd.read_csv(input_unit)
    unit_defs = {}
    for r in df_unit.itertuples(index=False):
        unit_defs[int(r.unit_id)] = {
            "name": str(r.name),
            "dens_range": (float(r.dens_min), float(r.dens_max)),
            "susc_range": (float(r.susc_min), float(r.susc_max)),
        }

    dens_flat = dens_core_3d.ravel()
    susc_flat = susc_core_3d.ravel()

    # Optional direct unit labels from clustering; when provided, this
    # preserves the exact unsupervised classification without re-binning
    # through rectangular density-susceptibility intervals.
    if input_unit_id is not None and input_unit_id.exists():
        unit_id_3d = np.load(input_unit_id).astype(np.int16, copy=False)
        if unit_id_3d.shape != dens_core_3d.shape:
            raise ValueError(
                f"unit_id_npy shape mismatch: {unit_id_3d.shape} vs model {dens_core_3d.shape}"
            )
        print(f"Loaded unit_id_npy: {input_unit_id}")
        unit_flat = unit_id_3d.ravel()
        # Exclude cells above topography from unit statistics and downstream grouping.
        unit_flat = unit_flat.copy()
        unit_flat[~active_flat] = 0
        unit_id_3d = unit_flat.reshape(dens_core_3d.shape)
    else:
        unit_flat = np.zeros_like(dens_flat, dtype=np.int16)
        for unit_id, info in unit_defs.items():
            dmin, dmax = info["dens_range"]
            smin, smax = info["susc_range"]
            mask = (
                active_flat &
                (dens_flat >= dmin) & (dens_flat < dmax) &
                (susc_flat >= smin) & (susc_flat < smax)
            )
            unit_flat[mask] = unit_id
        unit_id_3d = unit_flat.reshape(dens_core_3d.shape)

    unique, counts = np.unique(unit_id_3d, return_counts=True)
    print("voxel count per unit_id:", dict(zip(unique, counts)))
    unit_flat = unit_id_3d.ravel()

    # 3) Crossplot density vs susceptibility colored by unit_id.
    plt.figure(figsize=(9, 6))
    dens_scatter = dens_flat[active_flat]
    susc_scatter = susc_flat[active_flat]
    unit_scatter = unit_flat[active_flat]
    if dens_scatter.size == 0:
        raise ValueError("No active cells found for scatter analysis.")

    unique_ids = np.unique(unit_scatter)
    has_unclassified = 0 in unique_ids
    unique_ids = unique_ids[unique_ids != 0]
    unique_ids = np.sort(unique_ids)

    n_units = len(unique_ids)
    cmap = mpl.colormaps.get_cmap("coolwarm").resampled(n_units)

    for i, uid in enumerate(unique_ids):
        mask = unit_scatter == uid
        if not np.any(mask):
            continue
        label = f"Unit {uid}: {unit_defs.get(uid, {}).get('name', '')}"
        color = cmap(i)
        plt.scatter(
            dens_scatter[mask],
            susc_scatter[mask],
            s=5,
            alpha=0.5,
            label=label,
            color=color,
        )

    if has_unclassified:
        mask0 = unit_scatter == 0
        if np.any(mask0):
            plt.scatter(
                dens_scatter[mask0],
                susc_scatter[mask0],
                s=5,
                alpha=0.3,
                color="lightgray",
                label="Unclassified (0)",
            )

    plt.axhline(0.0, color="k", linewidth=0.5)
    plt.axvline(0.0, color="k", linewidth=0.5)

    # Auto-adjust axis limits based on actual data range with 10% padding
    dens_min, dens_max = np.nanmin(dens_scatter), np.nanmax(dens_scatter)
    susc_min, susc_max = np.nanmin(susc_scatter), np.nanmax(susc_scatter)

    dens_pad = (dens_max - dens_min) * 0.1 if dens_max != dens_min else 0.1
    susc_pad = (susc_max - susc_min) * 0.1 if susc_max != susc_min else 0.01

    plt.xlim(dens_min - dens_pad, dens_max + dens_pad)
    plt.ylim(susc_min - susc_pad, susc_max + susc_pad)

    plt.xlabel("Density (g/cm$^3$)")
    plt.ylabel("Susceptibility (SI)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(loc="lower left", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(out_geo_dir / "density_susceptibility_scatter_by_unit.png", dpi=300)
    plt.close()

    # 4) Map units to coarser geo groups.
    df_groups = pd.read_csv(input_unit_groups)
    unit_to_geo = {
        int(r.unit_id): int(r.geo_id)
        for r in df_groups.itertuples(index=False)
    }
    geo_defs = (
        df_groups[["geo_id", "geo_name"]]
        .drop_duplicates()
        .sort_values("geo_id")
        .set_index("geo_id")["geo_name"]
        .to_dict()
    )
    geo_defs = {int(k): str(v) for k, v in geo_defs.items()}

    print("unit_to_geo:", unit_to_geo)
    print("geo_defs:", geo_defs)

    unit_flat = unit_id_3d.ravel()
    geo_flat = np.zeros_like(unit_flat, dtype=np.int16)
    for u, g in unit_to_geo.items():
        geo_flat[unit_flat == u] = g
    # Keep air/above-topography cells excluded from geo-group assignment.
    geo_flat[~active_flat] = 0

    geo_id_3d = geo_flat.reshape(unit_id_3d.shape)
    unique_geo, counts_geo = np.unique(geo_id_3d, return_counts=True)
    print("geo_id voxel count (raw):", dict(zip(unique_geo, counts_geo)))

    # 5) Clean geo groups: remove tiny components and fill small holes.
    geo_id_clean = geo_id_3d.copy()
    structure3d = np.ones((3, 3, 3), dtype=bool)

    for gid in np.unique(geo_id_3d):
        if gid == 0:
            continue
        mask = (geo_id_clean == gid) & active_mask_3d
        if not np.any(mask):
            continue

        labeled, ncomp = ndimage.label(mask, structure=structure3d)
        print(f"geo group {gid}: {ncomp} components")
        sizes = ndimage.sum(mask, labeled, index=range(1, ncomp + 1))
        small_labels = [lab for lab, size in enumerate(sizes, start=1)
                        if size < min_voxels]
        print(f"  remove {len(small_labels)} small components (< {min_voxels} voxels)")
        if small_labels:
            small_labels = np.array(small_labels)
            remove_mask = np.isin(labeled, small_labels)
            geo_id_clean[remove_mask] = 0
    geo_id_clean[~active_mask_3d] = 0

    unique_geo, counts_geo = np.unique(geo_id_clean, return_counts=True)
    print("after removing small components:", dict(zip(unique_geo, counts_geo)))

    labels = geo_id_clean.copy()
    structure3d_int = np.ones((3, 3, 3), dtype=int)

    def majority_func(block):
        center = block[block.size // 2]
        if center != 0:
            return center
        neigh = block[block != 0]
        if neigh.size == 0:
            return 0
        vals, counts = np.unique(neigh, return_counts=True)
        return vals[np.argmax(counts)]

    for it in range(fill_iterations):
        new_labels = ndimage.generic_filter(
            labels,
            function=majority_func,
            footprint=structure3d_int,
            mode="nearest",
        )
        # Never fill cells above topography.
        new_labels[~active_mask_3d] = 0
        if np.array_equal(new_labels, labels):
            print(f"fill iteration {it}: no change, stop.")
            break
        labels = new_labels
        print(f"fill iteration {it}: done.")

    geo_id_filled = labels
    geo_id_filled[~active_mask_3d] = 0
    unique_geo2, counts_geo2 = np.unique(geo_id_filled, return_counts=True)
    print("after filling zeros:", dict(zip(unique_geo2, counts_geo2)))

    geo_id_3d = geo_id_filled

    # A no-plot path is useful for automated tests and cheap comparison runs.
    # It still writes the deterministic label arrays and definitions, but does
    # not invoke Matplotlib or PyVista.
    if not make_plots:
        out_unit_id_3d_npy = out_geo_dir / "unit_id_3d.npy"
        out_geo_id_3d_npy = out_geo_dir / "geo_id_3d.npy"
        out_geo_defs_json = out_geo_dir / "geo_defs.json"
        np.save(out_unit_id_3d_npy, unit_id_3d.astype(np.int16, copy=False))
        np.save(out_geo_id_3d_npy, geo_id_3d.astype(np.int16, copy=False))
        out_geo_defs_json.write_text(
            json.dumps({str(int(k)): str(v) for k, v in geo_defs.items()}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "mesh_core": mesh_core,
            "dens_core_3d": dens_core_3d,
            "susc_core_3d": susc_core_3d,
            "unit_id_3d": unit_id_3d,
            "geo_id_3d": geo_id_3d,
            "paths": {
                "source_inversion_dir": str(inversion_dir),
                "interpretation_output_dir": str(output_dir),
                "output_root": str(output_dir),
                "geology_models_dir": str(out_geo_dir),
                "mesh_core_ubc": str(msh_core_UBC_path),
                "dens_core_npy": str(dens_core_path),
                "susc_core_npy": str(susc_core_path),
                "unit_id_3d_npy": str(out_unit_id_3d_npy),
                "geo_id_3d_npy": str(out_geo_id_3d_npy),
                "geo_defs_json": str(out_geo_defs_json),
                "unit_defs_csv": str(input_unit),
                "unit_groups_csv": str(input_unit_groups),
                "unit_id_npy": str(input_unit_id) if input_unit_id is not None else "",
                "geo_slices_dir": str(out_geo_slices),
                "geo_geo_id_pngs": [],
                "geo_combo_slice_pngs": [],
                "geo_combo_section_pngs": [],
                "geo_3d_png": "",
            },
            "geo_defs": geo_defs,
            "unit_defs": unit_defs,
            "source_inversion_dir": str(inversion_dir),
            "interpretation_output_dir": str(output_dir),
        }

    # 6) Render 3D geology figures with PyVista.
    def setup_bounds(p, grid, font_size=7):
        actor = p.show_bounds(
            grid="front",
            location="outer",
            all_edges=False,
            xtitle="Easting (m)",
            ytitle="Northing (m)",
            ztitle="Elevation (m)",
            font_size=font_size,
            n_xlabels=3,
            n_ylabels=3,
        )
        actor.label_offset = 10

    def setup_camera(p, grid, zoom=1.1):
        xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        p.camera_position = [
            (center[0] + 2 * dx, center[1] + dy, center[2] + 1.5 * dz),
            center,
            (0, 0, 1),
        ]
        p.camera.zoom(zoom)

    grid = pv.RectilinearGrid(x_nodes, y_nodes, z_nodes)
    geo_flat_plot = geo_id_3d.ravel(order="F").astype(np.int16)
    grid.cell_data["geo_id"] = geo_flat_plot

    mask_nz = (geo_flat_plot != 0) & (geo_flat_plot != 1)
    grid_nz = grid.extract_cells(mask_nz)

    all_geo_ids = np.unique(geo_flat_plot)
    max_id = int(all_geo_ids.max())
    min_id = 1
    n_colors = max_id - min_id + 1
    base = mpl.colormaps.get_cmap("coolwarm").resampled(n_colors)
    colors = [base(i) for i in range(n_colors)]
    cmap_geo = ListedColormap(colors)
    clim = (min_id, max_id)

    scalar_bar_args = dict(
        title="Geo ID",
        n_labels=n_colors,
        fmt="%.0f",
        vertical=True,
        position_x=0.88,
        position_y=0.15,
        width=0.08,
        height=0.7,
    )
   
    combo_slice_pngs = []
    combo_section_pngs = []
    geo_single_pngs = []

    # Overall 3D geology model without background group.
    p = pv.Plotter(off_screen=True)
    p.set_background("white")
    p.add_mesh(
        grid_nz,
        scalars="geo_id",
        cmap=cmap_geo,
        show_edges=False,
        opacity=0.95,
        clim=clim,
        scalar_bar_args=scalar_bar_args,
    )
    p.add_mesh(grid.outline(), color="black", line_width=2)
    setup_bounds(p, grid)
    setup_camera(p, grid, zoom=0.8)
    geo_all = out_geo_dir / "geo_model_without_background.jpg"
    p.screenshot(str(geo_all))
    p.close()
    print("Save overall figure:", geo_all)

    # Per-group 3D renders.
    single_ids = sorted(set(geo_flat_plot) - {0})
    for gid in single_ids:
        mask_show = (geo_flat_plot == gid)
        if not np.any(mask_show):
            continue
        grid_show = grid.extract_cells(mask_show)

        p = pv.Plotter(off_screen=True)
        p.set_background("white")
        p.add_mesh(
            grid_show,
            scalars="geo_id",
            cmap=cmap_geo,
            show_edges=False,
            opacity=1.0,
            clim=clim,
            scalar_bar_args=scalar_bar_args,
        )
        p.add_mesh(grid.outline(), color="black", line_width=2)
        setup_bounds(p, grid)
        setup_camera(p, grid, zoom=0.8)
        out_single = out_geo_dir / f"geo_id_{gid}.jpg"
        p.screenshot(str(out_single))
        p.close()
        print(f"Save geo_id={gid} figure:", out_single)
        geo_single_pngs.append(str(out_single))

    # 7) Export combined geology/density/susceptibility slices and sections.
    xc = mesh_core.cell_centers_x
    yc = mesh_core.cell_centers_y
    zc = mesh_core.cell_centers_z

    all_geo_plot = all_geo_ids[all_geo_ids != 0]
    n_geo = len(all_geo_plot)
    bounds = np.concatenate([all_geo_plot - 0.5,
                             [all_geo_plot[-1] + 0.5]])
    norm = BoundaryNorm(bounds, ncolors=n_geo)
    vmax_dens = float(np.nanmax(np.abs(dens_core_3d)))/5 if np.isfinite(dens_core_3d).any() else 1e-6
    vmax_susc = float(np.nanmax(np.abs(susc_core_3d)))/5 if np.isfinite(susc_core_3d).any() else 1e-6

    # XY slices.
    z_indices = np.linspace(0, nz - 1, 10, dtype=int)
    z_indices = np.unique(z_indices)
    for k in z_indices:
        z_val = zc[k]
        print(f"[XY slice] k={k}, z={z_val}")
        slice_xy = geo_id_3d[:, :, k]
        slice_xy_plot = np.ma.masked_where(slice_xy == 0, slice_xy)

        # Combined panel: geology + density + susceptibility.
        dens_xy = dens_core_3d[:, :, k]
        susc_xy = susc_core_3d[:, :, k]
        fig_c, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
        im0 = ax0.imshow(
            slice_xy_plot.T,
            origin="lower",
            cmap=cmap_geo,
            norm=norm,
            extent=[xc[0], xc[-1], yc[0], yc[-1]],
            interpolation="nearest",
        )
        fig_c.colorbar(im0, ax=ax0, boundaries=bounds, ticks=all_geo_plot, label="Geo group ID")
        ax0.set_title(f"Geo model z ~ {z_val:.0f} m")
        ax0.set_xlabel("Easting (m)")
        ax0.set_ylabel("Northing (m)")

        im1 = ax1.imshow(
            dens_xy.T,
            origin="lower",
            cmap="seismic",
            vmin=-vmax_dens,
            vmax=vmax_dens,
            extent=[xc[0], xc[-1], yc[0], yc[-1]],
        )
        fig_c.colorbar(im1, ax=ax1, label="Density contrast (g/cc)")
        ax1.set_title("Density inversion")
        ax1.set_xlabel("Easting (m)")

        im2 = ax2.imshow(
            susc_xy.T,
            origin="lower",
            cmap="seismic",
            vmin=-vmax_susc,
            vmax=vmax_susc,
            extent=[xc[0], xc[-1], yc[0], yc[-1]],
        )
        fig_c.colorbar(im2, ax=ax2, label="Susceptibility (SI)")
        ax2.set_title("Magnetics inversion")
        ax2.set_xlabel("Easting (m)")

        fig_c.tight_layout()
        fname_combo = out_geo_slices / f"combo_slice_xy_k{k:03d}_z{z_val:.0f}.png"
        fig_c.savefig(fname_combo, dpi=300)
        plt.close(fig_c)
        combo_slice_pngs.append(str(fname_combo))

    # XZ sections.
    Xg, Zg = np.meshgrid(x_nodes, z_nodes)
    y_indices = np.linspace(0, ny - 1, 10, dtype=int)
    y_indices = np.unique(y_indices)
    for j in y_indices:
        y_val = yc[j]
        print(f"[XZ section] j={j}, y={y_val}")
        slice_xz = geo_id_3d[:, j, :]
        slice_xz_plot = np.ma.masked_where(slice_xz == 0, slice_xz)

        # Combined panel: geology + density + susceptibility.
        dens_xz = dens_core_3d[:, j, :]
        susc_xz = susc_core_3d[:, j, :]
        fig_c, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)

        pc0 = ax0.pcolormesh(
            Xg, Zg, slice_xz_plot.T,
            cmap=cmap_geo,
            norm=norm,
            shading="auto",
        )
        fig_c.colorbar(pc0, ax=ax0, boundaries=bounds, ticks=all_geo_plot, label="Geo group ID")
        ax0.set_title(f"Geo model y ~ {y_val:.0f} m")
        ax0.set_xlabel("Easting (m)")
        ax0.set_ylabel("Elevation (m)")

        pc1 = ax1.pcolormesh(
            Xg, Zg, dens_xz.T,
            cmap="seismic",
            vmin=-vmax_dens,
            vmax=vmax_dens,
            shading="auto",
        )
        fig_c.colorbar(pc1, ax=ax1, label="Density contrast (g/cc)")
        ax1.set_title("Density inversion")
        ax1.set_xlabel("Easting (m)")

        pc2 = ax2.pcolormesh(
            Xg, Zg, susc_xz.T,
            cmap="seismic",
            vmin=-vmax_susc,
            vmax=vmax_susc,
            shading="auto",
        )
        fig_c.colorbar(pc2, ax=ax2, label="Susceptibility (SI)")
        ax2.set_title("Magnetics inversion")
        ax2.set_xlabel("Easting (m)")

        fig_c.tight_layout()
        fname_combo = out_geo_slices / f"combo_section_xz_j{j:03d}_y{y_val:.0f}.png"
        fig_c.savefig(fname_combo, dpi=300)
        plt.close(fig_c)
        combo_section_pngs.append(str(fname_combo))

    # -----------------------------------------
    # 8) Persist core pseudo-geology arrays and mapping metadata.
    # -----------------------------------------
    out_unit_id_3d_npy = out_geo_dir / "unit_id_3d.npy"
    out_geo_id_3d_npy = out_geo_dir / "geo_id_3d.npy"
    out_geo_defs_json = out_geo_dir / "geo_defs.json"

    np.save(out_unit_id_3d_npy, unit_id_3d.astype(np.int16, copy=False))
    np.save(out_geo_id_3d_npy, geo_id_3d.astype(np.int16, copy=False))
    with out_geo_defs_json.open("w", encoding="utf-8") as f:
        json.dump(
            {str(int(k)): str(v) for k, v in geo_defs.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Saved unit_id_3d:", out_unit_id_3d_npy)
    print("Saved geo_id_3d:", out_geo_id_3d_npy)
    print("Saved geo_defs:", out_geo_defs_json)

    paths = {
        "source_inversion_dir": str(inversion_dir),
        "interpretation_output_dir": str(output_dir),
        "output_root": str(output_dir),
        "geology_models_dir": str(out_geo_dir),
        "mesh_core_ubc": str(msh_core_UBC_path),
        "dens_core_npy": str(dens_core_path),
        "susc_core_npy": str(susc_core_path),
        "unit_id_3d_npy": str(out_unit_id_3d_npy),
        "geo_id_3d_npy": str(out_geo_id_3d_npy),
        "geo_defs_json": str(out_geo_defs_json),
        "unit_defs_csv": str(input_unit),
        "unit_groups_csv": str(input_unit_groups),
        "unit_id_npy": str(input_unit_id) if input_unit_id is not None and input_unit_id.exists() else "",
        "scatter_rho_kappa": str(out_geo_dir / "density_susceptibility_scatter_by_unit.png"),
        "geo_slices_dir": str(out_geo_slices),
        "geo_geo_id_pngs": geo_single_pngs,
        "geo_combo_slice_pngs": combo_slice_pngs,
        "geo_combo_section_pngs": combo_section_pngs,
        "geo_3d_png": str(geo_all),
    }

    return {
        "mesh_core": mesh_core,
        "dens_core_3d": dens_core_3d,
        "susc_core_3d": susc_core_3d,
        "unit_id_3d": unit_id_3d,
        "geo_id_3d": geo_id_3d,
        "paths": paths,
        "geo_defs": geo_defs,
        "unit_defs": unit_defs,
        "source_inversion_dir": str(inversion_dir),
        "interpretation_output_dir": str(output_dir),
    }
