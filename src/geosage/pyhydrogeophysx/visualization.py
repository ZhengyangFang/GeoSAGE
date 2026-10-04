"""Physical-coordinate exports for the host's existing image and 3D viewers.

All plotting uses copied/masked views. Original arrays, labels and coordinates
are never resampled or changed. Scientific colours keep the same meaning in
the Studio's light and dark appearances.
"""

import json
from pathlib import Path


def export_results(workflow, output):
    import numpy as np
    import pyvista as pv
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.figure import Figure
    from matplotlib import colormaps, rc_context
    from geosage.validation import require_labels, require_real_finite

    inv = workflow["inversion_result"]
    geo = workflow.get("geology_result") or {}
    mesh = inv["mesh_core"]
    output = Path(output)
    directory = output / "viewer"
    density = np.asarray(inv["dens_core_3d"])
    susceptibility = np.asarray(inv["susc_core_3d"])
    require_real_finite(density, "Density model")
    require_real_finite(susceptibility, "Susceptibility model")
    shape = tuple(mesh.shape_cells)
    if density.shape != shape or susceptibility.shape != shape:
        raise ValueError("Model shape does not match the physical mesh.")
    grid = pv.RectilinearGrid(mesh.nodes_x, mesh.nodes_y, mesh.nodes_z)
    fields = [
        ("Density contrast", density, "g/cm³", "RdBu_r"),
        ("Susceptibility", susceptibility, "SI", "viridis"),
    ]
    grid.cell_data["Density contrast (g/cm3)"] = density.ravel(order="F")
    grid.cell_data["Susceptibility (SI)"] = susceptibility.ravel(order="F")
    metadata = {
        "coordinate_units": "m",
        "z_convention": "elevation, positive up",
        "depth_note": "Depth below ground requires local topographic elevation minus z.",
        "array_order": "x,y,z; VTK cells flattened in Fortran order",
        "shape": list(shape),
        "fields": {},
    }
    for name, array, units, _ in fields:
        finite = array[np.isfinite(array)]
        if not finite.size:
            raise ValueError(f"{name} has no finite cells.")
        metadata["fields"][name] = {
            "units": units,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "finite_cells": int(finite.size),
        }
    ids = geo.get("geo_id_3d")
    if ids is not None:
        ids = np.asarray(ids)
        if ids.shape != shape:
            raise ValueError("Geology labels do not match the physical mesh.")
        require_labels(ids, "Geo IDs")
        unit_ids = require_labels(geo["unit_id_3d"], "Unit IDs")
        if unit_ids.shape != shape:
            raise ValueError("Unit labels do not match the physical mesh.")
        grid.cell_data["Geo ID"] = ids.ravel(order="F")
        grid.cell_data["Unit ID"] = np.asarray(geo["unit_id_3d"]).ravel(order="F")
        metadata["geo_names"] = {str(k): v for k, v in geo.get("geo_defs", {}).items()}
        fields.append(("Geological groups", ids, "Geo ID", "tab20"))
    model_path = directory / "geosage_models.vtk"
    directory.mkdir(exist_ok=False)
    grid.save(model_path)
    (directory / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    centers = (mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z)
    nodes = (mesh.nodes_x, mesh.nodes_y, mesh.nodes_z)
    midpoint = tuple(n // 2 for n in shape)
    figures = {}
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": "#1d1d1f",
        "text.color": "#1d1d1f",
        "axes.edgecolor": "#c7c7cc",
        "axes.titleweight": "medium",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    }
    with rc_context(style):
        for title, values, unit, cmap in fields:
            figure = Figure(figsize=(13, 4.2), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(1, 3)
            views = [
                (
                    values[:, :, midpoint[2]].T,
                    nodes[0],
                    nodes[1],
                    "Easting (m)",
                    "Northing (m)",
                    f"Z elevation {centers[2][midpoint[2]]:,.1f} m",
                ),
                (
                    values[:, midpoint[1], :].T,
                    nodes[0],
                    nodes[2],
                    "Easting (m)",
                    "Z elevation (m)",
                    f"Northing {centers[1][midpoint[1]]:,.1f} m",
                ),
                (
                    values[midpoint[0], :, :].T,
                    nodes[1],
                    nodes[2],
                    "Northing (m)",
                    "Z elevation (m)",
                    f"Easting {centers[0][midpoint[0]]:,.1f} m",
                ),
            ]
            kwargs = {"cmap": cmap}
            if unit == "Geo ID":
                unique = np.unique(values)
                # Compact colour positions support non-contiguous geological IDs.
                colors = colormaps["tab20"](np.linspace(0, 1, max(1, len(unique))))
                if 0 in unique:
                    colors[np.flatnonzero(unique == 0)[0]] = [0.94, 0.94, 0.96, 1]
                kwargs = {
                    "cmap": ListedColormap(colors),
                    "norm": BoundaryNorm(np.arange(len(unique) + 1) - 0.5, len(unique)),
                }
            else:
                kwargs.update(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
            for ax, (plane, x, y, xlabel, ylabel, subtitle) in zip(axes, views):
                data = (
                    np.searchsorted(unique, plane)
                    if unit == "Geo ID"
                    else np.ma.masked_invalid(plane)
                )
                artist = ax.pcolormesh(x, y, data, shading="flat", **kwargs)
                ax.set(xlabel=xlabel, ylabel=ylabel, title=subtitle)
                ax.set_aspect("equal", adjustable="box")
                ax.ticklabel_format(style="plain", useOffset=False)
                ax.tick_params(labelsize=7)
            bar = figure.colorbar(artist, ax=list(axes), shrink=0.8, label=unit)
            if unit == "Geo ID":
                bar.set_ticks(range(len(unique)), labels=[str(int(v)) for v in unique])
            figure.suptitle(title, fontsize=14)
            path = directory / (title.lower().replace(" ", "_") + "_slices.png")
            figure.savefig(path, dpi=160)
            figures[title] = str(path)
        figure = Figure(figsize=(7, 4.5), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        finite = np.isfinite(density) & np.isfinite(susceptibility)
        hist = ax.hexbin(
            density[finite], susceptibility[finite], gridsize=65, mincnt=1, bins="log", cmap="Blues"
        )
        figure.colorbar(hist, ax=ax, label="Cells per bin (log scale)")
        ax.set(
            xlabel="Density contrast (g/cm³)",
            ylabel="Susceptibility (SI)",
            title="Physical-property distribution",
        )
        path = directory / "property_distribution.png"
        figure.savefig(path, dpi=160)
        figures["Physical-property distribution"] = str(path)
        for key, title, unit in (
            ("gravity", "Gravity", "mGal"),
            ("magnetics", "Magnetics", "nT"),
        ):
            observed_path = inv.get("paths", {}).get(f"obs_{key}_ubc")
            predicted = inv.get(f"dpred_{key}")
            if predicted is None:
                predicted_path = inv.get("paths", {}).get(f"dpred_{key}_npy")
                if predicted_path and Path(predicted_path).is_file():
                    predicted = np.load(predicted_path)
            if not observed_path or predicted is None:
                continue
            # GeoSAGE's *.obs exports have five columns and no UBC header:
            # easting, northing, elevation, observation, standard deviation.
            observations = np.loadtxt(observed_path, ndmin=2)
            if observations.shape[1] != 5:
                raise ValueError(f"{title} observations need five columns: x y z value std.")
            require_real_finite(observations, f"{title} observations")
            observed = observations[:, 3]
            predicted = np.asarray(predicted).ravel()
            require_real_finite(predicted, f"{title} predictions")
            if observed.shape != predicted.shape:
                raise ValueError(f"{title} prediction and observation shapes differ.")
            coordinates = observations[:, :3]
            if len(coordinates) != len(observed):
                raise ValueError(f"{title} station coordinates do not match observations.")
            if key == "gravity":
                component = workflow["config"]["data"].get("gravity_component", "gz")
                unit = "Eötvös" if len(component) == 3 else "mGal"
            residual = observed - predicted
            finite = np.isfinite(observed) & np.isfinite(predicted)
            if not finite.any():
                continue
            rmse = float(np.sqrt(np.mean(residual[finite] ** 2)))
            metadata.setdefault("fit", {})[key] = {
                "rmse": rmse,
                "units": unit,
                "n_stations": int(finite.sum()),
                "residual_definition": "observed - predicted",
            }
            figure = Figure(figsize=(13, 4.2), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(1, 3)
            limits = (
                float(min(observed[finite].min(), predicted[finite].min())),
                float(max(observed[finite].max(), predicted[finite].max())),
            )
            bound = max(float(np.abs(residual[finite]).max()), 1e-12)
            for ax, values, label in zip(
                axes, (observed, predicted, residual), ("Observed", "Predicted", "Residual")
            ):
                vmin, vmax = (-bound, bound) if label == "Residual" else limits
                artist = ax.scatter(
                    coordinates[finite, 0],
                    coordinates[finite, 1],
                    c=values[finite],
                    s=9,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set(title=label, xlabel="Easting (m)", ylabel="Northing (m)")
                ax.set_aspect("equal", adjustable="box")
                ax.ticklabel_format(style="plain", useOffset=False)
                ax.tick_params(labelsize=7)
                figure.colorbar(artist, ax=ax, shrink=0.8, label=unit)
            figure.suptitle(f"{title} data fit · RMSE {rmse:.3g} {unit}", fontsize=14)
            path = directory / f"{key}_data_fit.png"
            figure.savefig(path, dpi=160)
            figures[f"{title} data fit"] = str(path)
    (directory / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "model": str(model_path),
        "metadata": str(directory / "model_metadata.json"),
        "figures": figures,
    }
