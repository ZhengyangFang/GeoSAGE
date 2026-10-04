"""Physical-coordinate exports for the host's existing image and 3D viewers.

All plotting uses copied/masked views. Original arrays, labels and coordinates
are never resampled or changed. Scientific colours keep the same meaning in
the Studio's light and dark appearances.
"""

import json
from pathlib import Path


def export_results(workflow, output):
    import matplotlib.pyplot as plt
    import numpy as np
    import pyvista as pv

    from geosage.plotting import render_data_fit, render_model_sections, render_property_crossplot
    from geosage.plotting.scales import category_colors, field_scale, middle_index
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
        ("Density contrast", density, "g/cm3"),
        ("Susceptibility", susceptibility, "SI"),
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
        "viewer_fields": {},
    }
    for name, array, units in fields:
        finite = array[np.isfinite(array)]
        if not finite.size:
            raise ValueError(f"{name} has no finite cells.")
        metadata["fields"][name] = {
            "units": units,
            "min": float(finite.min()),
            "max": float(finite.max()),
            "finite_cells": int(finite.size),
        }
        vtk_name = (
            "Density contrast (g/cm3)" if name == "Density contrast" else "Susceptibility (SI)"
        )
        metadata["viewer_fields"][vtk_name] = {
            **field_scale(array, centered=name == "Density contrast"),
            "units": units,
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
        for field_name, labels in (("Geo ID", ids), ("Unit ID", unit_ids)):
            metadata["viewer_fields"][field_name] = {
                "colors": category_colors(labels),
                "names": metadata["geo_names"] if field_name == "Geo ID" else {},
                "units": "category",
            }
    model_path = directory / "geosage_models.vtk"
    directory.mkdir(exist_ok=False)
    grid.save(model_path)
    centers = (mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z)
    case = inv.get("inversion_parameters", {}).get("project_name") or workflow.get(
        "config", {}
    ).get("project", {}).get("name", "Model")
    k_indices = [middle_index(mesh.nodes_z)]
    j_indices = [middle_index(mesh.nodes_y)]
    figures = {}
    metadata["display"] = {
        "implementation": "geosage.plotting.paper",
        "preset": "adaptive",
        "dpi": 160,
        "range_policy": "full range per run; observations and predictions share a scale",
        "z_indices": k_indices,
        "y_indices": j_indices,
        "z_positions_m": [float(centers[2][k]) for k in k_indices],
        "y_positions_m": [float(centers[1][j]) for j in j_indices],
    }
    path = directory / "model_sections.png"
    fig = render_model_sections(
        density,
        susceptibility,
        ids,
        mesh.nodes_x,
        mesh.nodes_y,
        mesh.nodes_z,
        centers[2],
        centers[1],
        k_indices,
        j_indices,
        output=path,
        dpi=160,
        adaptive=True,
    )
    plt.close(fig)
    figures["Model sections"] = str(path)
    if ids is not None:
        path = directory / "property_relationships.png"
        names = geo.get("unit_defs") or {}
        definitions = workflow.get("config", {}).get("geology", {}).get("unit_defs_csv")
        if not names and definitions:
            import csv

            with Path(definitions).open(encoding="utf-8-sig", newline="") as handle:
                names = {
                    int(row["unit_id"]): {"name": row["name"]} for row in csv.DictReader(handle)
                }
        names = {int(k): (v if isinstance(v, dict) else {"name": str(v)}) for k, v in names.items()}
        metadata["viewer_fields"]["Unit ID"]["names"] = {
            str(k): v.get("name", "") for k, v in names.items()
        }
        fig = render_property_crossplot(
            density, susceptibility, unit_ids, names, output=path, dpi=160, adaptive=True
        )
        plt.close(fig)
        figures["Physical-property distribution"] = str(path)
    fit_data = {}
    for key, title, unit in (("gravity", "Gravity", "mGal"), ("magnetics", "Magnetics", "nT")):
        observed_path = inv.get("paths", {}).get(f"obs_{key}_ubc")
        predicted = inv.get(f"dpred_{key}")
        if predicted is None:
            predicted_path = inv.get("paths", {}).get(f"dpred_{key}_npy")
            if predicted_path and Path(predicted_path).is_file():
                predicted = np.load(predicted_path)
        if not observed_path or predicted is None:
            continue
        observations = np.loadtxt(observed_path, ndmin=2)
        if observations.shape[1] != 5:
            raise ValueError(f"{title} observations need five columns: x y z value std.")
        require_real_finite(observations, f"{title} observations")
        predicted = np.asarray(predicted).ravel()
        require_real_finite(predicted, f"{title} predictions")
        observed = observations[:, 3]
        if observed.shape != predicted.shape:
            raise ValueError(f"{title} prediction and observation shapes differ.")
        if key == "gravity":
            component = workflow.get("config", {}).get("data", {}).get("gravity_component", "gz")
            unit = "E" if len(component) == 3 else "mGal"
        residual = predicted - observed  # Same convention as paper notebook 4_1.
        fit_data[key] = (observations[:, :2], observed, predicted, residual, unit)
        metadata.setdefault("fit", {})[key] = {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "units": unit,
            "n_stations": len(observed),
            "residual_definition": "predicted - observed",
        }
    if fit_data:
        path = directory / "data_fit.png"
        columns = [
            (case, key, component.upper() if key == "gravity" else "Magnetics") for key in fit_data
        ]
        render_data_fit(columns, lambda _case, key: fit_data[key], path, dpi=160, adaptive=True)
        figures["Data fit"] = str(path)
    (directory / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "model": str(model_path),
        "metadata": str(directory / "model_metadata.json"),
        "figures": figures,
        "viewer_fields": metadata["viewer_fields"],
    }
