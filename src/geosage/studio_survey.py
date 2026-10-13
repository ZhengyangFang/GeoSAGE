"""Deterministic survey discovery for conversational Studio setup."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .survey_io import (TABLE_SUFFIXES, import_settings, normalize_table, read_table,
                        table_summary, rows_in_region, resolve_gravity_component)

ROLE_SUFFIXES = {
    "gravity_file": "_gravity_data.csv",
    "magnetic_file": "_magnetic_data.csv",
    "topography_file": "_topo.tif",
    "mesh_file": "_mesh.msh",
    "core_mesh_file": "_mesh_core.msh",
    "unit_defs_file": "_unit_defs.csv",
    "unit_groups_file": "_unit_groups.csv",
    "reference_file": "_geology_context.txt",
}
REQUIRED_ROLES = (
    "gravity_file", "magnetic_file", "topography_file", "mesh_file", "core_mesh_file"
)

GRAVITY_COLUMN_ALIASES = (
    "ISO", "CBA", "SBA", "FAA", "OG", "GX", "GY", "GZ", "GXX", "GXY", "GXZ", "GYY", "GYZ", "GZZ",
    "gravity_gradient_zz", "gravity_gzz", "gravity", "gravity_anomaly",
    "bouguer", "bouguer_anomaly", "bouguer_residual",
)
MAGNETIC_COLUMN_ALIASES = (
    "TFMA", "TMI", "magnetic_anomaly", "total_field",
    "total_magnetic_intensity", "mag",
)
NON_MEASUREMENT_COLUMNS = {
    "easting", "northing", "longitude", "latitude", "height", "elevation",
    "x", "y", "z", "id", "index", "station", "station_id", "line", "line_id",
}


# Results and environment directories must never become raw survey candidates.
SKIP_DIRECTORIES = {
    "outputs", "output", "results", "runs", "reports", "archive", "scratch",
    "__pycache__", "node_modules", "venv", "env", "iteration_model",
    "inversion_result", "geology_models", "observed_data",
}
RESULT_MARKERS = {"inversion_params.json", "studio_result.json", "run_manifest.json"}
FORMAT_LABELS = {
    ".csv": "CSV table", ".tsv": "TSV table", ".txt": "Text",
    ".xyz": "XYZ table", ".xlsx": "Excel workbook", ".xls": "Excel workbook",
    ".tif": "Raster", ".tiff": "Raster", ".asc": "ASCII raster candidate",
    ".grd": "Grid candidate", ".nc": "NetCDF", ".msh": "Mesh candidate",
    ".geojson": "GeoJSON", ".shp": "Shapefile", ".gpkg": "GeoPackage",
    ".vtk": "VTK model", ".vtu": "VTK model", ".vtr": "VTK model",
    ".pdf": "PDF document", ".json": "JSON", ".md": "Text document",
}


def _survey_inventory(folder: Path, limit: int = 2000) -> tuple[list[Path], list[dict], list[str]]:
    """Bounded recursive enumeration without following links or entering run outputs."""
    import os

    files, skipped, errors = [], [], []
    for root, dirs, names in os.walk(folder, followlinks=False, onerror=lambda exc: errors.append(str(exc))):
        parent = Path(root)
        if RESULT_MARKERS.intersection(name.casefold() for name in names):
            skipped.append({"path": str(parent), "reason": "existing_result"})
            dirs[:] = []
            continue
        allowed = []
        for name in sorted(dirs, key=str.casefold):
            child = parent / name
            if (name.startswith(".") or name.casefold() in SKIP_DIRECTORIES
                    or "_inversion" in name.casefold() or child.is_symlink()
                    or getattr(child, "is_junction", lambda: False)()):
                skipped.append({"path": str(child), "reason": "output_environment_or_link"})
            else:
                allowed.append(name)
        dirs[:] = allowed
        for name in sorted(names, key=str.casefold):
            path = parent / name
            if name.startswith(".") or path.is_symlink():
                continue
            if path.suffix.casefold() not in FORMAT_LABELS:
                continue
            files.append(path)
            if len(files) >= limit:
                errors.append(f"Scan reached {limit} files. Select a smaller survey folder.")
                return files, skipped, errors
    return files, skipped, errors


def _named_role(path: Path, words: tuple[str, ...]) -> bool:
    import re

    tokens = re.split(r"[^a-z0-9]+", path.stem.casefold())
    return any(word in tokens for word in words)


def _measurement_candidates(summary: dict, aliases: tuple[str, ...]) -> list[str]:
    names = {alias.casefold() for alias in aliases}
    return [column for column in summary.get("columns", []) if column.casefold() in names]


def _discover_files(folder: Path, role_overrides: dict | None = None, settings=None) -> dict:
    settings = import_settings(settings)
    paths, skipped, scan_errors = _survey_inventory(folder)
    overrides = {}
    for role, value in (role_overrides or {}).items():
        if role not in ROLE_SUFFIXES or not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = folder / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Explicit {role} does not exist: {path}")
        overrides[role] = str(path)
        if path not in paths:
            paths.append(path)

    inventory, candidates, summaries = [], {role: [] for role in ROLE_SUFFIXES}, {}
    for path in paths:
        path = path.resolve()
        suffix = path.suffix.casefold()
        entry = {"path": str(path), "format": FORMAT_LABELS.get(suffix, "Unknown"),
                 "roles": [], "issues": []}
        evidence = {}
        # Conventional names remain useful evidence, but never outrank a second candidate.
        for role, ending in ROLE_SUFFIXES.items():
            if (path.name.casefold().endswith(ending) or (role in {"gravity_file", "magnetic_file", "topography_file"}
                    and path.stem.casefold().endswith(Path(ending).stem)
                    and path.suffix.casefold() in TABLE_SUFFIXES | {".tif", ".tiff", ".asc"})):
                evidence[role] = "conventional_filename"
        if suffix in TABLE_SUFFIXES:
            for role, words in (("gravity_file", ("gravity", "grav")), ("magnetic_file", ("magnetic", "mag"))):
                if _named_role(path, words):
                    evidence.setdefault(role, "descriptive_filename")
            try:
                opts = next((settings.get(role, {}) for role in ("gravity_file", "magnetic_file")
                             if overrides.get(role) == str(path) or role in evidence), {})
                raw_table, reader_meta = read_table(path, opts)
                entry["columns"] = list(raw_table.columns)
                entry["reader"] = reader_meta
                content_roles = set()
                for role, aliases, words in (
                    ("gravity_file", GRAVITY_COLUMN_ALIASES, ("gravity", "grav")),
                    ("magnetic_file", MAGNETIC_COLUMN_ALIASES, ("magnetic", "mag")),
                ):
                    columns_for_role = _measurement_candidates(entry, aliases)
                    if columns_for_role:
                        content_roles.add(role)
                        evidence[role] = "measurement_columns: " + ", ".join(columns_for_role)
                    elif _named_role(path, words):
                        evidence.setdefault(role, "descriptive_filename")
                if content_roles:
                    for role in ("gravity_file", "magnetic_file"):
                        if role in evidence and role not in content_roles:
                            del evidence[role]
                            entry["issues"].append(f"Filename suggests {role}, but the measurement columns identify another type.")
                entry["rows"] = len(raw_table)
            except (ValueError, OSError, UnicodeError, csv.Error) as exc:
                entry["issues"].append(f"Cannot read survey: {exc}")
        elif suffix in {".tif", ".tiff", ".asc"}:
            try:
                meta = _topography_summary(path, None, settings.get("topography_file"), settings.get("projected_crs"))
                entry["crs"] = meta.get("crs")
                entry["shape"] = meta["shape"]
                entry["format"] = "Georeferenced raster" if meta.get("crs") else "Raster without CRS"
                if _named_role(path, ("dem", "dtm", "dsm", "topo", "terrain", "topography", "elevation")):
                    evidence.setdefault("topography_file", "terrain_filename_and_readable_raster")
                else:
                    entry["issues"].append("Raster content does not establish elevation; specify its role if it is terrain.")
            except ValueError as exc:
                entry["issues"].append(str(exc))
        elif suffix == ".msh":
            try:
                meta = _ubc_mesh_summary(path)
                summaries[str(path)] = meta
                entry["format"] = "UBC 3-D tensor mesh"
                entry["bounds"] = meta["bounds"]
                if _named_role(path, ("core", "inner", "roi")):
                    evidence.setdefault("core_mesh_file", "core_filename_and_UBC_geometry")
                elif _named_role(path, ("full", "domain", "global")):
                    evidence.setdefault("mesh_file", "full_filename_and_UBC_geometry")
            except (ValueError, OSError, UnicodeError) as exc:
                entry["issues"].append(str(exc))
        elif suffix in {".tsv", ".txt", ".xyz", ".xlsx", ".xls", ".asc", ".grd", ".nc"}:
            if not evidence:
                entry["issues"].append("Format identified; a compatible reader or explicit conversion is still required for inversion.")
        for role, reason in evidence.items():
            candidates[role].append({"path": str(path), "evidence": reason})
        entry["roles"] = list(evidence)
        inventory.append(entry)

    # Unnamed mesh roles require geometric nesting, not a guess based on cell counts.
    meshes = [entry for entry in inventory if entry["format"] == "UBC 3-D tensor mesh"]
    if len(meshes) == 2:
        outer, inner = meshes
        if _contains(inner["bounds"], outer["bounds"], vertical=True):
            outer, inner = inner, outer
        if (outer["bounds"] != inner["bounds"]
                and _contains(outer["bounds"], inner["bounds"], vertical=True)
                and "core_mesh_file" not in outer["roles"]
                and "mesh_file" not in inner["roles"]):
            for role, entry in (("mesh_file", outer), ("core_mesh_file", inner)):
                if role not in entry["roles"]:
                    candidates[role].append({"path": entry["path"], "evidence": "strict_geometric_nesting"})
                    entry["roles"].append(role)

    matches, ambiguous, selection = {}, {}, {}
    for role, items in candidates.items():
        if role in overrides:
            matches[role] = overrides[role]
            selection[role] = "user_selected"
        elif len(items) == 1:
            matches[role] = items[0]["path"]
            selection[role] = items[0]["evidence"]
        elif items:
            ambiguous[role] = items
    prefixes = {
        Path(path).name[:-len(ROLE_SUFFIXES[role])]
        for role, path in matches.items() if role in REQUIRED_ROLES
        and Path(path).name.casefold().endswith(ROLE_SUFFIXES[role])
    }
    return {"files": matches, "project": next(iter(prefixes)) if len(prefixes) == 1 else folder.name,
            "inventory": inventory, "candidates": candidates, "ambiguous_roles": ambiguous,
            "selection_evidence": selection, "skipped": skipped, "scan_errors": scan_errors,
            "summaries": summaries}


def _detect_measurement_column(summary: dict, aliases: tuple[str, ...]) -> str | None:
    candidates = _measurement_candidates(summary, aliases)
    if candidates:
        return candidates[0] if len(candidates) == 1 else None
    # Do not reuse a recognized measurement of the other physical type.
    known = {name.casefold() for name in GRAVITY_COLUMN_ALIASES + MAGNETIC_COLUMN_ALIASES}
    candidates = [name for name in summary.get("numeric_columns", [])
                  if name.casefold() not in NON_MEASUREMENT_COLUMNS | known
                  and not name.casefold().startswith("unnamed")]
    return candidates[0] if len(candidates) == 1 else None


def _resolve_column(summary: dict, requested: str, label: str) -> str:
    lookup = {name.casefold(): name for name in summary.get("columns", [])}
    resolved = lookup.get(str(requested).strip().casefold())
    if not resolved:
        raise ValueError(f"{label} column '{requested}' is not in the selected table.")
    return resolved


def _validate_numeric_columns(path: Path, required: tuple[str, ...], options=None, target_crs=None) -> None:
    measurement = next((c for c in required if c not in {"Easting", "Northing", "Height", "Longitude", "Latitude"}), None)
    table, _ = normalize_table(path, "gravity_file" if "Height" in required else "survey",
                               options, target_crs=target_crs, measurement=measurement)
    missing = set(required) - set(table)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")


def _meaningful_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        text = raw.split("!", 1)[0].strip()
        if text:
            lines.append(text)
    return lines


def _expand_widths(tokens: list[str], expected: int, axis: str, path: Path) -> tuple[list[float], int]:
    values: list[float] = []
    consumed = 0
    while len(values) < expected and consumed < len(tokens):
        token = tokens[consumed]
        consumed += 1
        if "*" in token:
            count_text, value_text = token.split("*", 1)
            try:
                count, value = int(count_text), float(value_text)
            except ValueError as exc:
                raise ValueError(f"{path.name} has an invalid {axis}-cell token: {token}") from exc
            if count <= 0:
                raise ValueError(f"{path.name} has a non-positive {axis}-cell repetition.")
            values.extend([value] * count)
        else:
            try:
                values.append(float(token))
            except ValueError as exc:
                raise ValueError(f"{path.name} has an invalid {axis}-cell width: {token}") from exc
    if len(values) != expected:
        raise ValueError(
            f"{path.name} declares {expected} {axis}-cells but provides {len(values)} widths."
        )
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"{path.name} has non-positive or non-finite {axis}-cell widths.")
    return values, consumed


def _ubc_mesh_summary(path: Path) -> dict:
    """Read the geometry encoded in a UBC 3-D tensor mesh without loading models."""
    lines = _meaningful_lines(path)
    if len(lines) < 3:
        raise ValueError(f"{path.name} is not a complete UBC 3-D mesh.")
    try:
        shape = [int(value) for value in lines[0].split()]
        origin = [float(value) for value in lines[1].split()]
    except ValueError as exc:
        raise ValueError(f"{path.name} has an invalid UBC mesh header.") from exc
    if (len(shape) != 3 or len(origin) != 3 or any(value <= 0 for value in shape)
            or not all(math.isfinite(value) for value in origin)):
        raise ValueError(f"{path.name} needs three cell counts and a three-value origin.")
    tokens = " ".join(lines[2:]).split()
    widths = []
    offset = 0
    for count, axis in zip(shape, ("x", "y", "z"), strict=True):
        axis_values, used = _expand_widths(tokens[offset:], count, axis, path)
        widths.append(axis_values)
        offset += used
    if offset != len(tokens):
        raise ValueError(f"{path.name} contains unexpected values after its cell widths.")
    hx, hy, hz = widths
    min_e, min_n, top = origin
    bounds = {
        "min_e": min_e, "max_e": min_e + sum(hx),
        "min_n": min_n, "max_n": min_n + sum(hy),
        # UBC writes the upper z origin followed by positive downward widths.
        "min_z": top - sum(hz), "max_z": top,
    }
    return {
        "path": str(path),
        "shape": {"nx": shape[0], "ny": shape[1], "nz": shape[2]},
        "origin": {"easting": min_e, "northing": min_n, "top_elevation": top},
        "bounds": bounds,
        "cell_widths": {
            "x": {"min": min(hx), "max": max(hx)},
            "y": {"min": min(hy), "max": max(hy)},
            "z": {"min": min(hz), "max": max(hz)},
        },
        "cells": shape[0] * shape[1] * shape[2],
    }


def _topography_summary(path: Path, geographic_extent: dict | None, options=None, target_crs=None) -> dict:
    """Read GeoTIFF metadata and project its envelope when survey geography permits."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as exc:  # pragma: no cover - GeoSAGE installs rasterio
        raise ValueError("Reading GeoTIFF metadata requires rasterio.") from exc
    try:
        with rasterio.open(path) as dataset:
            from pyproj import CRS
            from .survey_io import _factor
            options = dict(options or {})
            crs = dataset.crs or (rasterio.crs.CRS.from_user_input(options['crs']) if options.get('crs') else None)
            if dataset.crs and options.get('crs') and CRS(dataset.crs) != CRS(options['crs']):
                raise ValueError('Terrain CRS override conflicts with file metadata.')
            band = options.get('band', 1)
            if type(band) is not int or not 1 <= band <= dataset.count or (dataset.count > 1 and 'band' not in options):
                raise ValueError('Choose one valid terrain elevation band.')
            _factor(options.get('height_unit', 'm'), 'length')
            bounds = dataset.bounds
            result = {
                "path": str(path),
                "shape": {"rows": dataset.height, "columns": dataset.width},
                "crs": crs.to_string() if crs else None,
                "is_geographic": bool(crs and crs.is_geographic),
                "native_bounds": {
                    "left": float(bounds.left), "bottom": float(bounds.bottom),
                    "right": float(bounds.right), "top": float(bounds.top),
                },
                "nodata": float(dataset.nodata) if dataset.nodata is not None and math.isfinite(dataset.nodata) else None,
            }
            if crs and target_crs:
                left, bottom, right, top = transform_bounds(crs, target_crs, *bounds, densify_pts=21)
                result['projected_crs'] = target_crs
                result['projected_bounds'] = dict(min_e=float(left), max_e=float(right), min_n=float(bottom), max_n=float(top))
            elif crs and crs.is_projected:
                result["projected_crs"] = crs.to_string()
                result["projected_bounds"] = {
                    "min_e": float(bounds.left), "max_e": float(bounds.right),
                    "min_n": float(bounds.bottom), "max_n": float(bounds.top),
                }
            elif crs and geographic_extent:
                mean_lon = (geographic_extent["min_lon"] + geographic_extent["max_lon"]) / 2
                mean_lat = (geographic_extent["min_lat"] + geographic_extent["max_lat"]) / 2
                zone = int(math.floor((mean_lon + 180) / 6) + 1)
                epsg = (32600 if mean_lat >= 0 else 32700) + zone
                left, bottom, right, top = transform_bounds(
                    crs, f"EPSG:{epsg}", *bounds, densify_pts=21
                )
                result["projected_crs"] = f"EPSG:{epsg}"
                result["projected_bounds"] = {
                    "min_e": float(left), "max_e": float(right),
                    "min_n": float(bottom), "max_n": float(top),
                }
            return result
    except Exception as exc:
        raise ValueError(f"Could not read topography metadata from {path.name}: {exc}") from exc


def _contains(outer: dict, inner: dict, *, vertical: bool = False) -> bool:
    keys = ["min_e", "max_e", "min_n", "max_n"]
    if vertical:
        keys += ["min_z", "max_z"]
    return all(
        outer[key] <= inner[key] if key.startswith("min_") else outer[key] >= inner[key]
        for key in keys
    )


def _rows_in_region(path: Path, region: dict, options=None, target_crs=None) -> int:
    table, _ = normalize_table(path, 'survey', options, target_crs=target_crs)
    return rows_in_region(table, region)


def inspect_survey_folder(
    path: str | Path, role_overrides: dict[str, str | Path] | None = None, data_import: dict | None = None
) -> dict:
    """Return useful partial evidence even when inputs are missing or invalid."""
    if not str(path).strip():
        raise ValueError("Provide the raw survey folder path.")
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Survey folder does not exist: {folder}")
    settings = import_settings(data_import)
    discovery = _discover_files(folder, role_overrides, settings)
    files = discovery["files"]
    missing = [role for role in REQUIRED_ROLES if role not in files]
    issues = [{"role": "folder", "message": msg} for msg in discovery["scan_errors"]]
    parsed = {}
    for role in REQUIRED_ROLES:
        if role not in files or role == "topography_file":
            continue
        source = Path(files[role])
        try:
            if role in {"gravity_file", "magnetic_file"}:
                table, metadata = normalize_table(source, role, settings.get(role), target_crs=settings.get('projected_crs'))
                parsed[role] = table_summary(table)
                parsed[role]['import'] = metadata
            else:
                parsed[role] = discovery["summaries"].get(str(source)) or _ubc_mesh_summary(source)
        except (ValueError, OSError, UnicodeError, csv.Error) as exc:
            issues.append({"role": role, "path": str(source), "message": str(exc)})
    gravity = parsed.get("gravity_file", {})
    magnetic = parsed.get("magnetic_file", {})
    mesh = parsed.get("mesh_file", {})
    core_mesh = parsed.get("core_mesh_file", {})
    topography = {}
    if files.get("topography_file"):
        source = Path(files["topography_file"])
        try:
            topography = _topography_summary(
                source, gravity.get("geographic_extent") or magnetic.get("geographic_extent"),
                settings.get("topography_file"), settings.get("projected_crs")
            )
            if not topography.get("crs"):
                raise ValueError("The topography has no coordinate reference system (CRS).")
            if topography.get("is_geographic") and not settings.get("projected_crs"):
                if not topography.get("projected_bounds"):
                    raise ValueError("Geographic terrain requires survey Longitude and Latitude for the current projection workflow.")
                if files.get("gravity_file"):
                    _validate_numeric_columns(Path(files["gravity_file"]), ("Longitude", "Latitude"), settings.get("gravity_file"))
        except (ValueError, OSError, UnicodeError, csv.Error) as exc:
            issues.append({"role": "topography_file", "path": str(source), "message": str(exc)})
    region = {key: core_mesh["bounds"][key] for key in ("min_e", "max_e", "min_n", "max_n")} if core_mesh else {}
    spatial_checks = {
        "core_mesh_inside_full_mesh": None, "topography_covers_full_mesh": None,
        "gravity_rows_in_core": None, "magnetic_rows_in_core": None,
    }
    if mesh and core_mesh:
        spatial_checks["core_mesh_inside_full_mesh"] = _contains(mesh["bounds"], core_mesh["bounds"], vertical=True)
        if not spatial_checks["core_mesh_inside_full_mesh"]:
            issues.append({"role": "core_mesh_file", "message": "The core mesh is not contained by the full inversion mesh."})
        else:
            from discretize import TensorMesh
            from .survey_io import core_mesh_slices
            try:
                core_mesh_slices(TensorMesh.read_UBC(files['mesh_file']), TensorMesh.read_UBC(files['core_mesh_file']))
            except ValueError as exc:
                issues.append({"role": "core_mesh_file", "message": str(exc)})
    if mesh and topography.get("projected_bounds"):
        spatial_checks["topography_covers_full_mesh"] = _contains(topography["projected_bounds"], mesh["bounds"])
        if not spatial_checks["topography_covers_full_mesh"]:
            issues.append({"role": "topography_file", "message": "The projected topography does not cover the full inversion mesh."})
    for role, summary in (("gravity", gravity), ("magnetic", magnetic)):
        if region and summary:
            try:
                count = _rows_in_region(Path(files[role + "_file"]), region, settings.get(role + "_file"), settings.get("projected_crs"))
                spatial_checks[role + "_rows_in_core"] = count
                if count == 0:
                    issues.append({"role": role + "_file", "message": f"The core mesh contains no usable {role} observations."})
            except (OSError, UnicodeError, ValueError, csv.Error, KeyError) as exc:
                issues.append({"role": role + "_file", "message": str(exc)})
    gravity_column = _detect_measurement_column(gravity, GRAVITY_COLUMN_ALIASES)
    magnetic_column = _detect_measurement_column(magnetic, MAGNETIC_COLUMN_ALIASES)
    missing_parameters = ["field_strength", "inclination", "declination"]
    if gravity and not gravity_column:
        missing_parameters.append("gravity_column")
    if magnetic and not magnetic_column:
        missing_parameters.append("magnetic_column")
    method_defaults = {
        "gravity_component": resolve_gravity_component(gravity_column), "std_grv": 0.25, "std_mag": 10.0,
        "flight_height_ft": 1000.0, "max_iterations": 50, "beta_cooling": 2.0,
    }
    if core_mesh:
        horizontal_cell_size = min(core_mesh["cell_widths"]["x"]["min"], core_mesh["cell_widths"]["y"]["min"])
        method_defaults["cross_gradient_lambda"] = horizontal_cell_size**4
    return {
        "folder": str(folder), "project": discovery["project"], "files": files, "data_import": settings,
        "status": "needs_inputs" if missing or issues else "needs_parameters",
        "inputs_valid": not missing and not issues,
        "missing_files": missing, "validation_issues": issues,
        "inventory": discovery["inventory"], "candidates": discovery["candidates"],
        "ambiguous_roles": discovery["ambiguous_roles"],
        "selection_evidence": discovery["selection_evidence"], "skipped": discovery["skipped"],
        "gravity": gravity, "magnetic": magnetic,
        "mesh": mesh, "core_mesh": core_mesh, "topography": topography,
        "spatial_checks": spatial_checks,
        "detected": {
            "gravity_column": gravity_column, "magnetic_column": magnetic_column,
            "region": region, "region_source": "core_mesh_file" if region else None,
        },
        "column_candidates": {
            "gravity_column": _measurement_candidates(gravity, GRAVITY_COLUMN_ALIASES),
            "magnetic_column": _measurement_candidates(magnetic, MAGNETIC_COLUMN_ALIASES),
        },
        "missing_parameters": missing_parameters, "method_defaults": method_defaults,
    }


def inspection_text(inspection: dict) -> str:
    """Compact setup preview; detailed evidence stays available to the conversation."""
    labels = {"gravity_file": "Gravity", "magnetic_file": "Magnetics",
              "topography_file": "Terrain", "mesh_file": "Full mesh", "core_mesh_file": "Core mesh"}
    files = inspection["files"]
    lines = [f"Identified {sum(role in files for role in REQUIRED_ROLES)}/{len(REQUIRED_ROLES)} required inputs."]
    for role in REQUIRED_ROLES:
        if role in files:
            lines.append(f"{labels[role]}: {Path(files[role]).name}")
        elif role in inspection["ambiguous_roles"]:
            lines.append(f"{labels[role]}: choose from {len(inspection['ambiguous_roles'][role])} candidates in the conversation.")
        else:
            lines.append(f"{labels[role]}: not identified.")
    if inspection["validation_issues"]:
        lines.append("Input checks: " + inspection["validation_issues"][0]["message"])
    elif inspection["inputs_valid"]:
        lines.append("Inputs checked. Confirm the remaining parameters in the conversation.")
    else:
        lines.append("Partial inspection saved. Resolve missing inputs in the conversation.")
    return "\n".join(lines)


def build_configuration(inspection: dict, parameters: dict) -> dict:
    """Build a full-run configuration from explicit user parameters."""
    missing_files = [role for role in REQUIRED_ROLES if not inspection.get("files", {}).get(role)]
    if missing_files or inspection.get("validation_issues"):
        details = ["Missing or ambiguous inputs: " + ", ".join(missing_files)] if missing_files else []
        details.extend(issue["message"] for issue in inspection.get("validation_issues", []))
        raise ValueError("Input checks need attention before configuring inversion. " + " ".join(details))
    required = ("min_e", "max_e", "min_n", "max_n", "field_strength", "inclination", "declination")
    resolved = dict(parameters)
    region_keys = ("min_e", "max_e", "min_n", "max_n")
    region_overridden = any(parameters.get(key) not in (None, "") for key in region_keys)
    detected_region = inspection.get("detected", {}).get("region") or {}
    for key in region_keys:
        if resolved.get(key) in (None, "") and detected_region.get(key) not in (None, ""):
            resolved[key] = detected_region[key]
    missing = [key for key in required if resolved.get(key) in (None, "")]
    if missing:
        raise ValueError("Still needed from the user: " + ", ".join(missing))
    try:
        values = {key: float(resolved[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise ValueError("Survey bounds and magnetic-field parameters must be numbers.") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Survey bounds and magnetic-field parameters must be finite.")
    if values["min_e"] >= values["max_e"] or values["min_n"] >= values["max_n"]:
        raise ValueError("Each region minimum must be below its maximum.")
    if values["field_strength"] <= 0:
        raise ValueError("Magnetic field strength must be positive.")
    if not -90 <= values["inclination"] <= 90:
        raise ValueError("Inclination must be between -90 and 90 degrees.")
    defaults = dict(inspection["method_defaults"])
    defaults.pop("gravity_component", None)
    optional = {key: parameters.get(key, default) for key, default in defaults.items()}
    try:
        optional = {key: float(value) for key, value in optional.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Method settings must be numeric.") from exc
    if not all(math.isfinite(value) for value in optional.values()):
        raise ValueError("Method settings must be finite.")
    if optional["std_grv"] <= 0 or optional["std_mag"] <= 0:
        raise ValueError("Gravity and magnetic standard deviations must be positive.")
    if optional["flight_height_ft"] < 0:
        raise ValueError("Flight height cannot be negative.")
    if optional["max_iterations"] < 1 or not optional["max_iterations"].is_integer():
        raise ValueError("Maximum iterations must be a positive whole number.")
    if optional["cross_gradient_lambda"] < 0:
        raise ValueError("Cross-gradient weight cannot be negative.")
    if optional["beta_cooling"] < 1:
        raise ValueError("Beta cooling must be at least 1 so beta does not grow each iteration.")
    optional["max_iterations"] = int(optional["max_iterations"])
    gravity_column = str(parameters.get("gravity_column") or inspection["detected"].get("gravity_column") or "").strip()
    if not gravity_column:
        raise ValueError("Choose the gravity data column.")
    gravity_column = _resolve_column(inspection["gravity"], gravity_column, "Gravity")
    gravity_component = resolve_gravity_component(gravity_column, parameters.get('gravity_component'))
    magnetic_column = str(
        parameters.get("magnetic_column")
        or inspection["detected"].get("magnetic_column") or ""
    ).strip()
    if not magnetic_column:
        raise ValueError("Choose the magnetic data column.")
    magnetic_column = _resolve_column(inspection["magnetic"], magnetic_column, "Magnetic")
    files = inspection["files"]
    settings = import_settings(inspection.get('data_import'))
    normalized = {}
    for role, column in (("gravity_file", gravity_column), ("magnetic_file", magnetic_column)):
        normalized[role], _ = normalize_table(Path(files[role]), role, settings.get(role),
            target_crs=settings.get('projected_crs'), measurement=column, component=gravity_component)
    mesh_bounds = inspection.get("mesh", {}).get("bounds")
    configured_region = {key: values[key] for key in ("min_e", "max_e", "min_n", "max_n")}
    if mesh_bounds and not _contains(mesh_bounds, configured_region):
        raise ValueError("The configured region extends outside the full inversion mesh.")
    region_observations = {
        "gravity": rows_in_region(normalized["gravity_file"], configured_region),
        "magnetic": rows_in_region(normalized["magnetic_file"], configured_region),
    }
    if not all(region_observations.values()):
        raise ValueError(
            "The configured region must contain at least one usable gravity and magnetic observation."
        )
    geology = {"mode": "gmm_only"}
    if files.get("unit_defs_file") and files.get("unit_groups_file"):
        geology.update(mode="csv_manual", unit_defs_csv=files["unit_defs_file"],
                       unit_groups_csv=files["unit_groups_file"])
    if files.get("reference_file"):
        geology["context_path"] = files["reference_file"]
    return {
        **({"data_import": settings} if settings else {}),
        "project": {
            "name": inspection["project"], "input_dir": inspection["folder"],
            "input_files": dict(files),
            "region_source": (
                "user_override" if region_overridden
                else inspection.get("detected", {}).get("region_source", "core_mesh_file")
            ),
            "region_observations": region_observations,
        },
        "region": {key: values[key] for key in ("min_e", "max_e", "min_n", "max_n")},
        "data": {
            "gravity_column": gravity_column,
            "magnetic_column": magnetic_column,
            "gravity_component": gravity_component,
            "std_grv": optional["std_grv"], "std_mag": optional["std_mag"],
            "flight_height_ft": optional["flight_height_ft"],
        },
        "inversion": {
            "field_strength": values["field_strength"],
            "inclination": values["inclination"], "declination": values["declination"],
            "cross_gradient_lambda": optional["cross_gradient_lambda"],
            "beta_cooling": optional["beta_cooling"],
            "optimization": {"maxGNCG": optional["max_iterations"]},
        },
        "geology": geology,
        "run": {"execution_mode": "full", "run_inversion": True,
                "run_geology_model": True, "make_plots": True},
    }
