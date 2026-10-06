"""Deterministic survey discovery for conversational Studio setup."""

from __future__ import annotations

import csv
import math
from pathlib import Path

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
    "ISO", "CBA", "SBA", "FAA", "OG", "GZZ", "gravity", "gravity_anomaly",
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


def _preferred_named(paths: list[Path], words: tuple[str, ...]) -> list[Path]:
    return [path for path in paths if any(word in path.stem.casefold() for word in words)]


def _match_files(
    folder: Path, role_overrides: dict[str, str | Path] | None = None
) -> tuple[dict[str, str], str]:
    """Resolve conventional names first, then discover unambiguous semantic roles."""
    files = [p for p in folder.iterdir() if p.is_file()]
    matches: dict[str, str] = {}
    prefixes: list[str] = []
    for role, value in (role_overrides or {}).items():
        if role not in ROLE_SUFFIXES or not value:
            continue
        candidate = Path(value).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f"Explicit {role} does not exist: {candidate}")
        matches[role] = str(candidate)
    for role, suffix in ROLE_SUFFIXES.items():
        if role in matches:
            continue
        candidates = [p for p in files if p.name.lower().endswith(suffix)]
        if len(candidates) > 1:
            raise ValueError(f"More than one candidate was found for {role}.")
        if candidates:
            path = candidates[0].resolve()
            matches[role] = str(path)
            prefixes.append(path.name[: -len(suffix)])
    required_prefixes = {
        Path(matches[role]).name[: -len(ROLE_SUFFIXES[role])].casefold()
        for role in REQUIRED_ROLES
        if role in matches and Path(matches[role]).name.lower().endswith(ROLE_SUFFIXES[role])
    }

    used = {Path(value).resolve() for value in matches.values()}
    if "topography_file" not in matches:
        candidates = [p for p in files if p.suffix.casefold() in {".tif", ".tiff"} and p.resolve() not in used]
        preferred = _preferred_named(candidates, ("topo", "terrain", "surface", "dem"))
        selected = preferred if len(preferred) == 1 else candidates
        if len(selected) == 1:
            matches["topography_file"] = str(selected[0].resolve())
            used.add(selected[0].resolve())

    mesh_roles = [role for role in ("mesh_file", "core_mesh_file") if role not in matches]
    mesh_candidates = [p for p in files if p.suffix.casefold() == ".msh" and p.resolve() not in used]
    if mesh_roles:
        core_named = _preferred_named(mesh_candidates, ("core", "inner", "roi", "active"))
        if "core_mesh_file" in mesh_roles and len(core_named) == 1:
            matches["core_mesh_file"] = str(core_named[0].resolve())
            used.add(core_named[0].resolve())
            mesh_candidates.remove(core_named[0])
            mesh_roles.remove("core_mesh_file")
        full_named = _preferred_named(mesh_candidates, ("full", "domain", "global"))
        if "mesh_file" in mesh_roles and len(full_named) == 1:
            matches["mesh_file"] = str(full_named[0].resolve())
            used.add(full_named[0].resolve())
            mesh_candidates.remove(full_named[0])
            mesh_roles.remove("mesh_file")
        if len(mesh_roles) == 2 and len(mesh_candidates) == 2:
            summaries = sorted(
                ((_ubc_mesh_summary(path)["cells"], path) for path in mesh_candidates),
                key=lambda item: item[0], reverse=True,
            )
            if summaries[0][0] != summaries[1][0]:
                matches["mesh_file"] = str(summaries[0][1].resolve())
                matches["core_mesh_file"] = str(summaries[1][1].resolve())
                mesh_roles.clear()
        if len(mesh_roles) == 1 and len(mesh_candidates) == 1:
            matches[mesh_roles[0]] = str(mesh_candidates[0].resolve())

    survey_roles = [role for role in ("gravity_file", "magnetic_file") if role not in matches]
    survey_candidates: list[tuple[Path, dict]] = []
    for candidate in files:
        if candidate.suffix.casefold() != ".csv" or candidate.resolve() in used:
            continue
        try:
            survey_candidates.append((candidate, _csv_extent(candidate)))
        except ValueError:
            continue
    if survey_roles:
        for role, words in (
            ("gravity_file", ("gravity", "grav")),
            ("magnetic_file", ("magnetic", "mag")),
        ):
            if role not in survey_roles:
                continue
            named = [(path, summary) for path, summary in survey_candidates
                     if any(word in path.stem.casefold() for word in words)]
            if len(named) == 1:
                matches[role] = str(named[0][0].resolve())
                survey_candidates.remove(named[0])
                survey_roles.remove(role)
        if len(survey_roles) == 2 and len(survey_candidates) == 2:
            gravity_like = [item for item in survey_candidates if _detect_measurement_column(
                item[1], GRAVITY_COLUMN_ALIASES
            )]
            magnetic_like = [item for item in survey_candidates if _detect_measurement_column(
                item[1], MAGNETIC_COLUMN_ALIASES
            )]
            if len(gravity_like) == 1 and len(magnetic_like) == 1 and gravity_like[0] != magnetic_like[0]:
                matches["gravity_file"] = str(gravity_like[0][0].resolve())
                matches["magnetic_file"] = str(magnetic_like[0][0].resolve())
                survey_roles.clear()
            else:
                with_height = [item for item in survey_candidates
                               if "height" in {c.casefold() for c in item[1]["columns"]}]
                without_height = [item for item in survey_candidates if item not in with_height]
                if len(with_height) == 1 and len(without_height) == 1:
                    matches["gravity_file"] = str(with_height[0][0].resolve())
                    matches["magnetic_file"] = str(without_height[0][0].resolve())
                    survey_roles.clear()
        if len(survey_roles) == 1 and len(survey_candidates) == 1:
            matches[survey_roles[0]] = str(survey_candidates[0][0].resolve())

    project = prefixes[0] if len(required_prefixes) == 1 else folder.name
    return matches, project


def _csv_extent(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [str(name or "").strip() for name in (reader.fieldnames or [])]
        lookup = {name.casefold(): name for name in fields}
        if "easting" not in lookup or "northing" not in lookup:
            raise ValueError(f"{path.name} needs Easting and Northing columns.")
        east, north = lookup["easting"], lookup["northing"]
        min_e = min_n = math.inf
        max_e = max_n = -math.inf
        longitude = lookup.get("longitude")
        latitude = lookup.get("latitude")
        min_lon = min_lat = math.inf
        max_lon = max_lat = -math.inf
        geographic_count = 0
        count = 0
        data_rows = 0
        numeric = {name: True for name in fields if name}
        for row in reader:
            data_rows += 1
            for name in numeric:
                try:
                    numeric[name] = numeric[name] and math.isfinite(float(row[name]))
                except (TypeError, ValueError, KeyError):
                    numeric[name] = False
            try:
                e, n = float(row[east]), float(row[north])
            except (TypeError, ValueError, KeyError):
                continue
            if not (math.isfinite(e) and math.isfinite(n)):
                continue
            min_e, max_e = min(min_e, e), max(max_e, e)
            min_n, max_n = min(min_n, n), max(max_n, n)
            count += 1
            if longitude and latitude:
                try:
                    lon, lat = float(row[longitude]), float(row[latitude])
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(lon) and math.isfinite(lat):
                    min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
                    min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
                    geographic_count += 1
    if not count:
        raise ValueError(f"{path.name} contains no finite Easting/Northing rows.")
    result = {
        "rows": count, "columns": fields,
        "numeric_columns": [name for name in fields if data_rows and numeric.get(name, False)],
        "extent": {"min_e": min_e, "max_e": max_e, "min_n": min_n, "max_n": max_n},
    }
    if geographic_count:
        result["geographic_extent"] = {
            "min_lon": min_lon, "max_lon": max_lon,
            "min_lat": min_lat, "max_lat": max_lat,
        }
    return result


def _detect_measurement_column(summary: dict, aliases: tuple[str, ...]) -> str | None:
    lookup = {name.casefold(): name for name in summary.get("columns", [])}
    for alias in aliases:
        if alias.casefold() in lookup:
            return lookup[alias.casefold()]
    candidates = [name for name in summary.get("numeric_columns", [])
                  if name.casefold() not in NON_MEASUREMENT_COLUMNS
                  and not name.casefold().startswith("unnamed")]
    return candidates[0] if len(candidates) == 1 else None


def _resolve_column(summary: dict, requested: str, label: str) -> str:
    lookup = {name.casefold(): name for name in summary.get("columns", [])}
    resolved = lookup.get(str(requested).strip().casefold())
    if not resolved:
        raise ValueError(f"{label} column '{requested}' is not in the selected CSV.")
    return resolved


def _validate_numeric_columns(path: Path, required: tuple[str, ...]) -> None:
    """Mirror the numerical runner's required CSV columns during setup."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}.")
        for line, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in required]
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"{path.name} has a non-numeric required value on row {line}."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path.name} has a non-finite required value on row {line}."
                )


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
    if len(shape) != 3 or len(origin) != 3 or any(value <= 0 for value in shape):
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


def _topography_summary(path: Path, geographic_extent: dict | None) -> dict:
    """Read GeoTIFF metadata and project its envelope when survey geography permits."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as exc:  # pragma: no cover - GeoSAGE installs rasterio
        raise ValueError("Reading GeoTIFF metadata requires rasterio.") from exc
    try:
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            result = {
                "path": str(path),
                "shape": {"rows": dataset.height, "columns": dataset.width},
                "crs": dataset.crs.to_string() if dataset.crs else None,
                "is_geographic": bool(dataset.crs and dataset.crs.is_geographic),
                "native_bounds": {
                    "left": float(bounds.left), "bottom": float(bounds.bottom),
                    "right": float(bounds.right), "top": float(bounds.top),
                },
                "nodata": None if dataset.nodata is None else float(dataset.nodata),
            }
            if dataset.crs and dataset.crs.is_projected:
                result["projected_crs"] = dataset.crs.to_string()
                result["projected_bounds"] = {
                    "min_e": float(bounds.left), "max_e": float(bounds.right),
                    "min_n": float(bounds.bottom), "max_n": float(bounds.top),
                }
            elif dataset.crs and geographic_extent:
                mean_lon = (geographic_extent["min_lon"] + geographic_extent["max_lon"]) / 2
                mean_lat = (geographic_extent["min_lat"] + geographic_extent["max_lat"]) / 2
                zone = int(math.floor((mean_lon + 180) / 6) + 1)
                epsg = (32600 if mean_lat >= 0 else 32700) + zone
                left, bottom, right, top = transform_bounds(
                    dataset.crs, f"EPSG:{epsg}", *bounds, densify_pts=21
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


def _rows_in_region(path: Path, region: dict) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        lookup = {str(name or "").strip().casefold(): name for name in (reader.fieldnames or [])}
        east, north = lookup["easting"], lookup["northing"]
        count = 0
        for row in reader:
            try:
                e, n = float(row[east]), float(row[north])
            except (TypeError, ValueError, KeyError):
                continue
            # Match the numerical kernel's strict study-area mask exactly.
            if (region["min_e"] < e < region["max_e"]
                    and region["min_n"] < n < region["max_n"]):
                count += 1
        return count


def inspect_survey_folder(
    path: str | Path, role_overrides: dict[str, str | Path] | None = None
) -> dict:
    """Identify files and evidence extents without guessing physical parameters."""
    if not str(path).strip():
        raise ValueError("Provide the raw survey folder path.")
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Survey folder does not exist: {folder}")
    files, project = _match_files(folder, role_overrides)
    missing = [role for role in REQUIRED_ROLES if role not in files]
    if missing:
        raise ValueError(
            "Could not identify required survey roles: " + ", ".join(missing)
            + ". Rename them descriptively or provide the file paths explicitly."
        )
    gravity = _csv_extent(Path(files["gravity_file"]))
    magnetic = _csv_extent(Path(files["magnetic_file"]))
    _validate_numeric_columns(
        Path(files["gravity_file"]),
        ("Easting", "Northing", "Height"),
    )
    _validate_numeric_columns(Path(files["magnetic_file"]), ("Easting", "Northing"))
    mesh = _ubc_mesh_summary(Path(files["mesh_file"]))
    core_mesh = _ubc_mesh_summary(Path(files["core_mesh_file"]))
    topography = _topography_summary(
        Path(files["topography_file"]),
        gravity.get("geographic_extent") or magnetic.get("geographic_extent"),
    )
    if not topography.get("crs"):
        raise ValueError("The topography GeoTIFF has no coordinate reference system (CRS).")
    if topography.get("is_geographic") and not topography.get("projected_bounds"):
        raise ValueError(
            "The topography uses geographic coordinates, so the gravity or magnetic CSV needs "
            "finite Longitude and Latitude columns to determine the projected survey CRS."
        )
    if topography.get("is_geographic"):
        _validate_numeric_columns(
            Path(files["gravity_file"]),
            ("Longitude", "Latitude"),
        )
    region = {
        key: core_mesh["bounds"][key]
        for key in ("min_e", "max_e", "min_n", "max_n")
    }
    topo_projected = topography.get("projected_bounds")
    spatial_checks = {
        "core_mesh_inside_full_mesh": _contains(mesh["bounds"], core_mesh["bounds"], vertical=True),
        "topography_covers_full_mesh": (
            _contains(topo_projected, mesh["bounds"]) if topo_projected else None
        ),
        "gravity_rows_in_core": _rows_in_region(Path(files["gravity_file"]), region),
        "magnetic_rows_in_core": _rows_in_region(Path(files["magnetic_file"]), region),
    }
    if not spatial_checks["core_mesh_inside_full_mesh"]:
        raise ValueError("The core mesh is not contained by the full inversion mesh.")
    if spatial_checks["gravity_rows_in_core"] == 0 or spatial_checks["magnetic_rows_in_core"] == 0:
        raise ValueError("The core mesh contains no usable gravity or magnetic observations.")
    if spatial_checks["topography_covers_full_mesh"] is False:
        raise ValueError("The projected topography does not cover the full inversion mesh.")
    gravity_column = _detect_measurement_column(gravity, GRAVITY_COLUMN_ALIASES)
    magnetic_column = _detect_measurement_column(magnetic, MAGNETIC_COLUMN_ALIASES)
    missing_parameters = ["field_strength", "inclination", "declination"]
    if not gravity_column:
        missing_parameters.append("gravity_column")
    if not magnetic_column:
        missing_parameters.append("magnetic_column")
    horizontal_cell_size = min(
        core_mesh["cell_widths"]["x"]["min"],
        core_mesh["cell_widths"]["y"]["min"],
    )
    return {
        "folder": str(folder), "project": project, "files": files,
        "gravity": gravity, "magnetic": magnetic,
        "mesh": mesh, "core_mesh": core_mesh, "topography": topography,
        "spatial_checks": spatial_checks,
        "detected": {
            "gravity_column": gravity_column,
            "magnetic_column": magnetic_column,
            "region": region,
            "region_source": "core_mesh_file",
        },
        "missing_parameters": missing_parameters,
        "method_defaults": {
            "gravity_component": "gz", "std_grv": 0.25, "std_mag": 10.0,
            "flight_height_ft": 1000.0, "max_iterations": 50,
            # Cross-gradient contains two spatial derivatives; scaling by dh**4
            # removes the mesh-length dependence used in the SimPEG joint example.
            "cross_gradient_lambda": horizontal_cell_size**4,
            # PairedBetaSchedule divides beta by this factor each update.
            "beta_cooling": 2.0,
        },
    }


def build_configuration(inspection: dict, parameters: dict) -> dict:
    """Build a full-run configuration from explicit user parameters."""
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
    gravity_component = str(
        parameters.get("gravity_component") or defaults.pop("gravity_component", "gz")
    ).strip()
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
    magnetic_column = str(
        parameters.get("magnetic_column")
        or inspection["detected"].get("magnetic_column") or ""
    ).strip()
    if not magnetic_column:
        raise ValueError("Choose the magnetic data column.")
    magnetic_column = _resolve_column(inspection["magnetic"], magnetic_column, "Magnetic")
    files = inspection["files"]
    _validate_numeric_columns(
        Path(files["gravity_file"]),
        ("Easting", "Northing", "Height", gravity_column),
    )
    _validate_numeric_columns(
        Path(files["magnetic_file"]),
        ("Easting", "Northing", magnetic_column),
    )
    if inspection.get("topography", {}).get("is_geographic"):
        _validate_numeric_columns(
            Path(files["gravity_file"]),
            ("Longitude", "Latitude"),
        )
    mesh_bounds = inspection.get("mesh", {}).get("bounds")
    configured_region = {key: values[key] for key in ("min_e", "max_e", "min_n", "max_n")}
    if mesh_bounds and not _contains(mesh_bounds, configured_region):
        raise ValueError("The configured region extends outside the full inversion mesh.")
    region_observations = {
        "gravity": _rows_in_region(Path(files["gravity_file"]), configured_region),
        "magnetic": _rows_in_region(Path(files["magnetic_file"]), configured_region),
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
