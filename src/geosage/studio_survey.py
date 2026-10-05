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


def _match_files(folder: Path) -> tuple[dict[str, str], str]:
    files = [p for p in folder.iterdir() if p.is_file()]
    matches: dict[str, str] = {}
    prefixes: list[str] = []
    for role, suffix in ROLE_SUFFIXES.items():
        candidates = [p for p in files if p.name.lower().endswith(suffix)]
        if len(candidates) > 1:
            raise ValueError(f"More than one candidate was found for {role}.")
        if candidates:
            path = candidates[0].resolve()
            matches[role] = str(path)
            prefixes.append(path.name[: -len(suffix)])
    required_prefixes = {
        Path(matches[role]).name[: -len(ROLE_SUFFIXES[role])].casefold()
        for role in REQUIRED_ROLES if role in matches
    }
    if len(required_prefixes) > 1:
        raise ValueError("The required survey files do not share one project prefix.")
    return matches, (prefixes[0] if required_prefixes else folder.name)


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
        count = 0
        for row in reader:
            try:
                e, n = float(row[east]), float(row[north])
            except (TypeError, ValueError, KeyError):
                continue
            if not (math.isfinite(e) and math.isfinite(n)):
                continue
            min_e, max_e = min(min_e, e), max(max_e, e)
            min_n, max_n = min(min_n, n), max(max_n, n)
            count += 1
    if not count:
        raise ValueError(f"{path.name} contains no finite Easting/Northing rows.")
    return {
        "rows": count, "columns": fields,
        "extent": {"min_e": min_e, "max_e": max_e, "min_n": min_n, "max_n": max_n},
    }


def inspect_survey_folder(path: str | Path) -> dict:
    """Identify files and evidence extents without guessing physical parameters."""
    if not str(path).strip():
        raise ValueError("Provide the raw survey folder path.")
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Survey folder does not exist: {folder}")
    files, project = _match_files(folder)
    missing = [role for role in REQUIRED_ROLES if role not in files]
    if missing:
        raise ValueError("Missing required survey files: " + ", ".join(missing))
    gravity = _csv_extent(Path(files["gravity_file"]))
    magnetic = _csv_extent(Path(files["magnetic_file"]))
    gravity_candidates = [
        name for name in gravity["columns"]
        if name.upper() in {"ISO", "CBA", "SBA", "FAA", "OG"}
    ]
    gravity_column = "ISO" if "ISO" in gravity_candidates else (
        gravity_candidates[0] if len(gravity_candidates) == 1 else None
    )
    return {
        "folder": str(folder), "project": project, "files": files,
        "gravity": gravity, "magnetic": magnetic,
        "detected": {"gravity_column": gravity_column},
        "missing_parameters": [
            "min_e", "max_e", "min_n", "max_n",
            "field_strength", "inclination", "declination",
        ],
        "method_defaults": {
            "gravity_component": "gz", "std_grv": 0.25, "std_mag": 10.0,
            "flight_height_ft": 1000.0, "max_iterations": 50,
        },
    }


def build_configuration(inspection: dict, parameters: dict) -> dict:
    """Build a full-run configuration from explicit user parameters."""
    required = ("min_e", "max_e", "min_n", "max_n", "field_strength", "inclination", "declination")
    missing = [key for key in required if parameters.get(key) in (None, "")]
    if missing:
        raise ValueError("Still needed from the user: " + ", ".join(missing))
    try:
        values = {key: float(parameters[key]) for key in required}
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
        optional["max_iterations"] = int(optional["max_iterations"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Method settings must be numeric.") from exc
    gravity_column = str(parameters.get("gravity_column") or inspection["detected"].get("gravity_column") or "").strip()
    if not gravity_column:
        raise ValueError("Choose the gravity data column.")
    if gravity_column not in inspection["gravity"]["columns"]:
        raise ValueError(f"Gravity column '{gravity_column}' is not in the gravity CSV.")
    files = inspection["files"]
    geology = {"mode": "gmm_only"}
    if files.get("unit_defs_file") and files.get("unit_groups_file"):
        geology.update(mode="csv_manual", unit_defs_csv=files["unit_defs_file"],
                       unit_groups_csv=files["unit_groups_file"])
    if files.get("reference_file"):
        geology["context_path"] = files["reference_file"]
    return {
        "project": {"name": inspection["project"], "input_dir": inspection["folder"]},
        "region": {key: values[key] for key in ("min_e", "max_e", "min_n", "max_n")},
        "data": {
            "gravity_column": gravity_column,
            "gravity_component": gravity_component,
            "std_grv": optional["std_grv"], "std_mag": optional["std_mag"],
            "flight_height_ft": optional["flight_height_ft"],
        },
        "inversion": {
            "field_strength": values["field_strength"],
            "inclination": values["inclination"], "declination": values["declination"],
            "optimization": {"maxGNCG": optional["max_iterations"]},
        },
        "geology": geology,
        "run": {"execution_mode": "full", "run_inversion": True,
                "run_geology_model": True, "make_plots": True},
    }
