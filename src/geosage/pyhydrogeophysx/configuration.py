"""Translate Studio file roles to an explicit, reproducible GeoSAGE run."""

from copy import deepcopy
import json
from pathlib import Path


FILE_ROLES = {
    "gravity_file": "gravity_data.csv",
    "magnetic_file": "magnetic_data.csv",
    "topography_file": "topo.tif",
    "mesh_file": "mesh.msh",
    "core_mesh_file": "mesh_core.msh",
}


def _reject_credentials(value):
    """Configuration is persisted; provider credentials belong only in settings."""
    if isinstance(value, dict):
        for key, child in value.items():
            if any(
                word in str(key).lower()
                for word in ("api_key", "apikey", "password", "secret", "token")
            ):
                raise ValueError(
                    "Keep credentials out of the JSON configuration; use Studio AI settings."
                )
            _reject_credentials(child)
    elif isinstance(value, list):
        for child in value:
            _reject_credentials(child)


def configure(payload):
    from geosage.runner import load_config

    inputs = dict(payload.get("inputs") or {})
    unknown = (
        set(inputs)
        - set(FILE_ROLES)
        - {
            "config_file",
            "source_inversion_dir",
            "input_dir",
            "unit_defs_file",
            "unit_groups_file",
            "reference_file",
        }
    )
    if unknown:
        raise ValueError(f"Unsupported GeoSAGE input roles: {', '.join(sorted(unknown))}")
    for key, value in inputs.items():
        if value and (not isinstance(value, str) or not Path(value).expanduser().exists()):
            raise ValueError(f"{key} must name one existing local file or folder.")
    inputs = {k: str(Path(v).expanduser().resolve()) for k, v in inputs.items() if v}
    config_file = inputs.get("config_file")
    raw = deepcopy(payload.get("config") or {})
    if config_file:
        if raw:
            raise ValueError("Supply either config_file or config, not both.")
        raw = json.loads(Path(config_file).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("GeoSAGE configuration must be a JSON object.")
    _reject_credentials(raw)
    source = inputs.get("source_inversion_dir") or raw.get("project", {}).get(
        "source_inversion_dir"
    )
    if not raw and not source:
        raise ValueError(
            "Add an existing GeoSAGE inversion folder, or a GeoSAGE JSON configuration "
            "with region and survey settings. An exploration goal alone cannot define them."
        )
    if not raw:
        raw = {
            "project": {"name": "GeoSAGE", "input_dir": source, "source_inversion_dir": source},
            "run": {"execution_mode": "interpret_existing", "run_inversion": False},
            "geology": {
                "mode": "reuse_existing_geology"
                if all(
                    (Path(source) / "geology_models" / name).is_file()
                    for name in ("geo_id_3d.npy", "unit_id_3d.npy")
                )
                else "gmm_only"
            },
        }
    # Do not inherit the default Hannah target or private reference when a
    # new project supplies no priors. Explicit case-study settings still win.
    neutral = {
        "unit_defs_csv": None,
        "unit_groups_csv": None,
        "context_path": None,
        "target_unit_ids": [],
        "target_geo_ids": [],
        "target_name": "",
    }
    raw["geology"] = {**neutral, **raw.get("geology", {})}
    raw.setdefault("run", {}).setdefault("make_plots", False)
    if "project" not in raw:
        raise ValueError("GeoSAGE configuration needs a project section.")
    if config_file:
        # Resolve config-relative workspace paths without writing a modified file.
        from geosage.paths import resolve_config_paths

        raw = resolve_config_paths(raw, config_path=Path(config_file))
    cfg = load_config(raw)
    project = cfg["project"]
    if inputs.get("input_dir"):
        project["input_dir"] = inputs["input_dir"]
    if inputs.get("source_inversion_dir"):
        project["source_inversion_dir"] = inputs["source_inversion_dir"]
        cfg["run"].update(execution_mode="interpret_existing", run_inversion=False)
    source = project.get("source_inversion_dir")
    mode = cfg["run"].get("execution_mode")
    if not mode:
        mode = "interpret_existing" if not cfg["run"].get("run_inversion", True) else "full"
    cfg["run"]["execution_mode"] = mode
    if mode not in {"full", "interpret_existing"}:
        raise ValueError(f"Unsupported execution mode: {mode}")
    if mode == "full":
        if not raw["project"].get("name") or not (
            raw["project"].get("input_dir")
            or inputs.get("input_dir")
            or all(k in inputs for k in FILE_ROLES)
        ):
            raise ValueError(
                "A full inversion needs an explicit project name and input folder or all five input roles."
            )
        region = raw.get("region", {})
        if not all(k in region for k in ("min_e", "max_e", "min_n", "max_n")):
            raise ValueError(
                "A full inversion needs explicit min_e/max_e/min_n/max_n in the configuration."
            )
    else:
        source = source or project.get("output_dir")
        if not source or not Path(source).is_dir():
            raise ValueError("Select an existing inversion folder to interpret.")
        project["source_inversion_dir"] = str(Path(source).resolve())
    output = Path(payload["output_dir"]).expanduser().resolve()
    protected = [Path(project["input_dir"]).resolve()]
    if source:
        protected.append(Path(source).resolve())
    if any(output == p or p in output.parents for p in protected):
        raise ValueError("Studio output must be outside the input and source inversion folders.")
    project["output_dir"] = str(output / "models")
    project["interpretation_output_dir"] = str(output / "interpretation")
    if not project["name"] or Path(project["name"]).name != project["name"]:
        raise ValueError("Project name must be a simple name, without path separators.")
    cfg["run"].update(overwrite=False, run_geology_model=True, run_inversion=mode == "full")
    for role, key in (
        ("reference_file", "context_path"),
        ("unit_defs_file", "unit_defs_csv"),
        ("unit_groups_file", "unit_groups_csv"),
    ):
        if role in inputs:
            cfg["geology"][key] = inputs[role]
    # Absence of explicit priors uses unsupervised clusters, not invented lithology.
    if cfg["geology"].get("mode") == "csv_manual" and not cfg["geology"].get("unit_defs_csv"):
        cfg["geology"]["mode"] = "gmm_only"
    cfg["studio_inputs"] = inputs
    cfg["user_request"] = str(payload.get("request") or "").strip()
    return cfg
