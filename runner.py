
from __future__ import annotations

import json
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Union, Dict, Any, Optional

from gravity_mag_joint_inversion import run_joint_inversion
from geo_modeling_workflow import build_geology_model
from existing_results import (
    build_source_manifest,
    load_existing_geology_result,
    load_existing_inversion_result,
)


# ==========================
# 1. Default configuration (can be overridden by JSON).
# ==========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "Hannah",
        "input_dir": "Hannah",         
        "output_dir": "Hannah_Inversion",
        "source_inversion_dir": None,
        "interpretation_output_dir": None,
    },
    "region": {
        "min_e": 510000.0,
        "max_e": 535000.0,
        "min_n": 4290000.0,
        "max_n": 4320000.0,
    },
    "data": {
        "gravity_column": "ISO",
        "gravity_component": "gz",  # Options: gx, gy, gz, gxx, gxy, gxz, gyy, gyz, gzz
        "std_grv": 0.25,
        "std_mag": 10.0,
        "std_grv_relative": False,
        "std_mag_relative": False,
        "flight_height_ft": 1000.0,
    },
    "inversion": {
        "inclination": 62.0,
        "declination": 15.0,
        "field_strength": 50686.0,
        "grv_alpha": [20.0, 1.0, 1.0, 1.0],
        "mag_alpha": [10.0, 1.0, 1.0, 1.0],
        # Overall regularization coefficients (gravity, magnetic). Keep for compatibility with previous reg_beta naming.
        "reg_coefficient": [1.0, 1.0],
        "reg_grv_norm": [1.0, 2.0, 2.0, 2.0],
        "reg_mag_norm": [1.0, 2.0, 2.0, 2.0],
        "cross_gradient_lambda": 1000.0,
        "beta0_ratio": 10.0,
        "beta_cooling": 0.8,
        "grv_bounds": [-10.0, 10.0],
        "mag_bounds": [-10.0, 10.0],
        "optimization": {
            "maxGNCG": 50,
            "maxLS": 10,
            "maxCG": 1000,
            "tolCG": 1e-2,
            "tolX": 1e-2
        },
        "irls": {
            "maxIRLSiter": 100,
            "IRLSstart": 5e4,
            "IRLS_mindelta": 1e-2,
            "IRLSbeta_tol": 1e-2
        }
    },
    "geology": {
        "mode": "csv_manual",  
        "unit_defs_csv": "Hannah/Hannah_unit_defs.csv",
        "unit_groups_csv": "Hannah/Hannah_unit_groups.csv",
        "context_path": "Hannah/Hannah_geology_context.pdf",  # Supports .txt and .pdf
        "target_unit_ids": [],
        "target_geo_ids": [10, 6, 5, 7],
        "target_name": "Serpentinite hydrogen play along Collayomi fault",
        "min_voxels": 10,
        "fill_iterations": 3
    },
    "run": {
        "run_inversion": True,
        "run_geology_model": True,
        "make_plots": True,
        "execution_mode": None,
        "skip_configuration_agents": False,
        "reuse_existing_geology": False,
        "overwrite": False,
        "review_enabled": False,
        "max_review_rounds": 1,
    }
}


# ==========================
# 2. Utility function: deep-merge configurations.
# ==========================

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively override values in base with those in updates; recurse only when both sides are dicts.
    The goal is to allow the JSON to specify only the changes, while using defaults for everything else.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Supports three forms:
    - dict: use directly and override DEFAULT_CONFIG
    - str/Path: treat as a JSON file path, read it, then override DEFAULT_CONFIG
    """
    cfg = deepcopy(DEFAULT_CONFIG)

    if isinstance(config, (str, Path)):
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        deep_update(cfg, user_cfg)
    elif isinstance(config, dict):
        deep_update(cfg, config)
    else:
        raise TypeError("config must be a dict or a path to JSON.")

    return cfg


def resolve_execution_mode(cfg: Dict[str, Any]) -> str:
    """Resolve the new execution mode while preserving old configurations."""

    run_cfg = cfg.setdefault("run", {})
    explicit = run_cfg.get("execution_mode")
    if explicit is None or str(explicit).strip() == "":
        mode = "full" if run_cfg.get("run_inversion", True) else "interpret_existing"
    else:
        mode = str(explicit).strip().lower()
    if mode not in {"full", "interpret_existing"}:
        raise ValueError(
            "run.execution_mode must be 'full' or 'interpret_existing'; "
            f"got {explicit!r}"
        )
    run_cfg["execution_mode"] = mode
    return mode


def _archived_parameters_to_config(
    cfg: Dict[str, Any],
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge archived physical settings into an effective reporting config."""

    effective = deepcopy(cfg)
    if not parameters:
        return effective

    data = effective["data"]
    inversion = effective["inversion"]
    region = effective["region"]

    if isinstance(parameters.get("select_region"), (list, tuple)) and len(parameters["select_region"]) == 4:
        region.update(
            dict(
                zip(
                    ("min_e", "max_e", "min_n", "max_n"),
                    [float(v) for v in parameters["select_region"]],
                    strict=True,
                )
            )
        )
    scalar_data = {
        "gravity_component": "gravity_component",
        "target_gravity_column": "gravity_column",
        "std_grv": "std_grv",
        "std_mag": "std_mag",
        "std_grv_relative": "std_grv_relative",
        "std_mag_relative": "std_mag_relative",
        "flight_height_ft": "flight_height_ft",
    }
    for source_key, target_key in scalar_data.items():
        if source_key in parameters:
            data[target_key] = parameters[source_key]

    inversion_keys = (
        "inclination",
        "declination",
        "field_strength",
        "cross_gradient_lambda",
        "beta0_ratio",
        "beta_cooling",
        "reg_coefficient",
        "reg_grv_norm",
        "reg_mag_norm",
    )
    for key in inversion_keys:
        if key in parameters:
            inversion[key] = parameters[key]
    if "weight_grv" in parameters:
        inversion["grv_alpha"] = parameters["weight_grv"]
    if "weight_mag" in parameters:
        inversion["mag_alpha"] = parameters["weight_mag"]
    bounds = parameters.get("inv_bound")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        inversion["grv_bounds"] = [bounds[0], bounds[2]]
        inversion["mag_bounds"] = [bounds[1], bounds[3]]
    if isinstance(parameters.get("optimization"), dict):
        deep_update(inversion.setdefault("optimization", {}), parameters["optimization"])
    if isinstance(parameters.get("irls"), dict):
        deep_update(inversion.setdefault("irls", {}), parameters["irls"])
    effective.setdefault("provenance", {})["archived_inversion_parameters"] = parameters
    return effective


def _safe_interpretation_output_dir(
    source_dir: Path,
    requested_dir: str | Path | None,
    overwrite: bool,
) -> Path:
    """Choose a write directory that cannot accidentally modify the source run."""

    source = source_dir.resolve()
    requested = Path(requested_dir).expanduser().resolve() if requested_dir else None
    requested_inside_source = requested is not None and (
        requested == source or source in requested.parents
    )
    if requested is not None and not requested_inside_source:
        if requested.exists() and any(requested.iterdir()) and not overwrite:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            requested = requested.parent / f"{requested.name}_{stamp}"
        requested.mkdir(parents=True, exist_ok=True)
        return requested

    if requested_inside_source:
        warnings.warn(
            "interpretation_output_dir is inside the source inversion. "
            "The source is read-only, so a sibling interpretation directory will be used.",
            RuntimeWarning,
            stacklevel=2,
        )
    base = source.parent / f"{source.name}_interpretations"
    if requested_inside_source and requested is not None and requested != source:
        base = base / requested.name
    destination = base / "latest"
    if destination.exists() and not overwrite:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = base / f"run_{stamp}"
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


# ==========================
# 3. Main entry point: run_workflow
# ==========================

def run_workflow(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Run a full inversion or a read-only interpretation of archived results."""

    cfg = load_config(config)
    execution_mode = resolve_execution_mode(cfg)
    project_cfg = cfg["project"]
    run_cfg = cfg["run"]
    inversion_result: Optional[Dict[str, Any]] = None
    geology_result: Optional[Dict[str, Any]] = None
    source_manifest: Dict[str, Any] = {}
    source_dir: Optional[Path] = None
    interpretation_dir: Optional[Path] = None

    def _fix_length(vec: Any, n: int, default_tail: float = 2.0) -> tuple[Any, ...]:
        values = list(vec)
        if len(values) >= n:
            return tuple(values[:n])
        fill = values[-1] if values else default_tail
        values.extend([fill] * (n - len(values)))
        return tuple(values)

    if execution_mode == "interpret_existing":
        source_dir = Path(
            project_cfg.get("source_inversion_dir")
            or project_cfg.get("output_dir")
            or f"{project_cfg['name']}_Inversion"
        ).expanduser().resolve()
        inversion_result = load_existing_inversion_result(source_dir)
        cfg = _archived_parameters_to_config(cfg, inversion_result["inversion_parameters"])
        cfg["project"]["source_inversion_dir"] = str(source_dir)
        interpretation_dir = _safe_interpretation_output_dir(
            source_dir,
            cfg["project"].get("interpretation_output_dir"),
            bool(run_cfg.get("overwrite", False)),
        )
        cfg["project"]["interpretation_output_dir"] = str(interpretation_dir)
        source_manifest = inversion_result["source_manifest"]
        print(f"[interpret_existing] SOURCE_DIR (read-only) = {source_dir}")
        print(f"[interpret_existing] OUTPUT_DIR = {interpretation_dir}")
    else:
        if not run_cfg.get("run_inversion", True):
            raise ValueError(
                "execution_mode='full' requires run.run_inversion=true. "
                "Use execution_mode='interpret_existing' for archived results."
            )

        project_output = Path(project_cfg["output_dir"]).expanduser().resolve()
        project_cfg["output_dir"] = str(project_output)
        source_dir = project_output
        interpretation_dir = project_output
        region_cfg = cfg["region"]
        data_cfg = cfg["data"]
        inv_cfg = cfg["inversion"]
        select_region = [
            region_cfg["min_e"],
            region_cfg["max_e"],
            region_cfg["min_n"],
            region_cfg["max_n"],
        ]
        inversion_result = run_joint_inversion(
            project_name=project_cfg["name"],
            input_dir=project_cfg["input_dir"],
            output_dir=project_output,
            select_region=select_region,
            target_grv_data=data_cfg["gravity_column"],
            gravity_component=data_cfg.get("gravity_component", "gz"),
            std_grv=data_cfg["std_grv"],
            std_mag=data_cfg["std_mag"],
            std_grv_relative=data_cfg.get("std_grv_relative", False),
            std_mag_relative=data_cfg.get("std_mag_relative", False),
            flight_height_ft=data_cfg["flight_height_ft"],
            inclination=inv_cfg["inclination"],
            declination=inv_cfg["declination"],
            field_strength=inv_cfg["field_strength"],
            grv_alpha=_fix_length(inv_cfg["grv_alpha"], 4),
            mag_alpha=_fix_length(inv_cfg["mag_alpha"], 4),
            reg_coefficient=_fix_length(
                inv_cfg.get("reg_coefficient", inv_cfg.get("reg_beta", [1.0, 1.0])),
                2,
                default_tail=1.0,
            ),
            reg_grv_norm=_fix_length(inv_cfg["reg_grv_norm"], 4),
            reg_mag_norm=_fix_length(inv_cfg["reg_mag_norm"], 4),
            cross_gradient_lambda=inv_cfg["cross_gradient_lambda"],
            beta0_ratio=inv_cfg["beta0_ratio"],
            beta_cooling=inv_cfg["beta_cooling"],
            grv_bounds=tuple(inv_cfg["grv_bounds"]),
            mag_bounds=tuple(inv_cfg["mag_bounds"]),
            maxGNCG=inv_cfg["optimization"]["maxGNCG"],
            maxLS=inv_cfg["optimization"]["maxLS"],
            maxCG=inv_cfg["optimization"]["maxCG"],
            tolCG=inv_cfg["optimization"]["tolCG"],
            tolX=inv_cfg["optimization"]["tolX"],
            maxIRLSiter=inv_cfg["irls"]["maxIRLSiter"],
            IRLSstart=inv_cfg["irls"]["IRLSstart"],
            IRLS_mindelta=inv_cfg["irls"]["IRLS_mindelta"],
            IRLSbeta_tol=inv_cfg["irls"]["IRLSbeta_tol"],
            make_plots=run_cfg.get("make_plots", True),
        )
        source_manifest = build_source_manifest(source_dir)

    # ---------- Pseudo-geological interpretation ----------
    if run_cfg.get("run_geology_model", True):
        geo_cfg = cfg["geology"]
        reuse_geo = bool(run_cfg.get("reuse_existing_geology", False)) or (
            str(geo_cfg.get("mode", "")).lower() == "reuse_existing_geology"
        )
        if reuse_geo:
            if source_dir is None or inversion_result is None:
                raise RuntimeError("Cannot reuse geology without an inversion source.")
            geology_result = load_existing_geology_result(source_dir, inversion_result)
        else:
            if source_dir is None or interpretation_dir is None:
                raise RuntimeError("Workflow directories were not resolved.")
            geology_result = build_geology_model(
                project_name=project_cfg["name"],
                input_dir=project_cfg["input_dir"],
                inversion_dir=source_dir,
                output_dir=interpretation_dir,
                min_voxels=geo_cfg["min_voxels"],
                fill_iterations=geo_cfg["fill_iterations"],
                unit_defs_csv=geo_cfg.get("unit_defs_csv"),
                unit_groups_csv=geo_cfg.get("unit_groups_csv"),
                unit_id_npy=geo_cfg.get("unit_id_npy"),
                make_plots=run_cfg.get("make_plots", True),
            )
            if execution_mode == "interpret_existing":
                geology_result["source_manifest"] = source_manifest

    if source_dir is None or interpretation_dir is None:
        raise RuntimeError("Workflow directories were not resolved.")
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": execution_mode,
        "inversion_reused": execution_mode == "interpret_existing",
        "inversion_recomputed": execution_mode == "full",
        "source_inversion_dir": str(source_dir),
        "interpretation_output_dir": str(interpretation_dir),
        "source_directory_read_only": execution_mode == "interpret_existing",
        "source_manifest": source_manifest,
    }
    if execution_mode == "interpret_existing":
        _write_json(interpretation_dir / "effective_config.json", cfg)
        _write_json(interpretation_dir / "source_manifest.json", source_manifest)
        _write_json(interpretation_dir / "run_manifest.json", run_manifest)
    elif source_manifest:
        _write_json(interpretation_dir / "source_manifest.json", source_manifest)
        _write_json(interpretation_dir / "run_manifest.json", run_manifest)

    return {
        "config": cfg,
        "effective_config": cfg,
        "inversion_result": inversion_result,
        "geology_result": geology_result,
        "source_manifest": source_manifest,
        "run_manifest": run_manifest,
        "interpretation_output_dir": str(interpretation_dir),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run gravity+mag joint inversion + geo modelling from a JSON config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_example.json",
        help="Path to JSON configuration file.",
    )
    args = parser.parse_args()

    result = run_workflow(args.config)
    print("\nWorkflow finished.")
    if result["inversion_result"] is not None:
        print("  Inversion output root:", result["inversion_result"]["paths"]["output_root"])
    if result["geology_result"] is not None:
        print("  Geology model figures in:",
              result["geology_result"]["paths"]["geo_slices_dir"])
