
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Union, Dict, Any, Optional

from gravity_mag_joint_inversion import run_joint_inversion
from geo_modeling_workflow import build_geology_model


# ==========================
# 1. Default configuration (can be overridden by JSON).
# ==========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "Hannah",
        "input_dir": "Hannah",         
        "output_dir": "Hannah_Inversion"
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
        "target_geo_ids": [10, 6, 5, 7],
        "target_name": "Serpentinite hydrogen play along Collayomi fault",
        "min_voxels": 10,
        "fill_iterations": 3
    },
    "run": {
        "run_inversion": True,
        "run_geology_model": True,
        "make_plots": True
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


# ==========================
# 3. Main entry point: run_workflow
# ==========================

def run_workflow(config: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run the workflow with a configuration (or a JSON path), including:
    - joint gravity-magnetic inversion
    - pseudo-geological modeling

    Returns a dict containing two sub-results for downstream agents to reuse.
    """
    cfg = load_config(config)

    project_cfg = cfg["project"]
    region_cfg = cfg["region"]
    data_cfg = cfg["data"]
    inv_cfg = cfg["inversion"]
    geo_cfg = cfg["geology"]
    run_cfg = cfg["run"]

    select_region = [
        region_cfg["min_e"],
        region_cfg["max_e"],
        region_cfg["min_n"],
        region_cfg["max_n"],
    ]

    inversion_result: Optional[Dict[str, Any]] = None
    geology_result: Optional[Dict[str, Any]] = None

    def _fix_length(vec, n: int, default_tail: float = 2.0):
        lst = list(vec)
        if len(lst) >= n:
            return lst[:n]
        fill = lst[-1] if lst else default_tail
        lst.extend([fill] * (n - len(lst)))
        return lst

    # ---------- 3.1 Joint Inversion ----------
    if run_cfg.get("run_inversion", True):
        grv_alpha = tuple(_fix_length(inv_cfg["grv_alpha"], 4))
        mag_alpha = tuple(_fix_length(inv_cfg["mag_alpha"], 4))
        reg_grv_norm = tuple(_fix_length(inv_cfg["reg_grv_norm"], 4))
        reg_mag_norm = tuple(_fix_length(inv_cfg["reg_mag_norm"], 4))
        reg_coefficient = tuple(
            _fix_length(
                inv_cfg.get("reg_coefficient", inv_cfg.get("reg_beta", [1.0, 1.0])),
                2,
                default_tail=1.0,
            )
        )

        inversion_result = run_joint_inversion(
            project_name=project_cfg["name"],
            input_dir=project_cfg["input_dir"],
            output_dir=project_cfg["output_dir"],
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
            grv_alpha=grv_alpha,
            mag_alpha=mag_alpha,
            reg_coefficient=reg_coefficient,
            reg_grv_norm=reg_grv_norm,
            reg_mag_norm=reg_mag_norm,
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
            make_plots=run_cfg["make_plots"],
        )

    # ---------- 3.2 pseudo-geological ----------
    if run_cfg.get("run_geology_model", True):
        if inversion_result is not None:
            inversion_dir = inversion_result["paths"]["output_root"]
        else:
            inversion_dir = project_cfg["output_dir"]

        geology_result = build_geology_model(
            project_name=project_cfg["name"],
            input_dir=project_cfg["input_dir"],
            inversion_dir=inversion_dir,
            min_voxels=geo_cfg["min_voxels"],
            fill_iterations=geo_cfg["fill_iterations"],
            unit_defs_csv=geo_cfg.get("unit_defs_csv"),
            unit_groups_csv=geo_cfg.get("unit_groups_csv"),
            unit_id_npy=geo_cfg.get("unit_id_npy"),
        )

    return {
        "config": cfg,
        "inversion_result": inversion_result,
        "geology_result": geology_result,
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
