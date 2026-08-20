"""Read-only loaders and provenance helpers for archived GeoSAGE results.

The functions in this module deliberately never write to ``source_dir``.  They
are used by the interpret-existing workflow so an expensive SimPEG inversion
can be reused by several independent geological interpretations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import discretize
import h5py
import numpy as np


REQUIRED_ARTIFACTS = {
    "mesh_core": Path("mesh/mesh_core.msh"),
    "density_core": Path("inversion_result/joint_density_core.npy"),
    "susceptibility_core": Path("inversion_result/joint_susceptibility_core.npy"),
}

OPTIONAL_ARTIFACTS = {
    "mesh": Path("mesh/mesh.msh"),
    "recovered_model": Path("inversion_result/inversion_result_dens_susc.npy"),
    "density_active": Path("inversion_result/inversion_result_density_active.npy"),
    "susceptibility_active": Path("inversion_result/inversion_result_susceptibility_active.npy"),
    "dpred_gravity": Path("inversion_result/dpred_gravity.npy"),
    "dpred_magnetics": Path("inversion_result/dpred_magnetics.npy"),
    "obs_gravity": Path("observed_data/gravity.obs"),
    "obs_magnetics": Path("observed_data/magnetics.obs"),
    "inversion_params": Path("inversion_params.json"),
    "paras_h5": Path("paras.h5"),
    "topography": Path("topo/topography.xyz"),
}


def _source_path(source_dir: str | Path) -> Path:
    return Path(source_dir).expanduser().resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _read_h5_group(group: h5py.Group) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in group.attrs.items():
        values[str(key)] = _json_safe(value)
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            values[key] = _json_safe(item[()])
        elif isinstance(item, h5py.Group):
            values[key] = _read_h5_group(item)
    return values


def load_inversion_parameters(source_dir: str | Path) -> dict[str, Any]:
    """Load the archived parameter snapshot, preferring JSON over HDF5."""

    root = _source_path(source_dir)
    json_path = root / OPTIONAL_ARTIFACTS["inversion_params"]
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in {json_path}")
        return payload

    h5_path = root / OPTIONAL_ARTIFACTS["paras_h5"]
    if h5_path.is_file():
        with h5py.File(h5_path, "r") as handle:
            return _read_h5_group(handle)
    return {}


def build_source_manifest(source_dir: str | Path) -> dict[str, Any]:
    """Build a JSON-serializable inventory of an archived source run."""

    root = _source_path(source_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, relative in {**REQUIRED_ARTIFACTS, **OPTIONAL_ARTIFACTS}.items():
        path = root / relative
        entry: dict[str, Any] = {
            "relative_path": relative.as_posix(),
            "path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            entry["size_bytes"] = int(path.stat().st_size)
        artifacts[name] = entry

    density_path = root / REQUIRED_ARTIFACTS["density_core"]
    shape: list[int] | None = None
    if density_path.is_file():
        shape = [int(v) for v in np.load(density_path, mmap_mode="r").shape]

    return {
        "source_dir": str(root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_artifacts": {key: artifacts[key] for key in REQUIRED_ARTIFACTS},
        "optional_artifacts": {key: artifacts[key] for key in OPTIONAL_ARTIFACTS},
        "artifacts": artifacts,
        "density_susceptibility_shape": shape,
    }


def _require_file(root: Path, relative: Path, label: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"Archived inversion is missing required {label}: {path}"
        )
    return path


def load_existing_inversion_result(source_dir: str | Path) -> dict[str, Any]:
    """Load a previously completed inversion without writing to its directory."""

    root = _source_path(source_dir)
    mesh_path = _require_file(root, REQUIRED_ARTIFACTS["mesh_core"], "core mesh")
    density_path = _require_file(root, REQUIRED_ARTIFACTS["density_core"], "density model")
    susc_path = _require_file(root, REQUIRED_ARTIFACTS["susceptibility_core"], "susceptibility model")

    mesh_core = discretize.TensorMesh._readUBC_3DMesh(mesh_path)
    density = np.load(density_path)
    susceptibility = np.load(susc_path)
    if density.shape != susceptibility.shape:
        raise ValueError(
            "Archived density and susceptibility shapes differ: "
            f"{density.shape} vs {susceptibility.shape}"
        )
    if density.ndim != 3 or susceptibility.ndim != 3:
        raise ValueError(
            "Archived density and susceptibility arrays must both be 3D; "
            f"got {density.ndim}D and {susceptibility.ndim}D"
        )
    if tuple(density.shape) != tuple(mesh_core.shape_cells):
        raise ValueError(
            "Archived model shape does not match mesh_core.shape_cells: "
            f"models={density.shape}, mesh={mesh_core.shape_cells}"
        )

    paths: dict[str, str] = {
        "output_root": str(root),
        "source_inversion_dir": str(root),
        "mesh_core_ubc": str(mesh_path),
        "mesh_core_msh": str(mesh_path),
        "density_core_npy": str(density_path),
        "dens_core_npy": str(density_path),
        "susceptibility_core_npy": str(susc_path),
        "susc_core_npy": str(susc_path),
    }
    optional_loads: dict[str, tuple[str, Any]] = {
        "mesh": ("mesh", discretize.TensorMesh._readUBC_3DMesh),
        "recovered_model": ("recovered_model", np.load),
        "density_active": ("density_active", np.load),
        "susceptibility_active": ("susceptibility_active", np.load),
        "dpred_gravity": ("dpred_gravity", np.load),
        "dpred_magnetics": ("dpred_magnetics", np.load),
    }
    loaded: dict[str, Any] = {}
    for name, (attribute, loader) in optional_loads.items():
        path = root / OPTIONAL_ARTIFACTS[name]
        if path.is_file():
            loaded[attribute] = loader(path)
            paths[f"{name}_path"] = str(path)

    for name in ("obs_gravity", "obs_magnetics", "inversion_params", "paras_h5", "topography"):
        path = root / OPTIONAL_ARTIFACTS[name]
        if path.is_file():
            paths[f"{name}_path"] = str(path)

    # Preserve the names emitted by run_joint_inversion.py.
    aliases = {
        "mesh": "mesh_ubc",
        "recovered_model": "recovered_model_npy",
        "density_active": "density_active_npy",
        "susceptibility_active": "susceptibility_active_npy",
        "dpred_gravity": "dpred_gravity_npy",
        "dpred_magnetics": "dpred_magnetics_npy",
        "obs_gravity": "obs_gravity_ubc",
        "obs_magnetics": "obs_magnetics_ubc",
        "inversion_params": "paras_json",
        "paras_h5": "paras_h5",
    }
    for name, alias in aliases.items():
        path = root / OPTIONAL_ARTIFACTS[name]
        if path.is_file():
            paths[alias] = str(path)

    manifest = build_source_manifest(root)
    return {
        "mesh": loaded.get("mesh", mesh_core),
        "mesh_core": mesh_core,
        "ind_active": None,
        "dens_core_3d": density,
        "susc_core_3d": susceptibility,
        "recovered_model": loaded.get("recovered_model"),
        "dpred_gravity": loaded.get("dpred_gravity"),
        "dpred_magnetics": loaded.get("dpred_magnetics"),
        "paths": paths,
        "runtime_hours": None,
        "reused_existing": True,
        "source_manifest": manifest,
        "inversion_parameters": load_inversion_parameters(root),
    }


def load_existing_geology_result(
    source_dir: str | Path,
    inversion_result: dict[str, Any],
) -> dict[str, Any]:
    """Load an archived geology model without rebuilding or modifying it."""

    root = _source_path(source_dir)
    geo_dir = root / "geology_models"
    unit_path = geo_dir / "unit_id_3d.npy"
    geo_path = geo_dir / "geo_id_3d.npy"
    defs_path = geo_dir / "geo_defs.json"
    for path in (unit_path, geo_path, defs_path):
        if not path.is_file():
            raise FileNotFoundError(f"Archived geology artifact not found: {path}")

    unit_ids = np.load(unit_path)
    geo_ids = np.load(geo_path)
    density = inversion_result["dens_core_3d"]
    if unit_ids.shape != density.shape or geo_ids.shape != density.shape:
        raise ValueError(
            "Archived geology label shape does not match inversion model: "
            f"unit={unit_ids.shape}, geo={geo_ids.shape}, model={density.shape}"
        )
    with defs_path.open("r", encoding="utf-8") as handle:
        geo_defs_raw = json.load(handle)
    geo_defs = {int(key): str(value) for key, value in geo_defs_raw.items()}

    paths = {
        "source_inversion_dir": str(root),
        "interpretation_output_dir": str(root),
        "geology_models_dir": str(geo_dir),
        "unit_id_3d_npy": str(unit_path),
        "geo_id_3d_npy": str(geo_path),
        "geo_defs_json": str(defs_path),
        "geo_slices_dir": str(geo_dir / "slices_and_sections"),
    }
    geo_3d_path = geo_dir / "geo_model_without_background.jpg"
    scatter_path = geo_dir / "density_susceptibility_scatter_by_unit.png"
    slices_dir = geo_dir / "slices_and_sections"
    paths.update(
        {
            "geo_3d_png": str(geo_3d_path) if geo_3d_path.is_file() else "",
            "scatter_rho_kappa": str(scatter_path) if scatter_path.is_file() else "",
            "geo_geo_id_pngs": [str(path) for path in sorted(geo_dir.glob("geo_id_*.jpg"))],
            "geo_combo_slice_pngs": (
                [str(path) for path in sorted(slices_dir.glob("combo_slice_xy_*.png"))]
                if slices_dir.is_dir()
                else []
            ),
            "geo_combo_section_pngs": (
                [str(path) for path in sorted(slices_dir.glob("combo_section_xz_*.png"))]
                if slices_dir.is_dir()
                else []
            ),
        }
    )
    return {
        "mesh_core": inversion_result["mesh_core"],
        "dens_core_3d": density,
        "susc_core_3d": inversion_result["susc_core_3d"],
        "unit_id_3d": unit_ids,
        "geo_id_3d": geo_ids,
        "paths": paths,
        "geo_defs": geo_defs,
        "unit_defs": {},
        "reused_existing": True,
        "source_manifest": inversion_result.get("source_manifest", {}),
    }
