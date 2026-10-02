"""Deterministic, JSON-serializable evidence bundles for report/review stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _axis_range(values: Iterable[float]) -> list[float]:
    values_array = np.asarray(list(values), dtype=float)
    if values_array.size == 0:
        return []
    return [float(np.nanmin(values_array)), float(np.nanmax(values_array))]


def _bounds_for_mask(mesh: Any, mask: np.ndarray) -> dict[str, list[float]]:
    if not np.any(mask):
        return {"x_m": [], "y_m": [], "z_m": []}
    centers = mesh.cell_centers.reshape((-1, 3), order="F")
    selected = centers[mask.reshape(-1, order="F")]
    return {
        "x_m": _axis_range(selected[:, 0]),
        "y_m": _axis_range(selected[:, 1]),
        "z_m": _axis_range(selected[:, 2]),
    }


def _percentile_summary(values: np.ndarray) -> dict[str, float]:
    """Return compact, JSON-safe distribution statistics in metres."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _topography_at_core_cells(
    mesh: Any,
    source_manifest: dict[str, Any],
    inversion: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Read archived topography and sample it at core-cell horizontal centres.

    The inversion writer stores topography as ``x y elevation`` in metres.  This
    helper deliberately returns *elevation*, rather than a depth proxy, so that
    every report/review stage uses the same explicit coordinate convention.
    """

    artifacts = source_manifest.get("artifacts", {}) if isinstance(source_manifest, dict) else {}
    artifact = artifacts.get("topography", {}) if isinstance(artifacts, dict) else {}
    topo_path = artifact.get("path") if isinstance(artifact, dict) else None
    if not topo_path:
        topo_path = (inversion.get("paths") or {}).get("topography_xyz")
    if not topo_path or not Path(topo_path).is_file():
        return None, {
            "available": False,
            "reason": "Archived topography.xyz was not available; relative depth below surface cannot be audited.",
        }

    try:
        topography = np.loadtxt(topo_path, usecols=(0, 1, 2))
        if topography.ndim == 1:
            topography = topography.reshape(1, -1)
        topography = topography[np.all(np.isfinite(topography), axis=1)]
        if topography.size == 0:
            raise ValueError("no finite x/y/elevation rows")

        # cKDTree avoids assuming that the archived topography is a perfect
        # rectangular grid, while nearest-neighbour sampling matches the
        # topography handling used to construct the active-cell mask.
        from scipy.spatial import cKDTree

        x_grid, y_grid = np.meshgrid(mesh.cell_centers_x, mesh.cell_centers_y, indexing="ij")
        query_xy = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        distances, indices = cKDTree(topography[:, :2]).query(query_xy, k=1)
        sampled = topography[indices, 2].reshape(x_grid.shape)
        return sampled, {
            "available": True,
            "topography_path": str(Path(topo_path)),
            "sampling": "nearest archived topography point at each core-cell x/y centre",
            "surface_elevation_m": _percentile_summary(sampled),
            "nearest_topography_distance_m": _percentile_summary(distances),
        }
    except Exception as exc:  # noqa: BLE001 - evidence generation must remain non-fatal
        return None, {
            "available": False,
            "reason": f"Could not read/sample archived topography: {exc}",
        }


def _target_depth_record(
    mask: np.ndarray,
    identity: dict[str, int],
    z_centres_m: np.ndarray,
    topo_elevation_m: np.ndarray,
) -> dict[str, Any]:
    """Summarize a target's elevation and local-topography-relative depth."""

    z_3d = np.broadcast_to(z_centres_m.reshape((1, 1, -1)), mask.shape)
    topo_3d = np.broadcast_to(topo_elevation_m[:, :, None], mask.shape)
    selected_z = z_3d[mask]
    selected_topo = topo_3d[mask]
    depth = selected_topo - selected_z
    bands = ((0.0, 1000.0), (1000.0, 1500.0), (1500.0, 4000.0), (4000.0, 7000.0))
    return {
        **identity,
        "voxel_count": int(np.sum(mask)),
        "elevation_m": _percentile_summary(selected_z),
        "local_surface_elevation_m": _percentile_summary(selected_topo),
        "depth_below_local_surface_m": _percentile_summary(depth),
        "depth_band_fraction": {
            **{
                f"{int(lo)}-{int(hi)} m": float(np.mean((depth >= lo) & (depth < hi)))
                for lo, hi in bands
            },
            ">=7000 m": float(np.mean(depth >= 7000.0)),
        },
    }


def build_evidence_bundle(
    workflow_result: dict[str, Any],
    result_summary: dict[str, Any],
    slice_analysis: dict[str, Any],
    geology_context: str = "",
    target_unit_ids: Iterable[int] | None = None,
    target_geo_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Build stable evidence records without exposing hidden model reasoning."""

    inversion = workflow_result.get("inversion_result") or {}
    geology = workflow_result.get("geology_result") or {}
    mesh = inversion.get("mesh_core") or geology.get("mesh_core")
    density = inversion.get("dens_core_3d")
    susceptibility = inversion.get("susc_core_3d")
    source_manifest = workflow_result.get("source_manifest") or inversion.get("source_manifest", {})
    parameters = inversion.get("inversion_parameters", {})
    unit_ids = geology.get("unit_id_3d")
    geo_ids = geology.get("geo_id_3d")
    target_units = sorted({int(v) for v in (target_unit_ids or [])})
    target_geos = sorted({int(v) for v in (target_geo_ids or [])})

    evidence: list[dict[str, Any]] = []

    def add(identifier: str, category: str, value: Any) -> None:
        evidence.append({"id": identifier, "category": category, "value": _safe(value)})

    add(
        "E_SOURCE_MANIFEST",
        "source",
        {
            "source_inversion_dir": source_manifest.get("source_dir"),
            "artifacts": source_manifest.get("artifacts", {}),
            "inversion_parameters": parameters,
            "reused_existing": bool(inversion.get("reused_existing", False)),
        },
    )
    if mesh is not None:
        z_centres_m = np.asarray(mesh.cell_centers_z, dtype=float)
        z_direction = (
            "k increases toward higher elevation (shallower for a fixed surface)"
            if z_centres_m.size < 2 or z_centres_m[-1] > z_centres_m[0]
            else "k increases toward lower elevation (deeper for a fixed surface)"
        )
        topo_elevation_m, topo_metadata = _topography_at_core_cells(mesh, source_manifest, inversion)
        coordinate_reference = {
            "vertical_coordinate": "Model z is elevation in metres, positive upward.",
            "depth_definition": "Depth below local surface (m) = local topographic elevation (m) - model z elevation (m).",
            "layer_index_rule": "k is an array index, not a physical depth and never a synonym for surface.",
            "k_index_direction": z_direction,
            "k_0_z_elevation_m": float(z_centres_m[0]),
            "k_last_z_elevation_m": float(z_centres_m[-1]),
            "topography": topo_metadata,
        }
        add(
            "E_MESH_GEOMETRY",
            "mesh",
            {
                "shape_cells": [int(v) for v in mesh.shape_cells],
                "x_range_m": _axis_range(mesh.cell_centers_x),
                "y_range_m": _axis_range(mesh.cell_centers_y),
                "z_range_m": _axis_range(z_centres_m),
            },
        )
        add("E_COORDINATE_REFERENCE", "coordinate", coordinate_reference)
    else:
        coordinate_reference = {
            "vertical_coordinate": "Unavailable because the core mesh was not available.",
            "topography": {"available": False, "reason": "Core mesh unavailable."},
        }
        topo_elevation_m = None
    if density is not None and susceptibility is not None:
        add(
            "E_INV_GLOBAL_RANGES",
            "inversion",
            {
                "density_min": float(np.nanmin(density)),
                "density_max": float(np.nanmax(density)),
                "susceptibility_min": float(np.nanmin(susceptibility)),
                "susceptibility_max": float(np.nanmax(susceptibility)),
                "shape": [int(v) for v in density.shape],
            },
        )
    for index, slice_record in enumerate(slice_analysis.get("inversion_xy_slices", []), 1):
        add(f"E_INV_XY_{index:03d}", "inversion_slice", slice_record)
    for index, section_record in enumerate(slice_analysis.get("inversion_xz_sections", []), 1):
        add(f"E_INV_XZ_{index:03d}", "inversion_section", section_record)

    unit_stats = geology.get("unit_stats") or []
    for record in unit_stats:
        uid = int(record.get("unit_id", 0))
        add(f"E_UNIT_{uid:03d}", "unit", record)

    geo_defs = {int(k): str(v) for k, v in (geology.get("geo_defs") or {}).items()}
    if geo_ids is not None:
        total = float(geo_ids.size)
        for gid in sorted(int(v) for v in np.unique(geo_ids) if int(v) != 0):
            mask = geo_ids == gid
            record = {
                "geo_id": gid,
                "name": geo_defs.get(gid, ""),
                "voxel_count": int(np.sum(mask)),
                "voxel_fraction": float(np.sum(mask)) / total if total else 0.0,
                "bounds": _bounds_for_mask(mesh, mask) if mesh is not None else {},
            }
            add(f"E_GEO_{gid:03d}", "geo_group", record)

    target_records: list[dict[str, Any]] = []
    if geo_ids is not None:
        for gid in target_geos:
            mask = geo_ids == gid
            target_records.append(
                {
                    "geo_id": gid,
                    "voxel_count": int(np.sum(mask)),
                    "voxel_fraction": float(np.mean(mask)) if mask.size else 0.0,
                    "bounds": _bounds_for_mask(mesh, mask) if mesh is not None else {},
                }
            )
    if unit_ids is not None:
        for uid in target_units:
            mask = unit_ids == uid
            target_records.append(
                {
                    "unit_id": uid,
                    "voxel_count": int(np.sum(mask)),
                    "voxel_fraction": float(np.mean(mask)) if mask.size else 0.0,
                    "bounds": _bounds_for_mask(mesh, mask) if mesh is not None else {},
                }
            )
    add(
        "E_TARGET_STATISTICS",
        "target",
        {
            "target_unit_ids": target_units,
            "target_geo_ids": target_geos,
            "records": target_records,
        },
    )
    target_depth_records: list[dict[str, Any]] = []
    if mesh is not None and topo_elevation_m is not None:
        z_centres_m = np.asarray(mesh.cell_centers_z, dtype=float)
        if geo_ids is not None:
            for gid in target_geos:
                mask = np.asarray(geo_ids == gid, dtype=bool)
                if np.any(mask):
                    target_depth_records.append(
                        _target_depth_record(mask, {"geo_id": gid}, z_centres_m, topo_elevation_m)
                    )
        if unit_ids is not None:
            for uid in target_units:
                mask = np.asarray(unit_ids == uid, dtype=bool)
                if np.any(mask):
                    target_depth_records.append(
                        _target_depth_record(mask, {"unit_id": uid}, z_centres_m, topo_elevation_m)
                    )
    add(
        "E_TARGET_DEPTH_AUDIT",
        "coordinate",
        {
            "depth_definition": "Depth below local surface (m) = local topographic elevation (m) - model z elevation (m).",
            "records": target_depth_records,
            "available": bool(target_depth_records),
            "reason": (
                "No target depth records were available; do not make a precise depth or drilling-depth claim."
                if not target_depth_records
                else ""
            ),
        },
    )
    excerpt = (geology_context or "").strip()
    add("E_PRIOR_CONTEXT_EXCERPT", "geological_context", excerpt[:4000])

    return {
        "schema_version": "1.0",
        "evidence": evidence,
        "source_manifest": _safe(source_manifest),
        "result_summary": _safe(result_summary),
        "slice_analysis": _safe(slice_analysis),
        "target_unit_ids": target_units,
        "target_geo_ids": target_geos,
        "coordinate_reference": _safe(coordinate_reference),
        "target_depth_audit": _safe({"records": target_depth_records}),
    }


def save_evidence_bundle(path: str | Path, bundle: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
