from __future__ import annotations
import ast
import base64
import csv
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import numpy as np

from openai import OpenAI
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from runner import DEFAULT_CONFIG, deep_update, run_workflow
from geo_modeling_workflow import build_geology_model


# =============================================================================
# 0. File helpers
# =============================================================================

def _resolve_geology_context_path(context_path: Path) -> Path:
    """
    Resolve the geology context path with lightweight fallbacks.
    Primary use-case: config points to *.txt but only *.pdf exists.
    """
    candidates: List[Path] = [context_path]
    suffix = context_path.suffix.lower()
    if suffix:
        for ext in (".txt", ".pdf"):
            if ext != suffix:
                candidates.append(context_path.with_suffix(ext))
    else:
        candidates.extend([context_path.with_suffix(".txt"), context_path.with_suffix(".pdf")])

    if not context_path.is_absolute():
        fallback_dir = Path("Geology report")
        candidates.extend([fallback_dir / c.name for c in candidates])

    seen = set()
    unique_candidates: List[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for candidate in unique_candidates:
        if candidate.exists():
            if candidate != context_path:
                print(f"[INFO] Geology context fallback: {context_path} -> {candidate}")
            return candidate

    tried = ", ".join(str(p) for p in unique_candidates)
    raise FileNotFoundError(
        f"Geology context file not found: {context_path}. Tried: {tried}"
    )


def read_geology_context(context_path: Path) -> str:
    resolved_path = _resolve_geology_context_path(context_path)
    suffix = resolved_path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_text(resolved_path)
    return resolved_path.read_text(encoding="utf-8")


def _read_pdf_text(path: Path) -> str:
    errors: List[str] = []

    def _join_pages(pages: List[str]) -> str:
        cleaned = [p.strip() for p in pages if p and p.strip()]
        return "\n\n".join(cleaned).strip()

    # 1) pypdf (preferred)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = _join_pages(pages)
        if text:
            return text
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    # 2) PyPDF2 (older)
    try:
        from PyPDF2 import PdfReader as PdfReader2

        reader = PdfReader2(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        text = _join_pages(pages)
        if text:
            return text
    except Exception as exc:
        errors.append(f"PyPDF2: {exc}")

    # 3) pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            pages = [(page.extract_text() or "") for page in pdf.pages]
        text = _join_pages(pages)
        if text:
            return text
    except Exception as exc:
        errors.append(f"pdfplumber: {exc}")

    # 4) External tools (pdftotext / mutool)
    for cmd in (
        ["pdftotext", "-layout", str(path), "-"],
        ["pdftotext", str(path), "-"],
        ["mutool", "draw", "-F", "text", str(path)],
    ):
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            text = (proc.stdout or "").strip()
            if text:
                return text
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as exc:
            errors.append(f"{cmd[0]}: {exc}")

    # 5) pandoc (if available)
    try:
        proc = subprocess.run(
            ["pandoc", str(path), "-t", "plain"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = (proc.stdout or "").strip()
        if text:
            return text
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as exc:
        errors.append(f"pandoc: {exc}")

    msg = (
        "Unable to extract text from PDF. "
        "Install 'pypdf' (recommended) or ensure 'pdftotext' is on PATH."
    )
    if errors:
        msg += f" Tried: {', '.join(errors)}"
    raise RuntimeError(msg)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _format_float(value: float) -> str:
    return f"{_safe_float(value):.6g}"


def _resolve_geo_output_path(cfg: Dict[str, Any], geo_cfg: Dict[str, Any], key: str, default_suffix: str) -> Path:
    cfg_path = geo_cfg.get(key)
    if cfg_path:
        p = Path(cfg_path)
    else:
        proj_dir = Path(cfg["project"]["input_dir"])
        p = proj_dir / f"{cfg['project']['name']}_{default_suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _first_existing_path(candidates: List[Path]) -> Optional[Path]:
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c
    return None


def _normalize_geo_input_paths(cfg: Dict[str, Any], geo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve geology input file paths with project-local priority.
    This prevents cross-project leakage (e.g., Iowa run reading Hannah files).
    """
    project_name = str(cfg.get("project", {}).get("name", "")).strip()
    project_dir = Path(str(cfg.get("project", {}).get("input_dir", ".")).strip() or ".")

    def _resolve_one(key: str, preferred_names: List[str], extra_candidates: Optional[List[Path]] = None) -> None:
        raw_val = geo_cfg.get(key)
        raw_path = Path(raw_val) if raw_val else None

        candidates: List[Path] = []

        # 1) Always prefer current project folder first.
        for name in preferred_names:
            candidates.append(project_dir / name)

        # 2) Then try user/LLM-provided path variants.
        if raw_path is not None:
            if raw_path.is_absolute():
                candidates.append(raw_path)
            else:
                candidates.append(project_dir / raw_path)
                candidates.append(project_dir / raw_path.name)
                candidates.append(raw_path)
                candidates.append(Path(raw_path.name))

        # 3) Workspace-level conventional names (fallback).
        for name in preferred_names:
            candidates.append(Path(name))

        # 4) Extra custom fallbacks (e.g., Geology report dir).
        if extra_candidates:
            candidates.extend(extra_candidates)

        resolved = _first_existing_path(candidates)
        if resolved is not None:
            prev = str(raw_path) if raw_path is not None else "(unset)"
            if raw_path is None or Path(str(raw_path)) != resolved:
                print(f"[INFO] {key} resolved for project '{project_name}': {prev} -> {resolved}")
            geo_cfg[key] = str(resolved)
        elif raw_path is None and preferred_names:
            # Keep a deterministic project-local default even if file does not exist yet.
            geo_cfg[key] = str(project_dir / preferred_names[0])

    _resolve_one(
        "unit_defs_csv",
        preferred_names=[f"{project_name}_unit_defs.csv", "unit_defs.csv"],
    )
    _resolve_one(
        "unit_groups_csv",
        preferred_names=[f"{project_name}_unit_groups.csv", "unit_groups.csv"],
    )
    _resolve_one(
        "context_path",
        preferred_names=[
            f"{project_name}_geology_context.txt",
            f"{project_name}_geology_context.pdf",
            "geology_context.txt",
            "geology_context.pdf",
        ],
        extra_candidates=[
            Path("Geology report") / f"{project_name}_geology_context.txt",
            Path("Geology report") / f"{project_name}_geology_context.pdf",
        ],
    )
    return geo_cfg


def _read_unit_defs_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r:
                continue
            try:
                unit_id = int(float(r.get("unit_id", "")))
            except Exception:
                continue
            rows.append(
                {
                    "unit_id": unit_id,
                    "name": str(r.get("name", "")).strip() or f"Unit {unit_id}",
                    "dens_min": _safe_float(r.get("dens_min"), 0.0),
                    "dens_max": _safe_float(r.get("dens_max"), 0.0),
                    "susc_min": _safe_float(r.get("susc_min"), 0.0),
                    "susc_max": _safe_float(r.get("susc_max"), 0.0),
                }
            )
    rows.sort(key=lambda x: int(x["unit_id"]))
    return rows


def _unit_defs_rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["unit_id", "name", "dens_min", "dens_max", "susc_min", "susc_max"])
    for row in sorted(rows, key=lambda x: int(x["unit_id"])):
        uid = int(row["unit_id"])
        writer.writerow(
            [
                uid,
                str(row.get("name", "")).strip() or f"Unit {uid}",
                _format_float(_safe_float(row.get("dens_min"), 0.0)),
                _format_float(_safe_float(row.get("dens_max"), 0.0)),
                _format_float(_safe_float(row.get("susc_min"), 0.0)),
                _format_float(_safe_float(row.get("susc_max"), 0.0)),
            ]
        )
    txt = out.getvalue()
    return txt if txt.endswith("\n") else (txt + "\n")


def _unit_stats_from_unit_rows(unit_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for row in sorted(unit_rows, key=lambda x: int(x["unit_id"])):
        dens_min = _safe_float(row.get("dens_min"), 0.0)
        dens_max = _safe_float(row.get("dens_max"), 0.0)
        susc_min = _safe_float(row.get("susc_min"), 0.0)
        susc_max = _safe_float(row.get("susc_max"), 0.0)
        stats.append(
            {
                "unit_id": int(row["unit_id"]),
                "name": str(row.get("name", "")).strip() or f"Unit {int(row['unit_id'])}",
                "voxel_count": 0,
                "voxel_fraction": 0.0,
                "dens_mean": 0.5 * (dens_min + dens_max),
                "dens_p10": dens_min,
                "dens_p90": dens_max,
                "susc_mean": 0.5 * (susc_min + susc_max),
                "susc_p10": susc_min,
                "susc_p90": susc_max,
                "depth_index_mean": 0.5,
            }
        )
    return stats


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def _target_scores(unit_stats: List[Dict[str, Any]]) -> np.ndarray:
    if not unit_stats:
        return np.array([], dtype=float)
    dens = np.array([_safe_float(s.get("dens_mean"), 0.0) for s in unit_stats], dtype=float)
    susc = np.array([_safe_float(s.get("susc_mean"), 0.0) for s in unit_stats], dtype=float)
    depth = np.array([_safe_float(s.get("depth_index_mean"), 0.5) for s in unit_stats], dtype=float)
    # Heuristic target tendency:
    # lower density + higher |susceptibility| + relatively shallow depth.
    return _zscore(-dens) + _zscore(np.abs(susc)) + 0.5 * _zscore(-depth)


def _build_fallback_unit_groups_csv(unit_stats: List[Dict[str, Any]], target_name: str = "") -> str:
    unit_stats_sorted = sorted(unit_stats, key=lambda x: int(x["unit_id"]))
    unit_ids = [int(s["unit_id"]) for s in unit_stats_sorted]
    n_units = len(unit_ids)
    if n_units == 0:
        return "unit_id,geo_id,geo_name\n"

    # Fixed target range: 3~5 groups (bounded by number of units).
    if n_units >= 9:
        n_groups = 5
    elif n_units >= 6:
        n_groups = 4
    else:
        n_groups = min(3, n_units)
    n_groups = max(1, min(n_groups, n_units))

    feats = np.array(
        [
            [
                _safe_float(s.get("dens_mean"), 0.0),
                _safe_float(s.get("susc_mean"), 0.0),
                _safe_float(s.get("depth_index_mean"), 0.5),
            ]
            for s in unit_stats_sorted
        ],
        dtype=float,
    )
    feats = (feats - feats.mean(axis=0)) / np.where(feats.std(axis=0) < 1e-9, 1.0, feats.std(axis=0))

    labels: np.ndarray
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_groups, n_init=10, random_state=42)
        labels = km.fit_predict(feats)
    except Exception:
        # Deterministic fallback: split by density ranking.
        order = np.argsort(feats[:, 0])
        labels = np.zeros(n_units, dtype=int)
        for i, idx in enumerate(order):
            labels[idx] = int(i * n_groups / max(1, n_units))
            labels[idx] = min(labels[idx], n_groups - 1)

    cluster_ids = sorted(np.unique(labels).tolist())
    cluster_order = sorted(cluster_ids, key=lambda c: float(np.mean(feats[labels == c, 0])))
    cluster_to_geo = {cid: i + 1 for i, cid in enumerate(cluster_order)}

    scores = _target_scores(unit_stats_sorted)
    primary_idx = int(np.argmax(scores)) if scores.size else 0
    primary_geo = cluster_to_geo.get(int(labels[primary_idx]), 1)

    target_label = target_name.strip() or "Inferred mineral system"
    geo_name_by_geo: Dict[int, str] = {}
    for geo_id in sorted(set(cluster_to_geo.values())):
        if geo_id == primary_geo:
            geo_name_by_geo[geo_id] = f"Primary target: {target_label}"
        else:
            geo_name_by_geo[geo_id] = f"Geo group {geo_id}"

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["unit_id", "geo_id", "geo_name"])
    for uid, lab in zip(unit_ids, labels, strict=False):
        gid = cluster_to_geo[int(lab)]
        writer.writerow([uid, gid, geo_name_by_geo.get(gid, f"Geo group {gid}")])
    txt = out.getvalue()
    return txt if txt.endswith("\n") else (txt + "\n")


def _normalize_unit_groups_csv(
    csv_text: str,
    unit_stats: List[Dict[str, Any]],
    target_name: str = "",
) -> str:
    unit_stats_sorted = sorted(unit_stats, key=lambda x: int(x["unit_id"]))
    unit_ids = [int(s["unit_id"]) for s in unit_stats_sorted]
    if not unit_ids:
        return "unit_id,geo_id,geo_name\n"

    if not csv_text or not csv_text.strip():
        return _build_fallback_unit_groups_csv(unit_stats_sorted, target_name=target_name)

    by_unit: Dict[int, Tuple[int, str]] = {}
    geo_name_first: Dict[int, str] = {}
    try:
        reader = csv.DictReader(io.StringIO(csv_text.strip()))
        for r in reader:
            if not r:
                continue
            try:
                uid = int(float(r.get("unit_id", "")))
                gid = int(float(r.get("geo_id", "")))
            except Exception:
                continue
            if uid not in unit_ids or gid <= 0:
                continue
            gname = str(r.get("geo_name", "")).strip()
            if uid not in by_unit:
                by_unit[uid] = (gid, gname)
            if gname and gid not in geo_name_first:
                geo_name_first[gid] = gname
    except Exception:
        return _build_fallback_unit_groups_csv(unit_stats_sorted, target_name=target_name)

    if not by_unit:
        return _build_fallback_unit_groups_csv(unit_stats_sorted, target_name=target_name)

    # Fill missing units into an existing geo group first.
    default_gid = next(iter(by_unit.values()))[0]
    for uid in unit_ids:
        if uid not in by_unit:
            by_unit[uid] = (default_gid, "")

    old_geo_ids = sorted({gid for gid, _ in by_unit.values()})
    if not (3 <= len(old_geo_ids) <= 5):
        return _build_fallback_unit_groups_csv(unit_stats_sorted, target_name=target_name)

    # Ensure there is a primary target group.
    has_primary = any("primary target" in (name or "").lower() for name in geo_name_first.values())
    if not has_primary:
        scores = _target_scores(unit_stats_sorted)
        primary_uid = unit_ids[int(np.argmax(scores))] if scores.size else unit_ids[0]
        primary_gid = by_unit[primary_uid][0]
        target_label = target_name.strip() or "Inferred mineral system"
        geo_name_first[primary_gid] = f"Primary target: {target_label}"

    # Re-index geo ids to contiguous 1..N.
    old_to_new = {old: i + 1 for i, old in enumerate(old_geo_ids)}
    new_geo_name: Dict[int, str] = {}
    for old, new in old_to_new.items():
        new_geo_name[new] = geo_name_first.get(old, f"Geo group {new}")

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["unit_id", "geo_id", "geo_name"])
    for uid in sorted(unit_ids):
        old_gid = by_unit[uid][0]
        new_gid = old_to_new[old_gid]
        writer.writerow([uid, new_gid, new_geo_name[new_gid]])
    txt = out.getvalue()
    return txt if txt.endswith("\n") else (txt + "\n")


def _normalize_unit_name_map(raw_name_map: Any, unit_ids: List[int]) -> Dict[int, str]:
    name_map: Dict[int, str] = {}
    if isinstance(raw_name_map, dict):
        for k, v in raw_name_map.items():
            try:
                uid = int(k)
            except Exception:
                continue
            if uid in unit_ids:
                name = str(v).strip()
                if name:
                    name_map[uid] = name
    for uid in unit_ids:
        name_map.setdefault(uid, f"Unit {uid}")
    return name_map


def _apply_unit_name_map(unit_rows: List[Dict[str, Any]], name_map: Dict[int, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in sorted(unit_rows, key=lambda x: int(x["unit_id"])):
        uid = int(row["unit_id"])
        row_new = dict(row)
        row_new["name"] = str(name_map.get(uid, row.get("name", f"Unit {uid}"))).strip() or f"Unit {uid}"
        out.append(row_new)
    return out


def _cluster_units_from_inversion_gmm(
    inversion_result: Dict[str, Any],
    unit_id_npy_path: Path,
    k_min: int = 6,
    k_max: int = 10,
    random_state: int = 42,
    max_fit_samples: int = 200000,
) -> Dict[str, Any]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for GMM+BIC clustering. "
            "Please install it (e.g., `uv pip install scikit-learn`)."
        ) from exc

    paths = inversion_result.get("paths", {})
    dens_path = Path(paths.get("density_core_npy") or paths.get("dens_core_npy") or "")
    susc_path = Path(paths.get("susceptibility_core_npy") or paths.get("susc_core_npy") or "")
    if not dens_path.exists() or not susc_path.exists():
        raise FileNotFoundError(
            f"Missing inversion core models for clustering: density={dens_path}, susceptibility={susc_path}"
        )

    dens = np.load(dens_path)
    susc = np.load(susc_path)
    if dens.shape != susc.shape:
        raise ValueError(f"density and susceptibility shape mismatch: {dens.shape} vs {susc.shape}")
    if dens.ndim != 3:
        raise ValueError(f"Expected 3D core models, got shape: {dens.shape}")

    nx, ny, nz = dens.shape
    dens_flat = dens.ravel()
    susc_flat = susc.ravel()
    z_index = np.broadcast_to(np.arange(nz, dtype=np.float32).reshape(1, 1, nz), dens.shape).ravel()
    z_norm = z_index / max(1.0, float(nz - 1))

    # Signed log compresses susceptibility while preserving polarity.
    susc_feat = np.sign(susc_flat) * np.log10(1.0 + np.abs(susc_flat))
    feats = np.column_stack([dens_flat, susc_feat, z_norm]).astype(np.float64)

    valid_mask = np.isfinite(feats).all(axis=1)
    if int(np.sum(valid_mask)) < 2:
        raise RuntimeError("Not enough valid voxels for clustering.")

    feats_valid = feats[valid_mask]
    med = np.median(feats_valid, axis=0)
    mad = np.median(np.abs(feats_valid - med), axis=0)
    scale = 1.4826 * mad
    std = np.std(feats_valid, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(std > 1e-9, std, 1.0))
    feats_valid_std = (feats_valid - med) / scale

    n_valid = feats_valid_std.shape[0]
    rng = np.random.default_rng(random_state)
    if n_valid > max_fit_samples:
        fit_idx = rng.choice(n_valid, size=max_fit_samples, replace=False)
        feats_fit = feats_valid_std[fit_idx]
    else:
        feats_fit = feats_valid_std

    k_lower = k_min if n_valid >= k_min else 2
    k_upper = min(k_max, max(2, n_valid - 1))
    if k_upper < k_lower:
        k_lower = max(2, min(k_upper, k_min))
    if k_upper < 2:
        raise RuntimeError("Insufficient valid samples for GMM.")

    best_model: Optional[Any] = None
    best_bic = float("inf")
    bic_scores: Dict[int, float] = {}
    for k in range(k_lower, k_upper + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=2,
            random_state=random_state,
            max_iter=300,
        )
        model.fit(feats_fit)
        bic = float(model.bic(feats_fit))
        bic_scores[k] = bic
        if bic < best_bic:
            best_bic = bic
            best_model = model

    if best_model is None:
        raise RuntimeError("GMM fitting failed: no valid model selected.")

    labels_valid = np.empty(n_valid, dtype=np.int16)
    chunk = 250000
    for i0 in range(0, n_valid, chunk):
        i1 = min(i0 + chunk, n_valid)
        labels_valid[i0:i1] = best_model.predict(feats_valid_std[i0:i1]).astype(np.int16)

    dens_valid = dens_flat[valid_mask]
    susc_valid = susc_flat[valid_mask]
    z_valid = z_norm[valid_mask]
    cluster_ids = sorted(np.unique(labels_valid).tolist())

    cluster_means: Dict[int, Tuple[float, float]] = {}
    for cid in cluster_ids:
        m = labels_valid == cid
        cluster_means[cid] = (
            float(np.mean(dens_valid[m])) if np.any(m) else 0.0,
            float(np.mean(susc_valid[m])) if np.any(m) else 0.0,
        )
    cid_sorted = sorted(cluster_ids, key=lambda c: cluster_means[c])
    cid_to_uid = {cid: i + 1 for i, cid in enumerate(cid_sorted)}

    unit_valid = np.array([cid_to_uid[int(c)] for c in labels_valid], dtype=np.int16)
    unit_flat = np.zeros_like(dens_flat, dtype=np.int16)
    unit_flat[valid_mask] = unit_valid
    unit_id_3d = unit_flat.reshape((nx, ny, nz))

    unit_id_npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(unit_id_npy_path, unit_id_3d.astype(np.int16))

    unit_rows: List[Dict[str, Any]] = []
    unit_stats: List[Dict[str, Any]] = []
    total = float(np.sum(valid_mask))
    for uid in sorted(set(cid_to_uid.values())):
        mask_u = unit_flat == uid
        count = int(np.sum(mask_u))
        if count == 0:
            continue
        dvals = dens_flat[mask_u]
        svals = susc_flat[mask_u]
        zvals = z_norm[mask_u]

        dens_min = float(np.percentile(dvals, 1.0))
        dens_max = float(np.percentile(dvals, 99.0))
        susc_min = float(np.percentile(svals, 1.0))
        susc_max = float(np.percentile(svals, 99.0))
        if dens_max <= dens_min:
            dens_max = dens_min + 1e-6
        if susc_max <= susc_min:
            susc_max = susc_min + 1e-6

        unit_rows.append(
            {
                "unit_id": uid,
                "name": f"GMM Unit {uid}",
                "dens_min": dens_min,
                "dens_max": dens_max,
                "susc_min": susc_min,
                "susc_max": susc_max,
            }
        )
        unit_stats.append(
            {
                "unit_id": uid,
                "name": f"GMM Unit {uid}",
                "voxel_count": count,
                "voxel_fraction": count / total if total > 0 else 0.0,
                "dens_mean": float(np.mean(dvals)),
                "dens_p10": float(np.percentile(dvals, 10.0)),
                "dens_p90": float(np.percentile(dvals, 90.0)),
                "susc_mean": float(np.mean(svals)),
                "susc_p10": float(np.percentile(svals, 10.0)),
                "susc_p90": float(np.percentile(svals, 90.0)),
                "depth_index_mean": float(np.mean(zvals)),
            }
        )

    unit_rows.sort(key=lambda x: int(x["unit_id"]))
    unit_stats.sort(key=lambda x: int(x["unit_id"]))
    return {
        "unit_rows": unit_rows,
        "unit_stats": unit_stats,
        "unit_id_npy": str(unit_id_npy_path),
        "best_k": int(best_model.n_components),
        "bic_scores": bic_scores,
    }


# =============================================================================
# 0. Lightweight LLM client + ContextAgent (inlined from agents_runner.py)
# =============================================================================


class LLMClient:
    """
    A simple LLM client wrapper:
      - Uses OPENAI_API_KEY for authentication by default
      - If LLM_BASE_URL is set, it routes to a local/self-hosted inference service
      - The model can be specified via the LLM_MODEL environment variable
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        raw_base_url = base_url or os.getenv("LLM_BASE_URL")
        # OpenRouter requires the /v1 suffix; add it if missing so we don't hit the HTML landing page.
        if raw_base_url and "openrouter.ai" in raw_base_url and not raw_base_url.rstrip("/").endswith("/v1"):
            raw_base_url = raw_base_url.rstrip("/") + "/v1"
        self.base_url = raw_base_url
        base_lower = (self.base_url or "").lower()
        is_openrouter = "openrouter.ai" in base_lower

        # Resolve API key with provider-aware fallback:
        # - OpenRouter: prefer OPENROUTER_API_KEY, fallback OPENAI_API_KEY
        # - OpenAI/default: OPENAI_API_KEY
        resolved_key = api_key
        if not resolved_key:
            if is_openrouter:
                resolved_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            else:
                resolved_key = os.getenv("OPENAI_API_KEY")
        self.api_key = resolved_key or "EMPTY"

        if self.api_key == "EMPTY":
            raise ValueError(
                "Missing API key for LLMClient. Set api_key explicitly, or set OPENAI_API_KEY "
                "(or OPENROUTER_API_KEY when using OpenRouter)."
            )
        if is_openrouter and self.api_key.startswith("sk-proj"):
            raise ValueError(
                "You are using OpenRouter base_url but provided an OpenAI project key (sk-proj...). "
                "Use an OpenRouter key (sk-or-v1...) or switch base_url to https://api.openai.com/v1."
            )

        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

        # OpenAI SDK: if base_url points to a local endpoint, requests are sent directly to the local service.
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or None,
        )

        # Detect whether the endpoint supports OpenAI-style response_format
        _base = (self.base_url or "").lower()
        self._supports_response_format = (
            not self.base_url
            or any(host in _base for host in ("openai.com", "openrouter.ai", "azure.com"))
        )

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Call /v1/chat/completions and force the model to return JSON.
        """
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        # Only add response_format for endpoints that support it
        if self._supports_response_format:
            request_params["response_format"] = {"type": "json_object"}
        else:
            # For non-OpenAI APIs, add explicit JSON instruction to system prompt
            request_params["messages"][0]["content"] = (
                system_prompt + "\n\nIMPORTANT: You must respond with valid JSON only. Do not include any text outside the JSON structure."
            )

        resp = self.client.chat.completions.create(**request_params)

        # Handle different response formats
        if isinstance(resp, str):
            # Some APIs return raw string directly
            text = resp
        elif hasattr(resp, 'choices'):
            text = resp.choices[0].message.content
        elif hasattr(resp, 'content'):
            text = resp.content
        else:
            raise ValueError(f"Unexpected response type: {type(resp)}, value: {resp}")

        # Debug: print response if empty or problematic
        if not text or not text.strip():
            print(f"[DEBUG] Empty API response! Response type: {type(resp)}")
            print(f"[DEBUG] Full response object: {resp}")
            raise ValueError("API returned empty response. Check your API key, endpoint URL, and model name.")
        if len(text) < 50:
            print(f"[DEBUG] Short response: {text}")

        return self._safe_json_loads(text)

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        """
        Try to parse the JSON. If the model includes extra text before/after, use a regex to extract the first {...} block.
        Also handles cases where a list is returned instead of a dict.
        """
        if not text:
            raise ValueError("Empty response from LLM")

        text = text.strip()
        if not text:
            raise ValueError("Empty response from LLM (after stripping)")

        # Guard: if we accidentally hit an HTML page (common when base_url is missing /v1)
        if text.lstrip().startswith("<!DOCTYPE") or text.lstrip().lower().startswith("<html"):
            raise ValueError(
                "LLM endpoint returned HTML instead of JSON. Check that LLM_BASE_URL includes the API path (e.g., https://openrouter.ai/api/v1)."
            )

        def _try_load(candidate: str) -> Optional[Dict[str, Any]]:
            """Attempt to parse with JSON first, then Python literal for single-quoted dicts."""
            for parser in (json.loads, ast.literal_eval):
                try:
                    result = parser(candidate)
                    if isinstance(result, list):
                        return {"result": result}
                    if isinstance(result, dict):
                        return result
                except Exception:
                    continue
            return None

        # Direct attempt
        parsed = _try_load(text)
        if parsed is not None:
            return parsed

        # Try to extract JSON from markdown code blocks
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if m:
            parsed = _try_load(m.group(1))
            if parsed is not None:
                return parsed

        # Try to find first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            parsed = _try_load(m.group(0))
            if parsed is not None:
                return parsed

        raise ValueError(f"LLM returns illegal JSON (first 500 chars): \n{text[:500]}")


class VisionClient:
    """
    A dedicated vision client that always uses OpenAI API for image analysis.
    This allows using a different LLM provider (e.g., Gemini) for text while
    keeping GPT-4o-mini for vision tasks.
    """

    DEFAULT_VISION_MODEL: str = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        # Vision always uses OpenAI API (supports images)
        self.api_key = api_key or os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.getenv("VISION_MODEL", self.DEFAULT_VISION_MODEL)

        # Vision client always uses OpenAI (no custom base_url for vision)
        self.client = OpenAI(api_key=self.api_key)


class ContextAgent:
    """
    Generate a complete configuration JSON based on the user's natural-language input plus the default DEFAULT_CONFIG.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def build_config(self, user_request: str) -> Dict[str, Any]:
        """
        Input: the user's natural-language requirements for gravity magnetic inversion + quasi-geological modeling.
        Output: a complete configuration (modified from DEFAULT_CONFIG)
        """
        default_cfg_str = json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2)

        system_prompt = """You are a geophysical workflow configuration assistant for joint gravity magnetic inversion and quasi-geological modeling.

Your tasks:
1. Read the user's natural-language requirements
2. Use the default configuration JSON I provide as the baseline, and modify fields only when necessary.
3. Output the complete configuration JSON. The field structure must match the default configuration exactly; do not add any extraneous fields.
4. Units:
   - Spatial coordinates are in meters (UTM projection).
   - Gravity in mGal (E if Gravity Gradient); magnetics in nT.
5. For any parameters the user did not mention, keep the default values; do not change them arbitrarily.

Output requirements:
- Output JSON only. The top-level object must contain exactly six keys: project / region / data / inversion / geology / run.
- Do not output any explanatory text or add comments."""

        user_prompt = f"""Below is the current default configuration JSON (please carefully review the field structure).
{default_cfg_str}

The user requirements are as follows:
\"\"\"{user_request}\"\"\"

Based on the default configuration, generate a new complete configuration JSON:
- If the user specifies the work area, data column names, regularization parameters, etc., update the corresponding fields.
- Parameters not mentioned should remain at their defaults.
- Keep the field structure unchanged.
"""

        cfg_from_llm = self.llm.chat_json(system_prompt, user_prompt)
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        deep_update(cfg, cfg_from_llm)
        return cfg


# =============================================================================
# 1. BaseLLMAgent
# =============================================================================

@dataclass
class BaseLLMAgent:
    llm: LLMClient
    name: str
    system_prompt: str

    def ask_json(self, user_prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
        """Return JSON (for LLMClient.chat_json)"""
        return self.llm.chat_json(self.system_prompt, user_prompt, temperature=temperature)

    def ask_text(self, user_prompt: str, temperature: float = 0.2) -> str:
        """Return free-text output (used for report generation)."""
        resp = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        # Handle different response formats
        if isinstance(resp, str):
            return resp.strip()
        elif hasattr(resp, 'choices'):
            return resp.choices[0].message.content.strip()
        elif hasattr(resp, 'content'):
            return resp.content.strip()
        else:
            raise ValueError(f"Unexpected response type: {type(resp)}, value: {resp}")

    def ask_text_with_images(
        self,
        user_prompt: str,
        image_paths: List[str],
        temperature: float = 0.3,
        max_images: int = 6,
        detail: str = "low",
    ) -> str:
        """
        Send text prompt along with images to a vision-capable LLM.
        Useful for models like GPT-4o, Claude 3.5 Sonnet, etc.
        """
        def _encode_image(p: str) -> str:
            path = Path(p)
            if not path.exists():
                return ""
            suffix = path.suffix.lower()
            mime = "image/png"
            if suffix in [".jpg", ".jpeg"]:
                mime = "image/jpeg"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"

        # Build content with text and images
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        # Add up to max_images
        for img_path in image_paths[:max_images]:
            uri = _encode_image(img_path)
            if uri:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": uri, "detail": detail},
                })

        resp = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
        )

        # Handle different response formats
        if isinstance(resp, str):
            return resp.strip()
        elif hasattr(resp, 'choices'):
            return resp.choices[0].message.content.strip()
        elif hasattr(resp, 'content'):
            return resp.content.strip()
        else:
            raise ValueError(f"Unexpected response type: {type(resp)}, value: {resp}")


# =============================================================================
# 2. Multi-agent roles: Data / Inversion / Geo / Report
# =============================================================================

class DataAgent(BaseLLMAgent):
    """Modify only `config["region"]` and `config["data"]`."""

    def refine(self, cfg: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        region = cfg["region"]
        data_cfg = cfg["data"]

        user_prompt = f"""
The current region and data configuration is as follows (JSON):
region = {json.dumps(region, ensure_ascii=False, indent=2)}
data   = {json.dumps(data_cfg, ensure_ascii=False, indent=2)}

The user requirements are as follows:
\"\"\"{user_request}\"\"\"

Based on these, please provide a new JSON:
{{
  "region": {{ ... }},
  "data":   {{ ... }}
}}

requirements:
1. Preserve the original field structure, such as min_e, max_e, min_n, max_n, gravity_column, std_grv, std_mag.
2. Parameters not explicitly requested by the user should remain at their original values, unless you have a well-justified reason to change them.
3. Do not output any extra fields and do not provide explanations; output JSON only
"""
        new_sub_cfg = self.ask_json(user_prompt)
        deep_update(cfg["region"], new_sub_cfg.get("region", {}))
        deep_update(cfg["data"], new_sub_cfg.get("data", {}))
        return cfg


class InversionAgent(BaseLLMAgent):
    """Only modify config["inversion"]"""

    def refine(self, cfg: Dict[str, Any], user_request: str,
               data_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        inv_cfg = cfg["inversion"]

        user_prompt = f"""
The current inversion configuration is as follows (JSON):
{json.dumps(inv_cfg, ensure_ascii=False, indent=2)}

Data summary (may be empty):
{json.dumps(data_summary or {}, ensure_ascii=False, indent=2)}

The user requirements are as follows:
\"\"\"{user_request}\"\"\"

Please output only a single JSON corresponding to the new inversion configuration. The field structure must be exactly the same as the inversion above.
Key considerations:
- target_grv_data, gravity_component
- reg_coefficient, reg_grv_norm, reg_mag_norm,
- cross_gradient_lambda, beta0_ratio, beta_cooling,
- optimization subfields (maxGNCG, maxLS, maxCG, tolCG, tolX),
- irls subfields (maxIRLSiter, IRLSstart, IRLS_mindelta, IRLSbeta_tol).

Rules:
1. Parameters not specifically emphasized by the user may remain at their defaults.
2. If the user mentions "smoother", "sparser", or "stronger structural coupling", adjust parameters appropriately
3. Output JSON only; no explanations.
"""
        new_inv = self.ask_json(user_prompt)
        deep_update(cfg["inversion"], new_inv)
        return cfg


class GeoAgent(BaseLLMAgent):
    """Only modify config["geology"]"""

    def refine(self, cfg: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        geo_cfg = cfg["geology"]

        user_prompt = f"""
The current geology configuration is as follows (JSON):
{json.dumps(geo_cfg, ensure_ascii=False, indent=2)}

The user requirements are as follows:
\"\"\"{user_request}\"\"\"

Please output only a single JSON corresponding to the new geology configuration. The field structure must be exactly the same as the geology above.
Key considerations:
- min_voxels: the minimum number of voxels used to remove small speckles
- fill_iterations: fill_iterations: the number of iterations for majority-vote hole filling
- unit_defs_csv / unit_groups_csv are usually generated by convention based on project.name, and generally should not be changed arbitrarily.

Output JSON only; no explanations.
"""
        new_geo = self.ask_json(user_prompt)
        deep_update(cfg["geology"], new_geo)
        return cfg


class ReportAgent(BaseLLMAgent):
    def write_report(
        self,
        cfg: Dict[str, Any],
        result_summary: Dict[str, Any],
        slice_analysis: Dict[str, Any],
        vision_analysis: Dict[str, str],
        geology_context: str,
        target_info: Dict[str, Any],
        fig_paths: Dict[str, Any],
        user_request: str,
        key_figures: Optional[List[str]] = None,
        use_vision: bool = False,
        max_images: int = 6,
    ) -> str:
        """
        Write a technical report based on inversion and geology modeling results.

        Parameters
        ----------
        max_images : int, default 6
            Maximum number of images to send to the vision model.
        """

        # Prepare the text prompt
        user_prompt = f"""
You are an exploration geophysicist and structural geologist specializing in mineral exploration,
and you need to write a technical report based on joint gravity-magnetic inversion and 3D pseudo-geological modeling.

[I. User Requirements]
\"\"\"{user_request}\"\"\"

[II. Project Configuration Summary]
project = {json.dumps(cfg["project"], ensure_ascii=False, indent=2)}
region  = {json.dumps(cfg["region"], ensure_ascii=False, indent=2)}
data    = {json.dumps(cfg["data"], ensure_ascii=False, indent=2)}
inversion = {json.dumps(cfg["inversion"], ensure_ascii=False, indent=2)}
geology   = {json.dumps(cfg["geology"], ensure_ascii=False, indent=2)}

[III. Geological Background (from external documents)]
{geology_context}

[IV. Summary of Numerical Results]
{json.dumps(result_summary, ensure_ascii=False, indent=2)}

[V. Automated Slice/Section Analysis (Quantitative Features)]
{json.dumps(slice_analysis, ensure_ascii=False, indent=2)}

[VI. Target Body Information]
{json.dumps(target_info, ensure_ascii=False, indent=2)}

[VII. Available Figures (File Paths)]
{json.dumps(fig_paths, ensure_ascii=False, indent=2)}

[VIII. Rapid Interpretation of the Visual Model (if available, for reference)]
{json.dumps(vision_analysis, ensure_ascii=False, indent=2)}

Please write a detailed Markdown report in English. The structure should include at least the following section:

## 1. Regional Geology and Exploration Background
- Briefly describe the study area's location, tectonic setting, and major lithologies.
- Cite information from the geological background that is relevant to the target body.

## 2. Data and Methods
- Describe which datasets were used, including data volume and spatial coverage.
- Summarize the joint inversion workflow (cross-gradient constraints, sparse regularization, etc.) and the mesh setup.

## 3. Pseudo-Geological Modeling Methodology
- Explain the principle of classifying geological units based on physical properties (density contrast vs. magnetic susceptibility).
- Describe how the 2D parameter space (density, susceptibility) is partitioned into Units and then merged into Geo groups.
- Reference the density-susceptibility scatter plot (scatter_rho_kappa) to illustrate the classification scheme.
- Explain the geological interpretation behind each unit/group (e.g., high-density high-susceptibility = mafic intrusive, low-density low-susceptibility = sedimentary, etc.).
- If the scatter plot is available in fig_paths, embed it using `![](figures/density_susceptibility_scatter_by_unit.png)`.

## 4. Inversion and Pseudo-Geological Results
- Describe the overall characteristics of the density and magnetic susceptibility models (e.g., gravity low-anomaly zones, magnetic high belts).
- Summarize the spatial distribution of the main geological bodies by geo group (volume, depth range, and geometry).
- Reference relevant figures (combo slices, 3D visualization) to support your description.

## 5. Target Body Analysis (Key Section)
- For the geological bodies/body groups corresponding to target_geo_ids,
create a dedicated subsection for each (primary vs. secondary targets must include separate conclusions and priority rankings):
  - Plan-view extent, strike, width, and depth extent.
  - Spatial relationships with faults, volcanic centers, geothermal fields, etc.
  - Consistency with observations, surface exposures, well-temperature data, and other constraints.
  - Mechanism/genesis interpretation (cite descriptions in the Geological Background (from external documents)that are relevant to the above elements as evidence).
- Evaluate the favorable conditions and unfavorable factors for these bodies as mineral exploration targets.
- If multiple targets exist (primary and secondary), discuss them separately and provide priority recommendations.
- Figure citations (required): include 2~3 key figure references in this chapter. If fig_paths is provided, prioritize figures most relevant to the target bodies.
- When you mention a figure filename, also embed it immediately using Markdown image syntax on the next line, e.g., `![](figures/your_file.png)`.

## 6. Uncertainty and Limitations
- Discuss the impacts of inversion non-uniqueness, data noise, and uncertainties in physical-property contrasts.
- Explain the subjectivity in unit classification during geology differentiation and the potential for misclassification.

## 7. Recommendations for Follow-up Work
- Suggested additional data (e.g., denser gravity/magnetic survey lines, physical-property sampling, seismic/electrical constraints, etc.)
- Recommended next steps (e.g., refined inversion, optimization of 2D target profiles, drilling planning recommendations including directions and depth intervals, etc.).

Writing requirements:
- Use a professional technical report style, and you may provide appropriate explanations of key concepts.
- Output in Markdown format; do not output JSON or code.
- Do not fabricate specific well IDs, coordinates, or numerical values; only make reasonable inferences based on information provided in result_summary and the geological background.
- Incorporate the numerical summary (result_summary), automated slice analysis (slice_analysis), and visual interpretation (vision_analysis, if available), and include more quantitative descriptions.
- In "Pseudo-Geological Modeling Methodology," explain the classification principle and reference the scatter plot if available.
- In "Inversion and Pseudo-Geological Results," separately summarize the density and magnetic susceptibility models, and describe the spatial distribution and volume proportions of the main geo_ids.
- In "Target Body Analysis," clearly state the priority ranking and spatial relationships of primary vs. secondary targets, and reference specific figure numbers or filenames to support the descriptions.
"""

        # Use vision if enabled and key figures are provided
        if use_vision and key_figures:
            user_prompt += """

\n\n[IMPORTANT: You have been provided with visual figures. Please analyze them directly while writing the report.]
"""
            return self.ask_text_with_images(
                user_prompt,
                key_figures,
                temperature=0.3,
                max_images=max_images,
            )
        else:
            return self.ask_text(user_prompt, temperature=0.3)


# =============================================================================
# 3. Helper: build numerical summaries for ReportAgent.
# =============================================================================

def build_result_summary(workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract some basic numerical information (e.g., volumes, ranges) 
    from the return output of run_workflow, to be used as input for the report agent.
    """
    inv = workflow_result.get("inversion_result")
    geo = workflow_result.get("geology_result")

    summary: Dict[str, Any] = {}

    if inv is not None:
        dens_core = inv["dens_core_3d"]
        susc_core = inv["susc_core_3d"]
        mesh_core = inv["mesh_core"]

        summary["inversion"] = {
            "runtime_hours": inv.get("runtime_hours"),
            "dens_min": float(np.nanmin(dens_core)),
            "dens_max": float(np.nanmax(dens_core)),
            "susc_min": float(np.nanmin(susc_core)),
            "susc_max": float(np.nanmax(susc_core)),
            "mesh_core_shape": tuple(int(v) for v in mesh_core.shape_cells),
        }

    if geo is not None:
        geo_id_3d = geo["geo_id_3d"]
        geo_defs = geo["geo_defs"]

        unique_ids, counts = np.unique(geo_id_3d, return_counts=True)
        geo_stats = []
        for gid, cnt in zip(unique_ids, counts):
            if gid == 0:
                continue
            geo_stats.append({
                "geo_id": int(gid),
                "name": geo_defs.get(int(gid), ""),
                "voxel_count": int(cnt),
            })

        summary["geology"] = {
            "n_geo_groups": len(geo_stats),
            "geo_groups": geo_stats,
        }

    return summary


def build_slice_analysis(workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform simple quantitative analyses on slices and cross-sections of the inversion/pseudo-geology results for convenient citation in the report.
    Do not rely on the visual model; use only NumPy data.
    """
    inv = workflow_result.get("inversion_result")
    geo = workflow_result.get("geology_result")
    analysis: Dict[str, Any] = {}

    if inv is not None:
        dens = inv.get("dens_core_3d")
        susc = inv.get("susc_core_3d")
        mesh_core = inv.get("mesh_core")
        if dens is not None and susc is not None and mesh_core is not None:
            xc = mesh_core.cell_centers_x
            yc = mesh_core.cell_centers_y
            zc = mesh_core.cell_centers_z
            nx, ny, nz = mesh_core.shape_cells

            def _stats_slice(arr2d, x_axis, y_axis):
                vmin = float(np.nanmin(arr2d))
                vmax = float(np.nanmax(arr2d))
                vmean = float(np.nanmean(arr2d))
                idx_min = np.unravel_index(np.nanargmin(arr2d), arr2d.shape)
                idx_max = np.unravel_index(np.nanargmax(arr2d), arr2d.shape)
                return {
                    "min": vmin,
                    "min_at": [float(x_axis[idx_min[0]]), float(y_axis[idx_min[1]])],
                    "max": vmax,
                    "max_at": [float(x_axis[idx_max[0]]), float(y_axis[idx_max[1]])],
                    "mean": vmean,
                    "p10": float(np.nanpercentile(arr2d, 10)),
                    "p90": float(np.nanpercentile(arr2d, 90)),
                }

            # 10 horizontal slices (consistent with the plotting)
            z_indices = np.unique(np.linspace(0, nz - 1, 10, dtype=int))
            inv_xy = []
            for k in z_indices:
                inv_xy.append({
                    "k": int(k),
                    "z": float(zc[k]),
                    "density": _stats_slice(dens[:, :, k], xc, yc),
                    "susceptibility": _stats_slice(susc[:, :, k], xc, yc),
                })

            # 10 vertical sections (XZ sections taken along the y direction)
            y_indices = np.unique(np.linspace(0, ny - 1, 10, dtype=int))
            inv_xz = []
            for j in y_indices:
                inv_xz.append({
                    "j": int(j),
                    "y": float(yc[j]),
                    "density": _stats_slice(dens[:, j, :], xc, zc),
                    "susceptibility": _stats_slice(susc[:, j, :], xc, zc),
                })

            analysis["inversion_xy_slices"] = inv_xy
            analysis["inversion_xz_sections"] = inv_xz

    # ---- Quasi-Geology model slices ----
    if geo is not None:
        geo_id_3d = geo.get("geo_id_3d")
        geo_defs = geo.get("geo_defs", {})
        mesh_core = geo.get("mesh_core")
        if geo_id_3d is not None and mesh_core is not None:
            xc = mesh_core.cell_centers_x
            yc = mesh_core.cell_centers_y
            zc = mesh_core.cell_centers_z
            nx, ny, nz = mesh_core.shape_cells

            def _geo_slice(arr2d):
                ids, counts = np.unique(arr2d, return_counts=True)
                stats = []
                for gid, cnt in zip(ids, counts):
                    if gid == 0:
                        continue
                    stats.append({
                        "geo_id": int(gid),
                        "name": geo_defs.get(int(gid), ""),
                        "voxel_count": int(cnt),
                        "fraction": float(cnt) / float(arr2d.size),
                    })
                stats = sorted(stats, key=lambda x: x["voxel_count"], reverse=True)
                return stats[:5]

            z_indices = np.unique(np.linspace(0, nz - 1, 10, dtype=int))
            geo_xy = []
            for k in z_indices:
                geo_xy.append({
                    "k": int(k),
                    "z": float(zc[k]),
                    "top_geo_groups": _geo_slice(geo_id_3d[:, :, k]),
                })

            y_indices = np.unique(np.linspace(0, ny - 1, 10, dtype=int))
            geo_xz = []
            for j in y_indices:
                geo_xz.append({
                    "j": int(j),
                    "y": float(yc[j]),
                    "top_geo_groups": _geo_slice(geo_id_3d[:, j, :]),
                })

            analysis["geo_xy_slices"] = geo_xy
            analysis["geo_xz_sections"] = geo_xz

    return analysis


def describe_images_with_vision(fig_paths: Dict[str, Any], vision_client: VisionClient, max_images: int = 6) -> Dict[str, str]:
    """
    Use OpenAI vision-capable model (via VisionClient) to provide brief descriptions of key figures.

    Parameters
    ----------
    fig_paths : dict
        Dictionary containing paths to generated figures
    vision_client : VisionClient
        Dedicated vision client that uses OpenAI API (always supports images)
    max_images : int
        Maximum number of images to analyze

    Returns
    -------
    dict
        Dictionary mapping figure keys to their vision-generated descriptions
    """

    picks: List[Tuple[str, str]] = []
    candidate_keys = [
        "density_slice",
        "susc_slice",
        "geo_3d",
    ]
    list_keys = [
        ("geo_3d_png", 1),
        ("geo_slices_dir", 1),
        ("geo_geo_id_pngs", 1),
        ("geo_combo_slice_pngs", 1),
        ("geo_combo_section_pngs", 1),
    ]
    for k in candidate_keys:
        p = fig_paths.get(k)
        if isinstance(p, str) and p:
            picks.append((k, p))
    for k, n in list_keys:
        lst = fig_paths.get(k, [])
        if isinstance(lst, list):
            for p in lst[:n]:
                if p:
                    picks.append((k, p))
    picks = picks[:max_images]

    def _encode_image(p: str) -> str:
        path = Path(p)
        if not path.exists():
            return ""
        suffix = path.suffix.lower()
        mime = "image/png"
        if suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    outputs: Dict[str, str] = {}
    for key, path in picks:
        uri = _encode_image(path)
        if not uri:
            continue
        try:
            resp = vision_client.client.chat.completions.create(
                model=vision_client.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a geophysical and pseudo-geological figure interpretation assistant. "
                        "In English, briefly describe the main anomalies, geometries, and possible geological implications in the image in 50-80 characters. Note the image embedding format in Markdown files.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Figure ID: {key}; Please briefly describe the main features and their relationship to the geology/targets."},
                            {"type": "image_url", "image_url": {"url": uri, "detail": "low"}},
                        ],
                    },
                ],
                temperature=0.3,
            )
            outputs[key] = resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            outputs[key] = f"(Visual interpretation failed: {exc})"

    return outputs

@dataclass
class PetrologyAgent(BaseLLMAgent):
    """
    An agent responsible for designing/adjusting unit_defs.csv and unit_groups.csv.
    Note: it uses only the geological background you provide and does not invent lithologies.
    """

    def design_unit_csvs(
        self,
        geology_text: str,
        project_name: str,
        mode: str = "csv_auto",
    ) -> Dict[str, str]:
        """
        Return two strings:
          - 'unit_defs_csv': CSV text including the header
          - 'unit_groups_csv': CSV text including the header
        """
        user_prompt = f"""
You are a geophysicist/structural geologist specializing in joint gravity-magnetic and geological interpretation.

Below is the geological background description for a study area (from a report/paper):

\"\"\"{geology_text}\"\"\"

Project objective: in the 2D parameter space of "density contrast (relative to the background density) + magnetic susceptibility,"
partition the model into 5 to 10 Units, then merge these Units into a smaller number of Geo groups for 3D pseudo-geological modeling (pseudo-geology / geology differentiation).

Please strictly follow the constraints and output requirements below:

[A. Naming and Interpretation Constraints]
1) You may only use lithology/stratigraphy/structural types or geologic unit names explicitly mentioned in geology_text (or reasonable abbreviations thereof) for naming and interpretation:
2) Do not invent new lithologies or new geologic unit names that are not mentioned in geology_text.
   - If geology_text only provides conceptual categories (e.g., "basement / intrusive / volcanic / sedimentary / melange / fault zone / alteration"), you may use those categories for naming.
   - Each unit name must be a short English term, but the key noun(s) must be traceable to terms present in geology_text.


[B. Numerical Range and Physical Plausibility Constraints]
3) In unit_defs.csv, dens_min/dens_max define the "density-contrast" range:
   - By default, relative to a background density of 2.67 g/cm^3 (if geology_text explicitly specifies a different background reference,
     you may follow that reference in your interpretation, but the output fields must still be density contrasts).
   - Units: g/cm^3; intervals are left-closed, right-open [min, max).
4) susc_min/susc_max define the magnetic-susceptibility range:
   - Units: SI; intervals are left-closed, right-open [min, max).
5) Values must be physically reasonable: avoid obviously unrealistic numbers (e.g., susceptibility > 1 SI, density contrasts exceeding +/-0.5 g/cm^3).
6) Prefer using typical physical-property ranges or qualitative descriptions provided in geology_text (e.g., "high susceptibility" or "low density") to set the intervals.
   If no specific values are provided, use conservative, distinguishable, and not overly narrow intervals, and ensure different units overlap as little as possible in the 2D parameter space.

[C. Unit Design Requirements (5 to 10 Units)]
7) The schema of unit_defs.csv must be exactly:
   unit_id,name,dens_min,dens_max,susc_min,susc_max
   - unit_id must be positive integers 1,2,3,...; 0 is reserved for "unclassified" and must not be defined in unit_defs.csv.
8) Unit partitioning must be interpretable and usable:
   - Each unit should correspond to a lithology, structural environment, genetic end-member, or a combination thereof described in geology_text (e.g., "melange / altered zone / intrusive / volcanic" etc.).
   - Avoid purely mathematical binning that results in unit names with no geological interpretation.
   - The total number of units must be 5 to 10, preferably 6 to 10 unless geology_text is extremely simple.

[D. Geo-Group Merging Requirements (Fewer Groups)]
9) The schema of unit_groups.csv must be exactly:
   unit_id,geo_id,geo_name
   - geo_id must be a positive integer; the same geo_id may correspond to multiple unit_ids.
   - geo_name provides a coarser-scale geological interpretation
   - When defining groups, if basement/bedrock-related units exist, prioritize putting them in geo_id=1 as the first group.
10) Geo-group design must emphasize "targets"
   - If geology_text explicitly mentions target lithologies/target environments/key structural controls (e.g., serpentinized ultramafics, mineralized intrusions, fault damage zones, alteration zones, etc.), 
     you must create at least one geo group representing the "primary target."
   - You may add 1 ~ 2 additional geo groups representing "secondary targets" that are related to the primary target but more mixed, weaker, or more uncertain (e.g., mixed rocks, melange, weak alteration, marginal facies, etc.).
   - If geology_text does not explicitly define a "target," designate as the primary target the unit most likely related to mineralization/fluid pathways/structural control (commonly fault zones, alteration zones, intrusion-host contacts,
     characteristic high-susceptibility/low-density end-members, etc.), and make it explicit in geo_name that this is an "inferred target (from geology text)"

[E. Output Format (Required)]
11) Output only a single JSON object containing exactly the following two keys:
{{
  "unit_defs_csv": "Full unit_defs.csv content goes here (including the header)",
  "unit_groups_csv": "Full unit_groups.csv content goes here (including the header)"
}}

Do not include any extra text outside the JSON object. Represent CSV rows using newline characters.
[Suggested workflow steps (perform internally; do not output these steps)]
- Extract a list of permissible lithology/structure/unit terms from geology_text.
- Identify one primary-target keyword (if present) and 0 ~ 2 secondary related keywords.
- Partition the density-susceptibility space into 5 ~ 10 rectangular intervals as units (minimize overlap as much as possible; allow blank areas for "unclassified").
- Merge the units into 3 ~ 5 geo groups (must include at least the primary target group).
"""
        return self.ask_json(user_prompt)

    def design_unit_names_and_groups(
        self,
        geology_text: str,
        project_name: str,
        unit_stats: List[Dict[str, Any]],
        target_name: str = "",
        min_groups: int = 3,
        max_groups: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate unit names + unit_groups.csv from unit-level statistics.
        Expected JSON keys:
          - unit_name_map: { "<unit_id>": "<name>", ... }
          - unit_groups_csv: CSV text with header unit_id,geo_id,geo_name
        """
        stats_json = json.dumps(unit_stats, ensure_ascii=False, indent=2)
        target_text = target_name or "inferred mineral system from geology text"
        user_prompt = f"""
You are a geophysicist and economic geologist.

Geological background:
\"\"\"{geology_text}\"\"\"

Project: {project_name}
Target focus: {target_text}

Unit statistics from inversion-based classification:
{stats_json}

Task:
1) Assign one short geological name to each unit_id in unit_name_map.
2) Merge units into geo groups and output unit_groups.csv.

Hard constraints:
- Keep the original unit_id values exactly; do not create or drop unit_ids.
- unit_groups.csv schema must be exactly:
  unit_id,geo_id,geo_name
- Every unit_id must appear exactly once in unit_groups.csv.
- Number of unique geo_id must be between {min_groups} and {max_groups} (inclusive).
- Must include one explicit primary target group in geo_name (must contain phrase "Primary target").
- Grouping should reflect ore-system relevance using geology text + unit physical-property tendencies.
- Do not invent unit IDs or unsupported geology terms.

Output JSON only:
{{
  "unit_name_map": {{
    "1": "name",
    "2": "name"
  }},
  "unit_groups_csv": "unit_id,geo_id,geo_name\\n..."
}}
"""
        return self.ask_json(user_prompt, temperature=0.2)

# =============================================================================
# 4. Multi-agent orchestrator
# =============================================================================

class MultiAgentOrchestrator:
    """
      1. ContextAgent: natural language -> initial config
      2. DataAgent: refine region/data settings
      3. InversionAgent: refine inversion parameters
      4. GeoAgent: refine geology parameters
      5. run_workflow(config): run inversion + geology modeling workflow
      6. ReportAgent: generate technical report

    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        vision_client: Optional[VisionClient] = None,
        enable_vision: bool = False,
        use_llm_vision: bool = False,
        max_vision_images: int = 26,
        images_per_list: int = 10,
    ) -> None:
        """
        Parameters
        ----------
        llm : LLMClient, optional
            Main LLM client for text generation
        vision_client : VisionClient, optional
            Separate vision client (only used if enable_vision=True and use_llm_vision=False)
        enable_vision : bool, default False
            Whether to use vision analysis for figures
        use_llm_vision : bool, default False
            If True, use the main LLM's vision capability directly (e.g., GPT-4o, Claude).
            If False, use the separate VisionClient for pre-analysis.
        max_vision_images : int, default 26
            Maximum number of images to send to the vision model.
            Increase this for more detailed analysis (e.g., 12-18), but be aware of API costs.
        images_per_list : int, default 10
            Maximum number of images to take from each figure list (slices, sections, etc.).
        """
        self.llm = llm or LLMClient()
        self.vision_client = vision_client or VisionClient()
        self.enable_vision = enable_vision
        self.use_llm_vision = use_llm_vision
        self.max_vision_images = max_vision_images
        self.images_per_list = images_per_list
        self.context_agent = ContextAgent(self.llm)

        self.data_agent = DataAgent(
            llm=self.llm,
            name="DataAgent",
            system_prompt=(
                "You are the DataAgent, responsible for setting gravity/magnetic data-related parameters (noise, flight altitude) "
                "and the working area bounds (UTM coordinates). Only modify the region and data sections in the JSON."
            ),
        )
        self.inversion_agent = InversionAgent(
            llm=self.llm,
            name="InversionAgent",
            system_prompt=(
                "You are the InversionAgent, responsible for configuring the parameters for joint gravity-magnetic inversion. "
                "Only modify the inversion section in the JSON.\nPay attention to regularization coefficients (reg_coefficient for gravity vs magnetic), regularization weights/norms, cross-gradient structural coupling, beta cooling strategy, etc."
            ),
        )
        self.geo_agent = GeoAgent(
            llm=self.llm,
            name="GeoAgent",
            system_prompt=(
                "You are the GeoAgent, responsible for configuring post-processing parameters for geological grouping based on the density/susceptibility models. "
                "Only modify the geology section in the JSON (speckle removal, hole filling, etc.)."
            ),
        )
        self.unit_csv_agent = PetrologyAgent(
            llm=self.llm,
            name="PetrologyAgent",
            system_prompt="You are responsible for designing unit_defs.csv and unit_groups.csv for the joint gravity-magnetic inversion results based on the geological background."
        )
        self.report_agent = ReportAgent(
            llm=self.llm,
            name="ReportAgent",
            system_prompt=(
                "You are an expert in geophysics and structural geology, skilled at writing professional technical reports based on inversion and geological modeling results."
            ),
        )

    def run_from_prompt(self, user_request: str) -> Dict[str, Any]:
        # 0. ContextAgent
        cfg0 = self.context_agent.build_config(user_request)

        # 1. DataAgent: region/data
        cfg1 = self.data_agent.refine(cfg0, user_request)

        # 2. Optional data summary hook (for future automated QC).
        data_summary = {}  

        # 3. InversionAgent: inversion parameters
        cfg2 = self.inversion_agent.refine(cfg1, user_request, data_summary=data_summary)

        # 4. GeoAgent: geology parameters
        cfg3 = self.geo_agent.refine(cfg2, user_request)

        # === Geology CSV branching logic (requested behavior) ===
        # 1) has unit_defs + unit_groups -> use CSV directly
        # 2) has unit_defs only          -> auto-generate unit_groups
        # 3) no unit_defs                -> run inversion, GMM+BIC clustering to make unit_defs + unit_groups
        geo_cfg = cfg3["geology"]
        geo_cfg = _normalize_geo_input_paths(cfg3, geo_cfg)
        cfg3["geology"] = geo_cfg
        mode = geo_cfg.get("mode", "csv_manual")
        unit_defs_cfg = geo_cfg.get("unit_defs_csv")
        unit_groups_cfg = geo_cfg.get("unit_groups_csv")
        has_defs = bool(unit_defs_cfg and Path(unit_defs_cfg).exists())
        has_groups = bool(unit_groups_cfg and Path(unit_groups_cfg).exists())

        context_path = Path(geo_cfg["context_path"])
        geology_text = read_geology_context(context_path)

        if not has_defs:
            print("[INFO] unit_defs_csv missing -> run inversion first, then auto-build unit_defs via GMM+BIC.")

            cfg_inv = json.loads(json.dumps(cfg3))
            cfg_inv["run"]["run_inversion"] = True
            cfg_inv["run"]["run_geology_model"] = False
            inv_only = run_workflow(cfg_inv)
            inversion_result = inv_only.get("inversion_result")
            if inversion_result is None:
                raise RuntimeError("Inversion did not return results; cannot run GMM clustering without core models.")

            output_root = Path(inversion_result["paths"]["output_root"])
            unit_id_npy_path = output_root / "geology_models" / f"{cfg3['project']['name']}_unit_id_gmm.npy"
            cluster_out = _cluster_units_from_inversion_gmm(
                inversion_result=inversion_result,
                unit_id_npy_path=unit_id_npy_path,
                k_min=6,
                k_max=10,
                random_state=42,
            )
            print(
                f"[INFO] GMM+BIC selected k={cluster_out['best_k']}; "
                f"BIC={cluster_out['bic_scores']}"
            )

            unit_rows = cluster_out["unit_rows"]
            unit_stats = cluster_out["unit_stats"]
            unit_ids = [int(r["unit_id"]) for r in unit_rows]

            raw_name_map: Dict[str, str] | Dict[int, str] = {}
            raw_groups_csv = ""
            try:
                resp = self.unit_csv_agent.design_unit_names_and_groups(
                    geology_text=geology_text,
                    project_name=cfg3["project"]["name"],
                    unit_stats=unit_stats,
                    target_name=geo_cfg.get("target_name", ""),
                    min_groups=3,
                    max_groups=5,
                )
                raw_name_map = resp.get("unit_name_map", {}) if isinstance(resp, dict) else {}
                raw_groups_csv = str(resp.get("unit_groups_csv", "")) if isinstance(resp, dict) else ""
            except Exception as exc:
                print(f"[WARN] PetrologyAgent naming/grouping failed; using deterministic fallback. {exc}")

            name_map = _normalize_unit_name_map(raw_name_map, unit_ids)
            named_rows = _apply_unit_name_map(unit_rows, name_map)
            unit_defs_text = _unit_defs_rows_to_csv(named_rows)
            unit_groups_text = _normalize_unit_groups_csv(
                raw_groups_csv,
                unit_stats=unit_stats,
                target_name=geo_cfg.get("target_name", ""),
            )

            unit_defs_path = _resolve_geo_output_path(cfg3, geo_cfg, "unit_defs_csv", "unit_defs_auto.csv")
            unit_groups_path = _resolve_geo_output_path(cfg3, geo_cfg, "unit_groups_csv", "unit_groups_auto.csv")
            unit_defs_path.write_text(unit_defs_text, encoding="utf-8")
            unit_groups_path.write_text(unit_groups_text, encoding="utf-8")

            geo_cfg["unit_defs_csv"] = str(unit_defs_path)
            geo_cfg["unit_groups_csv"] = str(unit_groups_path)
            geo_cfg["unit_id_npy"] = str(unit_id_npy_path)
            geo_cfg["mode"] = "gmm_bic_auto"
            cfg3["geology"] = geo_cfg

            if cfg3["run"].get("run_geology_model", True):
                geology_result = build_geology_model(
                    project_name=cfg3["project"]["name"],
                    input_dir=cfg3["project"]["input_dir"],
                    inversion_dir=output_root,
                    min_voxels=geo_cfg["min_voxels"],
                    fill_iterations=geo_cfg["fill_iterations"],
                    unit_defs_csv=geo_cfg.get("unit_defs_csv"),
                    unit_groups_csv=geo_cfg.get("unit_groups_csv"),
                    unit_id_npy=geo_cfg.get("unit_id_npy"),
                )
            else:
                geology_result = None

            wf_result = {
                "config": cfg3,
                "inversion_result": inversion_result,
                "geology_result": geology_result,
            }
        else:
            # If unit_defs exists, force CSV-based classification as requested.
            geo_cfg.pop("unit_id_npy", None)

            if not has_groups:
                print("[INFO] unit_groups_csv missing -> auto-generate unit_groups.csv from existing unit_defs.csv.")
                unit_defs_path = Path(unit_defs_cfg)
                unit_rows = _read_unit_defs_rows(unit_defs_path)
                if not unit_rows:
                    raise RuntimeError(f"Failed to parse unit definitions from {unit_defs_path}")
                unit_stats = _unit_stats_from_unit_rows(unit_rows)

                raw_groups_csv = ""
                try:
                    resp = self.unit_csv_agent.design_unit_names_and_groups(
                        geology_text=geology_text,
                        project_name=cfg3["project"]["name"],
                        unit_stats=unit_stats,
                        target_name=geo_cfg.get("target_name", ""),
                        min_groups=3,
                        max_groups=5,
                    )
                    raw_groups_csv = str(resp.get("unit_groups_csv", "")) if isinstance(resp, dict) else ""
                except Exception as exc:
                    print(f"[WARN] PetrologyAgent group generation failed; using deterministic fallback. {exc}")

                unit_groups_text = _normalize_unit_groups_csv(
                    raw_groups_csv,
                    unit_stats=unit_stats,
                    target_name=geo_cfg.get("target_name", ""),
                )
                unit_groups_path = _resolve_geo_output_path(cfg3, geo_cfg, "unit_groups_csv", "unit_groups_auto.csv")
                unit_groups_path.write_text(unit_groups_text, encoding="utf-8")
                geo_cfg["unit_groups_csv"] = str(unit_groups_path)
                geo_cfg["mode"] = "csv_auto_groups"
                cfg3["geology"] = geo_cfg
            else:
                if mode == "csv_auto":
                    print("[INFO] Both geology CSVs exist; using provided CSV classification directly.")
                cfg3["geology"] = geo_cfg

            wf_result = run_workflow(cfg3)

        # 6. Report
        result_summary = build_result_summary(wf_result)
        slice_analysis = build_slice_analysis(wf_result)
        geo_cfg = cfg3["geology"]
        context_path = Path(geo_cfg["context_path"])
        geology_text = read_geology_context(context_path)

        # Target information
        target_info = {
        "name": geo_cfg.get("target_name", "unknown target"),
        "geo_ids": geo_cfg.get("target_geo_ids", []),
        }
        # Figure paths
        inv_res = wf_result.get("inversion_result") or {}
        geo_res = wf_result.get("geology_result") or {}
        # Prefer the geo-modeling combo plots (geology + density + susceptibility)
        combo_slices = (
            geo_res.get("paths", {}).get("geo_combo_slice_pngs", [])
            or inv_res.get("paths", {}).get("inversion_result_slice_list", [])
            or []
        )
        combo_sections = (
            geo_res.get("paths", {}).get("geo_combo_section_pngs", [])
            or inv_res.get("paths", {}).get("inversion_result_section_list", [])
            or []
        )
        fig_paths = {
            "geo_3d": geo_res.get("paths", {}).get("geo_3d_png", ""),
            "scatter_rho_kappa": geo_res.get("paths", {}).get("scatter_rho_kappa", ""),
            "combo_slice_list": combo_slices,
            "combo_section_list": combo_sections,
            "geo_geo_id_pngs": geo_res.get("paths", {}).get("geo_geo_id_pngs", []),
        }
        # Output report
        output_root = (
            inv_res.get("paths", {}).get("output_root")
            or cfg3["project"].get("output_dir")
            or f"{cfg3['project'].get('name', 'project')}_Inversion"
        )
        output_root_path = Path(output_root)
        reports_dir = output_root_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Normalize figure paths to be relative to reports_dir or output_root_path
        def _norm_path(p: Any) -> str:
            if not p:
                return ""
            pp = Path(p)
            try:
                rel = pp.resolve().relative_to(reports_dir.resolve())
                return rel.as_posix()
            except Exception:
                try:
                    rel = pp.resolve().relative_to(output_root_path.resolve())
                    return (Path("..") / rel).as_posix()
                except Exception:
                    pass
            try:
                    rel = pp.relative_to(output_root_path)
                    return (Path("..") / rel).as_posix()
            except Exception:
                return pp.as_posix()

        fig_paths_normalized: Dict[str, Any] = {}
        for k, v in fig_paths.items():
            if isinstance(v, list):
                fig_paths_normalized[k] = [_norm_path(x) for x in v if x]
            else:
                fig_paths_normalized[k] = _norm_path(v)

        # Copy figures into the report folder to keep references stable in PDFs
        fig_dir = reports_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        def _copy_and_rewrite(p: str) -> str:
            if not p:
                return ""
            src = Path(p)
            if not src.is_absolute():
                src = (reports_dir / p).resolve()
            if not src.exists():
                return ""
            dst = fig_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            return f"figures/{dst.name}"

        fig_paths_md: Dict[str, Any] = {}
        for k, v in fig_paths_normalized.items():
            if isinstance(v, list):
                fig_paths_md[k] = [_copy_and_rewrite(x) for x in v if x]
            else:
                fig_paths_md[k] = _copy_and_rewrite(v)

        vision_analysis: Dict[str, str] = {}
        key_figures: List[str] = []

        if self.enable_vision:
            if self.use_llm_vision:
                # Collect key figures for direct LLM vision analysis
                # Use the original absolute paths for vision input
                for k, v in fig_paths.items():
                    if isinstance(v, str) and v:
                        key_figures.append(str(Path(reports_dir / v).resolve()))
                    elif isinstance(v, list):
                        for item in v[:self.images_per_list]:  # Configurable limit
                            if item:
                                key_figures.append(str(Path(reports_dir / item).resolve()))
                # Limit total images
                key_figures = key_figures[:self.max_vision_images]
            else:
                # Use separate VisionClient for pre-analysis
                vision_analysis = describe_images_with_vision(
                    fig_paths=fig_paths,
                    vision_client=self.vision_client,
                    max_images=self.max_vision_images,
                )

        report_text = self.report_agent.write_report(
            cfg=cfg3,
            result_summary=result_summary,
            slice_analysis=slice_analysis,
            vision_analysis=vision_analysis,
            geology_context=geology_text,
            target_info=target_info,
            fig_paths=fig_paths_md,
            user_request=user_request,
            key_figures=key_figures if self.use_llm_vision else None,
            use_vision=self.use_llm_vision,
            max_images=self.max_vision_images,
        )
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add a top-level title block for the report
        proj_name = cfg3["project"].get("name", "Project")
        target_name = target_info.get("name") or geo_cfg.get("target_name") or ""
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title_lines = [
            f"# {proj_name} Joint Gravity-Magnetic Inversion Report",
            f"### Target: {target_name}" if target_name else "",
            f"### Generated: {generated_at}",
        ]
        title_block = "\n".join([line for line in title_lines if line])
        # Auto-insert figure markdown.
        def _figures_markdown(fig_paths: Dict[str, Any]) -> str:
            blocks: List[str] = []

            def add_one(title: str, path: str):
                if path:
                    blocks.append(f"#### {title}\n\n![]({path})\n")

            def add_many(prefix: str, paths: Any, max_n: int = 6):
                if isinstance(paths, list):
                    for i, p in enumerate(paths[:max_n], 1):
                        if p:
                            blocks.append(f"#### {prefix} #{i}\n\n![]({p})\n")

            # 3D model
            geo3d = fig_paths.get("geo_3d")
            if isinstance(geo3d, str) and geo3d:
                add_one("Quasi Model (3D)", geo3d)

            # Combo slices / sections
            add_many("Combo Slice (XY)", fig_paths.get("combo_slice_list", []), max_n=6)
            add_many("Combo Section (XZ)", fig_paths.get("combo_section_list", []), max_n=6)

            # Individual geo_id 3D bodies
            add_many("Single unit 3D", fig_paths.get("geo_geo_id_pngs", []), max_n=10)

            if not blocks:
                return ""
            return "## Figures\n\n" + "\n".join(blocks)

        auto_fig_md = _figures_markdown(fig_paths_md)
        report_md_full = f"{title_block}\n\n{report_text}"
        if auto_fig_md:
            report_md_full = f"{title_block}\n\n{report_text}\n\n{auto_fig_md}"

        report_path = reports_dir / f"{cfg3['project']['name']}_report_{stamp}.md"
        report_path.write_text(report_md_full, encoding="utf-8")

        pdf_generated = False
        pdf_path = reports_dir / f"{cfg3['project']['name']}_report_{stamp}.pdf"
        # Use names + cwd to avoid duplicating the reports dir in the path
        pandoc_workdir = report_path.parent.resolve()
        base_pandoc_cmd = ["pandoc", report_path.name, "-o", pdf_path.name]

        env = os.environ.copy()
        # Add common Windows install paths so users do not need to edit PATH manually.
        extra_paths: List[Path] = []
        if os.name == "nt":
            extra_paths.extend(
                [
                    Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64",
                    Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin",
                    Path("C:/Program Files/Pandoc"),
                ]
            )
            # Winget often installs Pandoc under a versioned packages directory.
            winget_pkg_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
            if winget_pkg_root.exists():
                for exe in winget_pkg_root.glob("JohnMacFarlane.Pandoc_*/*/pandoc.exe"):
                    extra_paths.append(exe.parent)
        existing_path = env.get("PATH", "")
        prepend = [str(p) for p in extra_paths if p.exists()]
        if prepend:
            env["PATH"] = os.pathsep.join(prepend + [existing_path])

        if not shutil.which("pandoc", path=env.get("PATH")):
            print(
                "[WARN] Pandoc was not found in PATH; skipped PDF export. "
                f"Markdown report saved to: {report_path}"
            )
        else:
            engine_candidates = (
                "xelatex",
                "pdflatex",
                "lualatex",
                "tectonic",
                "wkhtmltopdf",
                "weasyprint",
            )
            available_engines = [
                e for e in engine_candidates if shutil.which(e, path=env.get("PATH"))
            ]
            if not available_engines:
                print(
                    "[WARN] No supported PDF engine found; skipped PDF export. "
                    "Looked for: "
                    + ", ".join(engine_candidates)
                    + f". Markdown report saved to: {report_path}"
                )
            else:
                engine_errors: List[str] = []
                for pdf_engine in available_engines:
                    font_args: List[str] = []
                    if pdf_engine == "xelatex" and os.name == "nt":
                        fonts_dir = Path("C:/Windows/Fonts")
                        if fonts_dir.exists():
                            has_times = any(fonts_dir.glob("times*.tt*"))
                            has_cambria = any(fonts_dir.glob("cambria*.tt*"))
                            if has_times and has_cambria:
                                font_args = [
                                    "-V",
                                    "mainfont=Times New Roman",
                                    "-V",
                                    "mathfont=Cambria Math",
                                ]

                    pandoc_cmd = [*base_pandoc_cmd, "--pdf-engine", pdf_engine, *font_args]
                    try:
                        proc = subprocess.run(
                            pandoc_cmd,
                            check=True,
                            cwd=pandoc_workdir,
                            env=env,
                            text=True,
                            capture_output=True,
                        )
                        pdf_generated = True
                        print(f"PDF report saved to: {pdf_path} (engine: {pdf_engine})")
                        stderr_text = (proc.stderr or "").strip()
                        if stderr_text:
                            print(f"Pandoc warnings ({pdf_engine}): {stderr_text}")
                        break
                    except subprocess.CalledProcessError as exc:
                        stderr = (exc.stderr or "").strip()
                        engine_errors.append(
                            f"{pdf_engine} failed (exit {exc.returncode})"
                            + (f": {stderr[:400]}" if stderr else "")
                        )

                if not pdf_generated:
                    detail = " | ".join(engine_errors) if engine_errors else "No engine attempted."
                    print(
                        "[WARN] PDF generation failed with all available engines. "
                        f"Tried: {', '.join(available_engines)}. Details: {detail}. "
                        f"Markdown report saved to: {report_path}"
                    )

        print("Report saved to:", report_path)
        if pdf_generated:
            print("PDF report saved to:", pdf_path)

        return {
            "config": cfg3,
            "workflow_result": wf_result,
            "summary": result_summary,
            "slice_analysis": slice_analysis,
            "vision_analysis": vision_analysis,
            "report": report_text,
            "report_path": str(report_path),
            "pdf_path": str(pdf_path) if pdf_generated else None,
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-agent: natural-language-driven joint gravity-magnetic inversion + pseudo-geological modeling"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User task description",
    )
    args = parser.parse_args()

    orchestrator = MultiAgentOrchestrator()
    result = orchestrator.run_from_prompt(args.prompt)

    print("\n=== Report ===\n")
    print(result["report"])
    print("\n=== Output Path ===")
    inv = result["workflow_result"]["inversion_result"]
    geo = result["workflow_result"]["geology_result"]
    if inv is not None:
        print("  Inversion output root:", inv["paths"]["output_root"])
    if geo is not None:
        print("  Geology slices dir   :", geo["paths"]["geo_slices_dir"])


