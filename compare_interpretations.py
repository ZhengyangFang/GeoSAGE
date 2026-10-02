"""Run low-cost post-inversion interpretation comparisons.

Every scenario uses one immutable inversion source and writes to its own
directory.  This module never calls ``run_joint_inversion`` directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from existing_results import build_source_manifest, load_existing_inversion_result
from multi_agent_runner import (
    LLMClient,
    MultiAgentOrchestrator,
    _cluster_units_from_inversion_gmm,
    _unit_defs_rows_to_csv,
    unit_stats_from_unit_id,
)
from runner import load_config


def _shared_unit_partition(
    inversion_result: dict[str, Any],
    comparison_root: Path,
    requested: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    shared_dir = comparison_root / "_shared_units"
    shared_dir.mkdir(parents=True, exist_ok=True)
    if requested:
        source = Path(str(requested))
        if not source.is_absolute():
            source = comparison_root / source
        if not source.is_file():
            raise FileNotFoundError(f"Shared unit partition not found: {source}")
        target = shared_dir / "unit_id_shared.npy"
        target.write_bytes(source.read_bytes())
        stats = unit_stats_from_unit_id(inversion_result, target)
        (shared_dir / "unit_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        return target, {"unit_stats": stats}

    target = shared_dir / "unit_id_gmm.npy"
    cluster_out = _cluster_units_from_inversion_gmm(
        inversion_result,
        target,
        k_min=6,
        k_max=10,
        random_state=42,
    )
    (shared_dir / "unit_stats.json").write_text(
        json.dumps(cluster_out["unit_stats"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (shared_dir / "unit_defs.csv").write_text(
        _unit_defs_rows_to_csv(cluster_out["unit_rows"]), encoding="utf-8"
    )
    (shared_dir / "bic_scores.json").write_text(
        json.dumps({str(k): float(v) for k, v in cluster_out["bic_scores"].items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return target, cluster_out


def _target_metrics(geo_result: dict[str, Any], target_geo_ids: list[int]) -> dict[str, Any]:
    labels = geo_result.get("geo_id_3d")
    if labels is None:
        return {"target_voxel_count": None, "target_voxel_fraction": None}
    mask = np.isin(labels, np.asarray(target_geo_ids, dtype=labels.dtype))
    return {
        "target_voxel_count": int(np.sum(mask)),
        "target_voxel_fraction": float(np.mean(mask)) if mask.size else 0.0,
    }


def _reference_metrics(
    geo_result: dict[str, Any],
    target_geo_ids: list[int],
    reference_path: str | Path | None,
) -> dict[str, Any]:
    if not reference_path:
        return {}
    path = Path(reference_path)
    if not path.is_file():
        return {"reference_error": f"Reference mask not found: {path}"}
    reference = np.asarray(np.load(path), dtype=bool)
    labels = geo_result.get("geo_id_3d")
    if labels is None or reference.shape != labels.shape:
        return {"reference_error": "Reference mask shape does not match scenario labels."}
    predicted = np.isin(labels, np.asarray(target_geo_ids, dtype=labels.dtype))
    intersection = int(np.sum(predicted & reference))
    union = int(np.sum(predicted | reference))
    pred_count = int(np.sum(predicted))
    ref_count = int(np.sum(reference))
    metrics: dict[str, Any] = {
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / (pred_count + ref_count) if pred_count + ref_count else 1.0,
        "target_volume_ratio": pred_count / ref_count if ref_count else None,
    }
    pred_center = np.argwhere(predicted).mean(axis=0) if pred_count else None
    ref_center = np.argwhere(reference).mean(axis=0) if ref_count else None
    metrics["centroid_offset_voxels"] = (
        float(np.linalg.norm(pred_center - ref_center))
        if pred_center is not None and ref_center is not None
        else None
    )
    return metrics


def run_comparison(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Run every scenario in a comparison JSON and write summary CSV/JSON."""

    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8-sig") as handle:
            comparison = json.load(handle)
    else:
        comparison = deepcopy(config)
    source_dir = Path(comparison["source_inversion_dir"]).expanduser().resolve()
    comparison_root = Path(comparison.get("comparison_root", "comparisons/geosage")).expanduser().resolve()
    if comparison_root == source_dir or source_dir in comparison_root.parents:
        raise ValueError("comparison_root must be outside the read-only source inversion directory")
    scenario_ids = [str(scenario["scenario_id"]) for scenario in comparison.get("scenarios", [])]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("scenario_id values must be unique")
    comparison_root.mkdir(parents=True, exist_ok=True)
    base_config = load_config(comparison.get("base_config", {}))
    base_config["project"]["source_inversion_dir"] = str(source_dir)
    base_config["project"]["output_dir"] = str(source_dir)
    base_config["run"]["execution_mode"] = "interpret_existing"
    base_config["run"]["run_inversion"] = False

    # Load the shared archived source once before scenario execution.
    inversion_result = load_existing_inversion_result(source_dir)
    source_manifest = build_source_manifest(source_dir)
    shared_unit_path: Path | None = None
    if comparison.get("shared_unit_partition") is not False:
        shared_unit_path, _ = _shared_unit_partition(
            inversion_result,
            comparison_root,
            comparison.get("shared_unit_partition") if comparison.get("shared_unit_partition") is not True else None,
        )

    rows: list[dict[str, Any]] = []
    for scenario in comparison.get("scenarios", []):
        scenario_id = str(scenario["scenario_id"])
        output_dir = Path(scenario.get("output_dir", comparison_root / scenario_id)).expanduser().resolve()
        cfg = deepcopy(base_config)
        cfg["project"]["interpretation_output_dir"] = str(output_dir)
        cfg["geology"]["mode"] = str(scenario.get("geology_mode", cfg["geology"].get("mode", "gmm_only")))
        if scenario.get("context_path"):
            cfg["geology"]["context_path"] = scenario["context_path"]
        if scenario.get("unit_id_npy"):
            cfg["geology"]["unit_id_npy"] = scenario["unit_id_npy"]
        elif shared_unit_path is not None and cfg["geology"]["mode"] in {
            "fixed_units_llm_groups",
            "fixed_units_fixed_groups",
            "gmm_only",
        }:
            cfg["geology"]["unit_id_npy"] = str(shared_unit_path)
        cfg["run"]["review_enabled"] = bool(scenario.get("review_enabled", False))
        cfg["run"]["max_review_rounds"] = 1
        cfg["run"]["reuse_existing_geology"] = cfg["geology"]["mode"] == "reuse_existing_geology"
        if "write_reports" in scenario:
            cfg["run"]["write_reports"] = bool(scenario["write_reports"])

        model = scenario.get("model")
        base_url = scenario.get("base_url")
        llm = None
        if model or base_url or cfg["run"]["review_enabled"] or cfg["geology"]["mode"] in {
            "gmm_bic_auto",
            "fixed_units_llm_groups",
        }:
            llm = LLMClient(model=model, base_url=base_url)
        orchestrator = MultiAgentOrchestrator(llm=llm)
        prompt = str(comparison.get("user_request", ""))
        result = orchestrator.run_from_config(cfg, user_request=prompt)
        workflow = result["workflow_result"]
        geo_result = workflow.get("geology_result") or {}
        final_cfg = result.get("config", cfg)
        target_ids = final_cfg["geology"].get("target_geo_ids", [])
        row = {
            "scenario_id": scenario_id,
            "model": model or "deterministic",
            "review_decision": (result.get("review") or {}).get("decision", ""),
            **_target_metrics(geo_result, [int(v) for v in target_ids]),
            "report_path": result.get("report_path", ""),
            **_reference_metrics(geo_result, [int(v) for v in target_ids], comparison.get("reference_mask")),
        }
        rows.append(row)

    summary_json = comparison_root / "comparison_summary.json"
    summary_csv = comparison_root / "comparison_summary.csv"
    summary_json.write_text(
        json.dumps({"source_manifest": source_manifest, "scenarios": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"source_manifest": source_manifest, "rows": rows, "summary_json": str(summary_json), "summary_csv": str(summary_csv)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GeoSAGE post-inversion interpretations.")
    parser.add_argument("--config", required=True, help="Comparison JSON file")
    args = parser.parse_args()
    result = run_comparison(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
