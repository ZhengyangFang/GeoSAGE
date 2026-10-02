"""Validate real archived models without an inversion or any LLM requests.

Run from the repository root:
python examples/validate_archived_results.py --data-root /path/to/data --output-dir /path/to/new/check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from geosage.paths import data_path, result_path
from geosage.geo_modeling_workflow import build_geology_model
from geosage.multi_agent_runner import MultiAgentOrchestrator


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def validate(data_root: Path, output: Path, baseline_root: Path | None = None) -> list[dict]:
    data_root, output = data_root.resolve(), output.resolve()
    sources = [result_path(f"{case}_Inversion_GPT", data_root) for case in ("Hannah", "Iowa")]
    if any(output == s or s in output.parents for s in sources):
        raise ValueError("Validation output must be outside both source directories")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case, source in zip(("Hannah", "Iowa"), sources, strict=True):
        before = inventory(source)
        config = {
            "project": {"name": case, "input_dir": str(data_path(case, data_root)),
                        "source_inversion_dir": str(source),
                        "interpretation_output_dir": str(output / case / "reuse")},
            "geology": {"mode": "reuse_existing_geology",
                        "unit_defs_csv": str(data_path(case, data_root) / f"{case}_unit_defs.csv"),
                        "context_path": str(data_path(case, data_root) / f"{case}_geology_context.txt"),
                        "target_unit_ids": [], "target_geo_ids": [5]},
            "run": {"execution_mode": "interpret_existing", "run_inversion": False,
                    "run_geology_model": True, "make_plots": False,
                    "write_reports": False, "review_enabled": False},
        }
        result = MultiAgentOrchestrator().run_from_config(config)
        reused = result["workflow_result"]["geology_result"]
        rebuilt = build_geology_model(project_name=case, input_dir=data_path(case, data_root),
            inversion_dir=source, output_dir=output / case / "rebuilt", make_plots=False)
        record = {"case": case, "source_files": len(before),
                  "shape": list(reused["geo_id_3d"].shape), "geo_defs": reused["geo_defs"],
                  "geo_defs_source": reused.get("geo_defs_source"), "arrays": {}}
        for key in ("unit_id_3d", "geo_id_3d"):
            path = source / "geology_models" / f"{key}.npy"
            archived = np.load(path)
            np.testing.assert_array_equal(reused[key], archived)
            values = {"reused_exactly": True, "archived_sha256": digest(path),
                      "rebuilt_array_sha256": hashlib.sha256(rebuilt[key].tobytes()).hexdigest(),
                      "rebuilt_vs_archive_changed_voxels": int(np.sum(rebuilt[key] != archived))}
            if baseline_root:
                baseline = np.load(baseline_root / f"before_{case}" / "geology_models" / f"{key}.npy")
                np.testing.assert_array_equal(rebuilt[key], baseline)
                values["equal_pre_fix_rebuild"] = True
            record["arrays"][key] = values
        if inventory(source) != before:
            raise AssertionError(f"Source files changed during validation: {source}")
        record["source_unchanged"] = True
        records.append(record)
    (output / "validation.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory")
    parser.add_argument("--baseline-root", type=Path, help="Optional pre-fix before_CASE model directories")
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root, args.output_dir, args.baseline_root), indent=2))
