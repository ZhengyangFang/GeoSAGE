from __future__ import annotations

from pathlib import Path

import numpy as np

from compare_interpretations import run_comparison
from test_interpret_existing import _make_source


def test_comparison_scenarios_share_frozen_source_and_units(tmp_path: Path) -> None:
    source = _make_source(tmp_path, shape=(3, 3, 3))
    tracked = [
        source / "mesh" / "mesh_core.msh",
        source / "inversion_result" / "joint_density_core.npy",
        source / "inversion_result" / "joint_susceptibility_core.npy",
    ]
    before = {str(path): path.read_bytes() for path in tracked}
    reference = tmp_path / "reference.npy"
    np.save(reference, np.zeros((3, 3, 3), dtype=bool))
    comparison = {
        "source_inversion_dir": str(source),
        "comparison_root": str(tmp_path / "comparison"),
        "user_request": "deterministic comparison",
        "reference_mask": str(reference),
        "base_config": {
            "project": {"name": "Tiny", "input_dir": str(source), "output_dir": str(source)},
            "geology": {"mode": "gmm_only"},
            "run": {"write_reports": False, "make_plots": False},
        },
        "scenarios": [
            {"scenario_id": "a", "geology_mode": "gmm_only"},
            {"scenario_id": "b", "geology_mode": "gmm_only"},
        ],
    }
    result = run_comparison(comparison)
    assert len(result["rows"]) == 2
    assert "iou" in result["rows"][0]
    assert "dice" in result["rows"][0]
    assert "source_manifest" not in result["rows"][0]
    assert before == {str(path): path.read_bytes() for path in tracked}
    assert Path(result["summary_csv"]).is_file()
