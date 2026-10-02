from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import discretize

from geosage import runner
from geosage.existing_results import load_existing_inversion_result
from geosage.multi_agent_runner import (
    MultiAgentOrchestrator,
    build_result_summary,
    build_slice_analysis,
)


def _make_source(root: Path, shape: tuple[int, int, int] = (2, 2, 2)) -> Path:
    source = root / "source"
    (source / "mesh").mkdir(parents=True)
    (source / "inversion_result").mkdir(parents=True)
    mesh = discretize.TensorMesh([np.ones(shape[0]), np.ones(shape[1]), np.ones(shape[2])])
    mesh.write_UBC(source / "mesh" / "mesh_core.msh")
    np.save(source / "inversion_result" / "joint_density_core.npy", np.arange(np.prod(shape)).reshape(shape))
    np.save(
        source / "inversion_result" / "joint_susceptibility_core.npy",
        np.arange(np.prod(shape), dtype=float).reshape(shape) / 10.0,
    )
    (source / "inversion_params.json").write_text(
        json.dumps({"select_region": [0, 2, 0, 2], "gravity_component": "gz"}),
        encoding="utf-8",
    )
    return source


def _config(source: Path, output: Path) -> dict:
    return {
        "project": {
            "name": "Tiny",
            "input_dir": str(source),
            "output_dir": str(source),
            "source_inversion_dir": str(source),
            "interpretation_output_dir": str(output),
        },
        "run": {
            "execution_mode": "interpret_existing",
            "run_inversion": False,
            "run_geology_model": False,
            "write_reports": False,
        },
    }


def _file_contents(paths: list[Path]) -> dict[str, bytes]:
    """Small-test helper that confirms read-only workflows preserve inputs."""

    return {str(path): path.read_bytes() for path in paths}


def test_interpret_existing_never_calls_inversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_source(tmp_path)
    monkeypatch.setattr(runner, "run_joint_inversion", lambda **_: (_ for _ in ()).throw(AssertionError("inversion called")))
    result = runner.run_workflow(_config(source, tmp_path / "out"))
    assert result["inversion_result"]["reused_existing"] is True
    assert result["inversion_result"]["dens_core_3d"].shape == (2, 2, 2)


def test_full_mode_still_calls_configured_inversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_source(tmp_path)
    called = {}

    def fake_inversion(**kwargs):
        called.update(kwargs)
        return {"paths": {"output_root": str(source)}}

    monkeypatch.setattr(runner, "run_joint_inversion", fake_inversion)
    cfg = _config(source, tmp_path / "full_out")
    cfg["run"].update({"execution_mode": "full", "run_inversion": True})
    result = runner.run_workflow(cfg)
    assert called
    assert result["run_manifest"]["execution_mode"] == "full"
    assert result["run_manifest"]["inversion_recomputed"] is True


def test_loaded_result_feeds_summary_and_slice_analysis(tmp_path: Path) -> None:
    source = _make_source(tmp_path, shape=(3, 3, 3))
    result = runner.run_workflow(_config(source, tmp_path / "out"))
    summary = build_result_summary(result)
    analysis = build_slice_analysis(result)
    assert summary["inversion"]["mesh_core_shape"] == (3, 3, 3)
    assert len(analysis["inversion_xy_slices"]) > 0
    assert len(analysis["inversion_xz_sections"]) > 0


def test_existing_result_shape_validation(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    np.save(source / "inversion_result" / "joint_density_core.npy", np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match="shape"):
        load_existing_inversion_result(source)


def test_source_directory_is_not_modified(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    tracked = [
        source / "mesh" / "mesh_core.msh",
        source / "inversion_result" / "joint_density_core.npy",
        source / "inversion_result" / "joint_susceptibility_core.npy",
        source / "inversion_params.json",
    ]
    before = _file_contents(tracked)
    runner.run_workflow(_config(source, tmp_path / "out"))
    after = _file_contents(tracked)
    assert before == after


def test_reuse_existing_geology_does_not_rebuild_source_model(tmp_path: Path) -> None:
    source = _make_source(tmp_path, shape=(2, 2, 2))
    geo_dir = source / "geology_models"
    geo_dir.mkdir()
    unit_ids = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=np.int16)
    geo_ids = np.where(unit_ids == 1, 10, 0).astype(np.int16)
    np.save(geo_dir / "unit_id_3d.npy", unit_ids)
    np.save(geo_dir / "geo_id_3d.npy", geo_ids)
    (geo_dir / "geo_defs.json").write_text(json.dumps({"10": "Archived target"}), encoding="utf-8")
    tracked_paths = list(geo_dir.iterdir())
    tracked = _file_contents(tracked_paths)

    cfg = _config(source, tmp_path / "reuse_out")
    cfg["run"]["run_geology_model"] = True
    cfg["geology"] = {"mode": "reuse_existing_geology"}
    result = runner.run_workflow(cfg)

    geology = result["geology_result"]
    assert geology["reused_existing"] is True
    np.testing.assert_array_equal(geology["unit_id_3d"], unit_ids)
    np.testing.assert_array_equal(geology["geo_id_3d"], geo_ids)
    assert tracked == _file_contents(tracked_paths)


def test_two_interpretations_share_source_inventory(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    first = runner.run_workflow(_config(source, tmp_path / "out_a"))
    second = runner.run_workflow(_config(source, tmp_path / "out_b"))
    assert first["source_manifest"]["artifacts"]["density_core"]["relative_path"] == (
        second["source_manifest"]["artifacts"]["density_core"]["relative_path"]
    )
    assert first["interpretation_output_dir"] != second["interpretation_output_dir"]


def test_gmm_only_uses_existing_arrays_without_inversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _make_source(tmp_path, shape=(3, 3, 3))
    monkeypatch.setattr(runner, "run_joint_inversion", lambda **_: (_ for _ in ()).throw(AssertionError("inversion called")))
    cfg = _config(source, tmp_path / "gmm_out")
    cfg["geology"] = {"mode": "gmm_only"}
    cfg["run"].update({"run_geology_model": True, "make_plots": False})
    orchestrator = MultiAgentOrchestrator()
    result = orchestrator.run_from_config(cfg)
    geo = result["workflow_result"]["geology_result"]
    assert Path(geo["paths"]["unit_id_3d_npy"]).is_file()
    assert Path(geo["paths"]["geo_id_3d_npy"]).is_file()
    assert Path(result["evidence_bundle_path"]).is_file()
    assert Path(result["agent_trace_path"]).is_file()


def test_gmm_bic_auto_uses_existing_arrays_without_inversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_source(tmp_path, shape=(3, 3, 3))
    monkeypatch.setattr(runner, "run_joint_inversion", lambda **_: (_ for _ in ()).throw(AssertionError("inversion called")))
    cfg = _config(source, tmp_path / "gmm_auto_out")
    cfg["geology"] = {"mode": "gmm_bic_auto"}
    cfg["run"].update({"run_geology_model": True, "make_plots": False})
    result = MultiAgentOrchestrator().run_from_config(cfg)
    geology_dir = Path(result["workflow_result"]["geology_result"]["paths"]["geology_models_dir"])
    assert (geology_dir / "unit_id_gmm.npy").is_file()
    assert (geology_dir / "bic_scores.json").is_file()
