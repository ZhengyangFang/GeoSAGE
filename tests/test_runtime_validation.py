"""Regressions for destructive reruns and silently corrupted scientific inputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from geosage import runner
from geosage.existing_results import load_existing_inversion_result, load_existing_geology_result
from geosage.geo_modeling_workflow import build_geology_model
from geosage.multi_agent_runner import MultiAgentOrchestrator, _normalize_unit_groups_csv
from test_interpret_existing import _make_source, _config


def test_full_run_refuses_existing_results_before_solver(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    output = tmp_path / "previous"
    output.mkdir()
    old = output / "important.npy"
    old.write_bytes(b"previous scientific result")
    cfg = _config(source, output)
    cfg["project"]["output_dir"] = str(output)
    cfg["run"].update(execution_mode="full", run_inversion=True, overwrite=False)
    monkeypatch.setattr(runner, "run_joint_inversion", lambda **kw: pytest.fail("solver started"))
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.run_workflow(cfg)
    assert old.read_bytes() == b"previous scientific result"


@pytest.mark.parametrize("value", [np.nan, np.inf, 1 + 2j, "bad"])
def test_archive_rejects_invalid_model_values(tmp_path, value):
    source = _make_source(tmp_path)
    np.save(source / "inversion_result/joint_density_core.npy", np.full((2, 2, 2), value))
    with pytest.raises(ValueError, match="finite real"):
        load_existing_inversion_result(source)


@pytest.mark.parametrize("value", [-1, 1.5, np.nan])
def test_archive_rejects_invalid_labels(tmp_path, value):
    source = _make_source(tmp_path)
    inv = load_existing_inversion_result(source)
    geo = source / "geology_models"
    geo.mkdir()
    for name in ("unit_id_3d", "geo_id_3d"):
        np.save(geo / f"{name}.npy", np.full((2, 2, 2), value))
    with pytest.raises(ValueError, match="IDs"):
        load_existing_geology_result(source, inv)


@pytest.mark.parametrize("value", [1.5, -1, 32768, np.nan])
def test_build_rejects_lossy_label_cast(tmp_path, value):
    source = _make_source(tmp_path)
    defs = tmp_path / "units.csv"
    defs.write_text("unit_id,name,dens_min,dens_max,susc_min,susc_max\n1,Unit,0,9,0,1\n")
    labels = tmp_path / "labels.npy"
    np.save(labels, np.full((2, 2, 2), value))
    with pytest.raises(ValueError, match="Unit partition"):
        build_geology_model(
            "Tiny", input_dir=tmp_path, inversion_dir=source, output_dir=tmp_path / "geo",
            unit_defs_csv=defs, unit_id_npy=labels, make_plots=False,
        )
    assert not (tmp_path / "geo/geology_models/unit_id_3d.npy").exists()


def test_final_config_records_geology_stage(tmp_path):
    source = _make_source(tmp_path, (3, 3, 3))
    cfg = _config(source, tmp_path / "out")
    cfg["run"].update(run_geology_model=True, make_plots=False)
    cfg["geology"] = {"mode": "gmm_only"}
    result = MultiAgentOrchestrator().run_from_config(cfg)
    saved = json.loads((Path(result["workflow_result"]["interpretation_output_dir"]) / "effective_config.json").read_text())
    assert saved["run"]["run_geology_model"] is True
    assert Path(saved["geology"]["unit_id_npy"]).is_file()


def test_invalid_recorded_llm_grouping_discloses_fallback(tmp_path):
    source = _make_source(tmp_path, (3, 3, 3))
    cfg = _config(source, tmp_path / "out")
    cfg["run"].update(run_geology_model=True, make_plots=False)
    cfg["geology"] = {"mode": "gmm_bic_auto"}

    class InvalidLLM:
        def chat_json(self, *args, **kwargs):
            return {"unit_groups_csv": "garbage", "unit_name_map": {}}

    result = MultiAgentOrchestrator(llm=InvalidLLM()).run_from_config(cfg)
    notes = result["workflow_result"]["geology_result"]["grouping_adjustments"]
    assert any("fallback" in note for note in notes)


def test_fractional_llm_group_id_is_not_silently_truncated():
    notes = []
    stats = [{"unit_id": i, "dens_mean": i, "susc_mean": i, "voxel_count": 1} for i in range(1, 5)]
    _normalize_unit_groups_csv(
        "unit_id,geo_id,geo_name\n1.5,1,One\n2,2,Two\n3,3,Three\n4,4,Four\n",
        stats, adjustments=notes,
    )
    assert any("invalid" in note for note in notes)


def test_major_review_issue_cannot_be_accepted(tmp_path):
    from test_review_loop import _FakeLLM

    class ContradictoryLLM(_FakeLLM):
        def chat_json(self, *args, **kwargs):
            return {"decision": "ACCEPT", "issues": [{"severity": "major", "problem": "Wrong depth"}]}

    source = _make_source(tmp_path)
    cfg = _config(source, tmp_path / "out")
    cfg["run"].update(write_reports=True, review_enabled=True, max_review_rounds=0)
    result = MultiAgentOrchestrator(llm=ContradictoryLLM()).run_from_config(cfg)
    assert result["review"]["decision"] == "REVISE_REPORT"


def test_full_report_config_uses_actual_solver_settings(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    output = tmp_path / "full"
    actual = {"gravity_component": "gzz", "target_gravity_column": "gzz", "select_region": [1, 3, 1, 3]}

    def solver(**kwargs):
        output.mkdir()
        (output / "inversion_params.json").write_text(json.dumps(actual))
        return {"paths": {"output_root": str(output)}}

    monkeypatch.setattr(runner, "run_joint_inversion", solver)
    cfg = _config(source, output)
    cfg["project"]["output_dir"] = str(output)
    cfg["run"].update(execution_mode="full", run_inversion=True)
    result = runner.run_workflow(cfg)
    assert result["effective_config"]["data"]["gravity_component"] == "gzz"
    assert result["effective_config"]["region"]["min_e"] == 1
    assert result["inversion_result"]["inversion_parameters"] == actual


def test_partial_archived_mapping_recovers_missing_names_without_editing_source(tmp_path):
    source = _make_source(tmp_path)
    geo = source / "geology_models"
    geo.mkdir()
    labels = np.array([1, 7] * 4).reshape(2, 2, 2)
    for name in ("geo_id_3d", "unit_id_3d"):
        np.save(geo / f"{name}.npy", labels)
    definitions = geo / "geo_defs.json"
    definitions.write_text('{"1": "Recorded unit"}')
    reports = source / "reports"
    reports.mkdir()
    (reports / "archived.md").write_text("1. **Geo Group 7 - Recovered unit**\n")
    result = load_existing_geology_result(source, load_existing_inversion_result(source))
    assert result["geo_defs"] == {1: "Recorded unit", 7: "Recovered unit"}
    assert definitions.read_text() == '{"1": "Recorded unit"}'


@pytest.mark.parametrize("unit_id", [1.5, 32768, -1])
def test_csv_unit_ids_cannot_be_truncated_or_overflowed(tmp_path, unit_id):
    source = _make_source(tmp_path)
    defs = tmp_path / "units.csv"
    defs.write_text(f"unit_id,name,dens_min,dens_max,susc_min,susc_max\n{unit_id},Unit,0,9,0,1\n")
    with pytest.raises(ValueError, match="Unit definition IDs"):
        build_geology_model(
            "Tiny", input_dir=tmp_path, inversion_dir=source, output_dir=tmp_path / "geo",
            unit_defs_csv=defs, make_plots=False,
        )
