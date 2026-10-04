"""Contract and numerical tests against the real host runtime (optional extra)."""

from pathlib import Path
import json

import numpy as np
import pytest

pytest.importorskip("PyHydroGeophysX.agents.assistants")

from geosage.pyhydrogeophysx import ASSISTANT
from geosage.pyhydrogeophysx.configuration import configure
from geosage.pyhydrogeophysx.workflow import run
from test_interpret_existing import _make_source


def archived_payload(tmp_path):
    from discretize import TensorMesh

    source = _make_source(tmp_path)
    TensorMesh([[2, 5], [3, 4], [1, 9]], x0=[1000, 2000, -10]).write_UBC(
        source / "mesh/mesh_core.msh"
    )
    geo = source / "geology_models"
    geo.mkdir()
    labels = np.array([0, 1, 1, 7, 7, 1, 7, 7]).reshape(2, 2, 2)
    np.save(geo / "geo_id_3d.npy", labels)
    np.save(geo / "unit_id_3d.npy", labels)
    (geo / "geo_defs.json").write_text(json.dumps({"1": "Unit one", "7": "Unit seven"}))
    return {
        "request": "Inspect archived property models",
        "output_dir": str(tmp_path / "run"),
        "inputs": {"source_inversion_dir": str(source)},
    }


def test_plugin_discovery_and_no_global_tool_pollution():
    from PyHydroGeophysX.agents.assistants import get_assistant

    aquah_tools = get_assistant("aquah").load_tools()

    assert ASSISTANT.availability() == (True, "")
    assert get_assistant("geosage") is ASSISTANT
    assert ASSISTANT.load_tools()["write_report"] is not aquah_tools["write_report"]
    assert "run_joint_inversion" not in aquah_tools


def test_archived_run_preserves_values_coordinates_and_source(tmp_path):
    import pyvista as pv

    payload = archived_payload(tmp_path)
    source = Path(payload["inputs"]["source_inversion_dir"])
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    events, approved = [], []
    payload["step_mode"] = True

    def approve(event):
        approved.append(event["tool"])
        return "proceed"

    result = run(payload, lambda *a: None, on_event=events.append, approve=approve)
    assert result["status"] == "needs_review"
    assert result["review_decision"] == "NOT_REVIEWED"
    assert result["source_files_unchanged"]
    assert approved == list(ASSISTANT.load_tools())
    assert [e["tool"] for e in events if e["phase"] == "done"] == approved
    assert before == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    model = pv.read(result["exports"]["model"])
    np.testing.assert_array_equal(
        model.cell_data["Density contrast (g/cm3)"], np.arange(8).reshape(2, 2, 2).ravel(order="F")
    )
    np.testing.assert_array_equal(
        model.cell_data["Geo ID"], np.load(source / "geology_models/geo_id_3d.npy").ravel(order="F")
    )
    np.testing.assert_array_equal(model.x, [1000, 1002, 1007])
    np.testing.assert_array_equal(model.y, [2000, 2003, 2007])
    np.testing.assert_array_equal(model.z, [-10, -9, 0])
    assert Path(result["report_files"]["report_markdown"]).is_file()
    assert "No LLM interpretation" in Path(result["report_files"]["report_markdown"]).read_text()
    assert set(result["exports"]["figures"]) == {"Model sections", "Physical-property distribution"}
    metadata = json.loads(Path(result['exports']['metadata']).read_text(encoding='utf-8'))
    assert metadata['display']['preset'] == 'adaptive'
    density_style = metadata['viewer_fields']['Density contrast (g/cm3)']
    assert density_style['limits'] == [-7., 7.]
    assert density_style['range_policy'] == 'full range'
    assert result['artifacts'][0]['label'] == '3D model'
    with pytest.raises(FileExistsError):
        run(payload, lambda *a: None, approve=approve)


def test_stop_and_skipped_dependencies_cannot_appear_successful(tmp_path):
    payload = archived_payload(tmp_path)
    payload["step_mode"] = True
    result = run(payload, lambda *a: None, approve=lambda e: "stop")
    assert result["status"] == "incomplete" and not result["report_files"]
    assert not list(Path(payload["output_dir"]).rglob("*.vtk"))


def test_step_mode_requires_real_approval_callback(tmp_path):
    payload = archived_payload(tmp_path)
    payload["step_mode"] = True
    with pytest.raises(ValueError, match="approval"):
        run(payload, lambda *a: None)
    assert not Path(payload["output_dir"]).exists()


def test_rejects_output_inside_source_before_writing(tmp_path):
    payload = archived_payload(tmp_path)
    payload["output_dir"] = str(Path(payload["inputs"]["source_inversion_dir"]) / "child")
    with pytest.raises(ValueError, match="outside"):
        run(payload, lambda *a: None)
    assert not Path(payload["output_dir"]).exists()


def test_config_has_no_implicit_hannah_target_and_uses_config_workspace(tmp_path):
    payload = archived_payload(tmp_path)
    config = {
        "project": {
            "workspace_dir": ".",
            "name": "Tiny",
            "input_dir": "source",
            "source_inversion_dir": "source",
        },
        "run": {"execution_mode": "interpret_existing", "run_inversion": False},
        "geology": {"mode": "reuse_existing_geology"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    payload["inputs"] = {"config_file": str(path)}
    cfg = configure(payload)
    assert cfg["project"]["source_inversion_dir"] == str(tmp_path / "source")
    assert cfg["geology"]["target_geo_ids"] == []
    assert cfg["geology"]["context_path"] is None


def test_offline_adapter_never_uses_environment_key(monkeypatch):
    from geosage.pyhydrogeophysx.providers import StudioLLM

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    with pytest.raises(RuntimeError, match="session API key"):
        StudioLLM({}).query("test")


@pytest.mark.parametrize("decision", ["ACCEPT", "REVISE_REPORT", "INSUFFICIENT_EVIDENCE"])
@pytest.mark.parametrize("provider", ["openai", "codex_cli", "claude_code"])
def test_report_and_review_are_distinct_approved_stages(tmp_path, monkeypatch, decision, provider):
    from geosage.multi_agent_runner import MultiAgentOrchestrator
    from geosage.pyhydrogeophysx import tools, providers
    from test_review_loop import _FakeLLM

    if provider not in ASSISTANT.providers:
        pytest.skip("Installed host does not provide this model backend")

    fake = _FakeLLM(decision)
    engine = MultiAgentOrchestrator(llm=fake, vision_client=object())
    monkeypatch.setattr(tools, "_orchestrator", lambda ctx: engine)
    order = iter(ASSISTANT.load_tools())
    monkeypatch.setattr(
        providers.StudioLLM,
        "query",
        lambda *a, **kw: json.dumps({"tool": next(order), "why": "Test stage"}),
    )
    payload = archived_payload(tmp_path)
    payload.update(provider=provider, studio_task="interpret", step_mode=True,
                   api_key="test-session-credential" if provider == "openai" else None)

    def approve(event):
        if event["tool"] == "write_report":
            assert fake.completions.calls == 0
        if event["tool"] == "review_report":
            assert fake.completions.calls == 1
            assert not fake.review_prompts
        return "proceed"

    result = run(payload, lambda *a: None, approve=approve)
    assert result["review_decision"] == decision
    assert result["status"] == ("success" if decision == "ACCEPT" else "needs_review")
    assert len(fake.review_prompts) == 1
    assert result["completion"]["interpretation"] == "generated"
    assert "test-session-credential" not in json.dumps(result)
    for path in Path(payload["output_dir"]).rglob("*.json"):
        assert "test-session-credential" not in path.read_text(encoding="utf-8")


def test_rejects_credentials_before_config_is_persisted(tmp_path):
    payload = archived_payload(tmp_path)
    payload["config"] = {"project": {"name": "Tiny"}, "provider": {"api_key": "private"}}
    with pytest.raises(ValueError, match="credentials"):
        run(payload, lambda *a: None)
    assert not Path(payload["output_dir"]).exists()


def test_full_adapter_matches_direct_synthetic_inversion(tmp_path, monkeypatch):
    from test_inversion_smoke import test_small_joint_inversion_writes_loadable_core
    from geosage.existing_results import load_existing_inversion_result
    from simpeg import directives

    # SimPEG's beta estimate uses random power iteration. Fix its seed in this
    # comparison only; neither production path changes its numerical defaults.
    monkeypatch.setattr(directives.PairedBetaEstimate_ByEig, "seed", 42)

    test_small_joint_inversion_writes_loadable_core(tmp_path)
    payload = {
        "request": "Analyze the tiny synthetic survey",
        "output_dir": str(tmp_path / "studio"),
        "config": {
            "project": {"name": "Tiny", "input_dir": str(tmp_path / "input")},
            "region": dict(min_e=500100, max_e=500500, min_n=4300100, max_n=4300500),
            "inversion": {
                "optimization": {"maxGNCG": 1, "maxCG": 10},
                "irls": {"maxIRLSiter": 0},
                "reg_grv_norm": [2, 2, 2, 2],
                "reg_mag_norm": [2, 2, 2, 2],
            },
            "geology": {"mode": "gmm_only", "min_voxels": 1},
            "run": {"execution_mode": "full", "make_plots": False},
        },
    }
    result = run(payload, lambda *a: None)
    assert result["status"] == "needs_review", result["warnings"]
    assert result["source_files_unchanged"]
    direct = load_existing_inversion_result(tmp_path / "output")
    adapted = load_existing_inversion_result(tmp_path / "studio/models")
    for key in ("dens_core_3d", "susc_core_3d"):
        np.testing.assert_allclose(adapted[key], direct[key], rtol=1e-10, atol=1e-12)
    assert "Data fit" in result["exports"]["figures"]
    metadata = json.loads(Path(result['exports']['metadata']).read_text())
    assert set(metadata['fit']) == {'gravity', 'magnetics'}
    assert all(row['residual_definition'] == 'predicted - observed' for row in metadata['fit'].values())
    assert result["iterations"]
    assert result["iterations"][0]["iteration"] >= 1
    assert "data_misfit" in result["iterations"][0]
    continuation = result['continuation']
    assert Path(continuation['source_inversion_dir']) == tmp_path / 'studio/models'
    config = json.loads(Path(continuation['config_file']).read_text())
    assert config['run']['execution_mode'] == 'interpret_existing'
    assert not config['run']['run_inversion']
    assert config['geology']['mode'] == 'reuse_existing_geology'
