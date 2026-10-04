"""Failure-path checks for the optional Studio adapter."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PyHydroGeophysX.agents.assistants")

from geosage.pyhydrogeophysx.configuration import configure
from geosage.pyhydrogeophysx.providers import StudioLLM
from geosage.pyhydrogeophysx.workflow import run
from test_studio_integration import archived_payload


@pytest.mark.parametrize("role", ["reference_file", "unit_defs_file", "config_file", "gravity_file"])
def test_output_must_not_contain_separately_selected_inputs(tmp_path, role):
    payload = archived_payload(tmp_path)
    output = Path(payload["output_dir"])
    output.mkdir()
    prior = output / "evidence_summary.md"
    content = '{"project": {"name": "Tiny"}}' if role == "config_file" else "private original"
    prior.write_text(content)
    payload["inputs"][role] = str(prior)
    with pytest.raises(ValueError, match="must not contain"):
        configure(payload)
    assert prior.read_text() == content
    assert not (output / ".geosage-run").exists()


def test_output_must_not_contain_source_folder(tmp_path):
    payload = archived_payload(tmp_path)
    payload["output_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        configure(payload)


def test_output_must_not_contain_configured_labels(tmp_path):
    payload = archived_payload(tmp_path)
    payload["config"] = {"project": {"name": "Tiny"}, "geology": {"unit_id_npy": str(Path(payload["output_dir"]) / "labels.npy")}}
    with pytest.raises(ValueError, match="must not contain"):
        configure(payload)


def test_existing_unrelated_output_is_not_overwritten(tmp_path):
    payload = archived_payload(tmp_path)
    output = Path(payload["output_dir"])
    output.mkdir()
    previous = output / "evidence_summary.md"
    previous.write_text("previous report")
    with pytest.raises(FileExistsError, match="already contains"):
        run(payload, lambda *a: None)
    assert previous.read_text() == "previous report"


def test_host_bootstrap_files_and_final_snapshot(tmp_path):
    payload = archived_payload(tmp_path)
    output = Path(payload["output_dir"])
    output.mkdir()
    for name in ("UNSAVED", "activity.log", "steering.jsonl"):
        (output / name).touch()

    def disconnected_display(label, *args):
        if label == "GeoSAGE complete":
            raise RuntimeError("display disconnected")

    result = run(payload, disconnected_display)
    saved = json.loads((output / "interpretation/effective_config.json").read_text())
    assert saved == result["workflow_config"]
    assert saved["run"]["run_geology_model"] is True
    assert saved["run"]["review_enabled"] is False
    assert result["status"] == "needs_review"


def test_stopped_before_inventory_does_not_claim_verified_sources(tmp_path):
    payload = archived_payload(tmp_path)
    payload["step_mode"] = True
    result = run(payload, lambda *a: None, approve=lambda e: "stop")
    assert result["source_files_unchanged"] is None
    assert result["status"] == "incomplete"


@pytest.mark.parametrize("response", [None, "", "   "])
def test_empty_provider_text_is_a_failure(response):
    llm = StudioLLM({"api_key": "session-secret"})
    llm._agent = SimpleNamespace(query_llm=lambda *a, **kw: response)
    with pytest.raises(RuntimeError, match="empty"):
        llm.query("write report")


def test_provider_exception_does_not_persist_session_key():
    llm = StudioLLM({"api_key": "session-secret"})

    def fail(*args, **kwargs):
        raise RuntimeError("Invalid credential: session-secret")

    llm._agent = SimpleNamespace(query_llm=fail)
    with pytest.raises(RuntimeError, match="REDACTED") as caught:
        llm.query("write report")
    assert "session-secret" not in str(caught.value)


def test_wrong_role_type_is_rejected_early(tmp_path):
    payload = archived_payload(tmp_path)
    payload["inputs"]["reference_file"] = str(tmp_path)
    with pytest.raises(ValueError, match="must name a file"):
        configure(payload)
