"""Use the host CLI bridge without real logins, subprocesses or model calls."""

import json

import pytest

pytest.importorskip("PyHydroGeophysX.llm.cli_providers")

from PyHydroGeophysX.llm import cli_providers as cli
from PyHydroGeophysX.llm.runtime_options import reasoning_effort

from geosage.pyhydrogeophysx import ASSISTANT
from geosage.pyhydrogeophysx.providers import StudioLLM, ai_enabled
from geosage.pyhydrogeophysx.workflow import run
from test_studio_integration import archived_payload


@pytest.mark.parametrize("provider", ["codex_cli", "claude_code"])
def test_report_backend_uses_host_cli_without_key(monkeypatch, provider):
    calls = []

    def complete(self, system, messages, specs, max_tokens=0):
        calls.append((self.id, self.model, self.reasoning_effort, messages, specs))
        assert self._api_key is None
        return {"content": '{"answer": "Geological evidence"}', "tool_calls": []}

    monkeypatch.setattr(cli.CliProvider, "complete", complete)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    backend = StudioLLM({"llm_provider": provider, "model": "default"})
    assert backend.api_key is None
    assert provider in ASSISTANT.providers
    token = reasoning_effort.set("high")
    try:
        assert backend.chat_json("Use evidence", "Explain the model") == {"answer": "Geological evidence"}
    finally:
        reasoning_effort.reset(token)
    assert calls[0][:3] == (provider, "default", "high")
    assert calls[0][4] == []  # Report backends do not execute CLI tools.


@pytest.mark.parametrize("provider", ["codex_cli", "claude_code"])
@pytest.mark.parametrize("task", ["inspect", "invert"])
def test_local_task_never_enables_cli(provider, task):
    settings = {"provider": provider, "studio_task": task, "api_key": "unused-key"}
    assert not ai_enabled(settings)
    with pytest.raises(RuntimeError, match="disabled"):
        StudioLLM(settings).query("Do not call a model")


@pytest.mark.parametrize("provider", ["codex_cli", "claude_code"])
def test_inspection_stays_offline_with_cli_selected(tmp_path, monkeypatch, provider):
    monkeypatch.setattr(cli.CliProvider, "complete", lambda *a, **kw: pytest.fail("Local task called CLI"))
    payload = archived_payload(tmp_path)
    payload.update(provider=provider, studio_task="inspect")
    result = run(payload, lambda *a: None)
    assert result["source_files_unchanged"] is True
    assert result["completion"]["interpretation"] == "not_run"
    assert result["review_decision"] == "NOT_REVIEWED"


@pytest.mark.parametrize("provider", ["codex_cli", "claude_code"])
def test_cli_failures_remain_actionable(monkeypatch, provider):
    def fail(*args, **kwargs):
        raise RuntimeError("Not logged in; sign in through Assistant settings")

    monkeypatch.setattr(cli.CliProvider, "complete", fail)
    with pytest.raises(RuntimeError, match="Not logged in"):
        StudioLLM({"provider": provider}).query("Explain the model")


@pytest.mark.parametrize("provider", ["codex_cli", "claude_code"])
def test_cli_chat_supports_auto_and_follow_up_turns(studio_app, monkeypatch, provider):
    from PySide6.QtCore import QEvent
    from PyHydroGeophysX.llm.providers import make_provider
    from PyHydroGeophysX.qt_apps.agent import chat_panel
    from PyHydroGeophysX.qt_apps.agent.controller import StudioController

    monkeypatch.setattr(chat_panel.assistant_registry, "active", lambda: ASSISTANT)
    monkeypatch.setattr(cli.CliProvider, "available", lambda self: (True, "Test CLI"))
    monkeypatch.setattr(chat_panel, "prewarm", lambda *a: None)

    def ready(self, backend, check_only=False):
        self._cli_states[backend.id] = {"state": "ready", "message": "Test login"}

    monkeypatch.setattr(chat_panel.AssistantChatPanel, "_start_cli_setup", ready)
    panel = chat_panel.AssistantChatPanel(StudioController(None), provider=make_provider(provider))
    started, histories = [], []
    monkeypatch.setattr(panel._controller, "run_to_report", lambda text, settings, *a, **kw:
                        started.append(settings) or "Workflow started.")
    try:
        panel._cli_states[provider]["state"] = "login"
        panel.start_workflow("Interpret these results")
        assert not started, "Task button must wait for CLI login, like the Send button"
        assert panel._settings_btn.isChecked()
        panel._cli_states[provider]["state"] = "ready"
        panel.start_workflow("Interpret these results")
        assert started[0]["provider"] == provider
        assert started[0]["api_key"] is None
        panel._on_workflow_finished("Numerical evidence available")
        panel._execution_mode.setCurrentIndex(panel._execution_mode.findData("guided"))
        monkeypatch.setattr(panel, "_start_request", lambda: histories.append(list(panel._messages)))
        panel._input.setPlainText("Explain the evidence")
        panel._on_send()
        panel._on_llm_ok({"content": "Inspect the density model next", "tool_calls": []})
        panel._input.setPlainText("Why that model?")
        panel._on_send()
        assert histories[-1][-1]["content"] == "Why that model?"
        assert any(m.get("content") == "Inspect the density model next" for m in histories[-1])
        panel._on_llm_ok({"content": "It contains the relevant anomaly", "tool_calls": []})
    finally:
        panel.close()
        panel.deleteLater()
        studio_app.sendPostedEvents(panel, QEvent.DeferredDelete)


@pytest.mark.parametrize("provider", ["codex_cli", "claude_code"])
def test_automatic_cli_workflow_has_an_ai_controller(tmp_path, monkeypatch, provider):
    from PyHydroGeophysX.agents.runtime import entry

    calls = []

    def drive(ctx, **kwargs):
        assert kwargs["ask"] is not None
        assert kwargs["on_step"] is None
        calls.append(kwargs["ask"]("Plan this run"))
        # Stop before scientific stages; report/review and approvals have their
        # own full workflow regression in test_studio_integration.
        ctx.ended = "stopped"

    monkeypatch.setattr(entry, "drive", drive)
    monkeypatch.setattr(cli.CliProvider, "complete", lambda *a, **kw: {
        "content": json.dumps({"tool": "prepare_data", "why": "Read inputs first"}),
        "tool_calls": [],
    })
    payload = archived_payload(tmp_path)
    payload.update(provider=provider, studio_task="interpret", step_mode=False)
    result = run(payload, lambda *a: None)
    assert json.loads(calls[0])["tool"] == "prepare_data"
    assert result["status"] == "incomplete"
    assert result["completion"]["interpretation"] == "not_run"
