"""Keep regression tests offline and independent of local PDF engines/keys."""
import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("multi_agent_runner.export_markdown_report_pdf", lambda *a, **kw: None)
