"""Keep regression tests offline and independent of local PDF engines/keys."""
import pytest


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEOSAGE_WORKSPACE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("geosage.multi_agent_runner.export_markdown_report_pdf", lambda *a, **kw: None)


@pytest.fixture(scope="session")
def studio_app():
    """Keep one Qt application alive until all desktop test widgets are freed."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
