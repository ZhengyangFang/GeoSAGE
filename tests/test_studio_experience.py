"""Task semantics and truthful completion states independent of desktop rendering."""

from pathlib import Path
import pytest

pytest.importorskip('PyHydroGeophysX.agents.assistants')
from geosage.pyhydrogeophysx.workflow import run
from geosage.pyhydrogeophysx.configuration import configure
from test_studio_integration import archived_payload


def test_local_task_ignores_supplied_key_and_does_not_claim_ai_review(tmp_path, monkeypatch):
    from geosage.pyhydrogeophysx.providers import StudioLLM
    monkeypatch.setattr(StudioLLM, 'query', lambda *a, **k: pytest.fail('Local task called AI'))
    payload = archived_payload(tmp_path)
    payload.update(studio_task='inspect', api_key='unused-provider-key')
    events = []
    result = run(payload, lambda *a: None, on_event=events.append)
    assert result['completion'] == {'numerical': 'complete', 'interpretation': 'not_run', 'review': 'NOT_REVIEWED'}
    labels = [e['label'] for e in events if e['phase'] == 'done']
    assert 'Load existing models' in labels
    assert 'Record review status' in labels
    assert 'Review report' not in labels
    assert result['exports']['viewer_fields']['Density contrast (g/cm3)']['limits'] == [-7, 7]
    assert result['exports']['viewer_fields']['Geo ID']['colors']['0'] == '#efeff5'


def test_new_inversion_task_cannot_silently_reuse_archive(tmp_path):
    payload = archived_payload(tmp_path)
    payload['studio_task'] = 'invert'
    with pytest.raises(ValueError, match='New inversion'):
        run(payload, lambda *a: None)
    assert not Path(payload['output_dir']).exists()


def test_ai_task_requires_session_provider_before_writing(tmp_path):
    payload = archived_payload(tmp_path)
    payload['studio_task'] = 'interpret'
    with pytest.raises(ValueError, match='session provider'):
        configure(payload)


def test_setup_preview_does_not_change_source_or_create_output(tmp_path):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from geosage.pyhydrogeophysx.setup import WorkflowSetup
    app = QApplication.instance() or QApplication([])
    payload = archived_payload(tmp_path)
    widget = WorkflowSetup()
    widget.update_inputs(payload['inputs'])
    prepared = widget.prepare_payload(payload)
    assert prepared['studio_task'] == 'inspect'
    widget.refresh_preview()
    assert 'Files ready' in widget.preview.toPlainText()
    assert not Path(payload['output_dir']).exists()
    widget.task.setCurrentIndex(widget.task.findData('invert'))
    assert 'gravity_file' in widget.allowed_roles()
    assert 'source_inversion_dir' not in widget.allowed_roles()
    widget.close()
    app.processEvents()


def test_configuration_editor_preserves_source_and_advanced_settings(tmp_path):
    pytest.importorskip('PySide6')
    import json
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLineEdit
    from geosage.pyhydrogeophysx.setup import WorkflowSetup

    app = QApplication.instance() or QApplication([])
    source = tmp_path / 'configuration.json'
    original = json.dumps({'inversion': {'maxGNCG': 17}})
    source.write_text(original, encoding='utf-8')
    widget = WorkflowSetup()
    widget.task.setCurrentIndex(widget.task.findData('invert'))
    widget.update_inputs({'config_file': str(source)})

    def edit_dialog():
        dialog = app.activeModalWidget()
        values = ('Demo', '0', '10', '0', '20', 'Gz', 'gz', '50000', '60', '0')
        for edit, value in zip(dialog.findChildren(QLineEdit), values):
            edit.setText(value)
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    QTimer.singleShot(0, edit_dialog)
    widget.edit_configuration()
    assert widget._configuration['project']['name'] == 'Demo'
    assert widget._configuration['inversion']['maxGNCG'] == 17
    assert widget._configuration['inversion']['inclination'] == 60
    assert source.read_text(encoding='utf-8') == original
    widget.close()
