"""Regression checks for task transitions and truthful desktop delivery."""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
pytest.importorskip('PyHydroGeophysX.agents.assistants')
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent
from geosage.pyhydrogeophysx import ASSISTANT
from geosage.pyhydrogeophysx.configuration import configure
from geosage.pyhydrogeophysx.setup import WorkflowSetup
from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule, OneClickWorker
from PyHydroGeophysX.qt_apps.state import StudioState
from test_studio_integration import archived_payload

@pytest.fixture(scope='module')
def app(studio_app):
    return studio_app

@pytest.fixture
def page(app, tmp_path):
    widget = OneClickModule(StudioState(output_dir=tmp_path / 'project'), lambda *a: None)
    widget.state.ensure_results_store()
    widget.set_assistant(ASSISTANT)
    yield widget
    widget._clock.stop()
    widget._worker = None
    widget._workers.clear()
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(widget, QEvent.DeferredDelete)
    app.processEvents()

def test_inactive_missing_input_does_not_block_archive_task(page, tmp_path, monkeypatch):
    payload = archived_payload(tmp_path)
    page._inputs = dict(payload['inputs'], gravity_file=str(tmp_path / 'removed.csv'))
    page._refresh_inputs()
    started = []
    monkeypatch.setattr(OneClickWorker, 'start', lambda self: started.append(json.loads(self._payload)))
    page._start_offline()
    assert len(started) == 1, page.status.text()
    assert 'gravity_file' not in started[0]['inputs']

def test_hidden_ai_goal_is_not_reused_for_local_task(app):
    setup = WorkflowSetup()
    setup.task.setCurrentIndex(setup.task.findData('interpret'))
    setup.goal.setText('Explain my ore target')
    setup.task.setCurrentIndex(setup.task.findData('inspect'))
    assert 'Explain my ore target' not in setup.request()
    setup.close()


@pytest.mark.parametrize('provider', ['codex_cli', 'claude_code'])
def test_cli_provider_survives_preflight_and_worker_payload(page, tmp_path, monkeypatch, provider):
    if provider not in ASSISTANT.providers:
        pytest.skip('Installed host does not provide CLI backends')
    setup = page._workflow_setup
    setup.task.setCurrentIndex(setup.task.findData('interpret'))
    page._inputs = archived_payload(tmp_path)['inputs']
    page._refresh_inputs()
    page._ai_settings = {'provider': provider, 'model': 'default', 'api_key': None}
    page._request_text = setup.request()
    started = []
    monkeypatch.setattr(OneClickWorker, 'start', lambda self: started.append(json.loads(self._payload)))
    page._start(step_mode=True)
    assert len(started) == 1, page.status.text()
    assert started[0]['provider'] == provider
    assert started[0]['api_key'] is None
    assert started[0]['studio_task'] == 'interpret'
    assert started[0]['step_mode'] is True

def test_ai_task_enables_interpretation_despite_local_config_flags(tmp_path):
    payload = archived_payload(tmp_path)
    payload.update(studio_task='interpret', api_key='test-only', config={
        'project': {'name': 'Demo'}, 'run': {'write_reports': False, 'review_enabled': False}})
    cfg = configure(payload)
    assert cfg['run']['write_reports'] and cfg['run']['review_enabled']

@pytest.mark.parametrize('section', ['project', 'run', 'geology', 'region', 'data', 'inversion'])
def test_malformed_sections_have_actionable_validation_errors(tmp_path, section):
    payload = archived_payload(tmp_path)
    payload['config'] = {'project': {'name': 'Demo'}, section: None}
    with pytest.raises(ValueError, match='object'):
        configure(payload)

def test_save_after_project_change_does_not_crash(page):
    page._current_run_id = 'run-from-a-different-project'
    page._save_current_run()
    assert 'saved' not in page.result_state.text().lower() or 'not saved' in page.result_state.text().lower()
    assert not page.save_result.isEnabled()


def test_upstream_report_preview_opens_the_integrated_results_tab(page, tmp_path):
    report = tmp_path / 'summary.md'
    report.write_text('# Local evidence\nReview these models.', encoding='utf-8')
    page._show_report(report)
    assert page.tabs.currentWidget() is page._result_page
    assert 'Review these models.' in page.report.toPlainText()

def test_completed_run_has_unified_results_and_external_save_stays_in_sync(page, tmp_path):
    from geosage.pyhydrogeophysx.workflow import run
    payload = archived_payload(tmp_path)
    handle = page.begin_persisted_run('unified', label='Synthetic archive')
    page._current_run_id = handle.run_id
    page._output = str(handle.outputs_dir)
    payload.update(output_dir=page._output, studio_task='inspect')
    result = run(payload, lambda *a: None)
    page._succeeded(result)
    assert page.tabs.currentWidget() is page._result_page
    assert page.view_result.isEnabled()
    assert not page.view_fit.isEnabled()
    assert page.read_report.text() == 'Read numerical summary'
    assert page.save_result.isEnabled()
    assert not page.next_step.isHidden()
    assert page.continue_result.isEnabled()
    page.continue_result.click()
    assert page.tabs.currentWidget() is page._data_tab
    assert page._workflow_setup.needs_ai
    assert page._worker is None
    assert page.progress.value() == 0
    assert page.header.headline.text() == 'Next task ready to configure'
    assert page._inputs == {'config_file': result['continuation']['config_file'],
                            'source_inversion_dir': result['continuation']['source_inversion_dir']}
    page.state.results_store.save_run(handle.run_id)
    page._sync_result_storage()
    assert not page.save_result.isEnabled()
    assert page.result_state.text() == 'Saved locally in Project history'
    original_store = page.state.results_store
    page.state.set_results_store(tmp_path / 'different-project')
    page._sync_result_storage()
    assert not page.view_result.isEnabled()
    assert not page.continue_result.isEnabled()
    page.state.results_store = original_store
    page._sync_result_storage()
    assert page.view_result.isEnabled()
    assert page.continue_result.isEnabled()

def test_task_submission_does_not_duplicate_its_ai_goal(page, tmp_path, monkeypatch):
    page._inputs = archived_payload(tmp_path)['inputs']
    page._request_text = 'Explain the target'
    monkeypatch.setattr(page, '_start', lambda **kw: None)
    page.submit_request('Explain the target', {'api_key': 'test-only'})
    assert page._request_text == 'Explain the target'

def test_stop_after_numerical_stage_does_not_erase_its_completion(tmp_path):
    from geosage.pyhydrogeophysx.workflow import run
    payload = archived_payload(tmp_path)
    payload.update(studio_task='inspect', step_mode=True)
    result = run(payload, lambda *a: None,
                 approve=lambda event: 'stop' if event['tool'] == 'build_quasi_geology' else 'proceed')
    assert result['status'] == 'incomplete'
    assert result['completion'] == {'numerical': 'complete', 'interpretation': 'not_run', 'review': 'not_run'}

def test_unknown_duration_does_not_advertise_a_percentage(page):
    page.progress.setRange(0, 0)
    page._on_progress('Joint inversion', .65, 'Iteration 3/20 · data misfit 25', 'joint_inversion')
    assert page._agent_status()['progress_percent'] is None
    assert 'Iteration 3/20' in page.status.text()


def test_compact_setup_reveals_advanced_controls_and_errors(page):
    setup = page._workflow_setup
    assert setup.advanced.isHidden() and page.role.isHidden()
    assert page.follow.isHidden() and page.step_through.isHidden()
    assert not page.tabs.isTabVisible(page.tabs.indexOf(page.details))
    setup.options.click()
    assert not page.role.isHidden() and not page.step_through.isHidden()
    assert page.tabs.isTabVisible(page.tabs.indexOf(page.details))
    setup.options.click()
    setup.show_error('Missing mesh')
    assert setup.options.isChecked() and not setup.advanced.isHidden()
    assert setup.preview.toPlainText() == 'Missing mesh'


def test_task_segments_select_the_correct_input_and_action(page):
    setup = page._workflow_setup
    setup.task_tabs.setCurrentIndex(1)
    assert setup.task.currentData() == 'invert'
    assert page.role.currentData() == 'config_file'
    assert page.run.text() == 'Run inversion'
    setup.task_tabs.setCurrentIndex(2)
    assert setup.needs_ai and page.role.currentData() == 'source_inversion_dir'
    assert page.run.text() == 'Generate interpretation'


def test_workflow_exposes_and_applies_conversational_setup_actions(page, tmp_path):
    folder = tmp_path / 'raw'
    folder.mkdir()
    (folder / 'Demo_gravity_data.csv').write_text(
        'Easting,Northing,ISO\n100,200,1\n300,500,2\n', encoding='utf-8')
    (folder / 'Demo_magnetic_data.csv').write_text(
        'Easting,Northing,TFMA\n150,250,1\n350,450,2\n', encoding='utf-8')
    (folder / 'Demo_gravity_data.csv').write_text(
        'Easting,Northing,Longitude,Latitude,Height,ISO\n'
        '125,225,-123,42,100,1\n300,500,-122.9,42.1,110,2\n', encoding='utf-8')
    (folder / 'Demo_magnetic_data.csv').write_text(
        'Easting,Northing,Longitude,Latitude,TFMA\n'
        '150,250,-123,42,1\n350,450,-122.9,42.1,2\n', encoding='utf-8')
    (folder / 'Demo_mesh.msh').write_text(
        '6 6 2\n50 150 100\n6*50\n6*50\n2*25\n', encoding='utf-8')
    (folder / 'Demo_mesh_core.msh').write_text(
        '4 4 2\n100 200 100\n4*50\n4*50\n2*25\n', encoding='utf-8')
    with rasterio.open(
        folder / 'Demo_topo.tif', 'w', driver='GTiff', height=10, width=10,
        count=1, dtype='float32', crs='EPSG:32610',
        transform=from_origin(0, 600, 60, 60),
    ) as dataset:
        dataset.write(np.ones((10, 10), dtype='float32'), 1)

    actions = {row['name']: row for row in page.agent_describe()['actions']}
    names = set(actions)
    assert {'prepare_inversion_folder', 'set_inversion_parameters',
            'start_confirmed_inversion'} <= names
    public_parameters = set(actions['set_inversion_parameters']['args'])
    assert 'cross_gradient_lambda' not in public_parameters
    assert 'beta_cooling' not in public_parameters
    scanned = page.agent_apply('prepare_inversion_folder', {'path': str(folder)})
    assert scanned['status'] == 'needs_input'
    assert page._inputs['input_dir'] == str(folder.resolve())
    assert {"gravity_file", "magnetic_file", "topography_file", "mesh_file",
            "core_mesh_file"} <= set(page._inputs)
    ready = page.agent_apply('set_inversion_parameters', {
        'min_e': 100, 'max_e': 200, 'min_n': 200, 'max_n': 300,
        'field_strength': 50000, 'inclination': 60, 'declination': 5,
    })
    assert ready['status'] == 'ready_for_confirmation'
    launched = []
    page.startAIRequested.connect(launched.append)
    result = page.agent_apply('start_confirmed_inversion', {'objective': 'Map the target'})
    assert result['launch_requested'] is True
    assert launched == ['Map the target']


def test_chat_inversion_preview_truthfully_shows_ai_report_and_review(page, tmp_path):
    setup = page._workflow_setup
    folder = tmp_path / 'raw'
    folder.mkdir()
    (folder / 'Demo_gravity_data.csv').write_text(
        'Easting,Northing,ISO\n100,200,1\n300,500,2\n', encoding='utf-8')
    (folder / 'Demo_magnetic_data.csv').write_text(
        'Easting,Northing,TFMA\n150,250,1\n350,450,2\n', encoding='utf-8')
    (folder / 'Demo_gravity_data.csv').write_text(
        'Easting,Northing,Longitude,Latitude,Height,ISO\n'
        '125,225,-123,42,100,1\n300,500,-122.9,42.1,110,2\n', encoding='utf-8')
    (folder / 'Demo_magnetic_data.csv').write_text(
        'Easting,Northing,Longitude,Latitude,TFMA\n'
        '150,250,-123,42,1\n350,450,-122.9,42.1,2\n', encoding='utf-8')
    (folder / 'Demo_mesh.msh').write_text(
        '6 6 2\n50 150 100\n6*50\n6*50\n2*25\n', encoding='utf-8')
    (folder / 'Demo_mesh_core.msh').write_text(
        '4 4 2\n100 200 100\n4*50\n4*50\n2*25\n', encoding='utf-8')
    with rasterio.open(
        folder / 'Demo_topo.tif', 'w', driver='GTiff', height=10, width=10,
        count=1, dtype='float32', crs='EPSG:32610',
        transform=from_origin(0, 600, 60, 60),
    ) as dataset:
        dataset.write(np.ones((10, 10), dtype='float32'), 1)
    page.agent_apply('prepare_inversion_folder', {'path': str(folder)})
    page.agent_apply('set_inversion_parameters', {
        'field_strength': 50000, 'inclination': 60, 'declination': 5,
        'max_iterations': 10,
    })
    setup.show_configuration({
        'inputs': dict(setup.inputs), 'request': 'Run and explain the model',
        'output_dir': str(tmp_path / 'runs' / 'preview'), 'studio_task': 'invert',
        'provider': 'openai', 'api_key': 'test-only', 'use_ai': True,
    })
    assert 'Interpretation + independent review' in setup.preview.toPlainText()
    plan = setup.agent_plan.toPlainText()
    assert 'Draft report' in plan and 'AI' in plan


def test_confirmed_inversion_default_request_includes_report_and_review(page, tmp_path):
    setup = page._workflow_setup
    setup._configuration = {'ready': True}
    result = setup.agent_apply('start_confirmed_inversion', {})
    assert result['start_workflow'] is True
    assert 'interpretation report' in result['request']
    assert 'independently review' in result['request']
