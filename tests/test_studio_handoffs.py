"""Agent handoffs survive later failures without recomputing the inversion."""
from dataclasses import replace
import json
from pathlib import Path
import pytest

pytest.importorskip('PyHydroGeophysX.agents.assistants')
from test_studio_integration import archived_payload
from geosage.pyhydrogeophysx.workflow import run
from geosage.pyhydrogeophysx.configuration import configure
from geosage.pyhydrogeophysx.lifecycle import plan_for


def test_plan_matches_execution_and_exposes_ai_geology(tmp_path):
    payload = archived_payload(tmp_path)
    config = configure(payload)
    assert not any(row['uses_ai'] for row in plan_for(config, False))
    config['geology']['mode'] = 'fixed_units_llm_groups'
    assert [r['tool'] for r in plan_for(config, True) if r['uses_ai']] == [
        'build_quasi_geology', 'write_report', 'review_report']
    result = run(payload, lambda *a: None)
    assert [r['label'] for r in result['planned_stages']] == [r['step'] for r in result['execution_plan']]
    report = Path(result['report_files']['report_markdown']).read_text(encoding='utf-8')
    assert '| Density contrast (g/cm³) | 0 to 7 |' in report
    assert '```json' not in report
    assert json.loads((Path(payload['output_dir']) / 'numerical_summary.json').read_text()) == json.loads(json.dumps(result['summary']))


def test_numerical_overview_escapes_labels_and_does_not_invent_missing_values():
    from geosage.pyhydrogeophysx.presentation import numerical_overview
    text = '\n'.join(numerical_overview({'geology': {'geo_groups': [
        {'geo_id': 7, 'name': '<script>|name\nnext', 'voxel_count': 0}]}}))
    assert 'Not recorded' in text
    assert '<script>' not in text and '&#124;' in text
    assert '| 0 |' in text


def test_report_failure_keeps_checkpoint_and_reusable_models(tmp_path, monkeypatch):
    from geosage.pyhydrogeophysx.tools import TOOLS
    payload = archived_payload(tmp_path)
    payload.update(studio_task='interpret', api_key='session-test-key')
    seen = []

    def fail_report(ctx):
        saved = json.loads((Path(ctx.output_dir) / 'studio_checkpoint.json').read_text())
        assert 'geo_model' in saved['available_artifacts']
        assert Path(saved['continuation']['config_file']).is_file()
        seen.append(True)
        raise RuntimeError('Simulated provider unavailable')

    monkeypatch.setitem(TOOLS, 'write_report', replace(TOOLS['write_report'], handler=fail_report))
    result = run(payload, lambda *a: None)
    assert seen and result['status'] == 'incomplete'
    assert result['completion']['numerical'] == 'complete'
    assert result['completion']['interpretation'] == 'not_run'
    continuation = result['continuation']
    assert continuation and not continuation['recomputes_inversion']
    saved = json.loads((Path(payload['output_dir']) / 'studio_checkpoint.json').read_text())
    assert saved['completed_steps'][-1]['error'] == 'RuntimeError: Simulated provider unavailable'
    for path in (Path(payload['output_dir']) / 'studio_checkpoint.json', Path(continuation['config_file'])):
        assert 'session-test-key' not in path.read_text()
    monkeypatch.undo()
    import geosage.runner as runner
    monkeypatch.setattr(runner, 'run_joint_inversion', lambda **kw: pytest.fail('Unexpected inversion'))
    resumed = run(dict(request='Inspect recovered numerical results', studio_task='inspect',
                       inputs={'config_file': continuation['config_file']},
                       output_dir=str(tmp_path / 'continued')), lambda *a: None)
    assert resumed['completion']['numerical'] == 'complete'
    assert resumed['source_files_unchanged']
    import pyvista as pv
    import numpy as np
    first, second = pv.read(result['exports']['model']), pv.read(resumed['exports']['model'])
    for field in first.cell_data:
        np.testing.assert_array_equal(first.cell_data[field], second.cell_data[field])


def test_stop_before_models_offers_no_continuation(tmp_path):
    payload = archived_payload(tmp_path)
    payload['step_mode'] = True
    result = run(payload, lambda *a: None, approve=lambda e: 'stop')
    assert result['continuation'] is None
    saved = json.loads((Path(payload['output_dir']) / 'studio_checkpoint.json').read_text())
    assert saved['ended'] == 'stopped' and saved['continuation'] is None
    assert not (Path(payload['output_dir']) / 'continue_config.json').exists()


def test_checkpoint_write_error_does_not_discard_models(tmp_path, monkeypatch):
    import geosage.pyhydrogeophysx.lifecycle as lifecycle
    def unavailable(*args):
        raise OSError('Read-only checkpoint')
    monkeypatch.setattr(lifecycle, '_write_json', unavailable)
    result = run(archived_payload(tmp_path), lambda *a: None)
    assert result['completion']['numerical'] == 'complete'
    assert result['continuation'] is None
    assert any('continuation record' in warning for warning in result['warnings'])
