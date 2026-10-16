"""Small, verified report handoffs. Never serialize or resume an optimizer."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil


def fingerprint(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}


def _safe(value):
    from geosage.existing_results import _json_safe
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    return _json_safe(value)


def _write(path, value):
    from geosage.multi_agent_runner import _redact_trace_value
    data = _redact_trace_value(_safe(value))
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def _paths(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _paths(item)
    elif isinstance(value, (str, Path)) and value and Path(value).is_file():
        yield Path(value).resolve()


def save_handoff(ctx, draft=None):
    """Persist only report inputs, not model arrays or provider credentials."""
    root = Path(ctx.output_dir)
    geo = ctx.get('geo_model')
    prepared = deepcopy({k: v for k, v in geo['prepared'].items()
                         if k not in {'geo_result', 'inversion_result'}})
    prepared['geo_result'] = {k: geo['prepared']['geo_result'][k] for k in
        ('paths', 'unit_defs', 'unit_to_geo', 'grouping_adjustments') if k in geo['prepared']['geo_result']}
    prepared['inversion_result'] = {k: geo['prepared']['inversion_result'][k] for k in
        ('paths', 'inversion_parameters', 'iteration_history') if k in geo['prepared']['inversion_result']}
    workflow = geo['workflow']
    small_workflow = {k: workflow[k] for k in ('source_manifest', 'run_manifest', 'interpretation_output_dir') if k in workflow}
    from geosage.existing_results import REQUIRED_ARTIFACTS, OPTIONAL_ARTIFACTS
    source = Path(workflow['run_manifest']['source_inversion_dir'])
    for relative in REQUIRED_ARTIFACTS.values():
        if not (source / relative).is_file():
            raise FileNotFoundError(f'Saved numerical model is missing: {source / relative}')
    paths = set(_paths(prepared['geo_result'].get('paths', {})))
    paths.update(_paths(prepared['inversion_result'].get('paths', {})))
    paths.update(_paths(geo['exports']))
    paths.add(Path(prepared['evidence_path']).resolve())
    paths.update((source / p).resolve() for p in [*REQUIRED_ARTIFACTS.values(), *OPTIONAL_ARTIFACTS.values()] if (source / p).is_file())
    record = dict(schema_version=1, config=ctx.config, goal=ctx.goal, prepared=prepared,
                  workflow=small_workflow, exports=geo['exports'],
                  files=[fingerprint(p) for p in sorted(paths)], draft=None)
    if draft and not draft.get('offline'):
        record['draft'] = {k: v for k, v in draft.items() if k != 'prepared'}
        record['files'] += [fingerprint(draft['draft_path'])]
        report_dir = Path(draft['draft_path']).parent
        record['files'] += [fingerprint(report_dir / p) for p in draft['allowed_figure_paths']]
    path = root / 'report_handoff.json'
    _write(path, record)
    # Reuse the host's existing continuation control and normal workflow worker.
    config = deepcopy(ctx.config)
    config.pop('studio_inputs', None)
    config.pop('user_request', None)
    config['project'].update(input_dir=str(source), source_inversion_dir=str(source),
                             output_dir=None, interpretation_output_dir=None)
    config['run'].update(execution_mode='interpret_existing', run_inversion=False)
    config['report_recovery'] = fingerprint(path)
    _write(root / 'retry_report_config.json', config)
    ctx.settings['report_handoff'] = fingerprint(path)
    return path


def read_handoff(reference):
    path = Path(reference['path']).resolve()
    if fingerprint(path)['sha256'] != reference['sha256']:
        raise ValueError('Report handoff changed; use a fresh interpretation of the models.')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema_version') != 1:
        raise ValueError('Unsupported report handoff version.')
    for old in record['files']:
        if fingerprint(old['path'])['sha256'] != old['sha256']:
            raise ValueError(f'Report source changed: {old["path"]}. Rebuild the interpretation before retrying.')
    return record


def restore(ctx, record):
    """Seed only the two report tools; completed numerical stages cannot run."""
    record = deepcopy(record)
    prepared = record['prepared']
    destination = Path(ctx.output_dir) / 'interpretation'
    destination.mkdir(parents=True, exist_ok=True)
    prepared['output_dir'] = destination
    prepared['evidence_path'] = destination / 'evidence_bundle.json'
    _write(prepared['evidence_path'], prepared['evidence'])
    workflow = record['workflow']
    workflow['interpretation_output_dir'] = str(destination)
    workflow['inversion_result'] = prepared['inversion_result']
    workflow['geology_result'] = prepared['geo_result']
    workflow['config'] = ctx.config
    ctx.put('field_data', {'files': record['files'], 'reused_report_evidence': True})
    ctx.put('property_models', workflow)
    ctx.put('geo_model', dict(workflow=workflow, prepared=prepared, exports=record['exports']))
    draft = record['draft']
    if draft:
        old_dir = Path(draft['draft_path']).parent
        report_dir = destination / 'reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        for name in draft['allowed_figure_paths']:
            source = (old_dir / name).resolve()
            target = (report_dir / name).resolve()
            if not source.is_relative_to(old_dir.resolve()) or not target.is_relative_to(report_dir.resolve()):
                raise ValueError('Report image must be local to its report directory.')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        draft['draft_path'] = report_dir / 'draft_report.md'
        draft['draft'] = append_record(draft['draft'], prepared, ctx.config, report_dir)
        audit_images(draft['draft'], report_dir, set(draft['allowed_figure_paths']))
        draft['draft_path'].write_text(draft['draft'], encoding='utf-8')
        draft['prepared'] = prepared
        draft['allowed_figure_paths'] = set(draft['allowed_figure_paths'])
        ctx.put('draft_report', draft)
    ctx.note('Reused saved report evidence and figures; no inversion or geological regrouping was run.')


def append_record(text, prepared, cfg, report_dir):
    """Attach a deterministic numerical record; not a claim to verify LLM prose."""
    marker = '<!-- geosage-run-record -->'
    end_marker = '<!-- /geosage-run-record -->'
    appendix = ('\n## Recorded run evidence\n\n'
        '[Actual parameters, numerical summary and provenance](run_record.json)\n\n'
        'This record is generated from the saved calculation. Interpretive claims still require scientific review.\n')
    conversions = cfg.get('input_recovery', [])
    if conversions:
        recovered = sum(item.get('status') == 'recovered' for item in conversions)
        appendix += (f'\nInput format recovery: {recovered} validated conversion(s) adopted after user confirmation. '
                     'The run record includes original and converted file fingerprints, validation results '
                     'and temporary-script outcomes. This is not a claim of scientific equivalence.\n')
    import re
    # Replace only our own block; revisions may add conclusions or images after it.
    text = re.sub(re.escape(marker) + r'.*?' + re.escape(end_marker), '', text, flags=re.DOTALL)
    text = text.replace(marker + appendix, '').rstrip()  # Earlier unbounded appendix.
    record = {'inversion_parameters': prepared['inversion_result'].get('inversion_parameters', {}),
              'result_summary': prepared['result_summary'],
              'effective_configuration': cfg,
              'coordinate_reference': prepared['evidence'].get('coordinate_reference', {}),
              'parameter_sources': cfg.get('parameter_sources', {})}
    _write(Path(report_dir) / 'run_record.json', record)
    return text + '\n\n' + marker + appendix + end_marker + '\n'


def report_context(prepared, cfg, report_dir, allowed):
    """Give drafting, review and revision the same locally verified evidence.

    Configuration describes settings, not proof of completed iterations. Counts
    come from saved observation rows, never from a filename or its byte size.
    """
    import numpy as np
    from geosage.multi_agent_runner import _redact_trace_value

    root = Path(report_dir).resolve()
    context = {
        'effective_configuration': cfg,
        'inversion_parameters': prepared['inversion_result'].get('inversion_parameters', {}),
        'iteration_history': prepared['inversion_result'].get('iteration_history', []),
        'coordinate_reference': prepared['evidence'].get('coordinate_reference', {}),
        'target_depth_audit': prepared['evidence'].get('target_depth_audit', {}),
        'unit_definitions': prepared['geo_result'].get('unit_defs', {}),
        'unit_to_geo': prepared['geo_result'].get('unit_to_geo', {}),
        'observations': {}, 'verified_figures': [],
        'scope': 'Settings are not execution history. Figure verification confirms local files, not scientific claims.',
    }
    for name in sorted(allowed):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Report figure is missing or outside its directory: {name}')
        context['verified_figures'].append({'relative_path': name, **fingerprint(path)})
    paths = prepared['inversion_result'].get('paths', {})
    for kind in ('gravity', 'magnetics'):
        source = paths.get(f'obs_{kind}_ubc')
        if not source:
            continue
        try:
            table = np.loadtxt(source, ndmin=2)
            if table.shape[1] < 5 or not np.all(np.isfinite(table)):
                raise ValueError('Expected finite x/y/z/value/uncertainty rows.')
            context['observations'][kind] = {'count': int(table.shape[0]), **fingerprint(source)}
        except (OSError, ValueError) as exc:
            context['observations'][kind] = {'available': False, 'reason': str(exc)}
    record = root / 'run_record.json'
    if record.is_file():
        context['generated_run_record'] = fingerprint(record)
    return _redact_trace_value(_safe(context))


def audit_images(text, report_dir, allowed):
    import re
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    files = []
    for ref in refs:
        if ref not in allowed or not (Path(report_dir) / ref).is_file():
            raise ValueError(f'Report image is missing or outside this run: {ref}')
        files.append(fingerprint(Path(report_dir) / ref))
    _write(Path(report_dir) / 'report_consistency.json', {
        'status': 'passed', 'scope': 'local image references and recorded numerical evidence; not LLM prose verification',
        'images': files, 'run_record': fingerprint(Path(report_dir) / 'run_record.json')})
