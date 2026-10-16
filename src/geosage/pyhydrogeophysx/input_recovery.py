"""Bridge GeoSAGE input validation to the host's approved temporary scripts.

Only a failed input preparation can be retried here. Numerical settings,
meshes, reference data and saved inversion results are never repair targets.
The host owns code generation, approval, execution and its one script fix.
"""

from copy import deepcopy
import json
from pathlib import Path


def recover_inputs(ctx, tools, ask, emit):
    """Try one approved format conversion, then validate before continuing.

    Called between controller runs, with no open step. A separate flat context
    exposes selected inputs to the host without changing GeoSAGE's schema.
    """
    from PyHydroGeophysX.agents.runtime.context import RunContext
    from PyHydroGeophysX.agents.runtime.modes import announcement, completion
    from PyHydroGeophysX.agents.runtime.tools import Tool, invoke
    from geosage.survey_io import FILE_SUFFIXES, fingerprint, normalize_table, configuration_crs
    from geosage.mesh_planning import validate_preflight_inputs, configuration_preflight

    failed = [s for s in ctx.steps if s.tool == 'prepare_data' and s.status == 'failed']
    if (not failed or ctx.has('field_data') or ctx.ended == 'stopped'
            or ctx.config['run']['execution_mode'] != 'full'
            or ctx.settings.get('input_recovery_attempted') or not callable(ask)
            or not callable(ctx.settings.get('ask_user'))):
        return False
    ctx.settings['input_recovery_attempted'] = True
    try:
        from PyHydroGeophysX.agents.runtime.recovery import diagnose, script_fix
    except ImportError:
        ctx.note('Input recovery needs a host with Temporary scripts; update PyHydroGeophysX.')
        return False

    original = deepcopy(ctx.config)
    project = original['project']
    sources = {**project.get('input_files', {}), **original.get('studio_inputs', {})}
    files = {role: str(Path(sources.get(role) or Path(project['input_dir']) /
                           f"{project['name']}_{suffix}").resolve())
             for role, suffix in FILE_SUFFIXES.items()}
    record = {'status': 'not_recovered', 'original_error': failed[-1].error}
    ctx.config.setdefault('input_recovery', []).append(record)
    audit_path = Path(ctx.output_dir) / 'input_recovery.json'

    # Share the transcript and questions, not the nested scientific configuration.
    # Every temporary script appears as a normal host activity event.
    class RecoveryContext(RunContext):
        def begin(self, tool, reason='', agent='', description='', purpose=''):
            offered = Tool(tool, purpose, lambda _: None, label=description, agent=agent)
            emit(dict(announcement(ctx, offered, reason), phase='start'))
            return super().begin(tool, reason, agent, description, purpose)

        def finish(self, *args, **kwargs):
            super().finish(*args, **kwargs)
            emit(dict(completion(ctx, self.steps[-1], tools), phase='done'))

    bridge = RecoveryContext(ctx.goal, config={}, output_dir=ctx.output_dir,
                             settings={**ctx.settings, 'assistant_name': 'GeoSAGE'})
    bridge.steps = ctx.steps
    bridge.questions = ctx.questions
    bridge.warnings = ctx.warnings

    def query(prompt, on_text=None):
        contract = (
            '\nGeoSAGE permits a format conversion of gravity_file or magnetic_file only. '
            'Preserve every observation, sign, unit, coordinate, row order and selected column. '
            'Do not invent missing values, remove rows, change physical parameters or modify inputs. '
            'Return give_up if those conditions cannot be met. A replacement must remain readable '
            'with these unchanged import settings and measurement columns:\n'
            + json.dumps({'data_import': original.get('data_import', {}),
                          'data': original.get('data', {})}, default=str))
        return ask(prompt + contract, on_text=on_text)

    try:
        # A source modified since the user's preflight needs fresh confirmation,
        # not a script that quietly legitimises the changed file.
        validate_preflight_inputs(original, files)
        before = [fingerprint(p) for p in files.values()]
        before += list(original.get('setup_preflight', {}).get('input_fingerprints', {}).values())
        record['original_files'] = before
        bridge.config = {r: files[r] for r in ('gravity_file', 'magnetic_file')}
        plan = diagnose(bridge, tools['prepare_data'], failed[-1].error, query)
        record['diagnosis'] = plan
        role = plan.get('replaces')
        if plan.get('action') != 'script' or role not in bridge.config:
            record['reason'] = plan.get('cause') or 'No supported format conversion was proposed.'
            return False

        def retry(name):
            replacement = Path(bridge.config[role]).resolve()
            scratch = (Path(ctx.output_dir) / 'scratch').resolve()
            if not replacement.is_file() or not replacement.is_relative_to(scratch):
                raise ValueError('Replacement must be a file produced in this run\'s scratch folder.')
            if any(fingerprint(item['path']) != item for item in before):
                raise ValueError('Source files changed during conversion; start a fresh preflight.')
            candidate = deepcopy(original)
            candidate.setdefault('studio_inputs', {})[role] = str(replacement)
            candidate['project'].setdefault('input_files', {})[role] = str(replacement)
            # Recompute checks instead of carrying the old input fingerprint over
            # to a new file. Parameters and import options remain unchanged.
            candidate['setup_preflight'] = configuration_preflight(candidate)
            record.update(replacement_role=role, replacement=fingerprint(replacement),
                          preflight=candidate['setup_preflight'])
            measurement = candidate['data']['gravity_column' if role == 'gravity_file' else 'magnetic_column']
            table, metadata = normalize_table(
                replacement, role, candidate.get('data_import', {}).get(role),
                target_crs=configuration_crs(candidate), measurement=measurement,
                component=candidate['data'].get('gravity_component', 'gz'))
            values = table[metadata['measurement_column']]
            preview = dict(role=role, path=str(replacement), rows=len(table),
                           measurement=metadata['measurement_column'],
                           unit=metadata.get('output_measurement_unit', ''),
                           minimum=float(values.min()), maximum=float(values.max()),
                           core_counts=candidate['setup_preflight']['resources']['stations_in_region'])
            record['preview'] = preview
            answer = ctx.ask(
                f'Converted {role}: {replacement.name}\n'
                f'{preview["rows"]} rows; {preview["core_counts"][role]} inside the region.\n'
                f'{preview["measurement"]}: {preview["minimum"]:.6g} to {preview["maximum"]:.6g} {preview["unit"]}.\n'
                f'File: {replacement}\n\n'
                'Check these values against your source. Structural checks passed; '
                'inversion parameters and units are unchanged.',
                [{'id': 'use', 'label': 'Use converted input', 'detail': 'Recheck all inputs and continue this run.'},
                 {'id': 'decline', 'label': 'Do not use it', 'detail': 'Keep the original configuration; no inversion starts.'}],
                default='decline')
            record['adoption_confirmed'] = answer == 'use'
            if answer != 'use':
                return 'failed'
            if any(fingerprint(item['path']) != item for item in before):
                raise ValueError('Source files changed while awaiting confirmation.')
            if fingerprint(replacement) != record['replacement']:
                raise ValueError('Converted input changed after its preview; review a fresh conversion.')
            candidate['input_recovery'] = ctx.config['input_recovery']
            ctx.config = candidate
            ctx.settings['input_stage_directory'] = str(Path(ctx.output_dir) / 'inputs_recovered')
            emit(dict(announcement(ctx, tools[name], 'Validate the approved converted input'), phase='start'))
            status = invoke(ctx, name, 'Validate the approved converted input', tools)
            emit(dict(completion(ctx, ctx.steps[-1], tools), phase='done'))
            ctx.settings.pop('input_stage_directory', None)
            if status != 'ok':
                ctx.config = original
                ctx.config['input_recovery'] = [record]
                return status
            if any(fingerprint(item['path']) != item for item in before):
                raise ValueError('Source files changed during validation; no conversion was adopted.')
            # Preserve both original and converted fingerprints in the run audit.
            manifest = ctx.get('field_data')
            known = {item['path'] for item in manifest['files']}
            for item in before:
                if item['path'] not in known:
                    manifest['files'].append(item)
                    known.add(item['path'])
            manifest['input_recovery'] = record
            return status

        status, note = script_fix(bridge, tools['prepare_data'], failed[-1].error, plan, query, retry)
        record.update(status='recovered' if status == 'ok' else 'not_recovered', detail=note)
        record['script_steps'] = [dict(tool=s.tool, status=s.status, summary=s.summary, error=s.error)
                                  for s in ctx.steps if s.tool == 'temporary_script']
        if status == 'ok':
            manifest_path = Path(ctx.config['project']['input_dir']) / 'import_manifest.json'
            manifest_path.write_text(json.dumps(ctx.get('field_data'), indent=2, ensure_ascii=False), encoding='utf-8')
        if note:
            ctx.note(note)
        return status == 'ok' and ctx.has('field_data')
    except Exception as exc:
        ctx.artifacts.pop('field_data', None)
        ctx.config = original
        ctx.config['input_recovery'] = [record]
        record['status'] = 'not_recovered'
        record['reason'] = f'{type(exc).__name__}: {exc}'
        ctx.note(f'Input conversion did not pass validation: {record["reason"]}')
        return False
    finally:
        ctx.settings.pop('input_stage_directory', None)
        record['script_steps'] = [dict(tool=s.tool, status=s.status, summary=s.summary, error=s.error)
                                  for s in ctx.steps if s.tool == 'temporary_script']
        audit_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
