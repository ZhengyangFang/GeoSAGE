"""Run planning and durable agent handoffs, independent of the desktop UI."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path


def tools_for(config, use_ai):
    """Use one task-specific registry for the preview and the actual controller."""
    from .tools import TOOLS

    tools = dict(TOOLS)
    if config['run']['execution_mode'] == 'interpret_existing':
        tools['run_joint_inversion'] = replace(tools['run_joint_inversion'], label='Load existing models')
    if not use_ai or not config['run'].get('write_reports', True):
        tools['write_report'] = replace(tools['write_report'], label='Export numerical summary')
        tools['review_report'] = replace(tools['review_report'], label='Record review status')
    return tools


def plan_for(config, use_ai):
    """Expose responsibilities and artifact dependencies without invoking agents."""
    ai_reports = use_ai and config['run'].get('write_reports', True)
    ai_geology = (use_ai and not config['run'].get('reuse_existing_geology') and
                  config.get('geology', {}).get('mode') in {'gmm_bic_auto', 'fixed_units_llm_groups'})
    return [dict(tool=t.name, agent=t.agent, label=t.label, purpose=t.description,
                 requires=list(t.requires), produces=list(t.produces),
                 uses_ai=bool((ai_reports and t.name in {'write_report', 'review_report'}) or
                              (ai_geology and t.name == 'build_quasi_geology')))
            for t in tools_for(config, use_ai).values()]


def _write_json(path, value):
    from geosage.multi_agent_runner import _redact_trace_value

    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(_redact_trace_value(value), indent=2,
                                    ensure_ascii=False, default=str), encoding='utf-8')
    temporary.replace(path)


def checkpoint(ctx, plan):
    """Save completed handoffs; never retry or modify a scientific calculation.

    The continuation is a fresh interpret-existing run, not an optimizer
    checkpoint. Reusable models are offered only after their producing stage
    succeeded and the required on-disk archive is present.
    """
    from geosage.existing_results import REQUIRED_ARTIFACTS

    root = Path(ctx.output_dir)
    continuation = None
    models = ctx.get('property_models') or {}
    source = models.get('run_manifest', {}).get('source_inversion_dir')
    if source and all((Path(source) / p).is_file() for p in REQUIRED_ARTIFACTS.values()):
        config = deepcopy(ctx.config)
        config.pop('studio_inputs', None)
        config.pop('user_request', None)
        config['project'].update(source_inversion_dir=source, input_dir=source,
                                 output_dir=None, interpretation_output_dir=None)
        config['run'].update(execution_mode='interpret_existing', run_inversion=False,
                             overwrite=False)
        # A full run stores its new labels alongside its numerical models.
        # Archive tasks otherwise keep their explicitly selected geology mode.
        if ctx.config['run']['execution_mode'] == 'full' and ctx.has('geo_model'):
            if all((Path(source) / 'geology_models' / p).is_file()
                   for p in ('geo_id_3d.npy', 'unit_id_3d.npy')):
                config['geology']['mode'] = 'reuse_existing_geology'
        config_path = root / 'continue_config.json'
        _write_json(config_path, config)
        continuation = dict(config_file=str(config_path), source_inversion_dir=source,
                            task='interpret', recomputes_inversion=False)
    # Large arrays and credentials never enter the agent handoff log.
    record = dict(schema_version=1, assistant='geosage', planned_stages=plan,
                  completed_steps=[dict(row, tool=step.tool, error=step.error)
                                   for row, step in zip(ctx.plan(), ctx.steps)], ended=ctx.ended,
                  available_artifacts=sorted(ctx.artifacts), continuation=continuation)
    _write_json(root / 'studio_checkpoint.json', record)
    return continuation
