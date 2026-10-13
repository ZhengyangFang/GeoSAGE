"""Readable numerical evidence without interpreting geological significance."""
from html import escape
import math


def _cell(value):
    return escape(str(value)).replace('|', '&#124;').replace('\n', ' ').replace('\r', ' ')


def _number(value):
    try:
        number = float(value)
        return f'{number:.6g}' if math.isfinite(number) else 'Not recorded'
    except (ValueError, TypeError):
        return 'Not recorded'


def numerical_overview(summary):
    """Markdown overview; unavailable values remain explicitly unavailable."""
    inv = summary.get('inversion') or {}
    geo = summary.get('geology') or {}
    shape = inv.get('mesh_core_shape')
    mesh = ' × '.join(_cell(n) for n in shape) if shape else 'Not recorded'
    lines = ['## Model overview', '', '| Quantity | Recorded value |', '| --- | --- |',
             f'| Core mesh (cells along x × y × z) | {mesh} |',
             f'| Density contrast (g/cm³) | {_number(inv.get("dens_min"))} to {_number(inv.get("dens_max"))} |',
             f'| Susceptibility (SI) | {_number(inv.get("susc_min"))} to {_number(inv.get("susc_max"))} |',
             f'| Geological groups | {_number(geo.get("n_geo_groups"))} |', '',
             'Ranges describe the recovered model; they are not uncertainty bounds.', '']
    if geo.get('geo_groups'):
        lines += ['## Recorded geological labels', '', '| ID | Name | Cells |', '| --- | --- | --- |']
        for group in geo['geo_groups']:
            lines.append(f'| {_cell(group.get("geo_id", "—"))} | {_cell(group.get("name") or "Not recorded")} | {_number(group.get("voxel_count"))} |')
        lines.append('')
    lines += ['[Full numerical record](numerical_summary.json)', '']
    return lines


def comparison_overview(result):
    """Compact comparison in the existing details card, with no new workspace."""
    def table(headers, rows):
        return '<table cellspacing="6"><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr>' + ''.join(
            '<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>' for row in rows) + '</table>'
    from pathlib import Path
    html = f'<b>{escape(result.get("left_label") or Path(result["left"]).name)} → {escape(result.get("right_label") or Path(result["right"]).name)}</b>'
    rows = []
    for kind, fit in result['fits'].items():
        rows.append((kind.title(), _number((fit['left'] or {}).get('rmse')) + ' ' + (fit['left'] or {}).get('units',''),
                     _number((fit['right'] or {}).get('rmse')) + ' ' + (fit['right'] or {}).get('units',''),
                     'Same observations' if fit['comparable'] else 'Not directly comparable'))
    html += table(('Data fit', 'Left RMSE', 'Right RMSE', 'Comparison'), rows)
    rows = [(name.replace('_core', '').title(), _number(model.get('rms_change')), model['units'])
            for name, model in result['models'].items()]
    html += table(('Model', 'RMS change', 'Unit'), rows)
    if result['parameter_differences']:
        html += '<b>Changed recorded parameters</b>' + table(('Parameter', 'Left', 'Right'),
            [(key, value['left'], value['right']) for key, value in result['parameter_differences'].items()])
    html += ''.join(f'<p>{escape(note)}</p>' for note in result['limitations'])
    return html
