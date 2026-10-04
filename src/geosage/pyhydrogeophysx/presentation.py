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
