"""Read-only survey diagnostics and bounded preview arrays shared with Studio.

Checks describe observations, not whether a geological anomaly is physically wrong.
Nothing here repairs, interpolates or discards scientific input files.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .survey_io import normalize_table, _factor


def analyze_table(table, column=None, region=None):
    """Full-table counts with source record IDs, independent of plot subsampling."""
    ids = table.index.to_numpy(dtype=int) + 1
    xy = table[['Easting', 'Northing']].to_numpy(dtype=float)
    findings = []

    def finding(code, mask, message):
        rows = ids[np.asarray(mask)]
        if rows.size:
            findings.append(dict(code=code, severity='info' if code == 'outside_region' else 'review', count=int(rows.size),
                                 data_rows=rows[:40].tolist(), rows_truncated=rows.size > 40, message=message))

    duplicate = table.duplicated(['Easting', 'Northing'], keep=False).to_numpy()
    finding('coincident_positions', duplicate,
            'Coincident horizontal positions; these may be repeat or different-height observations. Keep unless reviewed.')
    inside = np.ones(len(table), dtype=bool)
    if region:
        inside = ((xy[:, 0] > region['min_e']) & (xy[:, 0] < region['max_e']) &
                  (xy[:, 1] > region['min_n']) & (xy[:, 1] < region['max_n']))
        finding('outside_region', ~inside, 'Outside the selected inversion region; the solver will not use these rows.')
    values = table[column].to_numpy(dtype=float) if column else None
    stats = None
    if values is not None:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values-median)))
        stats = dict(minimum=float(values.min()), maximum=float(values.max()), median=median,
                     p02=float(np.percentile(values, 2)), p98=float(np.percentile(values, 98)))
        if np.ptp(values) == 0:
            finding('constant_values', np.ones(len(table), dtype=bool), 'All selected measurements are equal; verify the selected column.')
        elif len(values) >= 8 and mad > 0:
            finding('extreme_values', np.abs(values-median) > 6 * 1.4826 * mad,
                    'Beyond six robust deviations from the median (MAD × 1.4826). Review; genuine anomalies can be extreme.')
    if len(table) > 2 and np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
        finding('line_geometry', np.ones(len(table), dtype=bool), 'Stations form a line or a single location; cross-line structure may be weakly constrained.')
    return dict(rows=len(table), rows_in_region=int(inside.sum()), column=column,
                statistics=stats, findings=findings,
                row_numbering='1-based source data records after header/skiprows; not physical text-file line numbers')


def _terrain_preview(inspection, target_crs, region):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    from pyproj import Transformer

    path = inspection.get('files', {}).get('topography_file')
    if not path:
        return None
    options = inspection.get('data_import', {}).get('topography_file', {})
    with rasterio.open(path) as src:
        crs = src.crs or options.get('crs')
        if not crs or not target_crs:
            raise ValueError('Terrain overlay needs known terrain and survey coordinate systems.')
        band = options.get('band', 1)
        if type(band) is not int or not 1 <= band <= src.count or (src.count > 1 and 'band' not in options):
            raise ValueError('Choose the terrain elevation band before previewing.')
        window = Window(0, 0, src.width, src.height)
        if region and not (src.transform.b or src.transform.d):
            bounds = transform_bounds(target_crs, crs, region['min_e'], region['min_n'],
                                      region['max_e'], region['max_n'], densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform).intersection(window)
        ratio = min(1., 160 / max(window.width, window.height))
        shape = (max(1, int(window.height*ratio)), max(1, int(window.width*ratio)))
        values = src.read(band, window=window, out_shape=shape, masked=True,
                          resampling=Resampling.nearest).astype(float).filled(np.nan)
        values *= _factor(options.get('height_unit', 'm'), 'length')
        affine = src.window_transform(window) * src.transform.scale(window.width/shape[1], window.height/shape[0])
        cols, rows = np.meshgrid(np.arange(shape[1]+1), np.arange(shape[0]+1))
        x = affine.a * cols + affine.b * rows + affine.c
        y = affine.d * cols + affine.e * rows + affine.f
        x, y = Transformer.from_crs(crs, target_crs, always_xy=True).transform(x, y, errcheck=True)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Terrain transformation produced invalid coordinates.')
        return dict(x=np.asarray(x), y=np.asarray(y), values=values,
                    valid_fraction_sampled=float(np.isfinite(values).mean()),
                    sample_shape=list(shape), sampling='nearest; overview only, not a full-resolution coverage check')


def build_survey_preview(inspection, parameters=None):
    """Partial previews survive missing inputs; failures remain explicit in diagnostics."""
    parameters = parameters or {}
    settings = inspection.get('data_import', {})
    region = dict(inspection.get('detected', {}).get('region') or {})
    region.update({key: parameters[key] for key in ('min_e', 'max_e', 'min_n', 'max_n') if parameters.get(key) is not None})
    if len(region) != 4:
        region = None
    result = dict(surveys={}, terrain=None, region=region,
                  mesh=inspection.get('mesh', {}).get('bounds'), findings=[],
                  source_files_read_only=True, coordinate_assumption=None)
    result['findings'].extend(dict(role=issue['role'], severity='error', message=issue['message'])
                              for issue in inspection.get('validation_issues', []))
    result['findings'].extend(dict(role=role, severity='action', message='Input missing or ambiguous; resolve before inversion.')
                              for role in inspection.get('missing_files', []))
    for role, column_key in (('gravity_file', 'gravity_column'), ('magnetic_file', 'magnetic_column')):
        path = inspection.get('files', {}).get(role)
        if not path:
            continue
        column = parameters.get(column_key) or inspection.get('detected', {}).get(column_key)
        options = deepcopy(settings.get(role, {}))
        # Diagnostic omission only. The actual run keeps the user's missing_policy.
        options['missing_policy'] = 'drop'
        try:
            table, meta = normalize_table(path, role, options, target_crs=settings.get('projected_crs'),
                                          measurement=column, component=parameters.get('gravity_component'))
            quality = analyze_table(table, meta.get('measurement_column'), region)
            missing = meta['dropped_data_rows']
            if missing:
                quality['findings'].insert(0, dict(code='invalid_required_values', severity='review' if settings.get(role, {}).get('missing_policy') == 'drop' else 'error',
                    count=len(missing), data_rows=missing[:40], rows_truncated=len(missing)>40,
                    message='Invalid required values omitted from this preview. Run-time exclusion requires an explicit import setting.'))
            if not column:
                quality['findings'].append(dict(code='choose_measurement', severity='action', count=0, data_rows=[],
                                                message='Select a measurement column to preview values; showing station positions only.'))
            quality.update(path=str(path), excluded_data_rows=meta['excluded_data_rows'],
                           preview_omitted_data_rows=missing[:40], unit=meta.get('output_measurement_unit'),
                           imported_units=settings.get(role, {}))
            # Uniform deterministic display sampling. Diagnostics above always use all rows.
            index = np.linspace(0, len(table)-1, min(len(table), 6000), dtype=int)
            sample = table.iloc[index]
            result['surveys'][role] = dict(quality=quality, x=sample.Easting.to_numpy(), y=sample.Northing.to_numpy(),
                 values=sample[meta['measurement_column']].to_numpy() if column else None,
                 data_rows=(sample.index.to_numpy()+1), sampled=len(sample)<len(table))
        except (ValueError, OSError) as exc:
            result['findings'].append(dict(role=role, severity='error', message=str(exc)))
    target_crs = settings.get('projected_crs')
    if not target_crs:
        # Match the existing solver's legacy terrain projection, while disclosing the assumption.
        target_crs = inspection.get('topography', {}).get('projected_crs')
        if target_crs:
            result['coordinate_assumption'] = 'Legacy survey/mesh CRS inferred from terrain and survey metadata; confirm if uncertain.'
    try:
        result['terrain'] = _terrain_preview(inspection, target_crs, inspection.get('mesh', {}).get('bounds') or region)
        if result['terrain'] and result['terrain']['valid_fraction_sampled'] < 1:
            result['findings'].append(dict(role='topography_file', severity='review',
                message='Terrain overview contains NoData gaps. Preview is sampled; inspect coverage before inversion.'))
    except (ValueError, OSError, RuntimeError) as exc:
        result['findings'].append(dict(role='topography_file', severity='review', message=str(exc)))
    return result


def preview_summary(preview):
    """Compact JSON evidence for chat and logs; keep image arrays out of the LLM."""
    terrain = preview.get('terrain')
    return dict(surveys={role: data['quality'] for role, data in preview['surveys'].items()},
                findings=preview['findings'], region=preview['region'],
                terrain={key: terrain[key] for key in ('valid_fraction_sampled', 'sample_shape', 'sampling')} if terrain else None,
                coordinate_assumption=preview.get('coordinate_assumption'), source_files_read_only=True)


def render_survey_preview(preview, view='coverage'):
    """CPU-rendered figure: true station values, no interpolation or gravity sign flip."""
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from .plotting.paper import DISPLAY_STYLE
    from .plotting.scales import field_scale, coordinate_scale

    surveys = preview['surveys']
    coordinates = [data[key] for data in surveys.values() for key in ('x', 'y')]
    divisor, unit = coordinate_scale(*coordinates) if coordinates else (1., 'm')
    with mpl.rc_context(DISPLAY_STYLE):
        fig = Figure(figsize=(7.4, 3.7), dpi=110, layout='constrained')
        ax = fig.add_subplot()
        terrain = preview.get('terrain')
        if view == 'coverage' and terrain and np.isfinite(terrain['values']).any():
            m = ax.pcolormesh(terrain['x']/divisor, terrain['y']/divisor, terrain['values'],
                              cmap='Greys', alpha=.35, shading='flat', rasterized=True)
            fig.colorbar(m, ax=ax, label='Elevation (m)', shrink=.8, pad=.025)
        for role, color, label in (('gravity_file', '#007aff', 'Gravity'), ('magnetic_file', '#c26b27', 'Magnetics')):
            data = surveys.get(role)
            if not data or (view != 'coverage' and role != view):
                continue
            kwargs = dict(s=8, linewidths=0, label=f"{label} · {data['quality']['rows']:,}", alpha=.85)
            if view == 'coverage':
                kwargs.update(s=7 if role == 'gravity_file' else 2, alpha=.8 if role == 'gravity_file' else .5,
                              zorder=3 if role == 'gravity_file' else 2)
            if view != 'coverage' and data['values'] is not None:
                scale = field_scale(data['values'], centered=True)
                # Limits derive from all observations, including unsampled extremes.
                stats = data['quality']['statistics']
                bound = max(abs(stats['minimum']), abs(stats['maximum'])) or 1.
                kwargs.update(c=data['values'], cmap=scale['cmap'], vmin=-bound, vmax=bound)
            else:
                kwargs['color'] = color
            m = ax.scatter(data['x']/divisor, data['y']/divisor, **kwargs)
            if view != 'coverage' and data['values'] is not None:
                fig.colorbar(m, ax=ax, label=f"{data['quality']['column']} ({data['quality']['unit']})", shrink=.8, pad=.025)
                flagged = {row for f in data['quality']['findings'] if f.get('code') == 'extreme_values' for row in f['data_rows']}
                mark = np.isin(data['data_rows'], list(flagged))
                if mark.any():
                    ax.scatter(data['x'][mark]/divisor, data['y'][mark]/divisor, s=32, facecolors='none', edgecolors='#202733', linewidths=.8, label='Review extremes')
        for key, label, color, linestyle in (('mesh', 'Full mesh', '#8b929c', '--'), ('region', 'Inversion region', '#202733', '-')):
            bounds = preview.get(key)
            if bounds:
                ax.add_patch(Rectangle((bounds['min_e']/divisor,bounds['min_n']/divisor),
                    (bounds['max_e']-bounds['min_e'])/divisor, (bounds['max_n']-bounds['min_n'])/divisor,
                    fill=False, edgecolor=color, linestyle=linestyle, linewidth=1, label=label))
        ax.autoscale_view()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'Easting ({unit})'); ax.set_ylabel(f'Northing ({unit})')
        ax.ticklabel_format(style='plain', useOffset=False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.04, 1.), frameon=False, fontsize=8)
        if not surveys:
            ax.text(.5,.5,'Resolve coordinate mappings to preview stations.',transform=ax.transAxes,ha='center',va='center')
        return fig
