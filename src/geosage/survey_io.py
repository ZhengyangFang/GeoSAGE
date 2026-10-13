"""Survey adapters over pandas, Rasterio and pyproj; original inputs stay read-only."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

TABLE_SUFFIXES = {'.csv', '.tsv', '.txt', '.xyz', '.xlsx', '.xls'}
FILE_SUFFIXES = {'gravity_file': 'gravity_data.csv', 'magnetic_file': 'magnetic_data.csv',
                 'topography_file': 'topo.tif', 'mesh_file': 'mesh.msh', 'core_mesh_file': 'mesh_core.msh'}
COORDINATE_ALIASES = {
    'Easting': ('easting', 'x', 'east', '东坐标'),
    'Northing': ('northing', 'y', 'north', '北坐标'),
    'Height': ('height', 'elevation', 'z', '高程'),
    'Longitude': ('longitude', 'lon', 'long', '经度'),
    'Latitude': ('latitude', 'lat', '纬度'),
}
TABLE_OPTIONS = {'sheet', 'header', 'skiprows', 'delimiter', 'encoding', 'decimal',
                 'na_values', 'columns', 'coordinate_unit', 'height_unit',
                 'measurement_unit', 'crs', 'missing_policy', 'exclude_rows', 'exclude_rows_source_sha256'}


def import_settings(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError('data_import must be an object.')
    options = dict(value or {})
    unknown = set(options) - {'gravity_file', 'magnetic_file', 'topography_file', 'projected_crs'}
    if unknown:
        raise ValueError(f'Unknown data_import options: {sorted(unknown)}')
    for role in ('gravity_file', 'magnetic_file', 'topography_file'):
        if role in options and not isinstance(options[role], dict):
            raise ValueError(f'data_import.{role} must be an object.')
    for role in ('gravity_file', 'magnetic_file'):
        unknown = set(options.get(role, {})) - TABLE_OPTIONS
        if unknown:
            raise ValueError(f'Unknown {role} import options: {sorted(unknown)}')
        if 'columns' in options.get(role, {}) and not isinstance(options[role]['columns'], dict):
            raise ValueError(f'{role}.columns must map canonical names to source column names.')
    unknown = set(options.get('topography_file', {})) - {'crs', 'height_unit', 'band'}
    if unknown:
        raise ValueError(f'Unknown terrain import options: {sorted(unknown)}')
    if options.get('projected_crs'):
        options['projected_crs'] = projected_crs(options['projected_crs']).to_string()
    return options


def projected_crs(value):
    crs = _crs(value)
    if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info[:2]):
        raise ValueError('The inversion/mesh CRS must be projected with metre axes.')
    return crs


def configuration_crs(cfg):
    """Resolve the same mesh CRS for preflight, import and reference validation."""
    imported = cfg.get('data_import', {}).get('projected_crs')
    configured = cfg.get('data', {}).get('projected_crs')
    if imported and configured and projected_crs(imported) != projected_crs(configured):
        raise ValueError('data_import.projected_crs conflicts with data.projected_crs; select one mesh CRS.')
    value = imported or configured
    return projected_crs(value).to_string() if value else None


def _crs(value):
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f'Invalid coordinate reference system: {value!r}') from exc


def _transform(source, target, x, y):
    try:
        return Transformer.from_crs(source, target, always_xy=True).transform(x, y, errcheck=True)
    except Exception as exc:
        raise ValueError(f'Cannot transform survey coordinates: {exc}') from exc


def read_table(path, options=None):
    """Delegate actual format parsing to pandas and its supported Excel engines."""
    path, options = Path(path), dict(options or {})
    if path.suffix.casefold() not in TABLE_SUFFIXES:
        raise ValueError(f'Unsupported survey table: {path.name}')
    header = options.get('header', 0)
    if header is not None and (type(header) is not int or header < 0):
        raise ValueError('header must be a zero-based row number or null for a headerless table.')
    if type(options.get('skiprows', 0)) is not int or options.get('skiprows', 0) < 0:
        raise ValueError('skiprows must be a non-negative number of leading lines.')
    kwargs = dict(header=header, skiprows=options.get('skiprows', 0),
                  na_values=options.get('na_values'), decimal=options.get('decimal', '.'))
    metadata = {'reader': 'pandas', 'format': path.suffix.casefold(), 'header': header}
    try:
        if path.suffix.casefold() in {'.xlsx', '.xls'}:
            engine = 'xlrd' if path.suffix.casefold() == '.xls' else 'openpyxl'
            with pd.ExcelFile(path, engine=engine) as book:
                sheets = book.sheet_names
                sheet = options.get('sheet')
                if sheet is None:
                    if len(sheets) != 1:
                        raise ValueError(f'{path.name} has multiple worksheets; choose sheet from {sheets}.')
                    sheet = sheets[0]
                if not isinstance(sheet, (str, int)) or isinstance(sheet, bool):
                    raise ValueError('Choose one worksheet by name or zero-based index.')
                table = pd.read_excel(book, sheet_name=sheet, **kwargs)
                if header is not None:
                    probe = pd.read_excel(book, sheet_name=sheet, header=None,
                                          skiprows=kwargs['skiprows'], nrows=header+1)
                    _check_headers(probe.iloc[header].tolist())
                metadata.update(engine=engine, sheet=sheet, sheets=sheets)
        else:
            encoding = options.get('encoding', 'utf-8-sig')
            # Sniff only the separator; pandas handles quoting, decimal and missing data.
            with path.open(encoding=encoding) as handle:
                sample = handle.read(65536)
            separator = options.get('delimiter')
            if separator is None:
                lines = [line for line in sample.splitlines()[int(kwargs['skiprows']):]
                         if line.strip() and not line.lstrip().startswith('#')]
                try:
                    separator = csv.Sniffer().sniff('\n'.join(lines[:20]), delimiters=',;\t').delimiter
                except csv.Error:
                    separator = r'\s+'
            table = pd.read_csv(path, sep=separator, encoding=encoding, comment='#',
                                engine='python', on_bad_lines='error', **kwargs)
            if header is not None:
                probe = pd.read_csv(path, sep=separator, encoding=encoding, comment='#',
                                    engine='python', header=None, skiprows=kwargs['skiprows'], nrows=header+1)
                _check_headers(probe.iloc[header].tolist())
            metadata.update(encoding=encoding, delimiter=separator)
    except ImportError as exc:
        raise ValueError('Excel import needs the excel extra: uv pip install "geosage[excel]".') from exc
    except Exception as exc:
        # Excel engines have their own corrupt-container exceptions (e.g. BadZipFile).
        raise ValueError(f'Cannot read {path.name}: {exc}') from exc
    if table.empty:
        raise ValueError(f'{path.name} contains no data rows.')
    if header is None:
        table.columns = [f'column_{i + 1}' for i in range(len(table.columns))]
    else:
        columns = [str(column).strip() for column in table.columns]
        if all(_numeric_name(name) for name in columns):
            raise ValueError(f'{path.name} appears headerless; specify header=null and map column_1, column_2, etc.')
        if len(set(name.casefold() for name in columns)) != len(columns):
            raise ValueError(f'{path.name} has duplicate column names after whitespace/case normalization.')
        table.columns = columns
    return table, metadata


def _check_headers(values):
    names = [str(value).strip().casefold() for value in values]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate source column names; rename them before importing.')


def _numeric_name(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def _factor(unit, kind):
    units = {
        'length': {'m': 1., 'metre': 1., 'meter': 1., 'km': 1000., 'ft': .3048},
        'gravity': {'mgal': 1., 'ugal': .001, 'µgal': .001, 'μgal': .001, 'm/s2': 1e5, 'm/s^2': 1e5},
        'gradient': {'e': 1., 'eotvos': 1., 'eötvös': 1., '1/s2': 1e9, 's^-2': 1e9},
        'magnetic': {'nt': 1., 'ut': 1000., 'µt': 1000., 'μt': 1000., 't': 1e9},
    }
    key = str(unit).strip().casefold()
    if key not in units[kind]:
        raise ValueError(f'Unsupported {kind} unit {unit!r}; supported units: {list(units[kind])}')
    return units[kind][key]


GRAVITY_COMPONENTS = {'gx', 'gy', 'gz', 'gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz'}
VERTICAL_GRAVITY_COLUMNS = {'iso', 'cba', 'sba', 'faa', 'og', 'gravity',
    'gravity_anomaly', 'bouguer', 'bouguer_anomaly', 'bouguer_residual'}


def resolve_gravity_component(column, component=None):
    """Infer recognizable observables; never silently override an explicit choice."""
    label = str(column or '').strip().casefold()
    inferred = label if label in GRAVITY_COMPONENTS else (
        'gzz' if label in {'gravity_gradient_zz', 'gravity_gzz'} else
        'gz' if label in VERTICAL_GRAVITY_COLUMNS else None)
    selected = str(component or inferred or 'gz').strip().casefold()
    if selected not in GRAVITY_COMPONENTS:
        raise ValueError('Select a supported gravity acceleration or gradient component.')
    if inferred and inferred != selected:
        raise ValueError(f'Gravity component {selected!r} conflicts with column {column!r} ({inferred}); select the matching column/component.')
    return selected


def normalize_table(path, role, options=None, *, target_crs=None, measurement=None, component=None):
    """Map and validate one survey using explicit units and CRS; never flip signs."""
    options = dict(options or {})
    table, provenance = read_table(path, options)
    original_columns = list(table.columns)
    lookup = {name.casefold(): name for name in table.columns}
    mapping = dict(options.get('columns') or {})
    if set(mapping) - set(COORDINATE_ALIASES):
        raise ValueError('columns maps canonical Easting/Northing/Height/Longitude/Latitude to source names.')
    resolved = {}
    for canonical, aliases in COORDINATE_ALIASES.items():
        if canonical in mapping:
            actual = lookup.get(str(mapping[canonical]).casefold())
            if actual is None:
                raise ValueError(f'Mapped column {mapping[canonical]!r} is absent from {Path(path).name}.')
            matches = [actual]
        else:
            matches = [lookup[alias] for alias in aliases if alias in lookup]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous {canonical} columns: {matches}; supply columns.{canonical}.')
        if matches:
            resolved[canonical] = matches[0]
    if len(set(resolved.values())) != len(resolved):
        raise ValueError('One source column cannot represent multiple coordinates.')
    rename = {source: canonical for canonical, source in resolved.items()}
    for source, canonical in rename.items():
        if source != canonical and canonical in table.columns:
            raise ValueError(f'Mapping {source} would overwrite existing {canonical}.')
    table = table.rename(columns=rename)
    excluded = options.get('exclude_rows', [])
    if (not isinstance(excluded, list) or any(type(row) is not int or row < 1 or row > len(table) for row in excluded)):
        raise ValueError('exclude_rows must list existing 1-based source data rows (excluding headers).')
    excluded = sorted(set(excluded))
    if excluded and options.get('exclude_rows_source_sha256') and fingerprint(path)['sha256'] != options['exclude_rows_source_sha256']:
        raise ValueError('Source file changed since row exclusions were selected. Review the source records and select exclusions again.')
    table = table.loc[~table.index.isin([row-1 for row in excluded])].copy()
    projected = all(key in table for key in ('Easting', 'Northing'))
    geographic = all(key in table for key in ('Longitude', 'Latitude'))
    if not projected and not geographic:
        raise ValueError('Map both Easting/Northing or Longitude/Latitude columns.')
    required = ['Easting', 'Northing'] if projected else ['Longitude', 'Latitude']
    if role == 'gravity_file':
        required.append('Height')
    selected = None
    if measurement:
        selected = {c.casefold(): c for c in table.columns}.get(str(measurement).casefold())
        if selected is None or selected in COORDINATE_ALIASES:
            raise ValueError(f'Select a measurement column present in {Path(path).name}: {measurement!r}')
        required.append(selected)
    numeric = list(dict.fromkeys(required + [c for c in COORDINATE_ALIASES if c in table]))
    missing = [c for c in numeric if c not in table]
    if missing:
        raise ValueError(f'{Path(path).name} is missing required columns: {missing}')
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors='coerce')
    valid = np.isfinite(table[numeric].to_numpy(dtype=float)).all(axis=1)
    dropped = (table.index[~valid] + 1).tolist()
    policy = options.get('missing_policy', 'error')
    if policy not in {'error', 'drop'}:
        raise ValueError('missing_policy must be error or drop.')
    if dropped and policy == 'error':
        raise ValueError(f'{Path(path).name} has invalid required values at data rows {dropped[:10]}; correct the source or explicitly choose missing_policy=drop.')
    table = table.loc[valid].copy()
    if table.empty:
        raise ValueError('No valid survey rows remain.')
    if 'Height' in table:
        table['Height'] *= _factor(options.get('height_unit', 'm'), 'length')
    if geographic:
        if not (table.Longitude.between(-180, 180).all() and table.Latitude.between(-90, 90).all()):
            raise ValueError('Longitude/Latitude must be in degrees within valid geographic bounds.')
    target = projected_crs(target_crs) if target_crs else None
    source = _crs(options['crs']) if options.get('crs') else None
    if projected:
        if source:
            if not source.is_projected:
                raise ValueError('A geographic source CRS needs Longitude/Latitude column mappings.')
            if target is None:
                raise ValueError('Specify projected_crs for the inversion mesh when supplying a source CRS.')
            if 'coordinate_unit' in options:
                factor = _factor(options['coordinate_unit'], 'length')
                for column, axis in zip(('Easting', 'Northing'), source.axis_info[:2]):
                    table[column] *= factor / axis.unit_conversion_factor
            x, y = _transform(source, target, table.Easting, table.Northing)
            table['Easting'], table['Northing'] = x, y
        else:
            table[['Easting', 'Northing']] *= _factor(options.get('coordinate_unit', 'm'), 'length')
    else:
        if target is None:
            raise ValueError('Geographic observations need an explicit projected_crs matching the inversion mesh.')
        source = source or CRS.from_epsg(4326)
        if not source.is_geographic:
            raise ValueError('Longitude/Latitude requires a geographic source CRS.')
        x, y = _transform(source, target, table.Longitude, table.Latitude)
        table['Easting'], table['Northing'] = x, y
    if not np.isfinite(table[['Easting', 'Northing']].to_numpy()).all():
        raise ValueError('Coordinate transformation produced invalid positions.')
    if selected:
        gravity_component = resolve_gravity_component(selected, component) if role == 'gravity_file' else None
        gradient = gravity_component is not None and len(gravity_component) == 3
        kind = 'magnetic' if role == 'magnetic_file' else ('gradient' if gradient else 'gravity')
        target_unit = {'gravity':'mGal','gradient':'Eotvos','magnetic':'nT'}[kind]
        table[selected] *= _factor(options.get('measurement_unit', target_unit), kind)
        provenance.update(measurement_column=selected, output_measurement_unit=target_unit)
    provenance.update(columns=resolved, options=options, rows=len(table), dropped_data_rows=dropped,
                      excluded_data_rows=excluded,
                      source_crs=source.to_string() if source else None,
                      projected_crs=target.to_string() if target else None,
                      coordinate_unit='m', height_unit='m',
                      unit_source='explicit_or_legacy_defaults',
                      columns_unchanged=list(table.columns)==original_columns)
    return table, provenance


def table_summary(table):
    numeric = [c for c in table if np.isfinite(pd.to_numeric(table[c], errors='coerce')).all()]
    result = {'rows': len(table), 'columns': list(table.columns), 'numeric_columns': numeric,
              'extent': {'min_e': float(table.Easting.min()), 'max_e': float(table.Easting.max()),
                         'min_n': float(table.Northing.min()), 'max_n': float(table.Northing.max())}}
    if all(c in table for c in ('Longitude', 'Latitude')):
        result['geographic_extent'] = {'min_lon': float(table.Longitude.min()), 'max_lon': float(table.Longitude.max()),
                                       'min_lat': float(table.Latitude.min()), 'max_lat': float(table.Latitude.max())}
    return result


def rows_in_region(table, region):
    return int(((table.Easting > region['min_e']) & (table.Easting < region['max_e']) &
                (table.Northing > region['min_n']) & (table.Northing < region['max_n'])).sum())


def fingerprint(path):
    path = Path(path)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size, 'sha256': digest}


def core_mesh_slices(mesh, core):
    """Locate an exact tensor submesh without assuming padding widths or depths."""
    slices = []
    for axis, full_nodes, core_nodes, widths in zip(
            'xyz', (mesh.nodes_x, mesh.nodes_y, mesh.nodes_z),
            (core.nodes_x, core.nodes_y, core.nodes_z), mesh.h):
        start = int(np.argmin(np.abs(full_nodes - core_nodes[0])))
        stop = start + len(core_nodes) - 1
        tolerance = max(1e-8, float(np.min(widths)) * 1e-6)
        if stop >= len(full_nodes) or not np.allclose(
                full_nodes[start:stop+1], core_nodes, rtol=0, atol=tolerance):
            raise ValueError(f'Core mesh {axis}-nodes must align with a contiguous subset of full mesh cells.')
        slices.append(slice(start, stop))
    return tuple(slices)


def _same_solver_csv(path, table):
    """Byte-copy only when the unmodified solver reader sees the normalized table."""
    try:
        raw = pd.read_csv(path)
        return (raw.shape == table.shape and raw.columns.equals(table.columns)
                and bool(((raw == table) | (raw.isna() & table.isna())).to_numpy().all()))
    except (ValueError, OSError, UnicodeError):
        return False


def stage_inputs(cfg, destination):
    """Shared GUI/CLI boundary: convert supported inputs into the solver's format."""
    import rasterio
    from rasterio.shutil import copy as raster_copy

    project, data = cfg['project'], cfg['data']
    settings = import_settings(cfg.get('data_import'))
    target_crs = configuration_crs(cfg)
    source_map = {**project.get('input_files', {}), **cfg.get('studio_inputs', {})}
    root = Path(project['input_dir'])
    name = project['name']
    if not name or Path(name).name != name:
        raise ValueError('Project name must not contain path separators.')
    files = {role: Path(source_map.get(role) or root / f'{name}_{suffix}').resolve() for role, suffix in FILE_SUFFIXES.items()}
    if cfg.get('setup_preflight'):
        from .mesh_planning import validate_preflight_inputs
        validate_preflight_inputs(cfg, files)
    if cfg.get('reference_data'):
        from .reference_data import validate_reference_configuration
        validate_reference_configuration(cfg, files)
    destination = Path(destination).resolve()
    if (root.resolve() == destination or root.resolve() in destination.parents
            or any(destination == p or destination in p.parents for p in files.values())):
        raise ValueError('Staged inputs must be separate from source files.')
    from discretize import TensorMesh
    core_mesh_slices(TensorMesh.read_UBC(files['mesh_file']), TensorMesh.read_UBC(files['core_mesh_file']))
    originals = [fingerprint(p) for p in files.values()]
    tables, imports, counts = {}, {}, []
    for role, key in (('gravity_file','gravity_column'), ('magnetic_file','magnetic_column')):
        table, meta = normalize_table(files[role], role, settings.get(role),
                                      target_crs=target_crs, measurement=data[key],
                                      component=data.get('gravity_component','gz'))
        count = rows_in_region(table, cfg['region'])
        if count == 0:
            raise ValueError(f'{role} has no stations inside the configured region.')
        counts.append(count)
        tables[role], imports[role] = table, meta
        from geosage.survey_quality import analyze_table
        imports[role]['quality'] = analyze_table(table, meta.get('measurement_column'), cfg['region'])
    terrain_options = settings.get('topography_file', {})
    with rasterio.open(files['topography_file']) as src:
        terrain_crs = src.crs or terrain_options.get('crs')
        if not terrain_crs:
            raise ValueError('Terrain CRS is absent; supply topography_file.crs.')
        if src.crs and terrain_options.get('crs') and CRS(src.crs) != CRS(terrain_options['crs']):
            raise ValueError('Terrain CRS override conflicts with file metadata.')
        band = terrain_options.get('band', 1)
        if type(band) is not int or band < 1 or band > src.count or (src.count > 1 and 'band' not in terrain_options):
            raise ValueError('Choose one valid terrain elevation band.')
        scale = _factor(terrain_options.get('height_unit', 'm'), 'length')
        if CRS(terrain_crs).is_geographic and not target_crs and not all(c in tables['gravity_file'] for c in ('Longitude','Latitude')):
            raise ValueError('Geographic terrain requires projected_crs or gravity Longitude/Latitude.')
        # Sidecar projection files are part of the original raster, too.
        originals.extend(fingerprint(p) for p in src.files if Path(p).resolve() != files['topography_file'])
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / f'{name}_topo.tif'
        if not terrain_options and len(src.files) == 1 and files['topography_file'].suffix.casefold() in {'.tif','.tiff'}:
            shutil.copy2(files['topography_file'], target)
        elif not terrain_options:
            raster_copy(src, target, driver='GTiff')
        else:
            profile = dict(driver='GTiff', width=src.width, height=src.height, count=1,
                           dtype='float64', crs=terrain_crs, transform=src.transform, nodata=float('nan'))
            with rasterio.open(target, 'w', **profile) as dst:
                for _, window in src.block_windows(band):
                    values = src.read(band, window=window, masked=True).astype('float64').filled(np.nan) * scale
                    dst.write(values, 1, window=window)
        imports['topography_file'] = {'reader':'rasterio/GDAL', 'source_crs':str(terrain_crs), 'band':band, 'height_unit':'m', 'options':terrain_options}
    for role, table in tables.items():
        target = destination / f'{name}_{FILE_SUFFIXES[role]}'
        meta = imports[role]
        if (not settings.get(role) and not settings.get('projected_crs')
                and files[role].suffix.casefold() == '.csv' and meta['columns_unchanged']
                and meta.get('delimiter') == ',' and meta.get('encoding') == 'utf-8-sig'
                and _same_solver_csv(files[role], table)):
            shutil.copy2(files[role], target)
        else:
            table.to_csv(target, index=False)
    for role in ('mesh_file', 'core_mesh_file'):
        shutil.copy2(files[role], destination / f'{name}_{FILE_SUFFIXES[role]}')
    for original in originals:
        if fingerprint(original['path']) != original:
            raise RuntimeError(f'Input changed while importing: {original["path"]}. Start a new run.')
    manifest = {'files': originals, 'station_counts': counts, 'imports': imports,
                'staged_files':[fingerprint(destination/f'{name}_{suffix}') for suffix in FILE_SUFFIXES.values()]}
    (destination/'import_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return manifest


def use_staged_inputs(cfg, destination, manifest):
    """Consume conversion settings once, retaining their full audit in the manifest."""
    target = cfg.get('data_import', {}).get('projected_crs')
    if target:
        cfg['data']['projected_crs'] = target
    cfg['project']['input_dir'] = str(Path(destination).resolve())
    cfg['project'].pop('input_files', None)
    cfg.pop('data_import', None)
    cfg['import_provenance'] = manifest
