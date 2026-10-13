"""Read-only comparisons of explicit saved runs; no automatic model resampling."""

import json
from pathlib import Path
import numpy as np


def _read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def _archive(path):
    from geosage.existing_results import REQUIRED_ARTIFACTS, load_inversion_parameters
    path = Path(path).expanduser().resolve()
    result = _read(path / 'studio_result.json')
    config = result.get('workflow_config', {})
    source = path
    if not (source / REQUIRED_ARTIFACTS['mesh_core']).is_file():
        candidate = config.get('project', {})
        source = Path(candidate.get('source_inversion_dir') or candidate.get('output_dir') or path / 'models').resolve()
    if not all((source / p).is_file() for p in REQUIRED_ARTIFACTS.values()):
        raise ValueError(f'No complete saved density/susceptibility model at {path}.')
    if not config:
        config = _read(source / 'effective_config.json')
    return source, config, load_inversion_parameters(source)


def _flatten(value, prefix=''):
    rows = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            rows.update(_flatten(item, name))
        else:
            rows[name] = item
    return rows


def compare_runs(left, right):
    import discretize
    from pyproj import CRS
    from geosage.existing_results import REQUIRED_ARTIFACTS, OPTIONAL_ARTIFACTS
    from geosage.validation import require_real_finite

    a, ca, pa = _archive(left)
    b, cb, pb = _archive(right)
    ap, bp = _flatten(pa), _flatten(pb)
    ignore = {'input_dir', 'output_dir', 'project_name'}
    differences = {key: {'left': ap.get(key), 'right': bp.get(key)}
                   for key in sorted(ap.keys() | bp.keys()) if key not in ignore and ap.get(key) != bp.get(key)}
    crsa = pa.get('survey_crs') or ca.get('data', {}).get('projected_crs')
    crsb = pb.get('survey_crs') or cb.get('data', {}).get('projected_crs')
    crs_ok = bool(crsa and crsb and CRS.from_user_input(crsa) == CRS.from_user_input(crsb))
    ma = discretize.TensorMesh.read_UBC(str(a / REQUIRED_ARTIFACTS['mesh_core']))
    mb = discretize.TensorMesh.read_UBC(str(b / REQUIRED_ARTIFACTS['mesh_core']))
    grid_ok = crs_ok and np.array_equal(ma.origin, mb.origin) and all(np.array_equal(x, y) for x, y in zip(ma.h, mb.h))
    result = {'status': 'ok', 'left': str(a), 'right': str(b), 'parameter_differences': differences,
              'left_label': Path(left).name, 'right_label': Path(right).name,
              'same_coordinate_reference': crs_ok, 'same_core_grid': grid_ok, 'models': {}, 'fits': {},
              'limitations': ['Lower data misfit alone does not establish a better geological interpretation.']}
    for side, params, cfg in (('left', pa, ca), ('right', pb, cb)):
        recorded, configured = params.get('survey_crs'), cfg.get('data', {}).get('projected_crs')
        if recorded and configured and CRS.from_user_input(recorded) != CRS.from_user_input(configured):
            result['limitations'].append(f'{side}: configuration CRS differs from the saved inversion; comparison uses recorded survey_crs {recorded}.')
    if not grid_ok:
        result['limitations'].append('Cell-by-cell changes unavailable: core grids or recorded CRS differ or CRS is missing; no resampling was performed.')
    for role, units in (('density_core', 'g/cm³'), ('susceptibility_core', 'SI')):
        x, y = [np.load(p / REQUIRED_ARTIFACTS[role], allow_pickle=False) for p in (a, b)]
        for array, mesh in ((x, ma), (y, mb)):
            require_real_finite(array, role)
            if array.shape != tuple(mesh.shape_cells):
                raise ValueError(f'{role} array does not match its core mesh.')
        row = {'units': units, 'left_range': [float(x.min()), float(x.max())], 'right_range': [float(y.min()), float(y.max())]}
        if grid_ok:
            delta = y - x
            row.update(rms_change=float(np.sqrt(np.mean(delta**2))), max_absolute_change=float(np.max(np.abs(delta))))
        result['models'][role] = row
    for kind in ('gravity', 'magnetics'):
        rows, obs = [], []
        for folder, parameters in ((a, pa), (b, pb)):
            obs_path, pred_path = folder / OPTIONAL_ARTIFACTS[f'obs_{kind}'], folder / OPTIONAL_ARTIFACTS[f'dpred_{kind}']
            if not obs_path.is_file() or not pred_path.is_file():
                rows.append(None); obs.append(None); continue
            table, pred = np.loadtxt(obs_path, ndmin=2), np.load(pred_path, allow_pickle=False).ravel()
            require_real_finite(table, f'{kind} observations'); require_real_finite(pred, f'{kind} prediction')
            if table.shape != (len(pred), 5) or np.any(table[:, 4] <= 0):
                raise ValueError(f'{kind} requires matching predictions and x/y/z/value/positive-std observations.')
            residual = pred - table[:, 3]
            rows.append({'rmse': float(np.sqrt(np.mean(residual**2))),
                         'normalized_rms': float(np.sqrt(np.mean((residual / table[:, 4])**2))), 'stations': len(pred),
                         'units': 'nT' if kind == 'magnetics' else (
                             'E' if len(str(parameters.get('gravity_component', ''))) == 3 else
                             'mGal' if parameters.get('gravity_component') in {'gx', 'gy', 'gz'} else 'unknown')})
            obs.append(table)
        keys = ('gravity_component',) if kind == 'gravity' else ('field_strength', 'inclination', 'declination')
        physical_ok = all(key in pa and key in pb and pa[key] == pb[key] for key in keys)
        same = crs_ok and physical_ok and all(item is not None for item in obs) and np.array_equal(*obs)
        units_present = {row['units'] for row in rows if row is not None}
        units = next(iter(units_present)) if len(units_present) == 1 else 'different / unknown'
        result['fits'][kind] = {'left': rows[0], 'right': rows[1], 'units': units,
                               'comparable': bool(same), 'reason': 'Same ordered observations, uncertainties, CRS and field/component.' if same else
                               'Missing or different observations, uncertainties, CRS, or field/component; do not rank these RMSE values.'}
    return result
