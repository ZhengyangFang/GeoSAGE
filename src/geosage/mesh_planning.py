"""Reviewable tensor-mesh proposals using discretize; never infer target depth."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from .reference_data import digest, seal, utc_now, validate_proposal
from .survey_io import normalize_table, projected_crs, fingerprint, core_mesh_slices, rows_in_region


def _number(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number.')
    try:
        value=float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Provide {name}; it cannot be inferred reliably from station positions.') from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive.' if positive else '.'))
    return value


def _bounds(mesh):
    return dict(min_e=float(mesh.nodes_x[0]),max_e=float(mesh.nodes_x[-1]),
                min_n=float(mesh.nodes_y[0]),max_n=float(mesh.nodes_y[-1]),
                min_z=float(mesh.nodes_z[0]),max_z=float(mesh.nodes_z[-1]))


def resource_estimate(full_cells, core_cells, counts, iterations=None):
    """Sensitivity allocation envelope only; not a peak-RAM or time prediction."""
    entries=int(full_cells)*sum(int(v) for v in counts.values())
    return dict(full_cells=int(full_cells),core_cells=int(core_cells),stations_in_region=counts,
                active_cells_upper_bound=int(full_cells),scalar_model_parameters_upper_bound=2*int(full_cells),
                sensitivity_bytes_float32=entries*4,sensitivity_bytes_float64=entries*8,
                maximum_iterations=iterations,
                note='Both scalar gravity and magnetic sensitivity matrices, assuming every full-mesh cell is active. '
                     'This is not peak RAM: solver copies, regularization, meshes and plotting need additional memory. '
                     'Topography may reduce active cells. No reliable runtime estimate is made.')


def _meshes(geometry):
    from discretize import TensorMesh
    h=geometry['cell_size_m'];nx,ny,nz=geometry['core_shape']
    p=geometry['padding_cells'];f=geometry['padding_factor']
    core=TensorMesh([[(h,nx)],[(h,ny)],[(h,nz)]],origin=geometry['core_origin'])
    pad=float(sum(h*f**i for i in range(1,p+1)))
    full=TensorMesh([[(h,p,-f),(h,nx),(h,p,f)],
                     [(h,p,-f),(h,ny),(h,p,f)],[(h,p,-f),(h,nz)]],
                    origin=np.asarray(core.origin)-pad)
    core_mesh_slices(full,core)
    return full,core


def propose_mesh(inspection, *, projected_crs_value, depth_m, top_elevation_m,
                 cell_size_m=None, bounds=None, padding_cells=5, padding_factor=1.4):
    """Use station geometry for horizontal coverage; expose every heuristic."""
    from scipy.spatial import cKDTree
    crs=projected_crs(projected_crs_value).to_string()
    depth=_number(depth_m,'depth_m below mesh top',True)
    top=_number(top_elevation_m,'top_elevation_m in the survey/terrain height datum')
    if type(padding_cells) is not int or not 0 <= padding_cells <= 20:
        raise ValueError('padding_cells must be an integer from 0 to 20.')
    expansion=_number(padding_factor,'padding_factor',True)
    if not 1 <= expansion <= 2:
        raise ValueError('padding_factor must be between 1 and 2.')
    tables={};spacing={};sampling={}
    for role in ('gravity_file','magnetic_file'):
        path=inspection.get('files',{}).get(role)
        if not path:
            raise ValueError(f'Select {role} before proposing a joint-inversion mesh.')
        table,_=normalize_table(path,role,inspection.get('data_import',{}).get(role),target_crs=crs)
        tables[role]=table
        xy=np.unique(table[['Easting','Northing']].to_numpy(),axis=0)
        sampling[role]=len(xy)>50000
        xy=xy[np.linspace(0,len(xy)-1,min(len(xy),50000),dtype=int)]
        if len(xy)>1:
            distances=cKDTree(xy).query(xy,k=2)[0][:,1]
            spacing[role]=float(np.median(distances))
    extent=dict(min_e=min(t.Easting.min() for t in tables.values()),max_e=max(t.Easting.max() for t in tables.values()),
                min_n=min(t.Northing.min() for t in tables.values()),max_n=max(t.Northing.max() for t in tables.values()))
    requested=deepcopy(bounds or extent)
    if bounds is not None and set(bounds) != {'min_e','max_e','min_n','max_n'}:
        raise ValueError('bounds must contain min_e, max_e, min_n and max_n.')
    requested={k:_number(v,k) for k,v in requested.items()}
    if requested['max_e']<=requested['min_e'] or requested['max_n']<=requested['min_n']:
        raise ValueError('Station geometry has zero area; provide an explicit nonzero study rectangle for a line survey.')
    automatic=cell_size_m is None
    if automatic:
        candidate=max(min(spacing.values(),default=0),
                      (requested['max_e']-requested['min_e'])/40,
                      (requested['max_n']-requested['min_n'])/40,depth/20)
        order=10.**math.floor(math.log10(candidate))
        cell_size_m=next(v*order for v in (1,2,5,10) if v*order>=candidate)
    h=_number(cell_size_m,'cell_size_m',True)
    if not all(math.isfinite(v/h) for v in [*requested.values(),depth]):
        raise ValueError('Coordinate/depth to cell-size ratio is too large; use a coarser mesh.')
    # Automatic coverage adds one core cell around stations so boundary samples
    # are inside the solver's strict region bounds. Explicit bounds snap outward.
    margin=1 if bounds is None else 0
    low=[(math.floor(requested[k]/h)-margin)*h for k in ('min_e','min_n')]
    high=[(math.ceil(requested[k]/h)+margin)*h for k in ('max_e','max_n')]
    shape=[max(2,math.ceil((high[i]-low[i])/h)) for i in range(2)]+[max(2,math.ceil(depth/h))]
    full_shape=[shape[0]+2*padding_cells,shape[1]+2*padding_cells,shape[2]+padding_cells]
    if max(full_shape)>10000 or math.prod(full_shape)>2_000_000:
        raise ValueError('Proposed mesh exceeds 2 million cells or 10,000 cells along an axis. Request a coarser cell size or smaller bounds.')
    geometry=dict(cell_size_m=h,core_shape=shape,core_origin=[*low,top-shape[2]*h],
                  padding_cells=padding_cells,padding_factor=expansion)
    full,core=_meshes(geometry)
    region={k:v for k,v in _bounds(core).items() if k in extent}
    counts={r:rows_in_region(t,region) for r,t in tables.items()}
    if not all(counts.values()):
        raise ValueError('Proposed core region must include both gravity and magnetic observations.')
    warnings=['Mesh resolution is a starting proposal, not a validated geological resolution or depth of investigation.',
              'Check that the top elevation and terrain use the same vertical datum and that the mesh covers the target volume.']
    if automatic:
        warnings.append('Automatic cell size rounds up the largest of the finer median station spacing, horizontal span/40 and requested depth/20 to a 1/2/5 scale.')
    if padding_cells==0:
        warnings.append('No padding requested; boundary effects need review.')
    if any(sampling.values()):
        warnings.append('Spacing is estimated from at most 50,000 unique positions per survey; all rows determine coverage and station counts.')
    terrain=inspection.get('topography',{}).get('projected_bounds')
    # This metadata can be in another projection; only compare identical CRSs.
    if terrain and inspection['topography'].get('projected_crs')==crs:
        from .studio_survey import _contains
        if not _contains(terrain,_bounds(full)):
            warnings.append('Existing terrain does not cover the padded full mesh. Supply or request wider terrain before running.')
    return seal(dict(kind='mesh',created_at=utc_now(),implementation='discretize.TensorMesh',
        source='https://discretize.simpeg.xyz/en/main/api/generated/discretize.TensorMesh.html',
        projected_crs=crs,geometry=geometry,core_bounds=_bounds(core),full_bounds=_bounds(full),
        requested=dict(bounds={k:float(v) for k,v in requested.items()},depth_m=depth,top_elevation_m=top),
        cell_size_source='geometry_heuristic' if automatic else 'user',bounds_source='station_union_with_margin' if bounds is None else 'user_snapped_outward',
        median_station_spacing_m=spacing,spacing_sampled=sampling,
        resources=resource_estimate(full.n_cells,core.n_cells,counts),warnings=warnings,requires_confirmation=True))


def materialize_mesh(proposal, cache_dir):
    """Create/reuse small UBC axis descriptions, without allocating cell matrices."""
    validate_proposal(proposal)
    full,core=_meshes(proposal['geometry'])
    folder=Path(cache_dir).resolve()/('mesh-'+digest(proposal['geometry']))
    folder.mkdir(parents=True,exist_ok=True)
    result={}
    from discretize import TensorMesh
    for role,mesh in (('mesh_file',full),('core_mesh_file',core)):
        path=folder/(role+'.msh')
        if path.exists():
            saved=TensorMesh.read_UBC(path)
            if (not np.allclose(saved.origin,mesh.origin,rtol=0,atol=1e-6)
                    or any(a.shape!=b.shape or not np.allclose(a,b,rtol=1e-10,atol=1e-8) for a,b in zip(saved.h,mesh.h))):
                raise ValueError('Cached mesh differs from the proposal. Choose another mesh cache directory and review again.')
        else:
            mesh.write_UBC(str(path))
        result[role]=fingerprint(path)
    return result


def configuration_preflight(cfg):
    """Estimate the exact selected meshes/rows and disclose parameter origins."""
    from discretize import TensorMesh
    from .survey_io import FILE_SUFFIXES, configuration_crs
    target_crs = configuration_crs(cfg)
    project=cfg['project'];settings=cfg.get('data_import',{})
    sources={**project.get('input_files',{}),**cfg.get('studio_inputs',{})}
    files={r:Path(sources.get(r) or Path(project['input_dir'])/f"{project['name']}_{suffix}") for r,suffix in FILE_SUFFIXES.items()}
    fingerprints={role:fingerprint(path) for role,path in files.items()}
    import rasterio
    with rasterio.open(files['topography_file']) as terrain:
        for index,path in enumerate(terrain.files):
            if Path(path).resolve()!=files['topography_file'].resolve():
                fingerprints[f'topography_sidecar_{index}']=fingerprint(path)
    full=TensorMesh.read_UBC(files['mesh_file']);core=TensorMesh.read_UBC(files['core_mesh_file'])
    core_mesh_slices(full,core)
    from .studio_survey import _contains
    if not _contains(_bounds(full),cfg['region']):
        raise ValueError('The configured region extends outside the full inversion mesh.')
    counts={}
    for role,key in (('gravity_file','gravity_column'),('magnetic_file','magnetic_column')):
        table,_=normalize_table(files[role],role,settings.get(role),target_crs=target_crs,
                               measurement=cfg['data'][key],component=cfg['data'].get('gravity_component','gz'))
        counts[role]=rows_in_region(table,cfg['region'])
    if not all(counts.values()):
        raise ValueError('The configured region must contain both gravity and magnetic stations.')
    if any(fingerprint(item['path']) != item for item in fingerprints.values()):
        raise ValueError('Input changed during preflight; resolve the settings again.')
    return dict(resources=resource_estimate(full.n_cells,core.n_cells,counts,cfg['inversion']['optimization']['maxGNCG']),
                input_fingerprints=fingerprints,
                parameter_sources=deepcopy(cfg.get('parameter_sources',{})),
                reference_sources={k:v['proposal']['source'] for k,v in cfg.get('reference_data',{}).items()},
                core_bounds=_bounds(core),full_bounds=_bounds(full),region=deepcopy(cfg['region']))


def validate_preflight_inputs(cfg, files=None):
    """Changing source files after reviewing a run needs another preflight."""
    for role, expected in cfg.get('setup_preflight',{}).get('input_fingerprints',{}).items():
        path=(files or {}).get(role) or cfg.get('project',{}).get('input_files',{}).get(role) or expected['path']
        actual=fingerprint(path)
        if actual['sha256'] != expected['sha256'] or actual['size_bytes'] != expected['size_bytes']:
            raise ValueError(f'{role} changed after run preflight. Resolve and confirm the settings again.')


def parameter_sources(cfg, parameters, accepted):
    """Describe how the conversational configuration acquired each setting."""
    sources={}
    for key in cfg['region']:
        sources[f'region.{key}']='user' if parameters.get(key) not in (None,'') else 'core_mesh_bounds'
    for key in cfg['data']:
        sources[f'data.{key}']=('user' if parameters.get(key) not in (None,'') else
                               'detected_column' if key.endswith('_column') else 'method_default')
    for key in ('field_strength','inclination','declination','cross_gradient_lambda','beta_cooling'):
        sources[f'inversion.{key}']=('user' if key in parameters else
                                    'core_cell_width_to_fourth_power' if key=='cross_gradient_lambda' else 'method_default')
    for key in accepted.get('geomagnetic',{}).get('applied_parameters',{}):
        sources[f'inversion.{key}']='confirmed_IGRF14'
    sources['inversion.optimization.maxGNCG']='user' if 'max_iterations' in parameters else 'method_default'
    return sources
