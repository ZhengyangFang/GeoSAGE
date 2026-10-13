"""Chat confirmation and asynchronous terrain acquisition for the optional plugin."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from html import escape
import os
from pathlib import Path
import tempfile

from PySide6.QtCore import QTimer

from geosage.reference_data import digest, geomagnetic_proposal, terrain_request, fetch_terrain, validate_proposal, utc_now


def _memory_range(resources):
    def size(value):
        for divisor,unit in ((1024**3,'GiB'),(1024**2,'MiB'),(1024,'KiB')):
            if value >= divisor:
                return f'{value/divisor:.2f} {unit}'
        return f'{value} bytes'
    return f"{size(resources['sensitivity_bytes_float32'])}–{size(resources['sensitivity_bytes_float64'])}"

ACTIONS = [
    {'name':'propose_inversion_mesh','args':{'projected_crs':'metre-based projected CRS matching the survey',
        'depth_m':'positive number; core thickness BELOW THE MESH TOP, supplied by user',
        'top_elevation_m':'mesh top elevation in the survey/terrain vertical datum; supplied by user',
        'cell_size_m':'optional positive number; otherwise a disclosed geometry heuristic',
        'bounds':'optional object: min_e,max_e,min_n,max_n in mesh CRS; otherwise station union plus margin',
        'padding_cells':'optional integer 0..20; default 5 on x/y sides and bottom',
        'padding_factor':'optional number 1..2; default 1.4'},
     'desc':'Propose a generic discretize tensor full/core mesh and sensitivity-memory envelope from survey geometry. Writes no files and changes no inputs; show bounds, top/bottom, heuristic, warnings and resources before confirming via confirm_reference_data.'},
    {'name':'propose_magnetic_field', 'args':{'longitude':'number, degrees east','latitude':'number, degrees north',
        'observation_date':'YYYY-MM-DD, actual date required','height_m':'number, WGS84 ellipsoidal height; not AGL',
        'projected_crs':'optional str matching mesh; required if not already known'},
     'desc':'Calculate an IGRF-14 reference F/I/D proposal using ppigrf. Show location/date/height, sources and direction conventions; changes no run parameters until confirmed.'},
    {'name':'request_reference_terrain','args':{'projected_crs':'str matching the full mesh','resolution_m':'30 or 90 (default 90)', 'offline':'bool (default false)'},
     'desc':'Fetch a cropped Copernicus DSM for the full mesh in the background. No existing terrain is replaced. Preview and sources appear when ready; use get_reference_data to check completion before requesting confirmation.'},
    {'name':'get_reference_data','args':{},'desc':'Read pending source proposals, fetch status and confirmed provenance; downloading is not adoption.'},
    {'name':'confirm_reference_data','args':{'proposal_id':'exact ID returned in proposal','confirmed':'true only after the user confirms',
       'height_datum_confirmed':'true for terrain only after user confirms EGM2008 compatibility',
       'replace_existing':'bool; true only if user explicitly requests replacing existing data/field values'},
     'desc':'Apply exactly the reviewed reference proposal after explicit user confirmation. Never implicitly replace user values. Run configuration must be resolved and confirmed again.'},
    {'name':'discard_reference_data','args':{'kind':'geomagnetic, terrain or mesh'},'desc':'Discard a pending candidate; existing adopted inputs remain unchanged.'},
]


class ReferenceData:
    def __init__(self, owner):
        self.owner = owner
        self.pending = {}
        self.accepted = {}
        self.job = None
        self.error = None
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='geosage-terrain')
        self.timer = QTimer(owner)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.poll)
        owner.destroyed.connect(lambda: self.pool.shutdown(wait=False,cancel_futures=True))

    def basis(self):
        scan = self.owner._survey_inspection or {}
        files = {}
        for role, value in scan.get('files',{}).items():
            if role == 'topography_file':
                continue
            path=Path(value); stat=path.stat()
            files[role]=[str(path.resolve()),stat.st_size,stat.st_mtime_ns]
        region=dict(scan.get('detected',{}).get('region') or {})
        region.update({k:v for k,v in self.owner._survey_parameters.items() if k in ('min_e','max_e','min_n','max_n')})
        settings={k:v for k,v in scan.get('data_import',{}).items() if k != 'topography_file'}
        return digest(dict(folder=scan.get('folder'),files=files,region=region,data_import=settings))

    def mesh_crs(self, supplied=None):
        from geosage.survey_io import projected_crs
        known = (self.owner._survey_inspection or {}).get('data_import', {}).get('projected_crs')
        if known and supplied and projected_crs(known) != projected_crs(supplied):
            raise ValueError('Reference CRS conflicts with the confirmed survey/mesh CRS. Rescan with explicit coordinate mappings before changing the mesh CRS.')
        return supplied or known

    def reset(self):
        self.pending.clear();self.accepted.clear();self.error=None
        # In-flight fetches finish in their own cache, but can no longer be adopted.
        if self.job:
            self.job['future'].cancel()
        self.job=None;self.timer.stop()
        self.owner.reference_summary.hide()

    def check_accepted(self):
        for kind, record in self.accepted.items():
            if kind != 'mesh' and record['basis'] != self.basis():
                raise ValueError(f'{kind} reference basis changed. Request and confirm a new proposal or supply your own values.')
            validate_proposal(record['proposal'])
            if kind == 'mesh':
                from geosage.survey_io import fingerprint
                if any(fingerprint(v['path']) != v for v in record['generated_files'].values()):
                    raise ValueError('Generated mesh changed; review a new mesh proposal.')

    def ready_to_run(self):
        if self.pending or self.job:
            raise ValueError('Confirm or discard pending reference proposals before starting the inversion.')
        self.check_accepted()
        if self.owner._configuration:
            from geosage.reference_data import validate_reference_configuration
            validate_reference_configuration(self.owner._configuration, self.owner.inputs)
            from geosage.mesh_planning import validate_preflight_inputs
            validate_preflight_inputs(self.owner._configuration, self.owner.inputs)

    def manual_parameters(self, args):
        record=self.accepted.get('geomagnetic')
        if record:
            record['applied_parameters']={k:v for k,v in record['applied_parameters'].items() if k not in args}
            if not record['applied_parameters']:
                self.accepted.pop('geomagnetic')

    def poll(self):
        if not self.job or not self.job['future'].done():
            return
        job=self.job;self.job=None;self.timer.stop()
        try:
            proposal=job['future'].result()
            if job['basis'] != self.basis():
                raise ValueError('Survey changed during download. Request a new terrain preview.')
            self.pending['terrain']={'proposal':proposal,'basis':job['basis']}
            self.refresh_preview()
        except Exception as exc:
            self.error=str(exc)
        self.show()

    def refresh_preview(self):
        """Overlay pending candidates without changing accepted input roles."""
        scan=deepcopy(self.owner._survey_inspection)
        if not scan:
            return
        parameters=deepcopy(self.owner._survey_parameters)
        terrain=self.pending.get('terrain',{}).get('proposal')
        if terrain:
            scan['files']['topography_file']=terrain['file']['path']
            scan.setdefault('data_import',{})['projected_crs']=terrain['request']['projected_crs']
            scan['data_import']['topography_file']={}
        mesh=self.pending.get('mesh',{}).get('proposal')
        if mesh:
            scan['mesh']={'bounds':mesh['full_bounds']}
            scan.setdefault('detected',{})['region']={k:mesh['core_bounds'][k] for k in ('min_e','max_e','min_n','max_n')}
            scan.setdefault('data_import',{})['projected_crs']=mesh['projected_crs']
            parameters={k:v for k,v in parameters.items() if k not in ('min_e','max_e','min_n','max_n')}
        self.owner._refresh_survey_preview(scan,parameters)

    def show(self):
        lines=[]
        if self.job:
            lines.append('Fetching terrain candidate… Existing inputs remain unchanged.')
        if self.error:
            lines.append(escape(self.error))
        for kind, record in self.pending.items():
            p=record['proposal']
            if kind=='geomagnetic':
                v=p['parameters'];i=p['inputs']
                lines.append(f"<b>IGRF-14 · awaiting confirmation</b><br>F {v['field_strength']:.1f} nT · I {v['inclination']:.2f}° · D {v['declination']:.2f}°<br>"
                    f"{i['observation_date']} · {i['latitude']:g}°, {i['longitude']:g}° · {i['height_m']:g} m above WGS84 ellipsoid. "
                    f"D relative to {'mesh grid north' if p.get('mesh_crs') else 'true north'}; reference model, not measured anomalies.")
            elif kind=='mesh':
                g=p['geometry'];r=p['resources'];b=p['core_bounds']
                lines.append(f"<b>Mesh proposal · awaiting confirmation</b><br>{g['cell_size_m']:g} m cells · "
                    f"{r['core_cells']:,} core / {r['full_cells']:,} full cells · elevation {b['min_z']:g} to {b['max_z']:g} m.<br>"
                    f"Sensitivity matrices up to {_memory_range(r)}; additional solver memory required.")
            else:
                lines.append(f"<b>Terrain candidate · awaiting confirmation</b><br>Copernicus {p['request']['resolution_m']} m DSM · EGM2008 heights. "
                    'Preview shows the candidate; confirm height datum compatibility in the conversation before use.')
            lines.append(f"<a href=\"{escape(p['source'])}\">Source and documentation</a>")
        preflight=(self.owner._configuration or {}).get('setup_preflight')
        if preflight and not self.pending:
            r=preflight['resources']
            lines.append(f"<b>Run size</b> · {r['full_cells']:,} cells · {sum(r['stations_in_region'].values()):,} stations · "
                         f"up to {r['maximum_iterations']} iterations.<br>Sensitivity matrices up to "
                         f"{_memory_range(r)}; not peak RAM.")
        self.owner.reference_summary.setHtml('<p>'+'</p><p>'.join(lines)+'</p>')
        self.owner.reference_summary.setVisible(bool(lines) and self.owner.task.currentData() == 'invert')

    def apply(self, action, args):
        o=self.owner
        try:
            if action=='get_reference_data':
                self.poll()
                return dict(status='fetching' if self.job else ('failed' if self.error else 'ok'),error=self.error,
                            proposals={k:deepcopy(v['proposal']) for k,v in self.pending.items()},accepted=deepcopy(self.accepted))
            if not o._survey_inspection:
                raise ValueError('Scan the survey folder first; partial inspections are supported.')
            if action=='propose_inversion_mesh':
                from geosage.mesh_planning import propose_mesh
                self.pending.pop('mesh',None);o._configuration=None
                basis=self.basis()
                p=propose_mesh(o._survey_inspection,projected_crs_value=self.mesh_crs(args.get('projected_crs')),
                    depth_m=args.get('depth_m'),top_elevation_m=args.get('top_elevation_m'),
                    **{k:args[k] for k in ('cell_size_m','bounds','padding_cells','padding_factor') if k in args})
                if basis != self.basis():
                    raise ValueError('Survey changed during mesh planning; request a new proposal.')
                self.pending['mesh']={'proposal':p,'basis':basis}
                self.refresh_preview()
                self.show()
                return dict(status='needs_confirmation',proposal=deepcopy(p))
            if action=='propose_magnetic_field':
                self.pending.pop('geomagnetic',None)
                o._configuration=None
                required=('longitude','latitude','observation_date','height_m')
                missing=[k for k in required if args.get(k) is None]
                if missing:
                    raise ValueError('Provide '+', '.join(missing)+'. No date or height is assumed.')
                crs=self.mesh_crs(args.get('projected_crs'))
                p=geomagnetic_proposal(**{k:args[k] for k in required},mesh_crs=crs)
                self.pending['geomagnetic']={'proposal':p,'basis':self.basis()}
                self.show()
                return dict(status='needs_confirmation',proposal=deepcopy(p))
            if action=='request_reference_terrain':
                if self.job:
                    return dict(status='fetching',detail='A terrain request is already running.')
                bounds=o._survey_inspection.get('mesh',{}).get('bounds')
                if not bounds:
                    raise ValueError('A full mesh is needed to define terrain coverage. Supply one or confirm a propose_inversion_mesh proposal first.')
                request=terrain_request(bounds,self.mesh_crs(args.get('projected_crs')),args.get('resolution_m',90))
                o._configuration=None
                cache=Path(os.environ.get('GEOSAGE_REFERENCE_CACHE',str(Path(tempfile.gettempdir())/'geosage-reference-cache')))
                cache = cache.resolve()
                raw = Path(o._survey_inspection['folder']).resolve()
                if cache == raw or raw in cache.parents:
                    raise ValueError('Terrain cache must be outside the raw input folder.')
                self.pending.pop('terrain',None);self.error=None
                self.job=dict(basis=self.basis(),future=self.pool.submit(fetch_terrain,request,cache,offline=args.get('offline',False)))
                self.timer.start();self.show()
                return dict(status='fetching',request=request,detail='Read get_reference_data when ready; preview alone does not adopt terrain.')
            if action=='discard_reference_data':
                kind=args.get('kind')
                self.pending.pop(kind,None)
                if kind=='terrain' and self.job:
                    self.job['future'].cancel();self.job=None;self.timer.stop()
                self.error=None;self.show();self.refresh_preview()
                return dict(status='ok',detail='Candidate discarded; original inputs unchanged.')
            if action=='confirm_reference_data':
                if args.get('confirmed') is not True:
                    raise ValueError('Explicit user confirmation is required before adopting a proposal.')
                match=next(((k,v) for k,v in self.pending.items() if v['proposal']['proposal_id']==args.get('proposal_id')),None)
                if not match:
                    raise ValueError('Proposal is absent or superseded; request a fresh candidate.')
                kind,record=match;p=record['proposal']
                if record['basis'] != self.basis():
                    raise ValueError('Survey context changed since preview; request a fresh proposal.')
                validate_proposal(p)
                if kind=='geomagnetic':
                    if not p.get('mesh_crs'):
                        raise ValueError('Supply the mesh projected_crs and request the field again so declination can be converted from true to grid north.')
                    values={k:v for k,v in p['parameters'].items() if args.get('replace_existing') is True or k not in o._survey_parameters}
                    if not values:
                        raise ValueError('All magnetic parameters already exist; replacement requires explicit user approval.')
                    from geosage.studio_survey import inspect_survey_folder
                    scan=o._survey_inspection
                    settings=deepcopy(scan.get('data_import',{}))
                    settings['projected_crs']=p['mesh_crs']
                    updated=inspect_survey_folder(scan['folder'],scan['files'],settings)
                    o._survey_inspection=updated
                    o._survey_parameters.update(values)
                    record=deepcopy(record);record['applied_parameters']=values
                elif kind=='mesh':
                    from geosage.mesh_planning import materialize_mesh
                    from geosage.studio_survey import inspect_survey_folder, REQUIRED_ROLES
                    if any(o.inputs.get(r) for r in ('mesh_file','core_mesh_file')) and args.get('replace_existing') is not True:
                        raise ValueError('Mesh inputs already exist. Replacing them needs explicit user approval.')
                    cache=Path(os.environ.get('GEOSAGE_MESH_CACHE',str(Path(tempfile.gettempdir())/'geosage-mesh-cache'))).resolve()
                    source=Path(o._survey_inspection['folder']).resolve()
                    if cache==source or source in cache.parents:
                        raise ValueError('Mesh cache must be outside the raw input folder.')
                    generated=materialize_mesh(p,cache)
                    scan=o._survey_inspection
                    settings=deepcopy(scan.get('data_import',{}));settings['projected_crs']=p['projected_crs']
                    files={**scan['files'],**{r:f['path'] for r,f in generated.items()}}
                    updated=inspect_survey_folder(scan['folder'],files,settings)
                    mesh_errors=[i for i in updated['validation_issues'] if i['role'] in ('mesh_file','core_mesh_file')]
                    if mesh_errors:
                        raise ValueError(mesh_errors[0]['message'])
                    o._survey_inspection=updated
                    o.inputs={'input_dir':updated['folder'],**{r:updated['files'][r] for r in REQUIRED_ROLES if r in updated['files']}}
                    # The reviewed new core replaces any earlier region selection.
                    for key in ('min_e','max_e','min_n','max_n'):
                        o._survey_parameters.pop(key,None)
                    record=deepcopy(record);record['generated_files']=generated
                else:
                    if args.get('height_datum_confirmed') is not True:
                        raise ValueError('Confirm that EGM2008 terrain heights are compatible with survey and mesh heights.')
                    if o.inputs.get('topography_file') and args.get('replace_existing') is not True:
                        raise ValueError('A terrain input already exists. Replacing it needs explicit user approval.')
                    from geosage.studio_survey import inspect_survey_folder, REQUIRED_ROLES
                    scan=o._survey_inspection
                    files={**scan['files'],'topography_file':p['file']['path']}
                    settings=deepcopy(scan.get('data_import',{}))
                    settings.update(projected_crs=p['request']['projected_crs'],topography_file={})
                    updated=inspect_survey_folder(scan['folder'],files,settings)
                    terrain_errors=[i for i in updated['validation_issues'] if i['role']=='topography_file']
                    if terrain_errors:
                        raise ValueError(terrain_errors[0]['message'])
                    o._survey_inspection=updated
                    o.inputs={'input_dir':updated['folder'],**{r:updated['files'][r] for r in REQUIRED_ROLES if r in updated['files']}}
                    # check_accepted blocks earlier field provenance if the CRS basis changed.
                    record=deepcopy(record);record['basis']=self.basis()
                    record['height_datum_confirmed']=True
                record['confirmed_at']=utc_now()
                record['basis']=self.basis()
                self.accepted[kind]=record
                self.pending.pop(kind);o._configuration=None
                self.refresh_preview();self.show()
                return dict(status='needs_parameters',inputs=dict(o.inputs),accepted=deepcopy(record),
                            missing_files=list(o._survey_inspection['missing_files']),
                            validation_issues=deepcopy(o._survey_inspection['validation_issues']),
                            next='Resolve and show the updated inversion settings before confirming the run.')
        except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
            if action in ('propose_inversion_mesh','propose_magnetic_field'):
                self.refresh_preview();self.show()
            return dict(status='failed',error=str(exc))
        return dict(status='failed',error='Unknown reference-data action.')
