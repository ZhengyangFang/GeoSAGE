"""Reference-data proposals; fetching/calculation never authorizes adoption."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from .survey_io import fingerprint, projected_crs

IGRF_SOURCE = 'https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field'
DEM_SOURCE = 'https://registry.opendata.aws/copernicus-dem/'
DEM_LICENSE = 'https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def seal(proposal):
    proposal['proposal_id'] = digest(proposal)
    return proposal


def validate_proposal(proposal):
    if proposal.get('proposal_id') != digest({k:v for k,v in proposal.items() if k != 'proposal_id'}):
        raise ValueError('Reference proposal changed; request a fresh preview before confirming.')
    if proposal['kind'] == 'terrain' and fingerprint(proposal['file']['path']) != proposal['file']:
        raise ValueError('Cached terrain changed; fetch and review a fresh candidate.')


def validate_reference_configuration(cfg, files=None):
    """Keep saved/CLI configurations consistent with their confirmed provenance."""
    for kind, record in cfg.get('reference_data', {}).items():
        proposal = record['proposal']
        validate_proposal(proposal)
        if not record.get('confirmed_at') or proposal['kind'] != kind:
            raise ValueError('Reference data must have a matching, confirmed proposal.')
        from .survey_io import configuration_crs
        crs = configuration_crs(cfg)
        expected_crs = (proposal.get('mesh_crs') if kind == 'geomagnetic' else
                        proposal['projected_crs'] if kind == 'mesh' else proposal['request']['projected_crs'])
        if not crs or not expected_crs or projected_crs(crs) != projected_crs(expected_crs):
            raise ValueError('Reference CRS differs from the run CRS; review the reference data again.')
        if kind == 'geomagnetic':
            for key, value in record['applied_parameters'].items():
                if cfg['inversion'].get(key) != value or value != proposal['parameters'].get(key):
                    raise ValueError(f'{key} differs from confirmed reference provenance. Supply manual values explicitly or confirm a new proposal.')
        elif kind == 'mesh':
            for role, expected in record['generated_files'].items():
                actual = fingerprint(files[role] if files else expected['path'])
                if actual['sha256'] != expected['sha256']:
                    raise ValueError('Mesh input differs from the confirmed proposal; review it again.')
        else:
            if record.get('height_datum_confirmed') is not True:
                raise ValueError('Confirm terrain height datum compatibility before running.')
            if files:
                actual = fingerprint(files['topography_file'])
                if actual['sha256'] != proposal['file']['sha256']:
                    raise ValueError('Terrain input differs from the confirmed candidate; review it again.')


def geomagnetic_proposal(*, longitude, latitude, observation_date, height_m, mesh_crs=None):
    """IGRF14 via IAGA ppigrf; input height is WGS84 ellipsoidal, not AGL."""
    try:
        import ppigrf
        from ppigrf.ppigrf import read_shc
    except ImportError as exc:
        raise ValueError('Reference-field calculation needs the reference extra. Install geosage[reference] with uv in the Studio environment.') from exc

    lon, lat, height = float(longitude), float(latitude), float(height_m)
    if not all(math.isfinite(v) for v in (lon, lat, height)) or not (-180 <= lon <= 180 and -90 < lat < 90):
        raise ValueError('Supply finite longitude [-180,180], latitude (-90,90) and ellipsoidal height in metres.')
    if not -1000 <= height <= 1000000:
        raise ValueError('This survey adapter supports ellipsoidal heights from -1000 m to 1000 km; verify the height unit.')
    if not isinstance(observation_date, str) or len(observation_date) != 10:
        raise ValueError('Supply the actual observation date as YYYY-MM-DD; no default date is assumed.')
    date = datetime.strptime(observation_date, '%Y-%m-%d')
    coefficients = Path(ppigrf.__file__).parent/'IGRF14.shc'
    g, _ = read_shc(str(coefficients))
    if not g.index.min().to_pydatetime() <= date <= g.index.max().to_pydatetime():
        raise ValueError(f'Date outside IGRF-14 coefficients: {g.index.min().date()} to {g.index.max().date()}. No extrapolation.')
    east, north, up = (float(np.asarray(v).item()) for v in ppigrf.igrf(lon, lat, height/1000., date, coeff_fn=str(coefficients)))
    down = -up
    horizontal = math.hypot(north, east)
    if not all(math.isfinite(v) for v in (east,north,down)) or horizontal < 1:
        raise ValueError('Reference field is invalid or declination is ill-defined near the magnetic pole.')
    true_declination=math.degrees(math.atan2(east,north))
    convergence=0.
    if mesh_crs:
        from pyproj import Proj
        mesh_crs=projected_crs(mesh_crs).to_string()
        convergence=float(Proj(mesh_crs).get_factors(lon,lat,errcheck=True).meridian_convergence)
        if not math.isfinite(convergence):
            raise ValueError('Cannot determine grid convergence for the mesh CRS.')
    return seal(dict(kind='geomagnetic', model='IGRF-14', source=IGRF_SOURCE,
        coefficients_source='https://doi.org/10.5281/zenodo.14012302',
        implementation='https://github.com/IAGA-VMOD/ppigrf', implementation_version=version('ppigrf'),
        coefficients_sha256=fingerprint(coefficients)['sha256'], created_at=utc_now(),
        inputs=dict(longitude=lon,latitude=lat,observation_date=observation_date,height_m=height,
                    height_datum='WGS84 ellipsoid; not height above ground'),
        components_nT=dict(north=north,east=east,down=down),
        mesh_crs=mesh_crs,true_north_declination=true_declination,grid_convergence_degrees=convergence if mesh_crs else None,
        parameters=dict(field_strength=math.sqrt(horizontal**2+down**2),
                        inclination=math.degrees(math.atan2(down,horizontal)),
                        declination=(true_declination-convergence+180)%360-180),
        conventions='F in nT; inclination positive down. IGRF declination is east of true north; solver declination subtracts mesh meridian convergence when CRS is supplied.',
        limitations=['Main-field model at one location/date, not measured anomalies or disturbance corrections.',
                     'A single representative field may be insufficient for large survey areas.'],
        requires_confirmation=True))


def terrain_request(bounds, crs, resolution_m=90):
    from rasterio.warp import transform_bounds
    crs = projected_crs(crs).to_string()
    values = [float(bounds[k]) for k in ('min_e','min_n','max_e','max_n')]
    if not all(math.isfinite(v) for v in values) or values[0]>=values[2] or values[1]>=values[3]:
        raise ValueError('Supply valid full-mesh bounds and its projected CRS.')
    if type(resolution_m) is not int or resolution_m not in (30,90):
        raise ValueError('Choose Copernicus resolution_m=90 or 30.')
    west,south,east,north = transform_bounds(crs,'EPSG:4326',*values,densify_pts=21)
    if not all(math.isfinite(v) for v in (west,south,east,north)) or west>=east or not (-180<=west<east<=180 and -85<south<north<85):
        raise ValueError('This terrain adapter requires one area within ±85° latitude without crossing the dateline.')
    res = (1 if resolution_m==30 else 3)/3600.
    # Snap outward with one-pixel margin so the downloaded raster covers the mesh.
    geographic = [math.floor(west/res)*res-res, math.floor(south/res)*res-res,
                  math.ceil(east/res)*res+res, math.ceil(north/res)*res+res]
    w,s,e,n = geographic
    if not -180 <= w < e <= 180:
        raise ValueError('Padded terrain bounds cross the dateline; supply a local DEM for this area.')
    pixels = math.ceil((e-w)/res)*math.ceil((n-s)/res)
    if pixels > 8_000_000:
        raise ValueError('Requested terrain exceeds 8 million pixels; choose 90 m or a smaller area.')
    tiles = []
    for lat in range(math.floor(s),math.ceil(n)):
        for lon in range(math.floor(w),math.ceil(e)):
            tile=f"Copernicus_DSM_COG_{'10' if resolution_m==30 else '30'}_{'N' if lat>=0 else 'S'}{abs(lat):02d}_00_{'E' if lon>=0 else 'W'}{abs(lon):03d}_00_DEM"
            tiles.append(f'https://copernicus-dem-{resolution_m}m.s3.amazonaws.com/{tile}/{tile}.tif')
    if len(tiles)>16:
        raise ValueError('Requested terrain spans more than 16 tiles; narrow the area.')
    return dict(mesh_bounds=dict(zip(('min_e','min_n','max_e','max_n'),values)),projected_crs=crs,
                geographic_bounds=geographic,resolution_m=resolution_m,resolution_degrees=res,
                estimated_uncompressed_bytes=pixels*4,tiles=tiles,release='Copernicus DEM 2021 AWS release')


def fetch_terrain(request, cache_dir, *, offline=False):
    """Read public COG windows via GDAL; cache only the clipped raster, not full tiles."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling

    if request != terrain_request(request['mesh_bounds'],request['projected_crs'],request['resolution_m']):
        raise ValueError('Terrain request was modified; generate it again from the mesh bounds.')
    cache = Path(cache_dir).resolve()
    key = digest(request)
    raster = cache/f'{key}.tif'
    manifest = cache/f'{key}.json'
    if raster.exists() and manifest.exists():
        try:
            proposal = json.loads(manifest.read_text(encoding='utf-8'))
            validate_proposal(proposal)
            if proposal['request'] != request:
                raise ValueError('Terrain cache request mismatch.')
            return proposal
        except (ValueError, KeyError, TypeError, OSError):
            if offline:
                raise
            # An online request can recover a corrupted cache without asking the
            # user to find and delete internal files. Replace only after success.
    if offline:
        raise ValueError('No matching cached terrain. Connect to download, or supply a local DEM.')
    cache.mkdir(parents=True,exist_ok=True)
    if sum(p.stat().st_size for p in cache.glob('*.tif')) + request['estimated_uncompressed_bytes'] > 256*1024**2:
        raise ValueError('Terrain cache would exceed 256 MiB. Remove unused cached candidates or supply a local DEM.')
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                          GDAL_HTTP_CONNECTTIMEOUT='10',GDAL_HTTP_TIMEOUT='40', GDAL_HTTP_MAX_RETRY='1',
                          GDAL_CACHEMAX=32*1024**2, CPL_VSIL_CURL_CACHE_SIZE=4*1024**2):
            with ExitStack() as stack:
                datasets=[stack.enter_context(rasterio.open(url)) for url in request['tiles']]
                if any(src.crs is None or src.crs.to_epsg()!=4326 or src.count!=1 for src in datasets):
                    raise ValueError('Unexpected Copernicus tile CRS or band count.')
                array, transform=merge(datasets,bounds=tuple(request['geographic_bounds']),
                    res=request['resolution_degrees'], nodata=-9999.,dtype='float32',
                    resampling=Resampling.bilinear,method='first')
        if not (np.isfinite(array)&(array!=-9999)).all():
            raise ValueError('Downloaded terrain contains gaps; supply suitable terrain. Missing tiles are not filled with zero.')
        with tempfile.NamedTemporaryFile(dir=cache,suffix='.tif',delete=False) as temp:
            part=Path(temp.name)
        try:
            with rasterio.open(part,'w',driver='GTiff',height=array.shape[1],width=array.shape[2],count=1,
                               dtype='float32',crs='EPSG:4326',transform=transform,nodata=-9999,
                               compress='deflate',tiled=True) as dst:
                dst.write(array)
                dst.update_tags(source=DEM_SOURCE, vertical_datum='EGM2008',dataset=request['release'])
            part.replace(raster)
        finally:
            part.unlink(missing_ok=True)
        proposal=seal(dict(kind='terrain',request=request,file=fingerprint(raster),source=DEM_SOURCE,
            license=DEM_LICENSE,retrieved_at=utc_now(),horizontal_crs='EPSG:4326',vertical_datum='EGM2008 orthometric height (m)',
            surface='Digital surface model; vegetation and buildings can be included.',
            value_range_m=[float(array.min()),float(array.max())],requires_confirmation=True,
            limitations=['Confirm mesh and survey height datum compatibility before adoption; no vertical datum conversion is applied.',
                         'Reference terrain is not survey-date-specific and is not guaranteed bare-earth ground.']))
        manifest.write_text(json.dumps(proposal,indent=2)+'\n',encoding='utf-8')
        return proposal
    except Exception as exc:
        raise ValueError(f'Terrain acquisition failed: {exc}. Existing inputs remain unchanged.') from exc
