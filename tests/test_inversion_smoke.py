"""Exercise the real SimPEG solver on a small synthetic survey, without an LLM."""
import numpy as np
import pandas as pd
import rasterio
from discretize import TensorMesh
from pyproj import Transformer
from rasterio.transform import from_bounds

from geosage.gravity_mag_joint_inversion import run_joint_inversion


def test_small_joint_inversion_writes_loadable_core(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    mesh = TensorMesh([np.full(6, 100.0), np.full(6, 100.0), np.full(4, 100.0)],
                      x0=[500000, 4300000, -400])
    mesh.write_UBC(input_dir / "Tiny_mesh.msh")
    core = TensorMesh([np.full(4, 100.0), np.full(4, 100.0), np.full(4, 100.0)],
                      x0=[500100, 4300100, -400])
    core.write_UBC(input_dir / "Tiny_mesh_core.msh")
    to_lonlat = Transformer.from_crs(32610, 4326, always_xy=True)
    x = np.array([500150, 500450, 500150, 500450], dtype=float)
    y = np.array([4300150, 4300150, 4300450, 4300450], dtype=float)
    lon, lat = to_lonlat.transform(x, y)
    survey = pd.DataFrame({"Easting": x, "Northing": y, "Longitude": lon,
                           "Latitude": lat, "Height": 100.0})
    survey.assign(ISO=[0.1, 0.2, 0.15, 0.3]).to_csv(input_dir / "Tiny_gravity_data.csv", index=False)
    survey.assign(MagneticResidual=[10., 20., 15., 30.]).to_csv(
        input_dir / "Tiny_magnetic_data.csv", index=False
    )
    west, south = to_lonlat.transform(499900, 4299900)
    east, north = to_lonlat.transform(500700, 4300700)
    with rasterio.open(input_dir / "Tiny_topo.tif", "w", driver="GTiff",
                       height=20, width=20, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_bounds(west, south, east, north, 20, 20)) as dst:
        dst.write(np.zeros((20, 20), dtype=np.float32), 1)
    result = run_joint_inversion(project_name="Tiny", input_dir=input_dir,
        output_dir=tmp_path / "output", select_region=[500100, 500500, 4300100, 4300500],
        target_mag_data="MagneticResidual",
        maxGNCG=1, maxCG=10, maxIRLSiter=0, reg_grv_norm=(2, 2, 2, 2),
        reg_mag_norm=(2, 2, 2, 2), make_plots=False)
    for key in ("dens_core_3d", "susc_core_3d"):
        assert result[key].shape == (4, 4, 4)
        assert np.isfinite(result[key]).all()
    from geosage.existing_results import load_existing_inversion_result
    loaded = load_existing_inversion_result(tmp_path / "output")
    np.testing.assert_array_equal(loaded["dens_core_3d"], result["dens_core_3d"])
    np.testing.assert_array_equal(loaded["susc_core_3d"], result["susc_core_3d"])
