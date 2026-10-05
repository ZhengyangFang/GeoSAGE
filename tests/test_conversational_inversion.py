"""Conversational setup must derive evidence locally and never invent survey physics."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geosage.studio_survey import build_configuration, inspect_survey_folder


def _survey(folder: Path):
    folder.mkdir()
    (folder / "Demo_gravity_data.csv").write_text(
        "Easting,Northing,Longitude,Latitude,Height,ISO,CBA\n"
        "125,225,-123.0,42.0,100,1,2\n300,500,-122.9,42.1,110,3,4\n",
        encoding="utf-8",
    )
    (folder / "Demo_magnetic_data.csv").write_text(
        "Easting,Northing,TFMA\n150,250,10\n350,450,20\n", encoding="utf-8"
    )
    (folder / "Demo_mesh.msh").write_text(
        "6 6 2\n50 150 100\n6*50\n6*50\n2*25\n", encoding="utf-8"
    )
    (folder / "Demo_mesh_core.msh").write_text(
        "4 4 2\n100 200 100\n4*50\n4*50\n2*25\n", encoding="utf-8"
    )
    with rasterio.open(
        folder / "Demo_topo.tif", "w", driver="GTiff", height=10, width=10,
        count=1, dtype="float32", crs="EPSG:32610", transform=from_origin(0, 600, 60, 60),
    ) as dataset:
        dataset.write(np.ones((10, 10), dtype="float32"), 1)
    for name in ("Demo_unit_defs.csv", "Demo_unit_groups.csv"):
        (folder / name).write_text("test", encoding="utf-8")
    (folder / "Demo_geology_context.txt").write_text("Observed geology", encoding="utf-8")
    return folder


def test_folder_scan_is_deterministic_and_leaves_scientific_parameters_missing(tmp_path):
    folder = _survey(tmp_path / "raw")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    found = inspect_survey_folder(folder)
    assert found["project"] == "Demo"
    assert found["gravity"]["extent"] == {
        "min_e": 125.0, "max_e": 300.0, "min_n": 225.0, "max_n": 500.0
    }
    assert found["detected"]["gravity_column"] == "ISO"
    assert found["detected"]["region"] == {
        "min_e": 100.0, "max_e": 300.0, "min_n": 200.0, "max_n": 400.0
    }
    assert found["core_mesh"]["shape"] == {"nx": 4, "ny": 4, "nz": 2}
    assert found["spatial_checks"] == {
        "core_mesh_inside_full_mesh": True,
        "topography_covers_full_mesh": True,
        "gravity_rows_in_core": 1,
        "magnetic_rows_in_core": 1,
    }
    assert found["missing_parameters"] == ["field_strength", "inclination", "declination"]
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_configuration_requires_user_supplied_field_and_accepts_region_override(tmp_path):
    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    with pytest.raises(ValueError, match="Still needed from the user"):
        build_configuration(found, {})
    config = build_configuration(found, {
        "min_e": 110, "max_e": 200, "min_n": 210, "max_n": 300,
        "field_strength": 50000, "inclination": 60, "declination": 5,
    })
    assert config["region"] == {"min_e": 110.0, "max_e": 200.0,
                                 "min_n": 210.0, "max_n": 300.0}
    assert config["project"]["region_source"] == "user_override"
    assert config["project"]["region_observations"] == {"gravity": 1, "magnetic": 1}
    assert config["inversion"]["field_strength"] == 50000.0
    assert config["data"]["gravity_column"] == "ISO"
    assert config["geology"]["mode"] == "csv_manual"


def test_configuration_uses_validated_core_mesh_region_when_omitted(tmp_path):
    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    config = build_configuration(found, {
        "field_strength": 50000, "inclination": 60, "declination": 5,
        "max_iterations": 10,
    })
    assert config["region"] == {
        "min_e": 100.0, "max_e": 300.0, "min_n": 200.0, "max_n": 400.0
    }
    assert config["project"]["region_source"] == "core_mesh_file"
    assert config["project"]["input_files"]["core_mesh_file"].endswith(
        "Demo_mesh_core.msh"
    )
    assert config["inversion"]["optimization"]["maxGNCG"] == 10


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"std_grv": 0}, "standard deviations"),
        ({"std_mag": -1}, "standard deviations"),
        ({"flight_height_ft": -1}, "Flight height"),
        ({"max_iterations": 0}, "positive whole number"),
        ({"max_iterations": 10.5}, "positive whole number"),
    ],
)
def test_configuration_rejects_invalid_method_settings(tmp_path, overrides, message):
    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    with pytest.raises(ValueError, match=message):
        build_configuration(found, {
            "field_strength": 50000, "inclination": 60, "declination": 5,
            **overrides,
        })


def test_configuration_rejects_a_region_without_both_data_types(tmp_path):
    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    with pytest.raises(ValueError, match="at least one usable gravity and magnetic"):
        build_configuration(found, {
            "min_e": 200, "max_e": 280, "min_n": 300, "max_n": 380,
            "field_strength": 50000, "inclination": 60, "declination": 5,
        })


def test_scan_rejects_missing_runner_columns_before_inversion(tmp_path):
    folder = _survey(tmp_path / "raw")
    (folder / "Demo_gravity_data.csv").write_text(
        "Easting,Northing,ISO\n125,225,1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Longitude, Latitude, Height"):
        inspect_survey_folder(folder)


def test_ai_backed_new_inversion_enables_report_but_local_run_does_not(tmp_path):
    pytest.importorskip("PyHydroGeophysX")
    from geosage.pyhydrogeophysx.configuration import configure

    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    config = build_configuration(found, {
        "min_e": 110, "max_e": 200, "min_n": 210, "max_n": 300,
        "field_strength": 50000, "inclination": 60, "declination": 5,
    })
    base = {"inputs": {"input_dir": found["folder"]}, "config": config,
            "studio_task": "invert", "request": "Run and review",
            "output_dir": str(tmp_path / "runs" / "one")}
    local = configure(dict(base, use_ai=False))
    assert not local["run"]["write_reports"] and not local["run"]["review_enabled"]
    ai = configure(dict(base, output_dir=str(tmp_path / "runs" / "two"),
                        provider="codex_cli", use_ai=True))
    assert ai["run"]["write_reports"] and ai["run"]["review_enabled"]


def test_preflight_accepts_hannah_style_magnetic_coordinates(tmp_path):
    pytest.importorskip("PyHydroGeophysX")
    from types import SimpleNamespace

    from geosage.pyhydrogeophysx.configuration import configure
    from geosage.pyhydrogeophysx.tools import prepare_data

    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    config = build_configuration(found, {
        "field_strength": 50000, "inclination": 60, "declination": 5,
        "max_iterations": 10,
    })
    output = tmp_path / "run"
    output.mkdir()
    cfg = configure({
        "inputs": {"input_dir": found["folder"]}, "config": config,
        "studio_task": "invert", "request": "Run locally", "use_ai": False,
        "output_dir": str(output),
    })
    summary, artifacts = prepare_data(SimpleNamespace(config=cfg, output_dir=output))
    assert "gravity and 1 magnetic" in summary
    assert artifacts["field_data"]["station_counts"] == [1, 1]
