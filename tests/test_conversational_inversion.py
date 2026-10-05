"""Conversational setup must derive evidence locally and never invent survey physics."""

from pathlib import Path

import pytest

from geosage.studio_survey import build_configuration, inspect_survey_folder


def _survey(folder: Path):
    folder.mkdir()
    (folder / "Demo_gravity_data.csv").write_text(
        "Easting,Northing,ISO,CBA\n100,200,1,2\n300,500,3,4\n", encoding="utf-8"
    )
    (folder / "Demo_magnetic_data.csv").write_text(
        "Easting,Northing,TFMA\n150,250,10\n350,450,20\n", encoding="utf-8"
    )
    for name in ("Demo_topo.tif", "Demo_mesh.msh", "Demo_mesh_core.msh",
                 "Demo_unit_defs.csv", "Demo_unit_groups.csv"):
        (folder / name).write_text("test", encoding="utf-8")
    (folder / "Demo_geology_context.txt").write_text("Observed geology", encoding="utf-8")
    return folder


def test_folder_scan_is_deterministic_and_leaves_scientific_parameters_missing(tmp_path):
    folder = _survey(tmp_path / "raw")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    found = inspect_survey_folder(folder)
    assert found["project"] == "Demo"
    assert found["gravity"]["extent"] == {
        "min_e": 100.0, "max_e": 300.0, "min_n": 200.0, "max_n": 500.0
    }
    assert found["detected"]["gravity_column"] == "ISO"
    assert found["missing_parameters"] == [
        "min_e", "max_e", "min_n", "max_n", "field_strength", "inclination", "declination"
    ]
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_configuration_requires_user_supplied_field_and_region(tmp_path):
    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    with pytest.raises(ValueError, match="Still needed from the user"):
        build_configuration(found, {})
    config = build_configuration(found, {
        "min_e": 120, "max_e": 280, "min_n": 220, "max_n": 480,
        "field_strength": 50000, "inclination": 60, "declination": 5,
    })
    assert config["region"] == {"min_e": 120.0, "max_e": 280.0,
                                 "min_n": 220.0, "max_n": 480.0}
    assert config["inversion"]["field_strength"] == 50000.0
    assert config["data"]["gravity_column"] == "ISO"
    assert config["geology"]["mode"] == "csv_manual"


def test_ai_backed_new_inversion_enables_report_but_local_run_does_not(tmp_path):
    pytest.importorskip("PyHydroGeophysX")
    from geosage.pyhydrogeophysx.configuration import configure

    found = inspect_survey_folder(_survey(tmp_path / "raw"))
    config = build_configuration(found, {
        "min_e": 120, "max_e": 280, "min_n": 220, "max_n": 480,
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
