from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from geosage import geo_modeling_workflow as geology
from geosage import runner
from geosage.compare_interpretations import run_comparison
from geosage.existing_results import load_existing_geology_result, load_existing_inversion_result
from test_interpret_existing import _make_source


def test_legacy_regularization_and_config_isolation(tmp_path):
    supplied = {"inversion": {"reg_beta": [7.0, 9.0]}, "custom": {"items": [1]}}
    original = deepcopy(supplied)
    cfg = runner.load_config(supplied)
    assert cfg["inversion"]["reg_coefficient"] == [7.0, 9.0]
    cfg["inversion"]["reg_beta"][0] = 0
    cfg["custom"]["items"].append(2)
    assert supplied == original
    supplied["inversion"]["reg_coefficient"] = [3.0, 4.0]
    assert runner.load_config(supplied)["inversion"]["reg_coefficient"] == [3.0, 4.0]
    path = tmp_path / "bom.json"
    path.write_text('{"project":{"name":"BOM"}}', encoding="utf-8-sig")
    assert runner.load_config(path)["project"]["name"] == "BOM"


def test_archived_dictionary_bounds_override_defaults():
    cfg = runner._archived_parameters_to_config(runner.load_config({}), {
        "inv_bound": {"grv_lb": -2, "grv_ub": 3, "mag_lb": 0, "mag_ub": 0.1}
    })
    assert cfg["inversion"]["grv_bounds"] == [-2, 3]
    assert cfg["inversion"]["mag_bounds"] == [0, 0.1]


@pytest.mark.parametrize("explicit", [True, False])
def test_repeated_output_allocation_never_overwrites(tmp_path, explicit):
    source = tmp_path / "source"
    source.mkdir()
    requested = tmp_path / "out" if explicit else None
    outputs = [runner._safe_interpretation_output_dir(source, requested, False) for _ in range(5)]
    assert len(set(outputs)) == 5
    assert all(path.is_dir() for path in outputs)


def test_comparison_cannot_write_shared_units_into_source(tmp_path):
    source = _make_source(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        run_comparison({"source_inversion_dir": str(source), "comparison_root": str(source / "comparison")})
    assert not (source / "comparison").exists()


def _csv_inputs(tmp_path):
    defs = tmp_path / "defs.csv"
    groups = tmp_path / "groups.csv"
    defs.write_text("unit_id,name,dens_min,dens_max,susc_min,susc_max\n1,Test,-1,10,-1,10\n")
    groups.write_text("unit_id,geo_id,geo_name\n1,2,Test group\n")
    return defs, groups


def test_no_plot_does_not_invoke_renderers(tmp_path, monkeypatch):
    source = _make_source(tmp_path)
    defs, groups = _csv_inputs(tmp_path)
    def fail(*args, **kwargs):
        raise AssertionError("renderer invoked with make_plots=False")
    monkeypatch.setattr(geology.plt, "figure", fail)
    monkeypatch.setattr(geology.pv, "Plotter", fail)
    out = tmp_path / "out"
    result = geology.build_geology_model(inversion_dir=source, output_dir=out,
        unit_defs_csv=defs, unit_groups_csv=groups, make_plots=False, min_voxels=1)
    np.testing.assert_array_equal(result["geo_id_3d"], np.full((2, 2, 2), 2))
    assert not list(out.rglob("*.png"))


def test_requested_missing_partition_is_not_silently_reclassified(tmp_path):
    source = _make_source(tmp_path)
    defs, groups = _csv_inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="partition"):
        geology.build_geology_model(inversion_dir=source, output_dir=tmp_path / "out",
            unit_defs_csv=defs, unit_groups_csv=groups,
            unit_id_npy=tmp_path / "missing.npy", make_plots=False)


def test_archived_labels_load_without_optional_names(tmp_path):
    source = _make_source(tmp_path)
    geo_dir = source / "geology_models"
    geo_dir.mkdir()
    labels = np.full((2, 2, 2), 7, dtype=np.int16)
    for name in ("unit_id_3d", "geo_id_3d"):
        np.save(geo_dir / f"{name}.npy", labels)
    with pytest.warns(RuntimeWarning, match="names unavailable"):
        result = load_existing_geology_result(source, load_existing_inversion_result(source))
    np.testing.assert_array_equal(result["geo_id_3d"], labels)
    assert result["geo_defs"][7] == "Geo ID 7 (name unavailable)"
    assert not (geo_dir / "geo_defs.json").exists()


def test_mapping_recovered_from_report_but_counts_from_arrays(tmp_path):
    source = _make_source(tmp_path)
    geo_dir = source / "geology_models"
    geo_dir.mkdir()
    labels = np.full((2, 2, 2), 5, dtype=np.int16)
    for name in ("unit_id_3d", "geo_id_3d"):
        np.save(geo_dir / f"{name}.npy", labels)
    reports = source / "reports"
    reports.mkdir()
    report = reports / "original.md"
    report.write_text("5. **Geo Group 5 \u2013 Primary Mafic Magmatic System**  \n   - Voxel count: 999\n", encoding="utf-8")
    result = load_existing_geology_result(source, load_existing_inversion_result(source))
    assert result["geo_defs"] == {5: "Primary Mafic Magmatic System"}
    assert result["geo_defs_source"]["paths"] == [str(report)]
    assert np.sum(result["geo_id_3d"] == 5) == 8
    assert not (geo_dir / "geo_defs.json").exists()
    (reports / "conflict.md").write_text("5. **Geo Group 5 - Different name**\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting"):
        load_existing_geology_result(source, load_existing_inversion_result(source))
