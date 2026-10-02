"""Configuration and notebook paths must survive a different launch directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import numpy as np
import pytest

from geosage.paths import data_path, result_path, workspace_root
from geosage.runner import load_config
from geosage.compare_interpretations import run_comparison
from test_interpret_existing import _make_source


def _workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "configs").mkdir(parents=True)
    (root / "notebooks").mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="geosage"\n')
    return root


def test_config_is_anchored_to_checkout_from_any_cwd(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    config = root / "configs" / "case.json"
    config.write_text(json.dumps({"project": {"input_dir": "data/Iowa", "output_dir": "outputs/new_run"}}))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_config(config)
    assert cfg["project"]["input_dir"] == str(root / "data" / "Iowa")
    assert cfg["project"]["output_dir"] == str(root / "outputs" / "new_run")
    assert Path.cwd() == elsewhere
    assert load_config(cfg) == cfg
    monkeypatch.chdir(root / "notebooks")
    assert workspace_root() == root


def test_workspace_override_and_legacy_input_fallback(tmp_path, monkeypatch):
    checkout = _workspace(tmp_path)
    external = tmp_path / "external"
    (external / "Hannah").mkdir(parents=True)
    (external / "Hannah_Inversion_GPT").mkdir()
    monkeypatch.setenv("GEOSAGE_WORKSPACE", str(external))
    monkeypatch.chdir(checkout / "notebooks")
    assert data_path("Hannah") == external / "Hannah"
    assert result_path("Hannah_Inversion_GPT") == external / "Hannah_Inversion_GPT"
    config = checkout / "configs" / "external.json"
    config.write_text(json.dumps({"project": {"input_dir": "data/Hannah",
        "source_inversion_dir": "outputs/Hannah_Inversion_GPT"},
        "run": {"execution_mode": "interpret_existing"}}))
    cfg = load_config(config)
    assert cfg["project"]["source_inversion_dir"] == str(external / "Hannah_Inversion_GPT")
    assert cfg["project"]["input_dir"] == str(external / "Hannah")
    (external / "data" / "Hannah").mkdir(parents=True)
    assert data_path("Hannah") == external / "data" / "Hannah"


def test_explicit_workspace_is_relative_to_config_and_overrides_environment(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    monkeypatch.setenv("GEOSAGE_WORKSPACE", str(tmp_path / "wrong"))
    config = root / "configs" / "case.json"
    config.write_text(json.dumps({"project": {"workspace_dir": "..", "input_dir": "data/Iowa"}}))
    assert load_config(config)["project"]["input_dir"] == str(root / "data" / "Iowa")


def test_modern_and_legacy_case_files_are_not_mixed(tmp_path):
    (tmp_path / "data" / "Iowa").mkdir(parents=True)
    (tmp_path / "Iowa").mkdir()
    (tmp_path / "Iowa" / "old.csv").write_text("old data")
    assert data_path("Iowa/old.csv", tmp_path) == tmp_path / "data" / "Iowa" / "old.csv"


def test_execution_mode_normalization_preserves_legacy_source_lookup(tmp_path):
    source = tmp_path / "Iowa_Inversion_GPT"
    source.mkdir()
    cfg = load_config({"project": {"workspace_dir": str(tmp_path), "output_dir": "outputs/Iowa_Inversion_GPT"},
                       "run": {"execution_mode": " INTERPRET_EXISTING "}})
    assert cfg["project"]["output_dir"] == str(source)


def test_cli_reuses_data_from_outside_checkout(tmp_path):
    root = _workspace(tmp_path)
    source = _make_source(root)
    (root / "outputs").mkdir()
    source.rename(root / "outputs" / "source")
    config = root / "configs" / "run.json"
    config.write_text(json.dumps({"project": {"source_inversion_dir": "outputs/source",
        "interpretation_output_dir": "outputs/interpretation"},
        "run": {"execution_mode": "interpret_existing", "run_geology_model": False}}))
    process = subprocess.run([sys.executable, "-m", "geosage.runner", "--config", str(config)],
                             cwd=tmp_path, capture_output=True, text=True, encoding="utf-8")
    assert process.returncode == 0, process.stderr
    manifest = root / "outputs" / "interpretation" / "run_manifest.json"
    assert json.loads(manifest.read_text())["inversion_reused"] is True


@pytest.mark.parametrize("name", ["geosage-run", "geosage-agents", "geosage-compare"])
def test_installed_commands_work_from_another_directory(tmp_path, name):
    executable = Path(sysconfig.get_path("scripts")) / (name + (".exe" if os.name == "nt" else ""))
    result = subprocess.run([str(executable), "--help"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_comparison_json_resolves_reference_partition_and_base_config(tmp_path, monkeypatch):
    root = _workspace(tmp_path)
    _make_source(root, shape=(3, 3, 3))
    np.save(root / "partition.npy", np.ones((3, 3, 3), dtype=np.int16))
    base_config = {"project": {"input_dir": "data/Test"},
                   "run": {"make_plots": False, "write_reports": False}}
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base_config))
    comparison = root / "configs" / "comparison.json"
    comparison.write_text(json.dumps({"source_inversion_dir": "source", "base_config": str(base_path),
        "shared_unit_partition": "partition.npy", "reference_mask": "partition.npy",
        "scenarios": [{"scenario_id": "a", "geology_mode": "gmm_only"}]}))
    monkeypatch.chdir(tmp_path)
    result = run_comparison(comparison)
    assert Path(result["summary_json"]) == root / "outputs" / "comparisons" / "comparison_summary.json"
    assert result["rows"][0].get("reference_error") is None

    effective = json.loads((root / "outputs" / "comparisons" / "a" / "effective_config.json").read_text())
    assert effective["project"]["input_dir"] == str(root / "data" / "Test")
