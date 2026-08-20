# GeoSAGE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19958815.svg)](https://doi.org/10.5281/zenodo.19958815)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GeoSAGE is a reproducible multi-agent workflow for geological reasoning from joint
gravity and magnetic inversion models. The project accompanies the manuscript
**A Multi-Agent Framework for Geological Reasoning From Joint Gravity and Magnetic
Inversion Models**.

This GitHub repository is intended to contain the source code, notebooks, project
metadata, and lightweight examples. The large case-study data and representative
outputs are archived on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.19958815
- Zenodo record: https://zenodo.org/records/19958815
- Version: 1.0.2

## What This Project Does

GeoSAGE supports an end-to-end workflow for two case studies, Hannah and Iowa:

- joint inversion of gravity and magnetic data;
- construction of pseudo-geological models from recovered density and susceptibility;
- manual or automatic grouping of inversion-derived units into geological groups;
- language-model-assisted interpretation and report generation;
- 2D slices, 3D views, cross-plots, comparison figures, and metric extraction.

The workflow can be used in two ways:

- **Code-first reproduction:** download the Zenodo data, install the Python environment,
  and rerun inversion/modeling scripts.
- **Result-first inspection:** download the Zenodo representative outputs and run the
  plotting notebooks without repeating the expensive inversion.

## Repository And Data Policy

Large files should not be committed to GitHub. The data archive contains `.zip`
packages with the input data, generated models, figures, and representative outputs.
Keep these downloaded folders untracked in Git.

Current GitHub repository contents:

| Path | Purpose |
|---|---|
| `gravity_mag_joint_inversion.py` | Joint gravity-magnetic inversion workflow. |
| `geo_modeling_workflow.py` | Pseudo-geological model construction. |
| `runner.py` | JSON-driven deterministic workflow runner. |
| `multi_agent_runner.py` | Natural-language multi-agent workflow and report generation. |
| `notebook_geo_helpers.py` | Shared helper functions used by plotting notebooks. |
| `*.ipynb` | Reproduction, plotting, comparison, and metric notebooks. |
| `Hannah_model_metrics_table.csv` | Lightweight summary table. |
| `Figure/` | Optional selected publication figures. |
| `pyproject.toml` | Python project metadata and dependencies. |
| `LICENSE` | MIT license for original code. |

Zenodo-only large data and output archives:

| Path or pattern | Reason |
|---|---|
| `Hannah/`, `Iowa/` | Case-study input data, including large topography rasters. |
| `*_Inversion_*/` | Inversion outputs, reports, meshes, intermediate models, and figures. |
| `*.tif`, `*.npy`, `*.h5`, large `*.txt` | Too large or too generated for normal Git history. |
| `iteration_model/` | Iteration snapshots from inversion runs. |
| `reports/*.pdf` | Generated report artifacts. |

## Zenodo Archive Contents

The Zenodo record contains both data archives and a snapshot of the code release.
When using GitHub as the code source, the most important files to download from
Zenodo are the case-study folders and representative output folders.

| Archive | Purpose | Size |
|---|---:|---:|
| `Hannah.zip` | Hannah input data and supporting files. | 370.9 MiB |
| `Iowa.zip` | Iowa input data and supporting files. | 277.0 MiB |
| `Hannah_Inversion_GPT.zip` | Hannah representative GPT output. | 131.7 MiB |
| `Hannah_Inversion_GPT_auto_group.zip` | Hannah GPT output with automatic grouping. | 112.0 MiB |
| `Hannah_Inversion_claude.zip` | Hannah representative Claude output. | 91.1 MiB |
| `Hannah_Inversion_gemini.zip` | Hannah representative Gemini output. | 105.7 MiB |
| `Hannah_Inversion_Qwen.zip` | Hannah representative Qwen output. | 81.0 MiB |
| `Iowa_Inversion_GPT.zip` | Iowa representative GPT output. | 327.2 MiB |
| `Figure.zip` | Publication and supplementary figures. | 45.2 MiB |

## Installation

Use Python 3.11, 3.12, or 3.13. Python 3.11 or 3.12 is recommended for the
widest availability of scientific and geospatial wheels.

### Option A: Install With `uv`

```bash
uv venv --python 3.11
uv pip install -e ".[notebooks,reports]"
```

On Windows PowerShell:

```powershell
uv venv --python 3.11
uv pip install -e ".[notebooks,reports]"
```

### Option B: Install With `pip`

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,reports]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,reports]"
```

The `notebooks` extra installs JupyterLab, IPython kernel support, nbconvert, and
Cartopy. The `reports` extra installs PDF text-extraction fallbacks used by the
multi-agent report workflow.

### External Tools For PDF Reports

The Python dependencies do not install the Pandoc command-line program or a PDF
engine. These are needed only if you want the multi-agent workflow to export PDF
reports.

Install:

- Pandoc: https://pandoc.org/installing.html
- One PDF engine, for example MiKTeX, TeX Live, Tectonic, wkhtmltopdf, or WeasyPrint

If these tools are missing, the workflow can still generate Markdown reports.

## Download The Zenodo Data

Run the download commands from the root of this GitHub repository. After extraction,
the folders such as `Hannah/`, `Iowa/`, and `Iowa_Inversion_GPT/` should sit next
to `runner.py` and `README.md`.

### Minimal Download For Rerunning Inversion

Download only the input folders:

```powershell
$files = @("Hannah.zip", "Iowa.zip")
foreach ($file in $files) {
    $url = "https://zenodo.org/api/records/19958815/files/$file/content"
    Invoke-WebRequest -Uri $url -OutFile $file
    Expand-Archive -Path $file -DestinationPath . -Force
}
```

Linux/macOS:

```bash
for file in Hannah.zip Iowa.zip; do
  curl -L -o "$file" "https://zenodo.org/api/records/19958815/files/${file}/content"
  unzip -o "$file"
done
```

### Download Representative Outputs For Fast Reproduction

Use this option if you want to run plotting notebooks and inspect model outputs
without recomputing the full inversion.

Windows PowerShell:

```powershell
$files = @(
    "Hannah.zip",
    "Iowa.zip",
    "Hannah_Inversion_GPT.zip",
    "Hannah_Inversion_GPT_auto_group.zip",
    "Hannah_Inversion_claude.zip",
    "Hannah_Inversion_gemini.zip",
    "Hannah_Inversion_Qwen.zip",
    "Iowa_Inversion_GPT.zip",
    "Figure.zip"
)

foreach ($file in $files) {
    $url = "https://zenodo.org/api/records/19958815/files/$file/content"
    Invoke-WebRequest -Uri $url -OutFile $file
    Expand-Archive -Path $file -DestinationPath . -Force
}
```

Linux/macOS:

```bash
for file in \
  Hannah.zip \
  Iowa.zip \
  Hannah_Inversion_GPT.zip \
  Hannah_Inversion_GPT_auto_group.zip \
  Hannah_Inversion_claude.zip \
  Hannah_Inversion_gemini.zip \
  Hannah_Inversion_Qwen.zip \
  Iowa_Inversion_GPT.zip \
  Figure.zip
do
  curl -L -o "$file" "https://zenodo.org/api/records/19958815/files/${file}/content"
  unzip -o "$file"
done
```

### Expected Directory Layout After Extraction

```text
GeoSAGE/
  README.md
  pyproject.toml
  runner.py
  multi_agent_runner.py
  notebook_geo_helpers.py
  gravity_mag_joint_inversion.py
  geo_modeling_workflow.py
  Hannah/
    Hannah_gravity_data.csv
    Hannah_magnetic_data.csv
    Hannah_topo.tif
    Hannah_mesh.msh
    Hannah_mesh_core.msh
    Hannah_unit_defs.csv
    Hannah_unit_groups.csv
    Hannah_geology_context.pdf
  Iowa/
    Iowa_gravity_data.csv
    Iowa_magnetic_data.csv
    Iowa_topo.tif
    Iowa_mesh.msh
    Iowa_mesh_core.msh
    Iowa_unit_defs.csv
    Iowa_unit_groups.csv
    Iowa_geology_context.txt
  Hannah_Inversion_GPT/
  Hannah_Inversion_GPT_auto_group/
  Hannah_Inversion_claude/
  Hannah_Inversion_gemini/
  Hannah_Inversion_Qwen/
  Iowa_Inversion_GPT/
  Figure/
```

## Configure LLM Access

The deterministic inversion and geology modeling scripts do not require an LLM
API. The multi-agent interpretation workflow does.

For OpenAI-compatible APIs, set these environment variables:

```bash
export OPENAI_API_KEY="your_api_key"
export LLM_MODEL="gpt-4o-mini"
```

If you use OpenRouter or another OpenAI-compatible endpoint:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="openai/gpt-4o-mini"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "your_api_key"
$env:LLM_MODEL = "gpt-4o-mini"
```

For notebook workflows, the first cells expose:

```python
API_BASE = ""
MODEL_NAME = ""
API_KEY = ""
```

Set these values before running the multi-agent notebooks. Do not commit real API
keys to GitHub.

## Quick Start: Rebuild Geology Models From Archived Inversion Results

This is the fastest end-to-end smoke test because it uses the downloaded
representative inversion output and reruns only the pseudo-geological modeling step.

After installing the project dependencies and before running the workflow itself,
you can run the lightweight example check:

```bash
python examples/quick_test.py
```

This verifies that the code imports, the example configuration loads, and the
expected Zenodo data folders are present. It does not run the full inversion.

Create `config_hannah_geology_only.json`:

```json
{
  "project": {
    "name": "Hannah",
    "input_dir": "Hannah",
    "output_dir": "Hannah_Inversion_GPT"
  },
  "geology": {
    "mode": "csv_manual",
    "unit_defs_csv": "Hannah/Hannah_unit_defs.csv",
    "unit_groups_csv": "Hannah/Hannah_unit_groups.csv",
    "context_path": "Hannah/Hannah_geology_context.pdf",
    "min_voxels": 10,
    "fill_iterations": 3
  },
  "run": {
    "run_inversion": false,
    "run_geology_model": true,
    "make_plots": true
  }
}
```

Run:

```bash
python runner.py --config config_hannah_geology_only.json
```

Expected outputs are written under:

```text
Hannah_Inversion_GPT/geology_models/
Hannah_Inversion_GPT/geology_models/slices_and_sections/
```

You can do the same for Iowa with `output_dir` set to `Iowa_Inversion_GPT` and
the input paths changed to `Iowa/...`.

## Reproduce The Hannah Workflow From Input Data

Full inversion is computationally expensive. Use a new output directory if you do
not want to overwrite the archived representative output.

Create `config_hannah_full.json`:

```json
{
  "project": {
    "name": "Hannah",
    "input_dir": "Hannah",
    "output_dir": "Hannah_Inversion_repro"
  },
  "region": {
    "min_e": 510000.0,
    "max_e": 535000.0,
    "min_n": 4290000.0,
    "max_n": 4320000.0
  },
  "data": {
    "gravity_column": "ISO",
    "gravity_component": "gz",
    "std_grv": 0.25,
    "std_mag": 10.0,
    "std_grv_relative": false,
    "std_mag_relative": false,
    "flight_height_ft": 1000.0
  },
  "inversion": {
    "inclination": 62.0,
    "declination": 15.0,
    "field_strength": 50686.0,
    "reg_coefficient": [1.0, 1.0],
    "reg_grv_norm": [1.0, 2.0, 2.0, 2.0],
    "reg_mag_norm": [1.0, 2.0, 2.0, 2.0],
    "cross_gradient_lambda": 1000000000000.0,
    "beta_cooling": 1.1,
    "optimization": {
      "maxGNCG": 100
    }
  },
  "geology": {
    "mode": "csv_manual",
    "unit_defs_csv": "Hannah/Hannah_unit_defs.csv",
    "unit_groups_csv": "Hannah/Hannah_unit_groups.csv",
    "context_path": "Hannah/Hannah_geology_context.pdf",
    "target_name": "Serpentinite hydrogen play along Collayomi fault",
    "min_voxels": 10,
    "fill_iterations": 3
  },
  "run": {
    "run_inversion": true,
    "run_geology_model": true,
    "make_plots": true
  }
}
```

Run:

```bash
python runner.py --config config_hannah_full.json
```

Main outputs:

```text
Hannah_Inversion_repro/observed_data/
Hannah_Inversion_repro/iteration_model/
Hannah_Inversion_repro/inversion_result/
Hannah_Inversion_repro/geology_models/
```

## Reproduce The Iowa Workflow From Input Data

Create `config_iowa_full.json`:

```json
{
  "project": {
    "name": "Iowa",
    "input_dir": "Iowa",
    "output_dir": "Iowa_Inversion_repro"
  },
  "region": {
    "min_e": 580000.0,
    "max_e": 615000.0,
    "min_n": 4785000.0,
    "max_n": 4825000.0
  },
  "data": {
    "gravity_column": "Gzz",
    "gravity_component": "gzz",
    "std_grv": 0.25,
    "std_mag": 10.0,
    "std_grv_relative": false,
    "std_mag_relative": false,
    "flight_height_ft": 1000.0
  },
  "inversion": {
    "inclination": 70.0,
    "declination": 0.0,
    "field_strength": 55068.0,
    "reg_coefficient": [108.0, 270.0],
    "reg_grv_norm": [2.0, 2.0, 2.0, 2.0],
    "reg_mag_norm": [2.0, 2.0, 2.0, 2.0],
    "cross_gradient_lambda": 3000000000000000.0,
    "optimization": {
      "maxGNCG": 50
    },
    "irls": {
      "maxIRLSiter": 50
    }
  },
  "geology": {
    "mode": "csv_manual",
    "unit_defs_csv": "Iowa/Iowa_unit_defs.csv",
    "unit_groups_csv": "Iowa/Iowa_unit_groups.csv",
    "context_path": "Iowa/Iowa_geology_context.txt",
    "target_name": "Iowa mineral exploration target",
    "min_voxels": 10,
    "fill_iterations": 3
  },
  "run": {
    "run_inversion": true,
    "run_geology_model": true,
    "make_plots": true
  }
}
```

Run:

```bash
python runner.py --config config_iowa_full.json
```

Main outputs:

```text
Iowa_Inversion_repro/observed_data/
Iowa_Inversion_repro/iteration_model/
Iowa_Inversion_repro/inversion_result/
Iowa_Inversion_repro/geology_models/
```

## Reinterpret an existing inversion without rerunning SimPEG

GeoSAGE can now treat a completed inversion directory as an immutable evidence source. In
`interpret_existing` mode, `run_joint_inversion()` is never called. The archived mesh, density,
susceptibility, optional predicted data, and inversion parameter snapshot are loaded read-only;
new geology models, reports, review files, and manifests are written to a separate interpretation
directory.

The additional configuration fields are:

```json
{
  "project": {
    "source_inversion_dir": "Hannah_Inversion_GPT",
    "interpretation_output_dir": "Hannah_Inversion_GPT_interpretations/my_run"
  },
  "run": {
    "execution_mode": "interpret_existing",
    "skip_configuration_agents": true,
    "reuse_existing_geology": false,
    "overwrite": false,
    "review_enabled": false,
    "max_review_rounds": 1
  }
}
```

The supported geology modes are `csv_manual`, `reuse_existing_geology`, `gmm_bic_auto`,
`fixed_units_llm_groups`, `fixed_units_fixed_groups`, and `gmm_only`. GMM artifacts and all
interpretation outputs are stored below the destination directory. Every run writes
`effective_config.json`, `source_manifest.json`, `run_manifest.json`, `evidence_bundle.json`, and
`agent_trace.json`; the source manifest records the available inversion artifacts and their sizes.
The evidence and trace files are also written when report generation is disabled. Agent traces
redact common API-key formats before they are saved.

Example commands:

```powershell
# Reuse the archived quasi-geological model and generate a reviewed report.
python multi_agent_runner.py --config configs/hannah_reuse_existing_geology.json --prompt-file prompts/hannah.txt

# Keep fixed unit labels, but let the PetrologyAgent propose names and Geo groups.
python multi_agent_runner.py --config configs/hannah_fixed_units_llm_groups.json --prompt-file prompts/hannah.txt

# Deterministic GMM/BIC baseline with one-to-one Geo groups.
python multi_agent_runner.py --config configs/hannah_gmm_only.json
```

For a comparison study, use `compare_interpretations.py` with a JSON containing one
`source_inversion_dir`, a shared request, and separate scenario output directories. The comparison
runner records the shared source inventory once and writes `comparison_summary.csv` and
`comparison_summary.json`. The four low-cost comparison designs map to scenario fields as follows:

| Comparison | Scenario setup |
|---|---|
| Report-only | `fixed_units_fixed_groups`, shared `unit_id_npy`/group CSV, `write_reports=true`; vary model or review. |
| Grouping-only | `fixed_units_llm_groups`, shared `unit_id_npy`, `write_reports=false`; vary model. |
| Prior sensitivity | Fixed Unit labels plus a different `context_path` for each scenario. |
| Review ablation | Duplicate the same scenario with `review_enabled=false` and `true`. |

Each scenario may set `write_reports`, `review_enabled`, `model`, `base_url`, `context_path`, and
`unit_id_npy` independently. API keys are read only from environment variables.

Existing full-run configurations remain valid. A legacy configuration with `run_inversion=false`
is interpreted as `interpret_existing`; if no separate output directory is supplied, GeoSAGE
creates a safe sibling `<source>_interpretations/` destination rather than modifying the source directory.

## Run The Multi-Agent Workflow

The multi-agent workflow converts a natural-language prompt into a workflow
configuration, runs the inversion/geological modeling steps, and writes an
interpretive report.

Example:

```bash
python multi_agent_runner.py --prompt "Please use the Hannah data to run a joint gravity-magnetic inversion, construct a 3D pseudo-geological model, and write a detailed report about hydrogen exploration potential. Use ./Hannah_Inversion_agent as the output directory."
```

For a notebook-based version, open:

```bash
jupyter lab
```

Then run one of:

| Notebook | Purpose |
|---|---|
| `1_1_language_driven_multi_agents_Hannah.ipynb` | Hannah multi-agent workflow with manual grouping. |
| `1_2_language_driven_multi_agents_Hannah_auto_group.ipynb` | Hannah multi-agent workflow with automatic grouping. |
| `1_3_language_driven_multi_agents_Iowa.ipynb` | Iowa multi-agent workflow. |

Before running these notebooks, set `API_BASE` and `MODEL_NAME` in the first configuration cell,
and set `OPENAI_API_KEY` (or the provider-specific environment variable) in the shell. Notebook
code reads the key from the environment and does not store credentials in the notebook.

## Reproduce Figures And Tables From Archived Outputs

Download the representative output archives first. Then run:

```bash
jupyter lab
```

Suggested notebook order:

| Step | Notebook | Output |
|---:|---|---|
| 1 | `2_plot_Study_Area_data.ipynb` | Study-area data maps and observed gravity/magnetic figures. |
| 2 | `3_1_plot_Hannah_result.ipynb` | Hannah model slices and 3D visualization. |
| 3 | `3_2_plot_Iowa_result.ipynb` | Iowa model slices and 3D visualization. |
| 4 | `4_plot_Hannah_result_discussion.ipynb` | Hannah comparison and discussion figures. |
| 5 | `5_extract_Hannah_model_metrics.ipynb` | Model metrics table. |

Generated figures are written to `Figure/` or to the relevant inversion output
folder, depending on the notebook.

## Main Input File Conventions

For a project named `ProjectName`, the deterministic workflow expects:

```text
ProjectName/
  ProjectName_gravity_data.csv
  ProjectName_magnetic_data.csv
  ProjectName_topo.tif
  ProjectName_mesh.msh
  ProjectName_mesh_core.msh
  ProjectName_unit_defs.csv
  ProjectName_unit_groups.csv
  ProjectName_geology_context.txt or ProjectName_geology_context.pdf
```

The gravity and magnetic CSV files must contain spatial coordinates and the data
columns referenced by the JSON configuration. The mesh files are UBC-style tensor
meshes read by `discretize`.

## Output File Conventions

For an output directory such as `Hannah_Inversion_repro/`, the workflow creates:

```text
Hannah_Inversion_repro/
  observed_data/
    gravity.obs
    magnetics.obs
  mesh/
    mesh.msh
    mesh_core.msh
  topo/
    topography.xyz
  iteration_model/
    InversionModel_*.npy
    Output_*.txt
  inversion_result/
    inversion_result_dens_susc.npy
    inversion_result_density_active.npy
    inversion_result_susceptibility_active.npy
    joint_density_core.npy
    joint_susceptibility_core.npy
    joint_density_*_UBC.txt
    joint_susceptibility_*_UBC.txt
    image/
  geology_models/
    unit_id_3d.npy
    geo_id_3d.npy
    density_susceptibility_scatter_by_unit.png
    geo_model_without_background.jpg
    slices_and_sections/
```

These generated folders can be large and should normally remain outside Git.

## Troubleshooting

### `FileNotFoundError` For `Hannah/...` Or `Iowa/...`

The Zenodo archives were likely extracted in the wrong directory. Make sure
`Hannah/` and `Iowa/` are directly inside the repository root.

### Missing API Key

The deterministic `runner.py` workflow does not require an API key. The
multi-agent workflow does. Set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY` when using OpenRouter)
in the environment.

### OpenRouter Returns HTML Instead Of JSON

Use an API base URL ending in `/v1`:

```bash
export LLM_BASE_URL="https://openrouter.ai/api/v1"
```

### PDF Report Export Fails

Check that Pandoc and a PDF engine are installed and visible on `PATH`. The
Markdown report is still useful even when PDF export fails.

### PyVista Or 3D Rendering Fails On A Headless Server

Try enabling off-screen rendering before running plotting code:

```bash
export PYVISTA_OFF_SCREEN=true
```

Windows PowerShell:

```powershell
$env:PYVISTA_OFF_SCREEN = "true"
```

### GitHub Rejects Large Files

Do not commit downloaded Zenodo archives or extracted data folders. Add them to
`.gitignore` before the first commit. Large files already committed to Git history
must be removed from history before pushing.

## Citation

If you use GeoSAGE, please cite the associated manuscript and the Zenodo release:

```text
Fang, Zhengyang, and Chen, Hang. 2026. GeoSAGE: A Multi-Agent Framework for
Geological Reasoning From Joint Gravity and Magnetic Inversion Models.
Zenodo. https://doi.org/10.5281/zenodo.19958815
```

Please also cite the original data and case-study sources where applicable:

```text
Su, Y., Wu, S., Sun, J., Wu, X., Huang, Y., Chen, J., et al. 2025.
Natural hydrogen exploration by joint sparse inversion of geophysical
measurements and integrated geological interpretation. International Journal
of Hydrogen Energy, 173, 151040. https://doi.org/10.1016/j.ijhydene.2025.151040

Sun, J., Melo, A. T., Kim, J. D., and Wei, X. 2020. Unveiling the 3D undercover
structure of a Precambrian intrusive complex by integrating airborne magnetic
and gravity gradient data into 3D quasi-geology model building. Interpretation,
8(4), SS15-SS29. https://doi.org/10.1190/INT-2019-0273.1
```

## License And Data Terms

The original GeoSAGE source code is released under the MIT License. See
`LICENSE`.

Third-party data, maps, figures, publications, and externally sourced materials
remain subject to their original licenses, access conditions, and copyright
restrictions. Users are responsible for checking and complying with those terms
before redistribution or reuse.

## Contact

For questions about this release:

- Zhengyang Fang: zhengyfang@uiowa.edu
- Hang Chen: hchen117@uiowa.edu
