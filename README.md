# GeoSAGE

- **Desktop host:** [PyHydroGeophysX Professional Studio](https://github.com/geohang/PyHydroGeophysX)
- **Start here:** [Install GeoSAGE and open it in PyHydroGeophysX](https://zhengyangfang.github.io/GeoSAGE/)

GeoSAGE combines joint gravity–magnetic inversion, pseudo-geological modeling,
and language-model-assisted interpretation for the Hannah and Iowa case studies.

This repository contains code, configuration templates, notebooks, and
documentation. Input data, calculated models, figures, reports, and notebook
outputs are not committed. Existing data archives are documented in the
[full workflow guide](docs/usage.md).

## Article

**GeoSAGE: A multi-agent workflow for geological reasoning from joint gravity
and magnetic inversion models**

Zhengyang Fang, Pu Yang, Yuxin Liu, Deshan Feng, and Hang Chen. *Computers &
Geosciences* 218 (2027), 106282.
[https://doi.org/10.1016/j.cageo.2026.106282](https://doi.org/10.1016/j.cageo.2026.106282)

> **New to GeoSAGE?** Follow the illustrated
> [installation and Professional Studio quick start](https://zhengyangfang.github.io/GeoSAGE/).
> You can also hand the setup to
> [Claude Code or Codex](docs/install-with-coding-agent.md).

## Project layout

| Task | Start here |
| --- | --- |
| Install and open the desktop interface | [Visual quick start](https://zhengyangfang.github.io/GeoSAGE/) |
| Explore models, figures and reports | [Studio guide](docs/studio.md) |
| Run or interpret a case | `configs/`, then the commands below |
| Reproduce paper figures and assessments | [Notebook guide](notebooks/README.md) |
| Connect private data and existing results | [Path guide](docs/paths.md) |

```text
GeoSAGE/
├── src/geosage/          # Installable Python package and command-line entry points
├── notebooks/           # Workflow, plotting, and assessment notebooks
├── configs/             # JSON examples and prompts/
├── examples/            # Quick checks and local archive validation
├── docs/                # Workflow guide and path conventions
├── pyproject.toml
└── constraints-tested.txt
```

Local inputs belong in `data/` and generated products in `outputs/`. Both are
ignored by Git. Existing flat folders such as `Hannah/`, `Iowa/`, and
`Hannah_Inversion_GPT/` are still recognized when reading older workspaces.

Inside `src/geosage/`, responsibilities are separated as follows:

| Module | Responsibility |
| --- | --- |
| `runner.py` | Configured numerical and geological workflow |
| `multi_agent_runner.py` | Agent orchestration, interpretation and review |
| `gravity_mag_joint_inversion.py`, `geo_modeling_workflow.py` | Scientific computation |
| `existing_results.py`, `paths.py`, `validation.py` | Result loading, paths and input checks |
| `plotting/` | Shared figures and colour scales for notebooks and Studio |
| `pyhydrogeophysx/` | Studio assistant, workflow and viewer integration |

Use `from geosage.plotting import render_model_sections, render_property_crossplot,
render_data_fit` for shared plotting. The default preset preserves paper exports;
`adaptive=True` selects compact views whose scales follow the current dataset.
The Studio adapter exports these figures and the interactive VTK model.

## Install

### With Claude Code or Codex

Open an empty workspace in either coding agent and paste the
[tested installation request](docs/install-agent-prompt.txt). It asks the agent
to clone the official GeoSAGE and PyHydroGeophysX repositories side by side,
create one `uv` environment, verify both packages, and open Professional Studio.
The complete walkthrough is in
[Install with Claude Code or Codex](docs/install-with-coding-agent.md).

### Manually

Use Python 3.11–3.13. From this checkout:

```bash
python -m pip install -e ".[notebooks]"
python examples/quick_test.py
```

Installing registers the package and these commands, which also work outside
the repository when given an absolute configuration path:

```bash
geosage-run --config configs/hannah_geology_only.json
geosage-agents --config configs/hannah_gmm_only.json
geosage-agents --config configs/hannah_reuse_existing_geology.json --prompt-file configs/prompts/hannah.txt
```

The first command rebuilds geology from existing inversion arrays. GMM-only mode
does not need an LLM. Report generation and review require configured credentials.
Use `reuse_existing_geology` to preserve archived geological labels exactly.

Python module entry points are also available, for example
`python -m geosage.runner --config configs/hannah_geology_only.json`.
Imports use the package name: `from geosage.runner import run_workflow`.

## Desktop Studio integration

GeoSAGE can run as an optional assistant in
[PyHydroGeophysX Professional Studio](https://github.com/geohang/PyHydroGeophysX), reusing its
workflow UI, light/dark theme, step approvals and result viewers. The integration
supports new configured inversions and exact reuse of archived models, with
offline numerical summaries or provider-backed interpretation and review.
See the [Studio integration guide](docs/studio.md) for the tested source version,
installation, supported data roles and current limitations.

## Work with local data

Open notebooks from `notebooks/` after installing the package. Paths are resolved
from the workspace rather than the notebook's current directory.

If your datasets already live elsewhere, point to that workspace without moving them:

```powershell
$env:GEOSAGE_WORKSPACE = 'D:\path\to\existing-workspace'
```

On Linux/macOS: `export GEOSAGE_WORKSPACE=/path/to/existing-workspace`.

See [path conventions and migration](docs/paths.md) for the full precedence rules,
custom configuration files, and layout examples. See [notebooks/README.md](notebooks/README.md)
for notebook purposes and prerequisites.

## Optional archive validation

```bash
python examples/validate_archived_results.py --data-root /path/to/workspace --output-dir /path/to/new/validation
```

The archive validator is optional and writes local outputs only. Historical CSV
rebuilds can differ from archived labels; it reports that comparison separately.

The numerical inversion and classification algorithms are unchanged by the package
reorganization. SimPEG remains constrained to the tested 0.25 API series.

## Documentation and license

- [Full workflow guide, archive links, and citation](docs/usage.md)
- [Install with Claude Code or Codex](docs/install-with-coding-agent.md)
- [What is and is not published](docs/open-source-scope.md)
- [Path conventions and migration](docs/paths.md)
- [Examples](examples/README.md)
- [MIT license](LICENSE)
