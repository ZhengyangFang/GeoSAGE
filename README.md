# GeoSAGE

GeoSAGE combines joint gravity–magnetic inversion, pseudo-geological modeling,
and language-model-assisted interpretation for the Hannah and Iowa case studies.

This repository contains code, configuration templates, notebooks, tests, and
documentation. Input data, calculated models, figures, reports, and notebook
outputs are not committed. Existing data archives are documented in the
[full workflow guide](docs/usage.md).

## Project layout

```text
GeoSAGE/
├── src/geosage/          # Installable Python package and command-line entry points
├── notebooks/           # Workflow, plotting, and assessment notebooks
├── configs/             # JSON examples and prompts/
├── examples/            # Quick checks and local archive validation
├── tests/               # Offline regression and synthetic inversion tests
├── docs/                # Workflow guide and path conventions
├── .github/workflows/   # Windows/Linux continuous integration
├── pyproject.toml
└── constraints-tested.txt
```

Local inputs belong in `data/` and generated products in `outputs/`. Both are
ignored by Git. Existing flat folders such as `Hannah/`, `Iowa/`, and
`Hannah_Inversion_GPT/` are still recognized when reading older workspaces.

## Install

Use Python 3.11–3.13. From this checkout:

```bash
python -m pip install -e ".[notebooks,dev]"
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

## Validate

```bash
python -m pytest
python examples/validate_archived_results.py --data-root /path/to/workspace --output-dir /path/to/new/validation
```

Tests include a small real SimPEG inversion and need no private data or API keys.
The archive validator is optional and writes local outputs only. Historical CSV
rebuilds can differ from archived labels; it reports that comparison separately.

The numerical inversion and classification algorithms are unchanged by the package
reorganization. SimPEG remains constrained to the tested 0.25 API series.

## Documentation and license

- [Full workflow guide, archive links, and citation](docs/usage.md)
- [Path conventions and migration](docs/paths.md)
- [Examples](examples/README.md)
- [MIT license](LICENSE)
