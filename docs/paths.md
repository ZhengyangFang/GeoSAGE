# Workspace paths and migration

Install the project with `python -m pip install -e ".[notebooks,dev]"` before running
notebooks or command-line tools. Source files now live in `src/geosage/`; imports
must start with `geosage.`. Replace `python runner.py` with `geosage-run` or
`python -m geosage.runner`; use `geosage-agents` and `geosage-compare` for the other
two entry points. The old root-level modules and `tools/` helper are removed.

## Preferred local layout

```text
workspace/
├── .geosage-workspace    # Optional marker for a shared local workspace
├── GeoSAGE/              # This source checkout
├── PyHydroGeophysX/      # Optional Studio source checkout
├── data/
│   ├── Hannah/
│   ├── Iowa/
│   └── logdata/
├── outputs/
│   ├── Hannah_Inversion_GPT/
│   ├── Iowa_Inversion_GPT/
│   ├── figures/
│   └── comparisons/
└── studio/               # Optional local Studio project
```

These data directories are local and not included in Git. No archive contents
need to be renamed. Extract each input case under `data/` and each result archive
under `outputs/`. The validator's `--data-root` argument names the workspace that
contains those two directories, not `data/` itself.

Flat legacy layouts are supported: when `data/Hannah` is absent, an existing
`Hannah` directory directly under the workspace is used; the same applies to
Iowa, logdata, and existing inversion result directories. Modern directories win
when both layouts exist. New inversion and interpretation outputs use `outputs/`.
An explicitly supplied absolute path is never relocated.

## Workspace selection

For workflow JSON configurations:

1. `project.workspace_dir`, if provided, selects the workspace. A relative value
   is relative to the JSON file's directory; for a Python dictionary it is relative
   to the current working directory.
2. Otherwise, `GEOSAGE_WORKSPACE` selects the workspace.
3. Otherwise, discovery walks upward from the configuration file (or cwd for a
   Python dictionary). A `.geosage-workspace` file selects its directory. A
   GeoSAGE `pyproject.toml` selects its checkout, except when the checkout's
   immediate parent contains that marker: then the parent is the workspace.
   This lets sibling code checkouts share private `data/` and `outputs/`.
4. If neither a marker nor a checkout is found, the configuration file's directory
   (or cwd) is used.

All other relative input/output paths are interpreted within that workspace.
The configuration is copied before path resolution, so caller-owned dictionaries
are not changed. Effective configurations contain absolute paths and remain stable
when loaded again from another directory. No library call changes `os.getcwd()`.

For example, a standalone JSON at `/work/configs/my_run.json` can contain:

```json
{
  "project": {
    "workspace_dir": "..",
    "name": "Iowa",
    "input_dir": "data/Iowa",
    "source_inversion_dir": "outputs/Iowa_Inversion_GPT",
    "interpretation_output_dir": "outputs/Iowa_interpretations"
  },
  "geology": {
    "mode": "reuse_existing_geology",
    "unit_defs_csv": "data/Iowa/Iowa_unit_defs.csv",
    "unit_groups_csv": "data/Iowa/Iowa_unit_groups.csv",
    "context_path": "data/Iowa/Iowa_geology_context.txt"
  },
  "run": {
    "execution_mode": "interpret_existing",
    "run_inversion": false,
    "make_plots": false,
    "write_reports": false,
    "review_enabled": false
  }
}
```

Run it from any directory with `geosage-agents --config /work/configs/my_run.json`.
CLI arguments such as `--config` and `--prompt-file` themselves are relative to the
shell's current directory, as usual; use absolute paths when running elsewhere.

Comparison JSON files accept `workspace_dir` at the top level. Their source,
base-config, shared partition, reference-mask, and scenario paths are relative to
that workspace. Notebook helpers use the environment override or discover the
checkout from the notebook directory. To use an external data workspace, set
`GEOSAGE_WORKSPACE` before starting the kernel (or before the first cell).

## Preserve previous results

Use `run.execution_mode="interpret_existing"` and
`geology.mode="reuse_existing_geology"` to reuse old arrays exactly. Rebuilding
from CSV is a different operation and retains the current topography masking and
cleanup behavior. New interpretation directories are separated from source results;
repeated runs reserve unique directories unless overwrite is explicitly enabled.
