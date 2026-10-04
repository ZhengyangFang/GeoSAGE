# Notebooks

Install GeoSAGE into the notebook kernel's Python environment first:
`python -m pip install -e ".[notebooks]"`. Notebooks import the `geosage` package
and discover the workspace independently of whether Jupyter starts in the checkout
root or this directory. Set `GEOSAGE_WORKSPACE` to use an external/legacy data workspace.

For sibling `GeoSAGE/` and `PyHydroGeophysX/` checkouts sharing private data,
create an empty `.geosage-workspace` file in their parent directory. Notebooks
and configuration files then resolve `data/` and `outputs/` from that parent,
without changing working directories or copying data into the public checkout.
An explicit `GEOSAGE_WORKSPACE` or config `project.workspace_dir` still takes
precedence. Older experimental notebooks are separate archival material; use
the maintained notebooks in this directory for the supported workflow.

| Notebook prefix | Purpose | Required local resources |
| --- | --- | --- |
| `1_1`, `1_3` | Prompt-driven inversion and interpretation | Case inputs and LLM credentials |
| `2_` | Study-area observations and map | Case inputs; optional map downloads |
| `3_1`, `3_2` | Plot saved Hannah/Iowa results | Corresponding inversion and geology archives |
| `4_1` | Assess data fit | Observed and predicted arrays |
| `4_2` | Sensitivity presentation | Meshes, recovered models, iteration logs |
| `4_3` | Compare misfit histories | Model-specific iteration logs |
| `4_4` | Well validation | Local `logdata/Hannah` records and saved model |
| `5_2` | Compare LLM interpretations | Saved model and model-specific credentials |

Input paths prefer `data/`; saved model paths prefer `outputs/`, with fallback to
legacy flat directories. New figures go to `outputs/figures/` and interpretation
comparisons to `outputs/Hannah_LLM_comparison/`. Keep notebook outputs cleared
before committing. No input datasets or calculated results are included here.

The 2D plotting cells in `3_1`, `3_2`, and `4_1` call
`geosage.plotting.paper`, also used by Studio. Edit that shared implementation
when maintaining these figures. Notebook paths, panel conventions and 600 dpi
exports are retained; the original 3D publication cells remain in the notebooks.
