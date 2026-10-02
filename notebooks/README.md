# Notebooks

Install GeoSAGE into the notebook kernel's Python environment first:
`python -m pip install -e ".[notebooks]"`. Notebooks import the `geosage` package
and discover the workspace independently of whether Jupyter starts in the checkout
root or this directory. Set `GEOSAGE_WORKSPACE` to use an external/legacy data workspace.

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
