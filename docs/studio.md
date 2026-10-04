# GeoSAGE in PyHydroGeophysX Studio

GeoSAGE is an optional assistant inside the existing PyHydroGeophysX Qt Studio.
It uses the host's light/dark theme, chat settings, workflow timeline, step
approvals, cancellation, result history, figure previews and 3D viewer. It does
not introduce a second desktop application or replace the inversion kernel.

## Current compatibility

The adapter targets the assistant entry-point API in PyHydroGeophysX 0.5.0,
tested against upstream commit `e8c0c5cec1e4bb6cb4bdcbfabf69f279814d0ac2`.
Use the source checkouts while this integration is under review: a release
carrying the same version number may predate that API.

The accompanying host changes add an explicit offline-workflow capability,
multi-property VTK selection, per-property colour defaults and a persistent
`needs_review` status. The plugin can register with the unmodified scaffold;
these UI enhancements and the ``geosage-studio`` launcher require the accompanying host branch. A packaged desktop
executable will need a rebuild to include this optional package.

## Install from two local checkouts

Use a separate Python 3.11–3.13 environment to preserve an existing scientific
environment. From the parent containing `GeoSAGE/` and `PyHydroGeophysX/`:

```powershell
python -m venv .venv-studio
.venv-studio\Scripts\python -m pip install -c GeoSAGE/constraints-tested.txt -e "./PyHydroGeophysX[desktop,desktop-3d]" -e "./GeoSAGE[dev]"
.venv-studio\Scripts\geosage-studio --module one_click
```

On Linux/macOS use `.venv-studio/bin/python` and `.venv-studio/bin/geosage-studio`.
No API key is needed for the numerical workflow. PyGIMLi, hydrological solvers,
CUDA and the host's other domain extras are not required for GeoSAGE.

## Inspect existing results

1. Open or create a **Project** outside the source data and inversion folders.
2. Select **GeoSAGE** in the assistant picker. In **Data & reports → Data**, add
   the folder with role **Existing GeoSAGE inversion folder**.
3. Choose **Inspect existing results · local**, use **Check inputs & preview configuration**,
   then select **Open results**. Enable **Approve each step before it runs** if
   you want to examine each stage. Stop preserves partial outputs in this run.
4. Inspect **Results & report**, or **Saved Results → Visualization**. The VTK
   property selector offers density contrast, susceptibility, Geo ID and Unit ID;
   the clipping plane exposes interior cells. Figures remain available without
   a compatible OpenGL renderer.
5. Use **Save this run to Project** to retain the run in Studio's history across sessions.
   An unsaved run's files exist locally but are not part of saved history.

At minimum, an archive needs:

```text
source/
  mesh/mesh_core.msh
  inversion_result/joint_density_core.npy
  inversion_result/joint_susceptibility_core.npy
```

With both `geology_models/unit_id_3d.npy` and `geo_id_3d.npy`, labels are reused
exactly. Existing GeoSAGE metadata recovery handles missing group definitions;
it does not infer new labels. Without labels, the default is unsupervised GMM
property grouping, explicitly not verified lithology. Supply a JSON config to
choose another geological mode or explicit targets.

## Run a new inversion

Choose **Run a new inversion · local**. Add a **GeoSAGE configuration (JSON)** with an explicit project name, region,
survey and inversion settings. Existing configs retain their workspace-relative
path rules. The adapter does not infer physical settings from a chat request.
Use **Create / edit inversion configuration** to edit physical settings for this run;
imported advanced settings are retained and the original JSON is not modified.
**View all resolved parameters** shows the complete effective configuration.
Supply an input folder with these names, or assign the five individual roles:

| File | Required contents |
|---|---|
| `<project>_gravity_data.csv` | Easting, Northing, Longitude, Latitude, Height, configured gravity column |
| `<project>_magnetic_data.csv` | Easting, Northing, Longitude, Latitude, TFMA |
| `<project>_topo.tif` | GeoSAGE-compatible georeferenced topography |
| `<project>_mesh.msh` | UBC inversion mesh |
| `<project>_mesh_core.msh` | UBC core mesh |

Inputs are copied into the new run before invoking the original SimPEG kernel.
Source files are hashed before and after processing. A run cannot overwrite an
existing GeoSAGE run or write inside the selected source/input folders. Review
the effective config and audit files alongside the results.

Repeated full inversions can vary because SimPEG's initial beta estimate uses
random power iteration. The adapter does not alter that existing behavior.
Use archived-result mode for exact preservation. The synthetic equivalence test
fixes the beta-estimation seed in the test only and compares both entry points.

## Interpretation and review

Choose **Interpret existing results · with AI**, enter your objective and select
**Generate interpretation**. The Assistant panel opens for provider configuration
when needed. Starting after configuration uses the same **Auto to report** route. The host's OpenAI or Anthropic client supplies the controller
and existing GeoSAGE report agents. The workflow has six separate dependency-bound
stages: prepare data, compile priors, inversion/reuse, geological model, draft,
review. Approval occurs before each stage, including before writing and reviewing.
Generic Gravity/Magnetics and Joint Inversion module forms are independent
tools; editing their fields does not edit the GeoSAGE JSON configuration. Those
pages display an explicit independence notice while GeoSAGE is selected.

Provider keys are session settings, never configuration fields. The offline
button ignores provider settings and environment keys. Local reference documents
use GeoSAGE's existing context handling; Studio's additional RAG/MCP retrieval
is not implemented for this workflow, so those controls are hidden for GeoSAGE.
The selected task determines whether report generation and review are enabled,
even when an imported configuration was originally used for a local run.
Do not put credentials in JSON configs.

An offline summary is always `needs_review` / `NOT_REVIEWED`. A revised report
is not described as accepted without another independent review. An interrupted
or failed dependency chain is `incomplete`. No live paid-provider call is needed
by the test suite; provider connectivity and scientific interpretation quality
still require a separate real-provider acceptance run.

## Workspace and result exploration

GeoSAGE opens with the workflow and relevant tools in focus. **Assistant** in the
toolbar opens chat; **View → Show all processing tools** restores the full module list. Logs remain
available from View. AQUAH keeps its normal workflow and retrieval controls.

A run shows separate numerical, interpretation and review states. Local runs do
not claim an AI review. Solver iterations and data misfit are reported as they
arrive; the activity bar does not imply a measured percentage or time estimate.
On completion the workflow opens **Results & report**, with separate numerical,
interpretation and review states. **View models & compare runs**, **View data fit**
and **Read interpretation** lead to the current run. Local runs label the last
action **Read numerical summary**; the fit action is disabled when observation
and prediction files are absent. **View models & compare runs** also works before saving.
Save it explicitly to retain history after closing the application. Saving through
Ctrl+S or Saved Results refreshes this status too. A result from another Project
cannot be saved into the current Project by a stale workflow button.

In the model viewer, **Linked sections & values** shows plan, east–elevation and
north–elevation sections. Enter physical coordinates, use arrow keys to step
between cells, or click a section to move the crosshairs. Coordinates snap to
actual cell centres, including on nonuniform meshes. Large plot axes use explicitly
labelled kilometres to keep ticks legible; coordinate controls always use metres. Right-click a point in the
3D model to inspect the nearest cell. Property changes keep the selected cell,
camera and clipping plane. Linked sections remain available without OpenGL
when PyVista is installed.

Select two runs with Ctrl-click, then **Compare two models**. Matching rectilinear
grids share a colour scale and show A, B and B − A at the selected elevation.
Different coordinates are rejected rather than silently resampled. Comparisons
cover continuous properties; geological IDs are categories, so subtracting them
is not presented as a scientific difference.

## Scientific display conventions

- Mesh coordinates are metres; **z is elevation, positive up**, not depth below
  the ground surface. Depth claims require the topography-based evidence audit.
- VTK uses the original rectilinear edges and Fortran cell ordering; arrays are
  neither resampled nor reordered in the source archive.
- Density contrast uses g/cm³ with a scale symmetric about zero, susceptibility SI. Sparse geological identifiers
  are categorical labels, not a continuous measurement. Category colours are shared
  between figures, sections and 3D; label 0 has a neutral background colour.
- Orthogonal sections use cell edges and shared limits for each property.
  Data-fit maps share observed/predicted limits, use `observed − predicted`
  residuals, and report RMSE in mGal (gravity gradients: Eötvös) or nT.
- No data-fit map is claimed when the observation/prediction files are absent.

## Agent handoffs and continuing a run

Expand **How the agents work together** in task setup to see the six stages,
their responsibilities and what each hands to the next. Checking inputs updates
the plan for the resolved geology configuration, including AI grouping when
configured. The preview and execution share the same tool registry and the
PyHydroGeophysX dependency controller; numerical parameters are not changed by
the continuation mechanism.

After numerical models are available, **Interpret these models…** prepares an
AI interpretation task with those models and the recorded configuration. Review
the inputs, objective and provider settings before pressing **Generate
interpretation**. Preparing the task does not contact a provider or run anything.
It works after a report failure as well as after a successful local run.

Each completed stage updates `studio_checkpoint.json` atomically, recording its
agent, outputs, outcome and error. When reusable model files exist, a local
`continue_config.json` is also saved. After restarting Studio, choose an existing
results task and add that file as the configuration to reuse the numerical models
in a **new output folder**. Keep the referenced model and prior files in place.
This reuses completed numerical work; it does not resume an interrupted optimizer
iteration or skip unfinished interpretation/review. The selected geology mode
still applies; new full-run labels are reused when they were successfully saved.
These generated records remain local and are not part of the code contribution.

Local numerical summaries present mesh dimensions, physical-property ranges and
recorded geological labels as tables. `numerical_summary.json` retains the full
record; displayed ranges are not uncertainty estimates or geological conclusions.

## Validate and share

```bash
python -m pytest tests
geosage-studio --self-test
```

Integration tests run when the optional host package is installed. They cover
discovery, approvals, stopping, source preservation, coordinate/ID export,
offline isolation, report review outcomes and a real small SimPEG inversion.

Each Studio run needs a new output folder. Only the host's initial `UNSAVED`,
`activity.log` and `steering.jsonl` files may already be present. Inputs,
configuration files and priors must be outside that folder. The command-line
full workflow also refuses nonempty output folders when `run.overwrite=false`.

Models must contain finite real values. Geological labels must be nonnegative
integers; newly built models retain the existing int16 format and reject IDs
above 32767 before casting. Invalid inputs fail explicitly rather than silently
changing labels. A partial archived name mapping is completed from recorded
reports where possible, without editing the source archive.

`effective_config.json` records the final geological settings and the physical
settings actually used by the solver. Deterministic repairs to an LLM grouping
are disclosed in the result warnings. A review reporting major issues cannot
mark the report accepted, and an empty provider response is a failure. Source
integrity is `null` when the run stops before any source inventory is collected.

This repository is code-only. Keep Project folders, input data, inversion
results, generated VTK, figures, reports, credentials and private configs outside
the repository. Synthetic test inputs are generated at runtime. Screenshots
used for local UI review are not part of the source contribution.
