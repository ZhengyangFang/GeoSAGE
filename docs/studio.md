# GeoSAGE in PyHydroGeophysX Studio

GeoSAGE is an optional assistant inside the existing PyHydroGeophysX Qt Studio.
It uses the host's light/dark theme, chat settings, workflow timeline, step
approvals, cancellation, result history, figure previews and 3D viewer. It does
not introduce a second desktop application or replace the inversion kernel.

## Current compatibility

The adapter targets the assistant entry-point API in PyHydroGeophysX 0.5.0.
Use the current source checkouts because the latest packaged release may predate
that API. Pair GeoSAGE `main` with the upstream PyHydroGeophysX `main` branch:

```bash
git clone https://github.com/ZhengyangFang/GeoSAGE.git
git clone https://github.com/geohang/PyHydroGeophysX.git
```

The optional GeoSAGE package registers with the host's assistant interface. A
prebuilt desktop executable must be rebuilt with GeoSAGE included before it can
discover the plugin, so the source installation below is currently recommended.

## Install from two local checkouts

Use a separate Python 3.11–3.13 environment to preserve an existing scientific
environment. From the parent containing `GeoSAGE/` and `PyHydroGeophysX/`:

```powershell
uv venv .venv-studio --python 3.12
uv pip install --python .venv-studio\Scripts\python.exe -c GeoSAGE/constraints-tested.txt -e "./PyHydroGeophysX[desktop,desktop-3d]" -e "./GeoSAGE"
.venv-studio\Scripts\geosage-studio.exe --module one_click
```

On Linux/macOS use `.venv-studio/bin/python` and `.venv-studio/bin/geosage-studio`.
No API key is needed for the numerical workflow. PyGIMLi, hydrological solvers,
CUDA and the host's other domain extras are not required for GeoSAGE.

## Inspect existing results

The default GeoSAGE workspace has two sidebar destinations: **Data & reports**
and **Saved Results**. Start with **View results**, **New inversion**, or
**AI interpretation** and choose the primary input. **Options** reveals additional
input roles, configuration checks, agent responsibilities, step approvals and
logs. In Saved Results, selecting a record opens its visualization; **Details**
reveals metadata, file management and display options. **View → Show all processing
tools** restores the host's other modules. Numerical behavior is unchanged.

In **Saved Results**, select a run and open **Report** to read its saved Markdown
report with fitted figures. This also works for existing runs whose reports were
recorded only in `result.json`. **Visualization** shows models and figures by
default; **Details** exposes JSON metadata and the complete file list. A local
Inspect run contains a numerical evidence summary, not an AI interpretation.

To use the agents for interpretation, choose **Data & reports → AI interpretation**,
select an existing inversion folder, enter an optional question, and configure a
provider in **Assistant → Settings**. **Generate interpretation** starts the
workflow. **Options → How the agents work together** shows the stage responsibilities
and dependencies; **Activity** shows execution progress. The controller coordinates
data validation, priors, model loading/inversion, geology, report writing and review.
Local viewing does not call an AI provider. Interpretation of existing models does
not rerun the inversion.

1. Select **GeoSAGE** in the assistant picker. In **Data & reports → Data**, add
   the folder with role **Existing GeoSAGE inversion folder**.
2. Choose **Inspect existing results · local**, use **Check inputs & preview configuration**,
   then select **Open results**. Enable **Approve each step before it runs** if
   you want to examine each stage. Stop preserves partial outputs in this run.
3. Inspect **Results & report**, or **Saved Results → Visualization**. The VTK
   property selector offers density contrast, susceptibility, Geo ID and Unit ID;
   the clipping plane exposes interior cells. Figures remain available without
   a compatible OpenGL renderer.
4. Use **Save this run** to retain the run in Studio's local history across sessions.
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

For the guided route, open **Assistant**, keep **Step-by-step assistance**, and say:

> Use `D:\path\to\survey` to prepare a new GeoSAGE inversion. Inspect the files,
> ask me for parameters that cannot be derived, then run it and review the report.

GeoSAGE identifies the survey files and reads coordinate extents locally. It
does not ask the model to guess file roles. The deterministic scanner parses the
full and core UBC meshes, reads GeoTIFF bounds and CRS, checks mesh/topography
coverage and counts observations inside the core. Validated core-mesh horizontal
bounds supply the proposed inversion region. Magnetic field strength, inclination
and declination remain unset until the user supplies them.
The assistant shows the resolved configuration and asks for confirmation before
starting. After confirmation, the same conversation switches to the full
inversion, report and independent-review workflow. Generated runs go to the
workspace's `outputs/studio_runs/`; no separate Project selection is required.

The manual route remains available under **Run a new inversion · local**. Add a
**GeoSAGE configuration (JSON)** with an explicit project name, region, survey
and inversion settings. Existing configs retain their workspace-relative path rules.
Use **Create / edit inversion configuration** to edit physical settings for this run;
imported advanced settings are retained and the original JSON is not modified.
**View all resolved parameters** shows the complete effective configuration.
For the conversational route, files may use any names. GeoSAGE first honors the
conventional names below, then uses descriptive filenames and file contents to
identify the five roles. If more than one assignment is possible, the assistant
asks for explicit paths instead of guessing. The numerical runner stages read-only
copies under conventional names, so this discovery does not change the solver.

| File | Required contents |
|---|---|
| `<project>_gravity_data.csv` | Easting, Northing, Height and a selected numeric gravity observation column; Longitude/Latitude are needed only with geographic topography |
| `<project>_magnetic_data.csv` | Easting, Northing and a selected numeric magnetic observation column |
| `<project>_topo.tif` | A georeferenced geographic or projected GeoTIFF covering the mesh |
| `<project>_mesh.msh` | UBC inversion mesh |
| `<project>_mesh_core.msh` | UBC core mesh |

Common observation names such as ISO, CBA, TFMA and TMI are recognized. A single
other numeric measurement column is also detected. When a CSV has several possible
measurement columns, the conversation asks which one to use and records that exact
choice in the run configuration and audit metadata.

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
when needed. Starting after configuration uses the same **Auto to report** route.
The host's API client or authenticated CLI provider supplies the controller
and existing GeoSAGE report agents. The workflow has six separate dependency-bound
stages: prepare data, compile priors, inversion/reuse, geological model, draft,
review. Approval occurs before each stage, including before writing and reviewing.
Generic Gravity/Magnetics and Joint Inversion module forms are independent
tools; editing their fields does not edit the GeoSAGE JSON configuration. Those
pages display an explicit independence notice while GeoSAGE is selected.

In **Assistant → Settings**, choose OpenAI/Anthropic with a session API key, or
**Codex CLI** / **Claude Code CLI** with the host's existing setup and sign-in
controls. CLI providers use their own saved login and need no copied API key.
GeoSAGE does not install another CLI, read login tokens, or launch its own CLI
subprocess. Older hosts without CLI support expose the API options only.

Both CLI providers support **Auto to report** and **Step-by-step assistance**.
Workflow stage approval is independently controlled by **Approve each step
before it runs**. The task button waits for the host's login readiness; provider
failures remain errors, rather than silently producing an offline interpretation.
The current upstream CLI bridge accepts text, not screenshot/image input.

Provider keys are session settings, never configuration fields. The offline
button ignores provider settings, saved CLI logins and environment keys. Local reference documents
use GeoSAGE's existing context handling; Studio's additional RAG/MCP retrieval
is not implemented for this workflow, so those controls are hidden for GeoSAGE.
An AI-backed new inversion generates and independently reviews its report in the
same run. The local task button remains numerical-only. The selected task and
provider determine whether report generation and review are enabled, even when
an imported configuration was originally used for a local run.
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

### Adaptive views and paper exports

The Studio and notebooks `3_1`, `3_2`, and `4_1` use the same functions in
`geosage.plotting.paper`: model sections, unit-coloured property crossplots and
observed/predicted/residual panels. The Studio selects `adaptive=True`: one
horizontal and one vertical section at the physical mesh midpoints, full-range
scales derived from this run, actual cell edges and equal axis units. Metres or
kilometres follow the domain size. Project names never select display limits.
Signed fields use a diverging palette; nonnegative susceptibility uses a
sequential palette. Geological IDs stay discrete, including sparse IDs.
The 3D model uses the same field scales and categorical colours.

Observed and predicted panels share one scale; residuals use a separate
zero-centred scale with the **predicted minus observed** convention. Maps do
not extrapolate beyond the station convex hull; line surveys show measured
station positions. Auto-ranging applies per run, so compare the displayed
colour bars before comparing different runs by colour. Supply a unit-definitions
file for legend names; missing names never change numerical labels.

Notebooks retain their original layouts, case-specific presets and 600 dpi
publication exports. Studio creates 160 dpi
previews and retains the interactive 3D model, rather than maintaining another
set of plotting implementations. Line surveys show station values instead of
attempting an undefined 2D interpolation. Saved runs retain the figures produced
at the time; inspect an archive again to create a run using the shared plots.

The host retains one model-viewer OpenGL widget across artifact and Project
changes, resetting project data in place. This avoids the Windows compositor
failure observed after native folder dialogs. Regression testing must include
real Windows folder-dialog acceptance and screen capture: an offscreen Qt render
does not detect this failure.

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
