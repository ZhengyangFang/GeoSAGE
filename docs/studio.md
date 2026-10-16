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
uv pip install --python .venv-studio\Scripts\python.exe -c GeoSAGE/constraints-tested.txt -e "./PyHydroGeophysX[desktop,desktop-3d]" -e "./GeoSAGE[excel,reference]"
.venv-studio\Scripts\pyhydrogeophysx-studio.exe
```

On Linux/macOS use `.venv-studio/bin/python` and `.venv-studio/bin/pyhydrogeophysx-studio`.
The host starts with its normal home page and AQUAH assistant. Choose **GeoSAGE**
in the existing assistant picker when needed. The optional `geosage-studio`
shortcut opens the same host explicitly with GeoSAGE selected.
No API key is needed for the numerical workflow. PyGIMLi, hydrological solvers,
CUDA and the host's other domain extras are not required for GeoSAGE.

## Inspect existing results

With GeoSAGE selected, the compact workspace has two sidebar destinations: **Data & reports**
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
and declination remain unset until the user supplies them or confirms a reference-field proposal.
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
For the conversational route, files may use any names and live in subdirectories.
GeoSAGE combines measurement columns, descriptive names and mesh geometry to
identify the five roles. Conventional names do not override ambiguous candidates
or contradictory measurement columns. If more than one assignment is possible,
the assistant asks for explicit paths instead of guessing. The numerical runner stages read-only
copies under conventional names, so this discovery does not change the solver.

Missing terrain or meshes no longer discard the inspection. The conversation
receives the identified files, candidate paths, missing roles and individual
validation issues; inversion remains blocked until these are resolved. Generated
run folders and Python environments are excluded from recursive discovery. A
plain raster is not automatically assumed to be elevation, and a Height column
does not determine whether a table contains gravity or magnetic measurements.
Survey tables can be CSV, TSV, delimited TXT/XYZ, XLSX or XLS. Terrain can be
GeoTIFF or ESRI ASCII (with a `.prj` or explicitly supplied CRS). UBC meshes remain
supported. Other recognized grid/vector formats still require conversion.
The supplied core mesh must be a contiguous subset of the full mesh with matching
cell boundaries on all three axes. Inspection rejects mismatched cells before
inversion. Result extraction uses the supplied core geometry, including its depth
range, rather than assuming a fixed number of padding cells.
The names below describe the solver's internal layout, not naming requirements
for the original files.

| File | Required contents |
|---|---|
| `<project>_gravity_data.csv` | Easting, Northing, Height and a selected numeric gravity observation column; geographic terrain needs an explicit projected CRS or legacy Longitude/Latitude |
| `<project>_magnetic_data.csv` | Easting, Northing and a selected numeric magnetic observation column |
| `<project>_topo.tif` | A georeferenced geographic or projected GeoTIFF covering the mesh |
| `<project>_mesh.msh` | UBC inversion mesh |
| `<project>_mesh_core.msh` | UBC core mesh |

Common observation names such as ISO, CBA, TFMA and TMI are recognized. A single
other numeric measurement column is also detected. When a CSV has several possible
measurement columns, the conversation asks which one to use and records that exact
choice in the run configuration and audit metadata.

### Preview and review inputs

After a survey-folder scan, a **Survey preview** card appears in the existing setup
page. Switch between **Coverage**, **Gravity**, and **Magnetics**. It shows station
positions, the full mesh and selected inversion region, and a terrain overview
when the coordinate systems are known. Choosing a measurement column in the
conversation updates the preview. No inversion or additional Project is created.

Ask: “Show the gravity data and tell me which records need checking.” Diagnostics
include invalid required values, coincident positions, constant columns, linear
station geometry, extreme values and the count outside the inversion region.
Extreme values are review hints, not automatic errors; legitimate anomalies are
retained. Measurement maps show the actual imported sign and use the full finite
range. No interpolation is applied to observations. Units follow explicit import
settings or the documented legacy defaults; they are not inferred from magnitude.

Findings identify **1-based source data records after header/skiprows**, not physical
text-file lines. The conversation returns up to 40 example record IDs per finding,
with the full count and a truncation flag. Only on explicit instruction can the
assistant call `set_survey_row_exclusions` with a role and replacement `rows` list.
An empty list restores the manually excluded records. This changes the import
configuration, never the source file. A source hash binds chat-selected exclusions
to that file version; editing the file requires reviewing the exclusion list again.
Changed exclusions invalidate the prior run configuration and require confirmation.

All table diagnostics use the complete normalized table. For responsiveness, maps
show at most 6,000 stations per survey and a nearest-sampled terrain overview of
at most 160 × 160 pixels. The UI discloses sampling; terrain gaps in an overview
do not constitute a full-resolution coverage assessment. Invalid records omitted
from a diagnostic preview are still blocked at launch unless an explicit import
policy excludes them. CLI and Studio staging record the same quality checks and
exclusions in `import_manifest.json`.

Rendering reuses GeoSAGE's display style and Matplotlib's CPU renderer. Terrain
previews use Rasterio [windowed reading](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html)
and [resampling](https://rasterio.readthedocs.io/en/stable/topics/resampling.html).

Inputs are normalized into the new run before invoking the original SimPEG kernel.
Source files are hashed before and after processing. A run cannot overwrite an
existing GeoSAGE run or write inside the selected source/input folders. Review
the effective config and audit files alongside the results.

### Missing terrain and magnetic field parameters

After scanning a survey folder, ask the assistant to propose reference data.
The candidate appears on the existing setup page and in the conversation;
calculating or downloading it does not authorize its use. No new Project is created.

For example, using your actual location, date, height and mesh CRS:

> Estimate the magnetic field at longitude …, latitude …, observed on YYYY-MM-DD,
> at … metres above the WGS84 ellipsoid. My mesh CRS is EPSG:…. Show the source
> and proposed F/I/D before using them.

GeoSAGE uses the maintained [IAGA ppigrf implementation](https://github.com/IAGA-VMOD/ppigrf)
with the packaged [IGRF-14 coefficients](https://doi.org/10.5281/zenodo.14012302).
The reference model runs locally after installation; the location is not sent to
an online magnetic-field service. It reports F in nT, inclination positive down,
true-north declination and the declination converted to mesh grid north using
the projection's meridian convergence. The solver uses the latter. The model
date range is checked against the packaged coefficients (1900–2030 for IGRF-14).
Date and ellipsoidal height are required: flight height above ground and terrain
orthometric height are not interchangeable with ellipsoidal height.
These are [main-field model values](https://www.ncei.noaa.gov/products/international-geomagnetic-reference-field),
not measured anomalies or survey-time disturbance corrections. One reference
field may not represent a very large survey adequately.

For missing terrain:

> Find reference terrain covering my full mesh in EPSG:…. Use 90 m resolution
> and show the coverage and source before adopting it.

The background reader uses Rasterio/GDAL HTTP range requests for public
[Copernicus DEM cloud-optimized GeoTIFFs](https://registry.opendata.aws/copernicus-dem/).
It caches only the clipped area. Choose 90 m (default) or 30 m nominal resolution;
one request is limited to 8 million pixels and 16 tiles, and the cache budget is
256 MiB. `GEOSAGE_REFERENCE_CACHE` selects a cache directory; otherwise the OS
temporary directory is used. A matching verified cache works with `offline=true`.
An unavailable service, missing tile or NoData gap returns a clear failure;
missing elevations are never replaced with zero.

The AWS dataset is the 2021 release of a **digital surface model**, potentially
including trees and buildings. Heights are EGM2008 orthometric metres; GeoSAGE
does not convert vertical datums. Review the candidate and confirm compatibility
with your survey and mesh heights before adoption. See the
[Copernicus product and licence information](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
Polar areas beyond ±85° and dateline-crossing areas currently need local terrain.
Existing terrain is replaced only with explicit approval. A full mesh defines
download coverage: supply one or confirm a generated mesh proposal first.

`propose_magnetic_field` and `request_reference_terrain` return proposals;
`get_reference_data` reports download completion, mesh candidates and provenance. After the user
confirms, `confirm_reference_data` adopts that exact proposal. Terrain also needs
`height_datum_confirmed=true`; replacement of existing values needs
`replace_existing=true`. `discard_reference_data` discards a pending candidate.
The configuration must then be resolved and confirmed before starting.
Changed input context, an altered cached file or a superseded proposal invalidates
the confirmation. Confirmed source metadata is retained in `reference_data` in
the effective configuration, including dates, model/coefficient or terrain hashes,
input coordinates and applied values. Manual field overrides remain possible and
remove model provenance only for the overridden values.

The `reference` extra adds only `ppigrf`; terrain uses the existing Rasterio/pyproj
dependencies. On an existing installation, use uv with its existing Python:

```powershell
uv pip install --python .venv-studio\Scripts\python.exe -e "./GeoSAGE[excel,reference]"
```

### Missing meshes and run-size preflight

No full/core meshes yet? Scan the survey folder, then say (use your actual values):

> Propose a mesh in EPSG:…. Its top elevation is … m in my terrain height datum,
> with a core extending … m below that top. Show the coverage, cell size and
> estimated computational size before generating anything.

`propose_inversion_mesh` uses the existing
[discretize TensorMesh](https://discretize.simpeg.xyz/en/main/api/generated/discretize.TensorMesh.html)
implementation. The proposal shows core and padded full-mesh coverage on the
existing survey preview. No files are created until `confirm_reference_data`
adopts the exact proposal. Existing mesh roles need explicit replacement approval;
their original files are not overwritten. The generated UBC axis descriptions
are cached outside the input folder and copied into the run by the shared importer.
`GEOSAGE_MESH_CACHE` can select the cache directory; the default is
`geosage-mesh-cache` beneath the OS temporary directory. Identical geometry reuses
the same files. No Project, environment, dependency or separate viewer is added.

The required **core thickness below the mesh top** and **top elevation** are not
inferred from station positions. Top/bottom elevations use the survey/terrain
vertical datum. This is a geometric volume, not a scientifically demonstrated
depth of investigation. Without explicit `bounds`, horizontal coverage is the
station union with a one-cell margin. Explicit bounds snap outward to cell edges.
Zero-area line surveys need an explicit nonzero study rectangle. Both observation
types must have stations within the proposed region.

Optional `cell_size_m` fixes the core cell width. Otherwise the initial heuristic
takes the largest of the finer survey's median nearest-neighbour spacing,
horizontal spans divided by 40, and requested depth divided by 20, then rounds up
to a 1/2/5 scale. Spacing uses at most 50,000 unique positions per survey;
coverage and station counts use all valid rows. This is a disclosed starting
proposal, not an optimal inversion grid. Confirm or change it in the conversation.
The core has at least two cells per axis and rounds its extent outward. Default
padding is five expanding cells on each horizontal side and below the core,
with factor 1.4 and no padding above the top. These can be changed via
`padding_cells` and `padding_factor`; they are not case-study presets.
Proposals exceeding 2 million cells or 10,000 cells on one axis are rejected
with a request to reduce the region or choose a coarser cell size. User values
are never silently coarsened. Supplied existing meshes are not subject to this
proposal-generation limit.

After adopting a mesh, GeoSAGE rescans the inputs. A DEM that covers the core
may still fail to cover the padded full mesh; supply wider terrain or request
a reference crop before resolving the run settings.

`set_inversion_parameters` returns `setup_preflight` for the exact configured
mesh, region and station counts. The compact page shows cell count, station
count, iteration limit and a sensitivity-matrix allocation envelope. It assumes
all full-mesh cells are active, and reports the combined scalar gravity/magnetic
matrix sizes at 4 and 8 bytes per entry. The current solver defaults to float32
sensitivities. This is **not peak RAM or a runtime estimate**: solver copies,
regularization, plotting and other arrays need additional memory, while terrain
may reduce the active cell count. For details, see the
[SimPEG integral simulation implementation](https://github.com/simpeg/simpeg/blob/v0.25.0/simpeg/potential_fields/base.py).

`parameter_sources` distinguishes user values, detected measurement columns,
core-mesh bounds, method defaults, the mesh-derived cross-gradient weight, and
confirmed IGRF values. Incremental chat changes retain earlier settings and
refresh the estimate. Source fingerprints, including raster sidecars, bind the
preflight to the reviewed files. If a source changes before launch, resolve and
confirm the configuration again. GUI and CLI staging enforce this same check.
Confirmed mesh geometry and parameter sources remain in `effective_config.json`.

Recognizable gravity columns such as `GXY` and `GZZ` determine the proposed
component before confirmation; vertical anomaly columns such as `ISO` use `gz`.
An explicit conflicting component is rejected in import and in the solver,
rather than silently changed during inversion. Custom column names can use an
explicit `gravity_component`; units must match acceleration or gradient data.

### Import options through conversation or JSON

Tell the assistant which sheet, coordinate columns, units or CRS to use. It passes
`data_import` to `prepare_inversion_folder` and retains those options in the
configuration. CLI runs use the same adapters through `project.input_files` and
`data_import`, for example (replace the file paths and CRS with your own):

```json
{
  "project": {
    "input_files": {
      "gravity_file": "data/site/gravity.xlsx",
      "magnetic_file": "data/site/magnetic.xyz",
      "topography_file": "data/site/elevation.asc"
    }
  },
  "data_import": {
    "projected_crs": "EPSG:32614",
    "gravity_file": {
      "sheet": "Survey",
      "columns": {"Easting": "X", "Northing": "Y", "Height": "Z"},
      "coordinate_unit": "m",
      "height_unit": "m",
      "measurement_unit": "mGal"
    },
    "magnetic_file": {
      "header": null,
      "columns": {"Easting": "column_1", "Northing": "column_2"},
      "measurement_unit": "nT"
    },
    "topography_file": {"crs": "EPSG:32614", "height_unit": "m"}
  },
  "data": {"gravity_column": "ISO", "magnetic_column": "column_3"}
}
```

This is an import fragment to merge into a complete run configuration; the
region, meshes and physical settings are still needed. `columns` maps canonical
coordinate names to source names; measurement columns are selected separately.
Multiple worksheets or competing coordinate columns require an explicit choice.
Headerless tables use `header: null` and one-based names `column_1`, `column_2`, etc.
Use `skiprows`, `delimiter`, `encoding` and `decimal` for nonstandard text exports.
UTF-8 is the default; other encodings must be specified rather than guessed.

`na_values` defines additional missing-value markers. Invalid required values
block import unless the user explicitly chooses `missing_policy: "drop"`;
removed data-row indices are recorded. Units default to the established GeoSAGE
conventions (metres, mGal or Eotvos for gravity, nT for magnetics). Explicit units
are converted without changing sign. Noise settings remain in the normalized
units. `crs` is the source CRS; `projected_crs` must be the metre-based CRS of the
inversion mesh. Geographic-only data require this target CRS. Horizontal
reprojection does not perform vertical datum conversion.

`inputs/import_manifest.json` records original and converted hashes, worksheet,
column mapping, import options and removed rows. GUI and CLI use the same
conversion function. Existing standard CSV/GeoTIFF files are copied unchanged
when their contents already match the solver format. Terrain NoData pixels are
masked rather than treated as elevations.

The adapters reuse [pandas](https://pandas.pydata.org/docs/user_guide/io.html),
[Rasterio/GDAL](https://rasterio.readthedocs.io/en/stable/), and
[pyproj](https://pyproj4.github.io/pyproj/stable/api/transformer.html).
The Studio extra includes the Excel engines; for a code-only installation use
`uv pip install -e ".[excel]"` in your existing environment.

Repeated full inversions can vary because SimPEG's initial beta estimate uses
random power iteration. The adapter does not alter that existing behavior.
Use archived-result mode when an existing numerical model must be preserved
exactly; source files are hashed before and after inspection.

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
or failed dependency chain is `incomplete`. The installation self-check does not
make a paid provider call. Provider connectivity and scientific interpretation
quality require a separate real-provider acceptance run.

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

### Approved temporary scripts

With a current PyHydroGeophysX 0.5 host, GeoSAGE uses the shared **Temporary
scripts** chat tool. If the existing import options cannot handle a survey
layout, ask: "Convert this file to a supported table, preserve the observations,
and show me the preview before running inversion." The host shows the purpose
and Python code; **Run this script** approves execution. The assistant then
rescans the returned file path through GeoSAGE's normal setup and preview.

An AI-enabled automatic run can also recover a failed **Prepare data** step.
It offers one format-conversion attempt for a selected gravity or magnetic
table, using the host's script generator, restricted subprocess and approval
card. A failed script may receive one corrected version, with fresh approval.
GeoSAGE validates the replacement against the unchanged measurement columns,
units, CRS, mesh and region. It shows counts and value ranges and asks **Use
converted input** before adopting it. Structural validation cannot prove a
conversion preserved the scientific meaning; review the preview and script.

The original files remain unchanged. The run retains the scripts and their
outputs under `scratch/`, fingerprints in `input_recovery.json`, and provenance
in the import manifest and report's `run_record.json`. Converted inputs are
staged within the same run. No new Project is created. This does not repair
missing measurements, change magnetic-field or inversion parameters, rewrite
meshes or restart a failed numerical optimizer. Report-only recovery is below.
Offline runs, absent approval, declined conversions and hosts without this
feature keep the failure visible instead of executing generated code.

### Retry only the report

Once figures and structured evidence are ready, Studio saves a small
`report_handoff.json` and `retry_report_config.json`. Model arrays are referenced,
not copied into the handoff. If drafting fails, the retry runs drafting and
review; if review fails after a draft was saved, it runs review only. The
existing continuation control selects this route after a report failure.
The controller cannot select inversion or geological regrouping in this mode.

You can also ask GeoSAGE:

> Retry the report from `D:\my-project\runs\run-01`. Keep its models and figures.

The assistant calls `prepare_report_retry`, shows readiness and waits for your
confirmation before the normal workflow start. After restarting, supply
`retry_report_config.json` as the configuration of an AI interpretation task.
The host saves the new report attempt within the current Project's normal run
history; it does not create another Project. Existing reports remain untouched.
An explicit report retry can also add AI interpretation to a completed local
numerical summary. Provider credentials come from the current session.

The saved objective, parameters and evidence are reused exactly. Changing the
scientific question or inputs requires a fresh interpretation instead. Hashes
check the handoff, saved numerical models, figures and draft before reuse.
Moved, missing or changed files require rebuilding the interpretation. Older
runs without a handoff still support model reuse through `continue_config.json`;
they do not gain report-only recovery retroactively. No optimizer state is
restored, and interrupted numerical inversion is not resumed.

After successful report recovery, `continue_config.json` starts a fresh
interpretation with the new question. It does not retain the report-retry mode.

Reports embed the standard available figures even when the AI response mentions
only filenames. A declared but missing figure now fails explicitly. Drafts,
final reports and local summaries link to `run_record.json`, containing the
actual archived inversion parameters, effective configuration and numerical
summary. `report_consistency.json` records verified image references and file
hashes. These checks establish file and record consistency; they do not certify
the accuracy of AI-written geological claims. Independent review status remains
separate from numerical completion.

### Compare saved runs through conversation

> Compare `D:\my-project\runs\run-01` and `D:\my-project\runs\run-02`.

`compare_saved_runs` reads Studio run folders or original GeoSAGE model folders.
It shows recorded parameter changes, physical-property ranges and data-fit RMSE
in the existing details card and conversation, without writing files or creating
a Project. RMS/max cell changes require identical core grids and a known matching
CRS. No resampling or subtraction of geological class IDs is performed.
Missing CRS is reported as unknown, including for legacy archives.
The saved inversion's `survey_crs` takes precedence over configuration metadata;
a disagreement is disclosed in the comparison limitations.

Fits are ranked as directly comparable only when ordered station coordinates,
observations, uncertainties, CRS and gravity component/magnetic field settings
match. Different surveys can still be inspected side by side, with an explicit
non-comparability notice. A lower RMSE alone does not establish a better
geological interpretation. The existing 3D model comparison remains available
for visual exploration.

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
failure observed after native folder dialogs. On Windows, confirm project and
artifact switching once after installation because an offscreen check cannot
detect compositor failures.

```bash
geosage-studio --self-test
```

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
the repository. Screenshots used for local UI review are not part of the source
contribution.
