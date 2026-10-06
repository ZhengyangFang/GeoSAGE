"""Optional PyHydroGeophysX assistant; numerical and GUI imports stay lazy."""

from dataclasses import replace

from PyHydroGeophysX.agents.assistants.geosage import ASSISTANT as _SCAFFOLD

from .providers import STUDIO_PROVIDERS

ASSISTANT = replace(
    _SCAFFOLD,
    workflow="geosage.pyhydrogeophysx.workflow:run",
    tools="geosage.pyhydrogeophysx.tools:TOOLS",
    status="ready",
    status_note="",
    providers=STUDIO_PROVIDERS,
    requires_packages=("geosage", "simpeg"),
    persona=_SCAFFOLD.persona
    + (
        "\n- For a new inversion, open Workflow and call prepare_inversion_folder on the user's raw survey folder. "
        "Report the deterministic file/extents summary, including full/core mesh and topography coverage. "
        "Use the scanner's core-mesh region instead of an observation union, and ask only for the listed missing physical parameters. "
        "Use automatic mesh-based numerical defaults unless the user explicitly supplies cross_gradient_lambda or beta_cooling; "
        "both are supported overrides, so never describe them as unavailable. "
        "call set_inversion_parameters, show the exact resolved summary, and call start_confirmed_inversion only after confirmation. "
        "Do not navigate to Model Viewer or 3D Mesh Builder to inspect raw survey geometry already returned by the scanner. "
        "An explicit GeoSAGE JSON configuration and existing-result interpretation remain supported. "
        "Do not substitute a different inversion engine or invent a region, magnetic field, mesh, or priors. "
        "Use model_viewer to inspect exported density, susceptibility, Geo ID and Unit ID. "
        "Run without AI produces numerical evidence only, never an AI-reviewed interpretation. "
        "Without geological priors, clusters are property groups rather than confirmed lithologies."
    ),
    input_roles=(
        ("GeoSAGE configuration (JSON)", "config_file"),
        ("Existing GeoSAGE inversion folder", "source_inversion_dir"),
        ("GeoSAGE survey input folder", "input_dir"),
        ("Gravity data (CSV)", "gravity_file"),
        ("Magnetic data (CSV)", "magnetic_file"),
        ("Terrain (GeoTIFF)", "topography_file"),
        ("Inversion mesh (UBC)", "mesh_file"),
        ("Core mesh (UBC)", "core_mesh_file"),
        ("Unit definitions (CSV)", "unit_defs_file"),
        ("Unit groups (CSV)", "unit_groups_file"),
        ("Geological reference (PDF / text)", "reference_file"),
    ),
    examples=(
        "Use my raw survey folder to prepare a new inversion; ask me for parameters you cannot infer",
        "Interpret my existing GeoSAGE inversion without changing the models",
        "Run the supplied GeoSAGE configuration and review the geological interpretation",
        "Compare density, susceptibility and geological groups in the model viewer",
    ),
    _cache={},
    **({"workflow_setup": "geosage.pyhydrogeophysx.setup:WorkflowSetup",
        "retrieval": (), "focused_workspace": True}
       if "workflow_setup" in _SCAFFOLD.__dataclass_fields__ else {}),
    **({"offline_workflow": True} if "offline_workflow" in _SCAFFOLD.__dataclass_fields__ else {}),
)

__all__ = ["ASSISTANT"]
