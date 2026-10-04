"""Optional PyHydroGeophysX assistant; numerical and GUI imports stay lazy."""

from dataclasses import replace

from PyHydroGeophysX.agents.assistants.geosage import ASSISTANT as _SCAFFOLD

ASSISTANT = replace(
    _SCAFFOLD,
    workflow="geosage.pyhydrogeophysx.workflow:run",
    tools="geosage.pyhydrogeophysx.tools:TOOLS",
    status="ready",
    status_note="",
    requires_packages=("geosage", "simpeg"),
    persona=_SCAFFOLD.persona
    + (
        "\n- GeoSAGE computations run through the Workflow page using an explicit GeoSAGE JSON configuration "
        "or an existing inversion folder. Generic processing-module parameters do not edit that configuration. "
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
