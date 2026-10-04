"""GeoSAGE stages on the host's dependency-driven agent controller."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from PyHydroGeophysX.agents.runtime.tools import Tool

from .configuration import FILE_ROLES


def _orchestrator(ctx):
    if "orchestrator" not in ctx.settings:
        from geosage.multi_agent_runner import MultiAgentOrchestrator
        from .providers import StudioLLM

        ctx.settings["orchestrator"] = MultiAgentOrchestrator(
            llm=StudioLLM(ctx.settings), vision_client=object()
        )
    return ctx.settings["orchestrator"]


def fingerprint(path):
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest}


def prepare_data(ctx):
    import numpy as np
    import pandas as pd
    from geosage.existing_results import REQUIRED_ARTIFACTS

    cfg = ctx.config
    project = cfg["project"]
    inputs = cfg["studio_inputs"]
    if cfg["run"]["execution_mode"] == "interpret_existing":
        source = Path(project["source_inversion_dir"])
        paths = [source / relative for relative in REQUIRED_ARTIFACTS.values()]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing archived result: {path}")
        paths = sorted(p for p in source.rglob("*") if p.is_file())
        return "Validated the archived models; inversion will be reused without recomputation.", {
            "field_data": {"source": str(source), "files": [fingerprint(p) for p in paths]}
        }
    name = project["name"]
    root = Path(project["input_dir"])
    files = {
        role: Path(inputs.get(role) or root / f"{name}_{suffix}")
        for role, suffix in FILE_ROLES.items()
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Full inversion requires two CSV surveys, a GeoTIFF and two UBC meshes: "
            + ", ".join(missing)
        )
    counts = []
    for role, column in (
        ("gravity_file", cfg["data"]["gravity_column"]),
        ("magnetic_file", "TFMA"),
    ):
        table = pd.read_csv(files[role])
        required = ["Easting", "Northing", "Longitude", "Latitude", column]
        if role == "gravity_file":
            required.append("Height")
        absent = set(required) - set(table.columns)
        if absent:
            raise ValueError(
                f"{role} missing columns: {', '.join(sorted(absent))}. "
                "Use projected metres and corresponding longitude/latitude degrees."
            )
        if not np.isfinite(table[required].to_numpy(dtype=float)).all():
            raise ValueError(f"{role} contains non-finite coordinates or observations.")
        r = cfg["region"]
        inside = (
            (table.Easting > r["min_e"])
            & (table.Easting < r["max_e"])
            & (table.Northing > r["min_n"])
            & (table.Northing < r["max_n"])
        )
        if not inside.any():
            raise ValueError(f"{role} has no stations inside the configured region.")
        counts.append(int(inside.sum()))
    # Preserve originals; the existing numerical kernel expects these names.
    stage = Path(ctx.output_dir) / "inputs"
    stage.mkdir(exist_ok=False)
    for role, path in files.items():
        shutil.copy2(path, stage / f"{name}_{FILE_ROLES[role]}")
    project["input_dir"] = str(stage)
    return f"Validated {counts[0]} gravity and {counts[1]} magnetic stations in the study area.", {
        "field_data": {"files": [fingerprint(p) for p in files.values()], "station_counts": counts}
    }


def compile_priors(ctx):
    geo = ctx.config["geology"]
    paths = [
        Path(geo[k]) for k in ("unit_defs_csv", "unit_groups_csv", "context_path") if geo.get(k)
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Configured geological prior does not exist: {path}")
    mode = geo["mode"]
    if mode == "reuse_existing_geology":
        summary = "Archived geological labels and their recorded definitions will be retained."
    elif mode == "gmm_only":
        summary = "Unsupervised property clusters; cluster IDs are not verified lithologies."
        ctx.note(summary)
    else:
        summary = f"Using {mode} with {len(paths)} supplied prior files."
    return summary, {"geological_priors": {"mode": mode, "files": [fingerprint(p) for p in paths]}}


def run_joint_inversion(ctx):
    from geosage.runner import run_workflow

    cfg = deepcopy(ctx.config)
    cfg["run"]["run_geology_model"] = False
    if cfg["run"]["execution_mode"] == "full" and Path(cfg["project"]["output_dir"]).exists():
        raise FileExistsError("Model output folder already exists; start a new Studio run.")
    result = run_workflow(cfg)
    effective = result["effective_config"]
    effective["run"] = deepcopy(ctx.config["run"])
    effective["geology"] = deepcopy(ctx.config["geology"])
    ctx.config.update(effective)
    result["config"] = effective
    shape = result["inversion_result"]["dens_core_3d"].shape
    verb = "Reused archived" if cfg["run"]["execution_mode"] == "interpret_existing" else "Computed"
    return f"{verb} density and susceptibility on a {shape[0]} × {shape[1]} × {shape[2]} mesh.", {
        "property_models": result
    }


def build_quasi_geology(ctx):
    from .visualization import export_results

    result = ctx.get("property_models")
    engine = _orchestrator(ctx)
    result["geology_result"] = engine._prepare_configured_geology(ctx.config, result)
    if ctx.config["geology"]["mode"] in {"gmm_bic_auto", "fixed_units_llm_groups"}:
        if not getattr(engine.unit_csv_agent, "interactions", []):
            ctx.note(
                "LLM geological naming did not produce a recorded response; deterministic grouping fallback was used."
            )
    prepared = engine._prepare_interpretation_artifacts(ctx.config, result)
    exports = export_results(result, Path(ctx.output_dir))
    # Feed the same physical-coordinate figures into the existing report
    # inventory so the report and the Studio visualize the same run.
    figures = exports["figures"]
    result["inversion_result"].setdefault("paths", {})["inversion_result_slice_list"] = [
        figures[k]
        for k in ("Density contrast", "Susceptibility", "Gravity data fit", "Magnetics data fit")
        if k in figures
    ]
    geo_paths = result["geology_result"].setdefault("paths", {})
    geo_paths["geo_combo_slice_pngs"] = (
        [figures["Geological groups"]] if "Geological groups" in figures else []
    )
    geo_paths["scatter_rho_kappa"] = figures["Physical-property distribution"]
    groups = len((result["geology_result"] or {}).get("geo_defs", {}))
    return (
        f"Prepared {groups} geological groups, numerical evidence, slices and a 3D viewer model.",
        {"geo_model": {"workflow": result, "prepared": prepared, "exports": exports}},
    )


def write_report(ctx):
    geo = ctx.get("geo_model")
    if ctx.settings.get("api_key") and ctx.config["run"].get("write_reports", True):
        draft = _orchestrator(ctx).write_report_draft(
            ctx.config, geo["workflow"], ctx.goal, geo["prepared"]
        )
        return "Wrote the interpretation draft; independent review is still required.", {
            "draft_report": draft
        }
    # A deterministic record is useful offline, but is never called AI-reviewed.
    prepared = geo["prepared"]
    lines = [
        "# GeoSAGE · Numerical evidence summary",
        "",
        "**No LLM interpretation or independent report review was performed.**",
        "",
        "## Recorded results",
        "",
        "```json",
        json.dumps(prepared["result_summary"], indent=2, default=str),
        "```",
        "",
        "Coordinates are mesh coordinates in metres. Z is elevation, not depth below ground.",
        "A precise target depth requires the topography-based evidence audit.",
        "",
        "Property clusters alone do not establish lithology or a mineral resource.",
        "",
    ]
    for title, path in geo["exports"]["figures"].items():
        lines += [
            f"## {title}",
            "",
            f"![{title}]({Path(path).relative_to(ctx.output_dir).as_posix()})",
            "",
        ]
    draft = Path(ctx.output_dir) / "evidence_summary.md"
    draft.write_text("\n".join(lines), encoding="utf-8")
    return "Wrote a numerical evidence summary; LLM interpretation is not enabled.", {
        "draft_report": {"offline": True, "path": str(draft)}
    }


def review_report(ctx):
    draft = ctx.get("draft_report")
    if draft.get("offline"):
        ctx.note(
            "Numerical outputs are ready. LLM interpretation and independent review were not performed."
        )
        return "Numerical evidence exported; geological interpretation needs review.", {
            "report_files": {"report_markdown": draft["path"]},
            "review_decision": "NOT_REVIEWED",
        }
    cfg = deepcopy(ctx.config)
    cfg["run"]["review_enabled"] = True
    result = _orchestrator(ctx).review_report_draft(cfg, ctx.get("property_models"), draft)
    decision = (result.get("review") or {}).get("decision", "NOT_REVIEWED")
    if decision != "ACCEPT":
        ctx.note(
            f"Report review: {decision}. Any single revision has not been independently re-reviewed."
        )
    return f"Report review completed: {decision}.", {
        "report_files": {
            "report_markdown": result["report_path"],
            **({"report_pdf": result["pdf_path"]} if result.get("pdf_path") else {}),
        },
        "review_decision": decision,
    }


TOOLS = {
    tool.name: tool
    for tool in (
        Tool(
            "prepare_data",
            "Validate the selected surveys or archived inversion, preserving original files.",
            prepare_data,
            produces=("field_data",),
            agent="DataAgent",
            label="Prepare data",
            module="gravmag",
        ),
        Tool(
            "compile_priors",
            "Validate supplied priors or explicitly select unsupervised grouping.",
            compile_priors,
            requires=("field_data",),
            produces=("geological_priors",),
            agent="PetrologyAgent",
            label="Compile priors",
        ),
        Tool(
            "run_joint_inversion",
            "Run the configured SimPEG inversion, or reuse the archived models unchanged.",
            run_joint_inversion,
            requires=("field_data", "geological_priors"),
            produces=("property_models",),
            agent="InversionAgent",
            label="Joint inversion",
            module="joint_inversion",
        ),
        Tool(
            "build_quasi_geology",
            "Build or reuse labels, compute evidence and export physical-coordinate views.",
            build_quasi_geology,
            requires=("property_models", "geological_priors"),
            produces=("geo_model",),
            agent="GeoAgent",
            label="Geological model",
            module="model_viewer",
        ),
        Tool(
            "write_report",
            "Write an interpretation draft or an explicitly unreviewed offline evidence summary.",
            write_report,
            requires=("geo_model",),
            produces=("draft_report",),
            agent="ReportAgent",
            label="Draft report",
        ),
        Tool(
            "review_report",
            "Review the draft against numerical evidence; disclose absent or unresolved review.",
            review_report,
            requires=("draft_report",),
            produces=("report_files",),
            agent="ReviewAgent",
            label="Review report",
            module="one_click",
        ),
    )
}
