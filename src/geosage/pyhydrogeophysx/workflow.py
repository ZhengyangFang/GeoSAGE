"""Studio workflow contract, with truthful outcomes and the shared runtime."""

import json
from pathlib import Path

from .configuration import configure


def run(payload, progress, *, approve=None, on_event=None, events=None, **_hooks):
    from PyHydroGeophysX.agents.runtime.context import RunContext
    from PyHydroGeophysX.agents.runtime.entry import drive
    from PyHydroGeophysX.agents.runtime.modes import step_by_step
    from PyHydroGeophysX.agents.assistants.geosage.workflow import CONTROLLER_PROMPT
    from geosage.multi_agent_runner import _redact_trace_value
    from .providers import StudioLLM
    from .tools import TOOLS, fingerprint

    if not str(payload.get("request") or "").strip():
        raise ValueError("Describe the exploration objective or requested analysis.")
    if payload.get("step_mode") and approve is None:
        raise ValueError("Step-by-step mode needs an approval callback; no work was started.")
    provider = {"anthropic": "claude"}.get(
        payload.get("provider"), payload.get("provider") or "openai"
    )
    if provider not in {"openai", "claude"}:
        raise ValueError(f"Unsupported Studio provider: {provider}")
    cfg = configure(payload)
    output = Path(payload["output_dir"]).expanduser().resolve()
    # Host may already create the run directory and its UNSAVED marker.
    # A private reservation prevents two starts from sharing any destinations.
    output.mkdir(parents=True, exist_ok=True)
    reservation = output / ".geosage-run"
    with reservation.open("x", encoding="utf-8") as handle:
        handle.write("GeoSAGE owns this run. Start a new output directory to run again.\n")
    settings = {
        "api_key": payload.get("api_key"),
        "model": payload.get("model"),
        "llm_provider": provider,
        "ask_user": approve,
    }
    ctx = RunContext(
        goal=cfg["user_request"], config=cfg, output_dir=str(output), settings=settings
    )
    if payload.get("use_rag") or payload.get("use_mcp"):
        ctx.note(
            "GeoSAGE uses the configured local geological reference; Studio RAG/MCP retrieval is not enabled for this workflow."
        )
    recorded = []

    def emit(event):
        recorded.append(event)
        if on_event:
            on_event(event)

    model = StudioLLM(settings)

    def ask(prompt, on_text=None):
        return model.query(prompt, temperature=0, max_tokens=500, on_text=on_text)

    drive(
        ctx,
        tools=TOOLS,
        ask=ask if settings["api_key"] else None,
        progress_callback=progress,
        on_step=step_by_step(approve) if payload.get("step_mode") else None,
        on_event=emit,
        prompt=CONTROLLER_PROMPT,
        finish="report_files",
        recovery=False,
    )
    warnings = list(ctx.warnings)
    warnings += [
        f"{s.description}: {s.error}" for s in ctx.steps if s.status in {"failed", "blocked"}
    ]
    field = ctx.get("field_data") or {}
    priors = ctx.get("geological_priors") or {}
    unchanged = True
    for original in field.get("files", []) + priors.get("files", []):
        try:
            current = fingerprint(original["path"])
            if current["sha256"] != original["sha256"]:
                unchanged = False
        except OSError:
            unchanged = False
    if not unchanged:
        warnings.append(
            "Source files changed during the run; inspect the provenance before using results."
        )
    complete = ctx.has("report_files") and ctx.ended != "stopped"
    status = "incomplete" if not complete else ("needs_review" if warnings else "success")
    geo = ctx.get("geo_model") or {}
    exports = geo.get("exports", {})
    artifacts = []
    for name, path in {
        "geosage_volume": exports.get("model"),
        **exports.get("figures", {}),
    }.items():
        if path:
            artifacts.append(
                {
                    "artifact_id": name,
                    "path": path,
                    "format": Path(path).suffix.lstrip("."),
                    "kind": "model" if name == "geosage_volume" else "figure",
                    "metadata": {
                        "title": name,
                        "coordinate_units": "m",
                        "scalar_cmaps": {
                            "Density contrast (g/cm3)": "RdBu_r",
                            "Susceptibility (SI)": "viridis",
                        },
                        "z_convention": "elevation, positive up",
                    },
                }
            )
    result = {
        "status": status,
        "interpretation": (
            "GeoSAGE completed. "
            + (
                "Review the listed limitations."
                if warnings
                else "The reviewed interpretation is ready."
            )
        )
        if complete
        else "GeoSAGE stopped before the final report. Inspect the completed steps and warnings.",
        "warnings": warnings,
        "report_files": ctx.get("report_files") or {},
        "output_dir": str(output),
        "execution_plan": ctx.plan(),
        "workflow_config": _redact_trace_value(ctx.config),
        "exports": exports,
        "artifacts": artifacts,
        "source_files_unchanged": unchanged,
        "review_decision": ctx.get("review_decision"),
        "summary": (geo.get("prepared") or {}).get("result_summary", {}),
    }
    audit = {
        "schema_version": "1",
        "assistant": "geosage",
        "events": recorded,
        "steps": ctx.plan(),
        "inputs": field.get("files", []),
        "priors": priors.get("files", []),
        "source_files_unchanged": unchanged,
        "status": status,
        "warnings": warnings,
    }
    for name, data in (("studio_audit.json", audit), ("studio_result.json", result)):
        (output / name).write_text(
            json.dumps(_redact_trace_value(data), indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    if complete:
        progress("GeoSAGE complete", 1.0, result["interpretation"])
    return result
