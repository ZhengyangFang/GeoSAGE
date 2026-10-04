"""Resolve workspace paths without changing the process working directory.

New workspaces use data/ for inputs and outputs/ for numerical results. Existing
flat workspaces are supported for reading archived files without moving them.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def workspace_root(start: str | Path | None = None) -> Path:
    """Use an override, marked workspace, enclosing checkout, or start directory.

    A .geosage-workspace marker can group sibling source checkouts with private
    data/ and outputs/. A checkout directly inside that workspace shares its
    data root; deeper independent projects retain their own root.
    """
    override = os.environ.get("GEOSAGE_WORKSPACE")
    if override:
        return Path(override).expanduser().resolve()
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / ".geosage-workspace").is_file():
            return parent
        metadata = parent / "pyproject.toml"
        if metadata.is_file():
            import tomllib
            with metadata.open("rb") as handle:
                if tomllib.load(handle).get("project", {}).get("name") == "geosage":
                    shared = parent.parent
                    return shared if (shared / ".geosage-workspace").is_file() else parent
    return current


def data_path(name: str | Path, workspace: str | Path | None = None) -> Path:
    """Resolve inputs in data/, falling back to an existing flat archive."""
    root = Path(workspace).resolve() if workspace is not None else workspace_root()
    relative = Path(name)
    if relative.is_absolute():
        return relative
    modern, legacy = root / "data" / relative, root / relative
    first = relative.parts[0] if relative.parts else ""
    return legacy if first and not (root / "data" / first).exists() and (root / first).exists() else modern


def result_path(name: str | Path, workspace: str | Path | None = None) -> Path:
    """Find a named result archive in outputs/ or an existing flat workspace."""
    root = Path(workspace).resolve() if workspace is not None else workspace_root()
    relative = Path(name)
    if relative.is_absolute():
        return relative
    modern, legacy = root / "outputs" / relative, root / relative
    first = relative.parts[0] if relative.parts else ""
    return legacy if first and not (root / "outputs" / first).exists() and (root / first).exists() else modern


def resolve_path(value: str | Path, root: Path, *, output: bool = False) -> Path:
    """Anchor a config path and recognize current and historical case layouts."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    first = path.parts[0] if path.parts else ""
    if first == "data":
        return data_path(Path(*path.parts[1:]), root).resolve()
    if first == "outputs":
        if output:
            return (root / path).resolve()
        return result_path(Path(*path.parts[1:]), root).resolve()
    if first in {"Hannah", "Iowa", "logdata"}:
        return data_path(path, root).resolve()
    if "_Inversion" in first or first == "Hannah_LLM_comparison":
        return ((root / "outputs" / path) if output else result_path(path, root)).resolve()
    return (root / path).resolve()


def resolve_config_paths(config: dict[str, Any], config_path: Path | None = None) -> dict[str, Any]:
    """Resolve declared file paths once, retaining absolute paths on later loads.

    project.workspace_dir, when present, is relative to the JSON file's directory
    (or cwd for dict configs). Otherwise GEOSAGE_WORKSPACE or checkout discovery
    supplies the workspace. Paths in all other sections are workspace-relative.
    """
    project = config["project"]
    base = config_path.parent if config_path is not None else Path.cwd()
    explicit = project.get("workspace_dir")
    root = (base / Path(explicit).expanduser()).resolve() if explicit else workspace_root(base)
    project["workspace_dir"] = str(root)
    mode = str(config.get("run", {}).get("execution_mode") or "").strip().lower()
    reuse = mode == "interpret_existing" or (not mode and not config.get("run", {}).get("run_inversion", True))
    for key in ("input_dir", "source_inversion_dir", "output_dir", "interpretation_output_dir"):
        if project.get(key):
            is_output = key == "interpretation_output_dir" or (key == "output_dir" and not reuse)
            project[key] = str(resolve_path(project[key], root, output=is_output))
    geology = config.get("geology", {})
    for key in ("unit_defs_csv", "unit_groups_csv", "context_path", "unit_id_npy"):
        if geology.get(key):
            raw = Path(geology[key]).expanduser()
            resolved = resolve_path(raw, root)
            if key == "unit_id_npy" and not raw.is_absolute() and not resolved.exists():
                source = project.get("source_inversion_dir") or project.get("output_dir")
                if source and (Path(source) / raw).is_file():
                    resolved = (Path(source) / raw).resolve()
            geology[key] = str(resolved)
    return config
