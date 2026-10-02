from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def status(path: Path) -> str:
    return "FOUND" if path.exists() else "MISSING"


def main() -> int:
    try:
        from geosage.runner import load_config
    except ModuleNotFoundError as exc:
        print("GeoSAGE quick test")
        print(f"Missing Python dependency while importing GeoSAGE: {exc.name}")
        print('Install the project first with: python -m pip install -e ".[notebooks,reports]"')
        return 1

    cfg_path = REPO_ROOT / "configs" / "hannah_geology_only.json"
    cfg = load_config(cfg_path)

    print("GeoSAGE quick test")
    print("Project:", cfg["project"]["name"])
    print("Run inversion:", cfg["run"]["run_inversion"])
    print("Run geology model:", cfg["run"]["run_geology_model"])

    required_paths = [
        Path(cfg["project"]["input_dir"]),
        Path(cfg["project"]["output_dir"]),
        Path(cfg["geology"]["unit_defs_csv"]),
        Path(cfg["geology"]["unit_groups_csv"]),
    ]

    missing = []
    for path in required_paths:
        resolved = path if path.is_absolute() else REPO_ROOT / path
        print(f"{path}: {status(resolved)}")
        if not resolved.exists():
            missing.append(path)

    if missing:
        print("Quick test completed with missing data paths.")
        print("Place local inputs in data/ and archived models in outputs/, or set GEOSAGE_WORKSPACE.")
    else:
        print("Quick test completed. Required data paths are present.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
