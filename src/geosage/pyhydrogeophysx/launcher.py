"""Launch GeoSAGE in the host Studio, using its theme and shared panels."""

import os
from pathlib import Path


def main(argv=None):
    from geosage.paths import workspace_root

    workspace = workspace_root(Path(__file__))
    os.environ.setdefault(
        "PYHYDROGEOPHYSX_OUTPUT_DIR",
        str(Path(workspace).resolve() / "outputs" / "studio_runs"),
    )
    from PyHydroGeophysX.qt_apps.launcher import main as launch

    return launch(argv, assistant="geosage")


if __name__ == "__main__":
    raise SystemExit(main())
