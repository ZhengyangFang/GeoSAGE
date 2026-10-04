"""Launch GeoSAGE in the host Studio, using its theme and shared panels."""


def main(argv=None):
    from PyHydroGeophysX.qt_apps.launcher import main as launch

    return launch(argv, assistant="geosage")


if __name__ == "__main__":
    raise SystemExit(main())
