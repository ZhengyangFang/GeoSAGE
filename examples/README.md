# Examples

Install GeoSAGE first with `python -m pip install -e .`.

- `python examples/quick_test.py` checks package imports, configuration loading,
  and local data paths without starting an inversion.
- `python examples/validate_archived_results.py --data-root /path/to/workspace --output-dir /path/to/new/validation`
  validates existing Hannah/Iowa arrays, checks source-file integrity, and compares
  CSV rebuilds with the archived labels. No LLM calls are made.

Configuration templates are in `configs/`, including `hannah_geology_only.json`.
Inputs can live under `data/` and archives under `outputs/`; legacy flat directories
are supported. Use `GEOSAGE_WORKSPACE` to point the quick test at another workspace.
The validator's `--data-root` explicitly selects the data workspace, independently
of that environment variable. Validation output must be a new directory.

All validation results and recovered mappings stay local. See
[the path guide](../docs/paths.md) for migration and path precedence.
