# Open-source scope

The public GeoSAGE repository contains the software and the material needed to
understand, test, and reproduce its workflow without publishing research data or
generated scientific products.

## Published

- Python source code and package metadata;
- reusable configuration and prompt templates;
- notebooks with execution counts and outputs cleared;
- documentation and the static installation website;
- synthetic and temporary-directory tests; and
- GitHub Actions definitions used to verify supported Python versions and the
  PyHydroGeophysX integration.

The `tests/` directory is intentionally public. Its fixtures are generated at
test time and contain no Hannah or Iowa measurements. These tests document
expected behavior and prevent regressions in sign conventions, path isolation,
configuration overrides, report images, and Studio compatibility.

## Kept local

- raw or processed survey inputs;
- inversion arrays, meshes, VTK files, figures, reports, and logs;
- Hannah and Iowa data archives restored from Zenodo;
- manuscript products and one-off analysis outputs;
- API keys, `.env` files, provider sessions, and local agent settings;
- virtual environments, caches, coverage files, IDE settings, and operating
  system metadata.

These categories are excluded by `.gitignore`. Before publishing a change, use
`git status --short` and `git ls-files` to confirm that no local research file
has been added explicitly.
