# Open-source scope

The public GeoSAGE repository contains the software and the material needed to
understand, test, and reproduce its workflow without publishing research data or
generated scientific products.

## Published

- Python source code and package metadata;
- reusable configuration and prompt templates;
- notebooks with execution counts and outputs cleared;
- documentation and the static installation website; and
- the GitHub Pages workflow required to publish that website.

## Kept local

- raw or processed survey inputs;
- inversion arrays, meshes, VTK files, figures, reports, and logs;
- Hannah and Iowa data archives restored from Zenodo;
- manuscript products and one-off analysis outputs;
- API keys, `.env` files, provider sessions, and local agent settings;
- virtual environments, caches, coverage files, IDE settings, and operating
  system metadata; and
- private development tests and their temporary fixtures.

These categories are excluded by `.gitignore`. Before publishing a change, use
`git status --short` and `git ls-files` to confirm that no local research file
has been added explicitly.
