# GeoSAGE Examples

This directory contains small examples for checking a GeoSAGE checkout before
running the full inversion workflow.

## Quick Test

After installing the project dependencies, run from the repository root:

```bash
python examples/quick_test.py
```

The quick test verifies that the project code can be imported, loads the Hannah
geology-only example configuration, and checks whether the required Zenodo data
folders have been extracted into the repository root.

It does not run the full gravity-magnetic inversion.

## Configuration

`config_hannah_geology_only.json` is a minimal configuration for rebuilding the
Hannah pseudo-geological model from the archived `Hannah_Inversion_GPT` output.
Download the data from https://doi.org/10.5281/zenodo.19958815 before running the
full workflow.
