"""Shared figures for notebooks and Studio.

The default preset preserves paper exports. Pass ``adaptive=True`` for compact,
data-driven Studio views. Both presets use the same plotting implementation.
"""

from .paper import render_data_fit, render_model_sections, render_property_crossplot

__all__ = ["render_data_fit", "render_model_sections", "render_property_crossplot"]
