"""Adaptive views preserve geometry and meaning across unrelated projects."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from geosage.plotting.paper import (
    render_model_sections,
    render_property_crossplot,
    render_data_fit,
    interpolate_map,
)
from geosage.plotting.scales import field_scale, category_colors, coordinate_scale, middle_index


@pytest.mark.parametrize("values", [[0, 0], [7, 7], [-7, -7], [-0.00001, 0.00002], [1, 2, 100000]])
def test_full_range_includes_constants_outliers_and_small_values(values):
    scale = field_scale(values)
    low, high = scale["limits"]
    assert low < high and low <= min(values) <= max(values) <= high
    if min(values) < 0 < max(values):
        assert low == -high and scale["cmap"] == "RdBu_r"


def test_physical_midpoint_and_coordinate_units_do_not_depend_on_cell_count():
    assert middle_index([0, 1, 2, 10]) == 2
    assert coordinate_scale([400000, 400050], [-30, 0]) == (1.0, "m")
    assert coordinate_scale([0, 50000]) == (1000.0, "km")


def test_adaptive_sections_keep_nonuniform_edges_and_discrete_ids():
    density = np.arange(12.0).reshape(3, 2, 2)
    labels = np.where(density % 2, 100, 2)
    args = (
        density,
        density * 0.01,
        labels,
        np.array([0, 1, 2, 10]),
        np.array([0, 2, 3]),
        np.array([-8, -2, 0]),
        np.array([-5, -1]),
        np.array([1, 2.5]),
        [0],
        [1],
    )
    fig = render_model_sections(*args, adaptive=True)
    np.testing.assert_array_equal(
        fig.axes[0].collections[0].get_coordinates()[0, :, 0], [0, 1, 2, 10]
    )
    np.testing.assert_array_equal(fig.axes[0].collections[0].get_array(), density[:, :, 0].T)
    geological = fig.axes[2].collections[0]
    assert geological.cmap.N == 2
    np.testing.assert_array_equal(
        geological.get_array(), np.searchsorted([2, 100], labels[:, :, 0].T)
    )
    assert fig.axes[0].get_aspect() == 1.0
    plt.close(fig)
    fig = render_model_sections(*args[:2], None, *args[3:], adaptive=True)
    assert len(fig.axes) == 6
    plt.close(fig)


def test_many_sparse_ids_have_unique_colors_and_crossplot_contains_all_values():
    ids = np.arange(0, 35) * 21
    colors = category_colors(ids)
    assert len(set(colors.values())) == len(ids)
    density = np.linspace(-400, 900, len(ids))
    susc = np.linspace(-0.02, 0.0001, len(ids))
    fig = render_property_crossplot(density, susc, ids, adaptive=True)
    ax = fig.axes[0]
    assert ax.get_xlim()[0] < min(density) and ax.get_xlim()[1] > max(density)
    assert sum(len(c.get_offsets()) for c in ax.collections) == len(ids)
    plt.close(fig)


def test_fit_never_extrapolates_outside_survey_in_adaptive_mode():
    xy = np.array([[0, 0], [0, 1], [1, 0]])
    x, y = np.meshgrid([0, 0.5, 1], [0, 0.5, 1])
    assert np.isnan(interpolate_map(xy, [0, 1, 2], x, y, extrapolate=False)[-1, -1])


@pytest.mark.parametrize("case", ["Hannah", "Iowa", "Unrelated project"])
def test_fit_scales_come_from_data_even_for_a_familiar_project_name(tmp_path, monkeypatch, case):
    from matplotlib.figure import Figure

    xy = np.array([[0, 0], [0, 20], [100, 0], [100, 20]])
    observed = np.array([-1000, -1, 0, 2000])
    predicted = observed + 300
    captured = []
    save = Figure.savefig

    def capture(fig, *a, **kw):
        captured.extend(ax.images[0].get_clim() for ax in fig.axes if ax.images)
        return save(fig, *a, **kw)

    monkeypatch.setattr(Figure, "savefig", capture)
    render_data_fit(
        [(case, "gravity", "Gravity")],
        lambda *_: (xy, observed, predicted, predicted - observed, "mGal"),
        tmp_path / "fit.png",
        adaptive=True,
        dpi=35,
    )
    assert captured[:2] == [(-2300.0, 2300.0)] * 2
    assert captured[2] == (-300.0, 300.0)
