"""Scientific conventions shared by paper notebooks and the Studio adapter."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from geosage.plotting.paper import render_model_sections, render_data_fit


def test_sections_preserve_elevation_orientation_and_sparse_geo_ids():
    density = np.arange(24.0).reshape(2, 3, 4)
    susceptibility = density / 1000
    labels = np.where(density % 2, 7, 1)
    before = [a.copy() for a in (density, susceptibility, labels)]
    fig = render_model_sections(
        density,
        susceptibility,
        labels,
        np.array([0, 100, 300]),
        np.array([0, 100, 200, 400]),
        np.array([-500, -300, -150, -50, 0]),
        np.array([-400, -225, -100, -25]),
        np.array([50, 150, 300]),
        [1],
        [2],
    )
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), density[:, :, 1].T)
    np.testing.assert_array_equal(fig.axes[2].images[0].get_array(), labels[:, :, 1].T)
    assert fig.axes[0].images[0].origin == "lower"
    np.testing.assert_array_equal(fig.axes[3].collections[0].get_array(), density[:, 2, :].T)
    for source, copy in zip((density, susceptibility, labels), before):
        np.testing.assert_array_equal(source, copy)
    plt.close(fig)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_line_surveys_render_station_values_without_fabricating_2d_maps(
    tmp_path, monkeypatch, count
):
    from matplotlib.figure import Figure

    xy = np.column_stack((np.arange(count), np.arange(count)))
    observed = np.arange(count, dtype=float)
    predicted = observed + 2
    arrays = []
    original = Figure.savefig

    def capture(fig, *args, **kwargs):
        for ax in fig.axes:
            assert not ax.images
            arrays.extend(
                np.asarray(c.get_array()) for c in ax.collections if c.get_array() is not None
            )
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = tmp_path / "fit.png"
    render_data_fit(
        [("Example", "gravity", "Gravity")],
        lambda *_: (xy, observed, predicted, predicted - observed, "mGal"),
        output,
        dpi=30,
    )
    assert output.is_file()
    for expected in (observed, predicted, np.full(count, 2)):
        assert any(np.array_equal(a, expected) for a in arrays)
