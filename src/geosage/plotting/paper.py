"""Shared plotting routines extracted from notebooks 3_1, 3_2 and 4_1.

The notebook default preserves the original paper exports. Studio selects the
adaptive preset: data-derived scales, true cell geometry and compact layouts.
Both use the same scientific arrays and predicted-minus-observed residuals.
No scientific inputs are written. Interactive VTK export remains separate.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.interpolate import griddata

from .scales import category_colors, coordinate_scale, field_scale

PAPER_STYLE = {"font.family": "Arial", "font.sans-serif": ["Arial"], "axes.unicode_minus": False}
FIT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
}

DISPLAY_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "text.color": "#202733",
    "axes.labelcolor": "#526070",
    "xtick.color": "#526070",
    "ytick.color": "#526070",
    "axes.edgecolor": "#cbd1da",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.unicode_minus": False,
}


def render_model_sections(
    dens,
    susc,
    unit_id,
    x_edges,
    y_edges,
    z_edges,
    z_centers,
    y_centers,
    k_indices,
    j_indices,
    *,
    output=None,
    dpi=600,
    x_ticks=None,
    y_ticks=None,
    density_limits=(-0.4, 0.4),
    susceptibility_limits=(-0.05, 0.05),
    adaptive=False,
):
    """Physical sections; adaptive display or the original paper export preset."""
    with mpl.rc_context(DISPLAY_STYLE if adaptive else PAPER_STYLE):
        labels = np.sort(np.unique(unit_id).astype(int)) if unit_id is not None else None
        fields = [dens, susc] + ([unit_id] if labels is not None else [])
        styles = (
            [field_scale(dens, centered=True), field_scale(susc)]
            if adaptive
            else [
                {"limits": density_limits, "cmap": "seismic"},
                {"limits": susceptibility_limits, "cmap": "seismic"},
            ]
        )
        if labels is not None:
            if adaptive:
                colors = category_colors(unit_id)
                cmap_geo = ListedColormap([colors[str(int(v))] for v in labels])
                bounds = np.arange(len(labels) + 1) - 0.5
            else:
                bounds = np.r_[labels[0] - 0.5, (labels[:-1] + labels[1:]) * 0.5, labels[-1] + 0.5]
                cmap_geo = ListedColormap(
                    plt.get_cmap("coolwarm")(np.linspace(0, 1, len(labels))),
                    name="coolwarm_discrete_geo",
                )
            styles.append({"cmap": cmap_geo, "norm": BoundaryNorm(bounds, cmap_geo.N, clip=True)})
        rows = [
            *(("xy", k, float(z_centers[k])) for k in k_indices),
            *(("xz", j, float(y_centers[j])) for j in j_indices),
        ]
        if not rows:
            raise ValueError("No slice/section rows found.")
        nrows, ncols = len(rows), len(fields)
        divisor, units = coordinate_scale(x_edges, y_edges, z_edges) if adaptive else (1000.0, "km")
        x, y, z = (np.asarray(e) / divisor for e in (x_edges, y_edges, z_edges))
        ratios = [
            float(np.clip(np.ptp(y if mode == "xy" else z) / np.ptp(x), 0.2, 3))
            for mode, _, _ in rows
        ]
        fig = plt.figure(
            figsize=(4 * ncols, 3.4 * sum(ratios) + 2) if adaptive else (10, 15),
            constrained_layout=False,
        )
        gs = fig.add_gridspec(
            nrows=nrows + 1,
            ncols=ncols,
            height_ratios=(ratios if adaptive else [1.0] * nrows) + [0.11],
            left=0.075 if adaptive else 0.16,
            right=0.97 if adaptive else 0.95,
            top=0.85 if adaptive else 0.94,
            bottom=0.12,
            wspace=0.22 if adaptive else 0.04,
            hspace=0.62 if adaptive else 0.15,
        )
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)])
        refs = [None] * ncols
        for r, (mode, idx, position) in enumerate(rows):
            vertical = y if mode == "xy" else z
            for c, (array, style) in enumerate(zip(fields, styles)):
                ax = axes[r, c]
                plane = array[:, :, idx].T if mode == "xy" else array[:, idx, :].T
                if adaptive and c == 2:
                    plane = np.searchsorted(labels, plane)
                options = {"cmap": style["cmap"]}
                if "norm" in style:
                    options["norm"] = style["norm"]
                else:
                    options.update(vmin=style["limits"][0], vmax=style["limits"][1])
                if mode == "xy" and not adaptive:
                    artist = ax.imshow(
                        plane,
                        origin="lower",
                        extent=[x[0], x[-1], y[0], y[-1]],
                        interpolation="nearest",
                        **options,
                    )
                else:
                    # Actual cell edges preserve nonuniform mesh geometry.
                    artist = ax.pcolormesh(x, vertical, plane, shading="auto", **options)
                if refs[c] is None:
                    refs[c] = artist
                ax.set_xlim(x[0], x[-1])
                ax.set_ylim(vertical[0], vertical[-1])
                ylabel = "Northing" if mode == "xy" else "Elevation"
                if adaptive:
                    ax.set_aspect("equal", adjustable="box")
                    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                    ax.ticklabel_format(useOffset=False, style="plain")
                    ax.set_xlabel(f"Easting ({units})")
                    ax.set_ylabel(f"{ylabel} ({units})" if c == 0 else "")
                    if c == 0:
                        section = "Elevation" if mode == "xy" else "Northing"
                        ax.set_title(
                            f"{section} {position / divisor:,.6g} {units}",
                            loc="left",
                            fontsize=10,
                            pad=12,
                        )
                else:
                    if x_ticks is not None:
                        ax.set_xticks(x_ticks)
                    ax.set_yticks(
                        y_ticks
                        if mode == "xy" and y_ticks is not None
                        else np.linspace(*ax.get_ylim(), 4)
                    )
                    ax.tick_params(direction="in", labelsize=12)
                    if r == nrows - 1:
                        ax.set_aspect("auto")
                        ax.set_box_aspect((y_edges[-1] - y_edges[0]) / (x_edges[-1] - x_edges[0]))
                        ax.set_xlabel("Easting (km)", labelpad=2, fontsize=14)
                    else:
                        ax.set_xlabel("")
                        ax.tick_params(labelbottom=False)
                    if c == 0:
                        ax.set_ylabel(f"{ylabel} (km)", fontsize=14)
                    else:
                        ax.set_ylabel("")
                        ax.tick_params(labelleft=False)
        titles = [
            "Density contrast" if adaptive else "Density",
            "Susceptibility",
            "Geology" if adaptive else "Geo Group",
        ]
        for c, title in enumerate(titles[:ncols]):
            pos = axes[0, c].get_position()
            fig.text(
                (pos.x0 + pos.x1) * 0.5,
                0.91 if adaptive else 0.955,
                title,
                ha="center",
                va="bottom",
                fontsize=13 if adaptive else 14,
                fontweight="medium" if adaptive else "bold",
            )
        if adaptive:
            fig.text(0.075, 0.98, "Model sections", va="top", fontsize=18, fontweight="medium")
            fig.text(
                0.97,
                0.98,
                "Full range · true geometry",
                ha="right",
                va="top",
                fontsize=9,
                color="#687080",
            )
        fig.canvas.draw()
        bar_labels = [
            "Density contrast (g/cm³)" if adaptive else r"Density (g/cm$^3$)",
            "Susceptibility (SI)",
            "Group ID",
        ]
        for c, artist in enumerate(refs):
            pos = axes[-1, c].get_position()
            # Anchor adaptive bars to the layout, not the height of a thin survey.
            if adaptive:
                pos = gs[-2, c].get_position(fig)
            bar_y = (
                max(0.055, axes[-1, c].get_position().y0 - 0.65 / fig.get_figheight())
                if adaptive
                else pos.y0 - 0.04
            )
            cax = fig.add_axes([pos.x0, bar_y, pos.width, 0.01])
            bar = fig.colorbar(artist, cax=cax, orientation="horizontal")
            bar.set_label(
                bar_labels[c] if adaptive or c != 2 else "Geo Group ID",
                fontsize=9 if adaptive else 12,
            )
            if c == 2:
                step = max(1, int(np.ceil(len(labels) / 12)))
                bar.set_ticks(np.arange(len(labels))[::step] if adaptive else labels[::step])
                if adaptive:
                    bar.set_ticklabels(labels[::step])
        if output is not None:
            fig.savefig(output, dpi=dpi, bbox_inches="tight")
        return fig


def render_property_crossplot(
    dens,
    susc,
    unit_id,
    unit_defs=None,
    *,
    output=None,
    dpi=600,
    density_limits=(-1, 1),
    susceptibility_limits=(-0.4, 0.4),
    legend_location="lower left",
    adaptive=False,
):
    """Paper Figures S1/S2: density/susceptibility coloured by original unit IDs."""
    dens_flat, susc_flat, unit_flat = (
        np.ravel(dens),
        np.ravel(susc),
        np.asarray(unit_id, dtype=int).ravel(),
    )
    unit_defs = unit_defs or {}
    with mpl.rc_context(DISPLAY_STYLE if adaptive else PAPER_STYLE):
        fig = plt.figure(figsize=(11, 6) if adaptive else (9, 6))
        unique_ids = np.unique(unit_flat)
        has_unclassified = 0 in unique_ids
        unique_ids = unique_ids[unique_ids != 0]
        unique_ids = np.sort(unique_ids)

        n_units = len(unique_ids)
        cmap = mpl.colormaps.get_cmap("coolwarm").resampled(max(n_units, 1))
        colors = category_colors(unit_flat) if adaptive else {}

        for i, uid in enumerate(unique_ids):
            mask = unit_flat == uid
            if not np.any(mask):
                continue
            label = f"Unit {uid}: {unit_defs.get(uid, {}).get('name', '')}"
            if adaptive:
                from textwrap import fill

                label = fill(label.rstrip(": "), width=32)
            color = colors[str(uid)] if adaptive else cmap(i)
            plt.scatter(
                dens_flat[mask],
                susc_flat[mask],
                s=1,
                alpha=0.55 if adaptive else 1,
                label=label,
                color=color,
            )

        if has_unclassified:
            mask0 = unit_flat == 0
            if np.any(mask0):
                plt.scatter(
                    dens_flat[mask0],
                    susc_flat[mask0],
                    s=1,
                    alpha=1,
                    color=colors["0"] if adaptive else "lightgray",
                    label="Unclassified (0)",
                )

        plt.axhline(0.0, color="k", linewidth=0.5)
        plt.axvline(0.0, color="k", linewidth=0.5)

        # Fixed display range for consistent visual comparison across runs
        dens_min, dens_max = field_scale(dens)["limits"] if adaptive else density_limits
        susc_min, susc_max = field_scale(susc)["limits"] if adaptive else susceptibility_limits

        dens_pad = (dens_max - dens_min) * 0.1 if dens_max != dens_min else 0.1
        susc_pad = (susc_max - susc_min) * 0.1 if susc_max != susc_min else 0.01

        plt.xlim(dens_min - dens_pad, dens_max + dens_pad)
        plt.ylim(susc_min - susc_pad, susc_max + susc_pad)

        plt.xlabel(
            r"Density contrast (g/cm$^3$)" if adaptive else r"Density (g/cm$^3$)",
            fontweight="normal" if adaptive else "bold",
        )
        plt.ylabel("Susceptibility (SI)", fontweight="normal" if adaptive else "bold")
        plt.grid(
            True,
            linestyle="-" if adaptive else "--",
            linewidth=0.4 if adaptive else 0.5,
            alpha=0.25 if adaptive else 1,
        )
        if adaptive:
            ax = plt.gca()
            ax.set_title("Property relationships", loc="left", fontsize=18, pad=28)
            ax.text(
                0,
                1.025,
                f"{len(dens_flat):,} cells · full range",
                transform=ax.transAxes,
                color="#687080",
                fontsize=9,
            )
            fig.subplots_adjust(left=0.10, right=0.65, top=0.84, bottom=0.14)
            if len(colors) <= 12:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1),
                    fontsize=9,
                    frameon=False,
                    markerscale=4,
                    labelspacing=1,
                )
            else:
                ids = np.sort(np.unique(unit_flat))
                cm = ListedColormap([colors[str(uid)] for uid in ids])
                norm = BoundaryNorm(np.arange(len(ids) + 1) - 0.5, len(ids))
                bar = fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, pad=0.08, fraction=0.04
                )
                ticks = np.unique(np.linspace(0, len(ids) - 1, min(12, len(ids))).astype(int))
                bar.set_ticks(ticks, labels=ids[ticks])
                bar.set_label(f"Unit ID · {len(ids)} categories")
        else:
            plt.legend(loc=legend_location, fontsize=8, frameon=True)
            plt.tight_layout()

        if output is not None:
            fig.savefig(output, dpi=dpi)
        return fig


def nice_limit(value):
    """Round a colour-bar range to a compact readable limit."""
    if value <= 0:
        return 1.0
    scale = 10.0 ** np.floor(np.log10(value))
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value / scale <= candidate:
            return candidate * scale
    return 10.0 * scale


def interpolate_map(xy, values, grid_x, grid_y, *, extrapolate=True):
    mapped = griddata(xy, values, (grid_x, grid_y), method="linear")
    if extrapolate and np.isnan(mapped).any():
        mapped = np.where(
            np.isnan(mapped), griddata(xy, values, (grid_x, grid_y), method="nearest"), mapped
        )
    return mapped


def render_data_fit(columns, load_property, output, *, dpi=600, adaptive=False):
    """Figure 8 layout; loader returns xy, observed, predicted, pred-obs, unit."""
    with mpl.rc_context(DISPLAY_STYLE if adaptive else FIT_STYLE):
        rows = ("Observed", "Predicted", "Residual")
        field_limits = {
            ("Hannah", "gravity"): 32.0,
            ("Hannah", "magnetics"): 420.0,
            ("Iowa", "gravity"): 80.0,
            ("Iowa", "magnetics"): 1500.0,
        }
        field_ticks = {
            ("Hannah", "gravity"): (-30.0, -15.0, 0.0, 15.0, 30.0),
            ("Hannah", "magnetics"): (-400.0, -200.0, 0.0, 200.0, 400.0),
        }
        fig = plt.figure(
            figsize=(12, 3.6 * len(columns) + 1.1) if adaptive else (6.5 * len(columns), 19.0)
        )
        grid = fig.add_gridspec(
            len(columns) if adaptive else 3,
            3 if adaptive else len(columns),
            left=0.08 if adaptive else 0.10,
            right=0.97 if adaptive else 0.99,
            bottom=0.10 if adaptive else 0.075,
            top=0.83 if adaptive else 0.92,
            wspace=0.35 if adaptive else 0.28,
            hspace=0.70 if adaptive else 0.10,
        )
        row_centres = (0.79, 0.50, 0.21)
        for row_title, row_centre in [] if adaptive else zip(rows, row_centres):
            fig.text(
                0.025,
                row_centre,
                row_title,
                ha="center",
                va="center",
                rotation=0,
                fontsize=24,
                fontweight="bold",
            )

        panel_axes = [[None for _ in columns] for _ in rows]
        for column_index, (case, property_name, column_title) in enumerate(columns):
            xy, observed, predicted, residual, unit = load_property(case, property_name)
            xy = np.asarray(xy)
            can_interpolate = len(xy) >= 3 and np.linalg.matrix_rank(xy - xy.mean(axis=0)) == 2
            min_e, max_e = np.min(xy[:, 0]), np.max(xy[:, 0])
            min_n, max_n = np.min(xy[:, 1]), np.max(xy[:, 1])
            grid_e, grid_n = np.linspace(min_e, max_e, 260), np.linspace(min_n, max_n, 260)
            grid_x, grid_y = np.meshgrid(grid_e, grid_n)
            field_limit = field_limits.get(
                (case, property_name),
                nice_limit(max(np.max(np.abs(observed)), np.max(np.abs(predicted)))),
            )
            residual_limit = nice_limit(np.max(np.abs(residual)))
            panels = (
                (observed, "coolwarm", -field_limit, field_limit),
                (predicted, "coolwarm", -field_limit, field_limit),
                (residual, "RdBu_r", -residual_limit, residual_limit),
            )
            divisor, coord_unit = (
                coordinate_scale(xy[:, 0], xy[:, 1]) if adaptive else (1000.0, "km")
            )
            if adaptive:
                shared = field_scale(np.r_[observed, predicted])
                difference = field_scale(residual, centered=True)
                panels = [
                    (a, scale["cmap"], *scale["limits"])
                    for a, scale in (
                        (observed, shared),
                        (predicted, shared),
                        (residual, difference),
                    )
                ]

            for row_index, (row_title, (values, cmap, vmin, vmax)) in enumerate(zip(rows, panels)):
                slot = grid[column_index, row_index] if adaptive else grid[row_index, column_index]
                panel_grid = slot.subgridspec(
                    1,
                    2,
                    width_ratios=(1.0, 0.04 if adaptive else 0.06),
                    wspace=0.08 if adaptive else 0.05,
                )
                ax = fig.add_subplot(panel_grid[0, 0])
                panel_axes[row_index][column_index] = ax
                if can_interpolate:
                    mapped = interpolate_map(xy, values, grid_x, grid_y, extrapolate=not adaptive)
                    image = ax.imshow(
                        mapped,
                        extent=[min_e / divisor, max_e / divisor, min_n / divisor, max_n / divisor],
                        origin="lower",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="bilinear",
                    )
                    if not adaptive:
                        ax.contour(
                            grid_x / divisor,
                            grid_y / divisor,
                            mapped,
                            levels=8,
                            colors="black",
                            linewidths=0.45,
                            linestyles="dashed",
                            alpha=0.55,
                        )
                else:
                    # A line survey cannot define a 2D interpolated field.
                    # Keep the measured stations visible without inventing a surface.
                    image = ax.scatter(
                        xy[:, 0] / divisor,
                        xy[:, 1] / divisor,
                        c=values,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                if adaptive:
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel(f"Easting ({coord_unit})")
                    if row_index == 0:
                        ax.set_ylabel(f"Northing ({coord_unit})")
                    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
                    ax.ticklabel_format(useOffset=False, style="plain")
                    ax.set_title(row_title, loc="left", fontsize=11, pad=10)
                    cax = fig.add_subplot(panel_grid[0, 1])
                    colorbar = fig.colorbar(image, cax=cax)
                    colorbar.ax.tick_params(labelsize=8)
                    colorbar.set_label(unit, fontsize=9)
                    continue
                ax.set_aspect("auto")
                ax.tick_params(labelsize=20, width=1.0, length=4.5)
                if row_index == 2:
                    ax.set_xlabel("Easting (km)", fontsize=23, fontweight="bold")
                else:
                    ax.tick_params(labelbottom=False)
                if column_index == 0:
                    ax.set_ylabel("Northing (km)", fontsize=23, fontweight="bold")
                else:
                    ax.tick_params(labelleft=False)
                for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                    tick_label.set_fontweight("bold")
                colorbar_axis = fig.add_subplot(panel_grid[0, 1])
                colorbar = fig.colorbar(image, cax=colorbar_axis)
                colorbar.set_ticks(
                    field_ticks[(case, property_name)]
                    if row_index < 2 and (case, property_name) in field_ticks
                    else np.linspace(vmin, vmax, 5)
                )
                colorbar.set_label(unit, fontsize=21, fontweight="bold")
                colorbar.ax.tick_params(labelsize=19, width=0.9, length=3.5)
                for tick_label in colorbar.ax.get_yticklabels():
                    tick_label.set_fontweight("bold")

        if adaptive:
            fig.text(0.08, 0.975, "Data fit", va="top", fontsize=18, fontweight="medium")
            fig.text(
                0.97,
                0.975,
                "Residual = predicted − observed",
                ha="right",
                va="top",
                fontsize=9,
                color="#687080",
            )
            for i, (case, key, title) in enumerate(columns):
                xy, observed, predicted, residual, unit = load_property(case, key)
                pos = grid[i, 0].get_position(fig)
                detail = f"{title}  ·  {len(observed):,} stations  ·  RMSE {np.sqrt(np.mean(residual**2)):.3g} {unit}"
                fig.text(0.08, pos.y1 + 0.07, detail, fontsize=11)
            fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return output
        fig.canvas.draw()
        for row_index, axes_in_row in enumerate(panel_axes):
            label_y = max(axis.get_position().y1 for axis in axes_in_row) + 0.006
            for column_index, axis in enumerate(axes_in_row):
                panel_label = chr(ord("a") + row_index * len(columns) + column_index)
                fig.text(
                    axis.get_position().x0,
                    label_y,
                    f"({panel_label})",
                    ha="left",
                    va="bottom",
                    fontsize=23,
                    fontweight="bold",
                )
        header_y = max(axis.get_position().y1 for axis in panel_axes[0]) + 0.038
        for column_index, (_, _, column_title) in enumerate(columns):
            axis = panel_axes[0][column_index]
            bounds = axis.get_position()
            fig.text(
                (bounds.x0 + bounds.x1) / 2.0,
                header_y,
                column_title,
                ha="center",
                va="bottom",
                fontsize=24,
                fontweight="bold",
            )

        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output
